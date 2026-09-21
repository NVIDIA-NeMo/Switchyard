// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prepare media in an owned outgoing provider body without modifying the routing request.

mod decode;
mod fetch;

use std::sync::Arc;
use std::time::Duration;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use futures_util::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use switchyard_protocol::WireFormat;
use tokio::sync::Semaphore;

/// Representation expected by the target endpoint.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VideoMode {
    /// Leave the encoded video block unchanged.
    #[default]
    Passthrough,
    /// Replace video with timestamped JPEG frames.
    Frames,
    /// Emit a Chat Completions `video_url` block.
    VideoUrl,
    /// Emit a Gemini-on-Hub Chat `file` block with a video MIME type.
    File,
    /// Replace videos with a text marker, without downloading them.
    Omit,
}

/// Optional per-target media settings. All limits apply independently to each call.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct MediaConfig {
    /// Downscale still images to fit this edge, preserving aspect ratio. Never upscale.
    pub image_max_edge: Option<u32>,
    /// Keep the newest N image blocks (including extracted frames) in request order.
    pub max_images: Option<usize>,
    /// Target video representation.
    pub video: VideoMode,
    /// Uniform samples per video, including clip endpoints; one sample uses the midpoint.
    pub video_max_frames: usize,
    /// Maximum edge for extracted video frames. Never upscale.
    pub frame_max_edge: u32,
    /// Maximum downloaded or decoded inline bytes per locally processed media source.
    pub max_input_bytes: usize,
    /// Maximum combined bytes of locally prepared images before base64 encoding.
    pub max_output_bytes: usize,
    /// Deadline for all preparation, including downloads, queueing, and video subprocesses.
    pub timeout_ms: u64,
}

impl Default for MediaConfig {
    fn default() -> Self {
        Self {
            image_max_edge: None,
            max_images: None,
            video: VideoMode::Passthrough,
            video_max_frames: 4,
            frame_max_edge: 640,
            max_input_bytes: 32 * 1024 * 1024,
            max_output_bytes: 32 * 1024 * 1024,
            timeout_ms: 30_000,
        }
    }
}

impl MediaConfig {
    /// Validate bounds and native video compatibility at client construction time.
    pub fn validate(&self, format: WireFormat) -> Result<()> {
        if self
            .image_max_edge
            .is_some_and(|edge| !(1..=4096).contains(&edge))
            || !(1..=4096).contains(&self.frame_max_edge)
            || !(1..=64).contains(&self.video_max_frames)
            || self.max_images.is_some_and(|count| count > 128)
            || !(1..=256 * 1024 * 1024).contains(&self.max_input_bytes)
            || !(1..=256 * 1024 * 1024).contains(&self.max_output_bytes)
            || !(1..=300_000).contains(&self.timeout_ms)
        {
            return Err(MediaError::Invalid(
                "invalid media limits: edges 1..4096, frames 1..64, images 0..128, bytes 1..256MiB, timeout 1..300000ms",
            ));
        }
        if matches!(self.video, VideoMode::VideoUrl | VideoMode::File)
            && format != WireFormat::OpenAiChat
        {
            return Err(MediaError::Invalid(
                "video_url and file media modes require an openai_chat target",
            ));
        }
        Ok(())
    }
}

/// Media failures contain no input URLs, credentials, or encoded media.
#[derive(Debug, thiserror::Error)]
pub enum MediaError {
    /// The complete preparation deadline expired.
    #[error("media preparation timed out")]
    Timeout,
    /// Invalid media, endpoint settings, or resource limit.
    #[error("{0}")]
    Invalid(&'static str),
    /// Image decode or encode failure.
    #[error("image decoding or encoding failed")]
    Image(#[from] image::ImageError),
    /// Temporary-file or child-process I/O failure.
    #[error("media I/O failed")]
    Io(#[from] std::io::Error),
}

/// Result of media preparation.
pub type Result<T> = std::result::Result<T, MediaError>;

/// Shared download client and bounded CPU/subprocess concurrency for one LLM client.
pub struct MediaProcessor {
    client: reqwest::Client,
    slots: Arc<Semaphore>,
}

impl MediaProcessor {
    /// Construct a processor with a separate unauthenticated HTTP client.
    pub fn new() -> Result<Self> {
        Ok(Self {
            client: fetch::client()?,
            slots: Arc::new(Semaphore::new(2)),
        })
    }

    /// Prepare an outgoing body. On error discard the body; changes may be partial.
    /// Only message content is traversed. Tools, JSON arguments, and other controls are retained.
    pub async fn prepare(
        &self,
        body: &mut Value,
        format: WireFormat,
        config: &MediaConfig,
    ) -> Result<()> {
        config.validate(format)?;
        tokio::time::timeout(Duration::from_millis(config.timeout_ms), async {
            let mut budget = Budget {
                blocks: 64,
                bytes: config.max_output_bytes,
                images: config.max_images,
            };
            // Visit newest content first; retained blocks keep their original order.
            for key in ["input", "messages", "system"] {
                if let Some(Value::Array(items)) = body.get_mut(key) {
                    self.prepare_items(items, format, config, &mut budget)
                        .await?;
                }
            }
            Ok(())
        })
        .await
        .map_err(|_| MediaError::Timeout)?
    }

    fn prepare_items<'a>(
        &'a self,
        items: &'a mut Vec<Value>,
        format: WireFormat,
        config: &'a MediaConfig,
        budget: &'a mut Budget,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let mut output = Vec::with_capacity(items.len());
            for mut item in std::mem::take(items).into_iter().rev() {
                if matches!(
                    item["type"].as_str(),
                    Some("image_url" | "input_image" | "image")
                ) {
                    if budget.keep_images(1) == 0 {
                        output.push(text_part(format, "[image omitted]"));
                        continue;
                    }
                    if let Some(edge) = config.image_max_edge {
                        let source = image_url(&item)
                            .ok_or(MediaError::Invalid("image source cannot be resized"))?;
                        budget.consume_block()?;
                        let bytes =
                            fetch::load(&self.client, &source, config.max_input_bytes).await?;
                        let (mime, encoded) =
                            decode::resize(bytes, edge, self.slots.clone()).await?;
                        budget.consume_bytes(encoded.len())?;
                        // Retain detail/cache hints and other fields on existing image blocks.
                        replace_image_source(&mut item, &data_uri(mime, &encoded))?;
                    }
                } else if is_video(&item) {
                    match config.video {
                        VideoMode::Passthrough => {}
                        VideoMode::Omit => item = text_part(format, "[video omitted]"),
                        mode => {
                            let keep = if mode == VideoMode::Frames {
                                budget.keep_images(config.video_max_frames)
                            } else {
                                0
                            };
                            if mode == VideoMode::Frames && keep == 0 {
                                output.push(text_part(format, "[video frames omitted]"));
                                continue;
                            }
                            budget.consume_block()?;
                            let (url, mime) = video_source(&item)
                                .ok_or(MediaError::Invalid("unrecognized video source"))?;
                            match mode {
                                VideoMode::Frames => {
                                    let count = config.video_max_frames;
                                    let bytes =
                                        fetch::load(&self.client, &url, config.max_input_bytes)
                                            .await?;
                                    let frames = decode::frames(
                                        bytes,
                                        count,
                                        keep,
                                        config.frame_max_edge,
                                        self.slots.clone(),
                                        budget.bytes,
                                    )
                                    .await?;
                                    // This group is reversed with the surrounding content below.
                                    for (time, jpeg) in frames.into_iter().rev() {
                                        budget.consume_bytes(jpeg.len())?;
                                        output.push(image_part(
                                            format,
                                            &data_uri("image/jpeg", &jpeg),
                                        ));
                                        output.push(text_part(
                                            format,
                                            &format!("[video sample {time:.3}s]"),
                                        ));
                                    }
                                    output.push(text_part(
                                        format,
                                        "[video frames; times are sample positions in seconds]",
                                    ));
                                    continue;
                                }
                                VideoMode::VideoUrl => {
                                    if item["type"] != "video_url" {
                                        item = json!({"type":"video_url","video_url":{"url":url}});
                                    }
                                }
                                VideoMode::File => {
                                    let key = if url.starts_with("data:") {
                                        "file_data"
                                    } else {
                                        "file_id"
                                    };
                                    if item["type"] != "file" {
                                        item =
                                            json!({"type":"file","file":{key:url,"format":mime}});
                                    }
                                }
                                _ => unreachable!(),
                            }
                        }
                    }
                }
                if let Some(key) = child_content_key(&item)
                    && let Some(Value::Array(children)) = item.get_mut(key)
                {
                    self.prepare_items(children, format, config, budget).await?;
                }
                output.push(item);
            }
            output.reverse();
            *items = output;
            Ok(())
        })
    }
}

struct Budget {
    blocks: usize,
    bytes: usize,
    images: Option<usize>,
}
impl Budget {
    fn keep_images(&mut self, requested: usize) -> usize {
        match &mut self.images {
            Some(remaining) => {
                let keep = requested.min(*remaining);
                *remaining -= keep;
                keep
            }
            None => requested,
        }
    }
    fn consume_block(&mut self) -> Result<()> {
        self.blocks = self.blocks.checked_sub(1).ok_or(MediaError::Invalid(
            "too many media sources; maximum 64 per call",
        ))?;
        Ok(())
    }
    fn consume_bytes(&mut self, bytes: usize) -> Result<()> {
        self.bytes = self.bytes.checked_sub(bytes).ok_or(MediaError::Invalid(
            "prepared media exceeds max_output_bytes",
        ))?;
        Ok(())
    }
}

fn child_content_key(item: &Value) -> Option<&'static str> {
    match item.get("type").and_then(Value::as_str) {
        Some("function_call_output") => Some("output"),
        Some("message" | "tool_result") => Some("content"),
        None if item.get("role").is_some() => Some("content"),
        _ => None,
    }
}

fn image_url(item: &Value) -> Option<String> {
    match item.get("type")?.as_str()? {
        "image_url" | "input_image" => item["image_url"]
            .as_str()
            .or_else(|| item["image_url"]["url"].as_str())
            .map(str::to_owned),
        "image" => source_url(&item["source"], "image/png").map(|(url, _)| url),
        _ => None,
    }
}

fn source_url(source: &Value, default_mime: &str) -> Option<(String, String)> {
    let mime = source["media_type"]
        .as_str()
        .unwrap_or(default_mime)
        .to_owned();
    let url = match source["type"].as_str() {
        Some("url") => source["url"].as_str()?.to_owned(),
        Some("base64") => format!("data:{mime};base64,{}", source["data"].as_str()?),
        _ => return None,
    };
    Some((url, mime))
}

fn is_video(item: &Value) -> bool {
    matches!(
        item["type"].as_str(),
        Some("video_url" | "input_video" | "video")
    ) || (item["type"] == "file"
        && item["file"]["format"]
            .as_str()
            .is_some_and(|mime| mime.starts_with("video/")))
}

fn video_source(item: &Value) -> Option<(String, String)> {
    if item["type"] == "video" {
        return source_url(&item["source"], "video/mp4");
    }
    if item["type"] == "file" {
        let file = &item["file"];
        return Some((
            file["file_data"]
                .as_str()
                .or_else(|| file["file_id"].as_str())?
                .to_owned(),
            file["format"].as_str()?.to_owned(),
        ));
    }
    let mime = item["media_type"].as_str().unwrap_or("video/mp4");
    if let Some(url) = item["video_url"]
        .as_str()
        .or_else(|| item["video_url"]["url"].as_str())
    {
        let inferred = if let Some(mime) = item["media_type"].as_str() {
            mime.to_owned()
        } else if let Some((mime, _)) = url
            .strip_prefix("data:")
            .and_then(|rest| rest.split_once(';'))
        {
            mime.to_owned()
        } else {
            let parsed = reqwest::Url::parse(url).ok();
            match parsed
                .as_ref()
                .map(|url| url.path().to_ascii_lowercase())
                .as_deref()
            {
                Some(path) if path.ends_with(".webm") => "video/webm".to_owned(),
                Some(path) if path.ends_with(".mov") => "video/quicktime".to_owned(),
                Some(path) if path.ends_with(".mkv") => "video/x-matroska".to_owned(),
                _ => mime.to_owned(),
            }
        };
        return Some((url.to_owned(), inferred));
    }
    let source = &item["video"];
    let mime = source["media_type"].as_str().unwrap_or(mime);
    Some((
        format!("data:{mime};base64,{}", source["data"].as_str()?),
        mime.to_owned(),
    ))
}

fn data_uri(mime: &str, bytes: &[u8]) -> String {
    format!("data:{mime};base64,{}", STANDARD.encode(bytes))
}

fn text_part(format: WireFormat, text: &str) -> Value {
    json!({"type": if format == WireFormat::OpenAiResponses { "input_text" } else { "text" }, "text":text})
}

fn image_part(format: WireFormat, url: &str) -> Value {
    match format {
        WireFormat::OpenAiChat => json!({"type":"image_url","image_url":{"url":url}}),
        WireFormat::OpenAiResponses => json!({"type":"input_image","image_url":url}),
        WireFormat::AnthropicMessages => {
            let (header, data) = url.split_once(',').expect("generated data URI");
            let mime = header
                .trim_start_matches("data:")
                .trim_end_matches(";base64");
            json!({"type":"image","source":{"type":"base64","media_type":mime,"data":data}})
        }
    }
}

fn replace_image_source(item: &mut Value, url: &str) -> Result<()> {
    match item["type"].as_str() {
        Some("image_url") if item["image_url"].is_object() => item["image_url"]["url"] = url.into(),
        Some("image_url" | "input_image") => item["image_url"] = url.into(),
        Some("image") => {
            item["source"] = image_part(WireFormat::AnthropicMessages, url)["source"].take()
        }
        _ => return Err(MediaError::Invalid("unrecognized image source")),
    }
    Ok(())
}

#[cfg(test)]
mod tests;
