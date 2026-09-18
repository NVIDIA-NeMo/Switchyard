// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Media payloads shared by the OpenAI Chat and Responses codecs.

use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::{Map, Value, json};

use crate::error::{Result, TranslationError};
use crate::format::WireFormat;
use crate::llm::{ContentBlock, FileSource, ImageSource, MediaSource};
use crate::util::json_string;

pub(super) struct ImagePayload {
    pub(super) url: String,
    pub(super) detail: Option<String>,
}

pub(super) fn image_payload(source: &ImageSource) -> Option<ImagePayload> {
    match source {
        ImageSource::Url { url, detail } => Some(ImagePayload {
            url: url.clone(),
            detail: detail.clone(),
        }),
        ImageSource::Base64 { media_type, data } => {
            media_type.as_ref().map(|media_type| ImagePayload {
                url: format!("data:{media_type};base64,{data}"),
                detail: None,
            })
        }
        ImageSource::Raw(raw) => raw_image_payload(raw),
    }
}

// Recognizes common raw image shapes emitted by Anthropic and Responses.
fn raw_image_payload(raw: &Value) -> Option<ImagePayload> {
    let object = raw.as_object()?;
    let object = if object.get("type").and_then(Value::as_str) == Some("image") {
        let source = object.get("source").and_then(Value::as_object)?;
        if !matches!(
            source.get("type").and_then(Value::as_str),
            Some("base64" | "url")
        ) {
            return None;
        }
        source
    } else {
        object
    };
    if let Some(url) = object.get("url").and_then(Value::as_str) {
        return Some(ImagePayload {
            url: url.to_string(),
            detail: None,
        });
    }
    if let Some(url) = object.get("image_url").and_then(Value::as_str) {
        return Some(ImagePayload {
            url: url.to_string(),
            detail: None,
        });
    }
    let data = object.get("data").and_then(Value::as_str)?;
    let media_type = object
        .get("media_type")
        .and_then(Value::as_str)
        .unwrap_or("application/octet-stream");
    Some(ImagePayload {
        url: format!("data:{media_type};base64,{data}"),
        detail: None,
    })
}

pub(super) fn image_source_text(source: &ImageSource) -> String {
    match source {
        ImageSource::Url { url, detail } => json_string(&json!({
            "url": url,
            "detail": detail,
        })),
        ImageSource::Base64 { media_type, data } => json_string(&json!({
            "media_type": media_type,
            "data": data,
        })),
        ImageSource::Raw(raw) => json_string(raw),
    }
}

pub(super) fn file_payload(source: &FileSource) -> Option<Map<String, Value>> {
    match source {
        FileSource::FileId(file_id) => {
            let mut payload = Map::new();
            payload.insert("file_id".to_string(), Value::String(file_id.to_string()));
            Some(payload)
        }
        FileSource::FileData { data, filename } => {
            Some(file_data_payload(data, filename.as_deref()))
        }
        FileSource::Raw(raw) => raw_file_payload(raw),
    }
}

fn file_data_payload(data: &str, filename: Option<&str>) -> Map<String, Value> {
    let mut payload = Map::new();
    payload.insert("file_data".to_string(), Value::String(data.to_string()));
    if let Some(filename) = filename {
        payload.insert("filename".to_string(), Value::String(filename.to_string()));
    }
    payload
}

// Maps portable fields from raw Anthropic documents without forwarding provider-managed IDs.
fn raw_file_payload(raw: &Value) -> Option<Map<String, Value>> {
    let block = raw.as_object()?;
    if block.get("type").and_then(Value::as_str) == Some("input_file") {
        let mut payload = block.clone();
        payload.remove("type");
        return Some(payload);
    }
    if block.get("type").and_then(Value::as_str) != Some("document") {
        return None;
    }
    let source = block.get("source").and_then(Value::as_object)?;
    let filename = block.get("title").and_then(Value::as_str);
    match source.get("type").and_then(Value::as_str)? {
        "url" => {
            let mut payload = Map::new();
            payload.insert("file_url".into(), source.get("url")?.clone());
            if let Some(filename) = filename {
                payload.insert("filename".into(), filename.into());
            }
            Some(payload)
        }
        "text" => Some(file_data_payload(
            &format!(
                "data:text/plain;base64,{}",
                STANDARD.encode(source.get("data")?.as_str()?)
            ),
            filename,
        )),
        "base64" => Some(file_data_payload(
            &format!(
                "data:{};base64,{}",
                source.get("media_type")?.as_str()?,
                source.get("data")?.as_str()?
            ),
            filename,
        )),
        _ => None,
    }
}

pub(super) fn audio_part(source: &MediaSource) -> Result<Value> {
    match source {
        MediaSource::Raw(raw)
            if raw.get("type").and_then(Value::as_str) == Some("input_audio")
                && raw["input_audio"]["data"].is_string()
                && matches!(raw["input_audio"]["format"].as_str(), Some("wav" | "mp3")) =>
        {
            Ok(raw.clone())
        }
        MediaSource::Base64 {
            media_type: Some(media_type),
            data,
        } => {
            let format = match media_type.as_str() {
                "audio/wav" => "wav",
                "audio/mpeg" => "mp3",
                _ => {
                    return Err(TranslationError::LossyConversion(
                        "unsupported input audio media type".into(),
                    ));
                }
            };
            Ok(json!({"type": "input_audio", "input_audio": {"data": data, "format": format}}))
        }
        MediaSource::Url { .. } => Err(TranslationError::LossyConversion(
            "OpenAI input audio requires base64 data, not a URL; provide MediaSource::Base64 instead".into(),
        )),
        _ => Err(TranslationError::LossyConversion(
            "unsupported input audio source".into(),
        )),
    }
}

// File IDs belong to their provider; a target cannot resolve another provider's image ID.
pub(super) fn validate_media(block: &ContentBlock, target: WireFormat) -> Result<()> {
    let is_unsupported = match block {
        ContentBlock::Image {
            source: ImageSource::Raw(raw),
        } => raw.get("file_id").is_some() && target != WireFormat::OpenAiResponses,
        ContentBlock::Audio { .. } => target == WireFormat::AnthropicMessages,
        ContentBlock::File {
            source: FileSource::FileId(_),
        } => target == WireFormat::AnthropicMessages,
        ContentBlock::File {
            source: FileSource::Raw(raw),
        } if target == WireFormat::OpenAiChat => {
            raw.get("file_url").is_some()
                || (raw.get("type").and_then(Value::as_str) == Some("document")
                    && raw["source"]["type"].as_str() == Some("url"))
        }
        ContentBlock::ToolResult(result) => {
            for block in &result.content {
                validate_media(block, target)?;
            }
            false
        }
        _ => false,
    };
    if is_unsupported {
        return Err(TranslationError::LossyConversion(format!(
            "input media cannot be represented in {target}"
        )));
    }
    Ok(())
}

pub(super) fn file_source_text(source: &FileSource) -> String {
    match source {
        FileSource::FileId(file_id) => json_string(&json!({"file_id": file_id})),
        FileSource::FileData { data, filename } => json_string(&json!({
            "file_data": data,
            "filename": filename,
        })),
        FileSource::Raw(raw) => json_string(raw),
    }
}
