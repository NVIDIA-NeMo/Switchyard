// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CPU image preparation and bounded FFmpeg frame extraction.

use std::io::Cursor;
use std::process::Stdio;
use std::sync::Arc;

use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader, metadata::Orientation};
use serde_json::Value;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::sync::Semaphore;

use crate::{MediaError, Result};

const MAX_DIMENSION: u32 = 16_384;
const MAX_ALLOC: u64 = 128 * 1024 * 1024;

pub(crate) async fn resize(
    bytes: Vec<u8>,
    edge: u32,
    slots: Arc<Semaphore>,
) -> Result<(&'static str, Vec<u8>)> {
    let permit = slots
        .acquire_owned()
        .await
        .map_err(|_| MediaError::Invalid("media worker closed"))?;
    tokio::task::spawn_blocking(move || {
        // Keep the permit in the worker even if the caller cancels its wait.
        let _permit = permit;
        // Adapted from Dynamo's ImageReaderBackend; see README.md for provenance.
        let mut reader = ImageReader::new(Cursor::new(&bytes)).with_guessed_format()?;
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(MAX_DIMENSION);
        limits.max_image_height = Some(MAX_DIMENSION);
        limits.max_alloc = Some(MAX_ALLOC);
        reader.limits(limits);
        let format = reader
            .format()
            .ok_or(MediaError::Invalid("unsupported image format"))?;
        let mut decoder = reader.into_decoder()?;
        if decoder.total_bytes() > MAX_ALLOC {
            return Err(MediaError::Invalid("image exceeds decode allocation limit"));
        }
        let orientation = decoder.orientation()?;
        let mut decoded = DynamicImage::from_decoder(decoder)?;
        if decoded.width() <= edge
            && decoded.height() <= edge
            && orientation == Orientation::NoTransforms
        {
            return Ok((format.to_mime_type(), bytes));
        }
        decoded.apply_orientation(orientation);
        let resized = if decoded.width() > edge || decoded.height() > edge {
            decoded.resize(edge, edge, image::imageops::FilterType::Triangle)
        } else {
            decoded
        };
        let mut output = Cursor::new(Vec::new());
        let mime = if resized.color().has_alpha() {
            resized.write_to(&mut output, ImageFormat::Png)?;
            "image/png"
        } else {
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut output, 85)
                .encode_image(&resized.to_rgb8())?;
            "image/jpeg"
        };
        Ok((mime, output.into_inner()))
    })
    .await
    .map_err(|_| MediaError::Invalid("image worker failed"))?
}

// Adapted from Dynamo's get_target_times. Avoid seeking past the last frame.
fn sample_times(count: usize, duration: f64, fps: f64) -> Result<Vec<f64>> {
    if count == 0 || !duration.is_finite() || duration <= 0.0 || !fps.is_finite() || fps <= 0.0 {
        return Err(MediaError::Invalid(
            "video has invalid duration or frame rate",
        ));
    }
    let last = (duration - 1.0 / fps - 0.001).max(0.0);
    Ok(if count == 1 {
        vec![last / 2.0]
    } else {
        (0..count)
            .map(|index| index as f64 * last / (count - 1) as f64)
            .collect()
    })
}

// Limit stdout while reading, and kill the child on timeout, error or cancellation.
async fn command_output(command: &mut Command, limit: usize) -> Result<Vec<u8>> {
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .map_err(|_| {
            MediaError::Invalid("cannot start FFmpeg/FFprobe; install both executables on PATH")
        })?;
    let mut output = Vec::new();
    child
        .stdout
        .take()
        .ok_or(MediaError::Invalid("missing media process output"))?
        .take(limit as u64 + 1)
        .read_to_end(&mut output)
        .await?;
    if output.len() > limit {
        return Err(MediaError::Invalid("media process output exceeds limit"));
    }
    if !child.wait().await?.success() {
        return Err(MediaError::Invalid("FFmpeg/FFprobe rejected the video"));
    }
    Ok(output)
}

pub(crate) async fn frames(
    bytes: Vec<u8>,
    count: usize,
    keep: usize,
    edge: u32,
    slots: Arc<Semaphore>,
    mut remaining_bytes: usize,
) -> Result<Vec<(f64, Vec<u8>)>> {
    let _permit = slots
        .acquire_owned()
        .await
        .map_err(|_| MediaError::Invalid("media worker closed"))?;
    let directory = tempfile::tempdir()?;
    let input = directory.path().join("input.video");
    tokio::fs::write(&input, bytes).await?;
    let probe = command_output(
        Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-protocol_whitelist",
                "file",
                "-format_whitelist",
                "mov,matroska,webm",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate:format=duration",
                "-of",
                "json",
            ])
            .arg(&input),
        16 * 1024,
    )
    .await?;
    let probe: Value = serde_json::from_slice(&probe)
        .map_err(|_| MediaError::Invalid("invalid FFprobe output"))?;
    let stream = &probe["streams"][0];
    let width = stream["width"].as_u64().unwrap_or(0);
    let height = stream["height"].as_u64().unwrap_or(0);
    if width == 0
        || height == 0
        || width > MAX_DIMENSION as u64
        || height > MAX_DIMENSION as u64
        || width * height > MAX_ALLOC / 4
    {
        return Err(MediaError::Invalid("video dimensions exceed decode limits"));
    }
    let duration = probe["format"]["duration"]
        .as_str()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0.0);
    let fps = stream["avg_frame_rate"]
        .as_str()
        .and_then(|rate| rate.split_once('/'))
        .and_then(|(numerator, denominator)| {
            Some(numerator.parse::<f64>().ok()? / denominator.parse::<f64>().ok()?)
        })
        .unwrap_or(0.0);
    let times = sample_times(count, duration, fps)?;
    let mut frames = Vec::with_capacity(keep);
    for time in times.into_iter().skip(count - keep) {
        let seek = format!("{time:.6}");
        let scale = format!(
            "scale=w='min(iw,{edge})':h='min(ih,{edge})':force_original_aspect_ratio=decrease"
        );
        let mut command = Command::new("ffmpeg");
        command
            .args([
                "-nostdin",
                "-v",
                "error",
                "-threads",
                "1",
                "-protocol_whitelist",
                "file",
                "-format_whitelist",
                "mov,matroska,webm",
                "-ss",
                &seek,
                "-i",
            ])
            .arg(&input)
            .args([
                "-map",
                "0:v:0",
                "-frames:v",
                "1",
                "-vf",
                &scale,
                "-threads",
                "1",
                "-f",
                "image2pipe",
                "-c:v",
                "mjpeg",
                "-q:v",
                "3",
                "pipe:1",
            ]);
        let jpeg = command_output(&mut command, remaining_bytes.min(8 * 1024 * 1024)).await?;
        if jpeg.is_empty() {
            return Err(MediaError::Invalid("video sample could not be decoded"));
        }
        remaining_bytes = remaining_bytes
            .checked_sub(jpeg.len())
            .ok_or(MediaError::Invalid(
                "prepared media exceeds max_output_bytes",
            ))?;
        frames.push((time, jpeg));
    }
    Ok(frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_covers_clip_and_centers_one_frame() {
        let times = sample_times(3, 7.0, 29.0).unwrap();
        assert_eq!(times[0], 0.0);
        assert!(times[2] > 6.9 && times[2] < 7.0);
        assert_eq!(sample_times(1, 7.0, 29.0).unwrap()[0], times[1]);
        for duration in [0.0, f64::NAN, f64::INFINITY] {
            assert!(sample_times(1, duration, 29.0).is_err());
        }
    }
}
