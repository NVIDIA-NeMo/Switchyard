// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Image and file payloads shared by the OpenAI Chat and Responses codecs.

use serde_json::{Map, Value, json};

use crate::llm::{FileSource, ImageSource};
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
    if block.get("type").and_then(Value::as_str) != Some("document") {
        return None;
    }
    let source = block.get("source").and_then(Value::as_object)?;
    if source.get("type").and_then(Value::as_str) != Some("base64") {
        return None;
    }
    let data = source.get("data").and_then(Value::as_str)?;
    Some(file_data_payload(
        data,
        block.get("title").and_then(Value::as_str),
    ))
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
