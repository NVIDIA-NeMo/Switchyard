// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Text projection for local text-only judges; serving requests retain their attachments.

use switchyard_protocol::ContentBlock;

/// A short marker retains attachment presence without carrying its payload into the judge.
pub(crate) fn attachment_marker(block: &ContentBlock) -> Option<&'static str> {
    match block {
        ContentBlock::Image { .. } => Some("[image attachment]"),
        ContentBlock::Audio { .. } => Some("[audio attachment]"),
        ContentBlock::Video { .. } => Some("[video attachment]"),
        ContentBlock::File { .. } => Some("[file attachment]"),
        ContentBlock::Unknown { .. } => Some("[unsupported content]"),
        _ => None,
    }
}

/// Projects an owned judge view, including nested tool results, without editing the source.
pub(crate) fn project(content: &mut [ContentBlock]) {
    for block in content {
        if let Some(marker) = attachment_marker(block) {
            *block = ContentBlock::Text {
                text: marker.into(),
            };
        } else if let ContentBlock::ToolResult(result) = block {
            project(&mut result.content);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use switchyard_protocol::{ImageSource, ToolResult};

    /// Tool IDs, ordering and the original serving payload survive the judge projection.
    #[test]
    fn replaces_nested_payloads_without_mutating_the_original() {
        let image = ContentBlock::Image {
            source: ImageSource::Base64 {
                media_type: Some("image/png".into()),
                data: "private-payload".into(),
            },
        };
        let original = vec![
            ContentBlock::Text {
                text: "inspect".into(),
            },
            image.clone(),
            ContentBlock::ToolResult(ToolResult {
                tool_call_id: "call-1".into(),
                content: vec![image],
                is_error: Some(false),
            }),
        ];
        let mut projected = original.clone();
        project(&mut projected);
        assert!(matches!(original[1], ContentBlock::Image { .. }));
        assert_eq!(projected[0], original[0]);
        let encoded = serde_json::to_string(&projected).unwrap();
        assert!(!encoded.contains("private-payload"));
        assert_eq!(encoded.matches("[image attachment]").count(), 2);
        assert!(encoded.contains("call-1"));
    }
}
