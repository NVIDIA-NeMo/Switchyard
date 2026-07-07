// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wire-format identifiers understood by the translation engine.

use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::fmt;

/// Built-in provider API formats.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum WireFormat {
    #[serde(rename = "openai_chat")]
    OpenAiChat,
    #[serde(rename = "anthropic_messages")]
    AnthropicMessages,
    #[serde(rename = "openai_responses")]
    OpenAiResponses,
    #[serde(rename = "gemini_generate_content")]
    GeminiGenerateContent,
}

impl WireFormat {
    /// Returns the stable string identifier for a built-in format.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OpenAiChat => "openai_chat",
            Self::AnthropicMessages => "anthropic_messages",
            Self::OpenAiResponses => "openai_responses",
            Self::GeminiGenerateContent => "gemini_generate_content",
        }
    }
}

impl fmt::Display for WireFormat {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Extensible wire-format identifier used by codec registries.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FormatId(String);

impl FormatId {
    /// Creates a format identifier from an arbitrary string.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Creates a format identifier for a built-in format.
    pub fn known(format: WireFormat) -> Self {
        Self(format.as_str().to_string())
    }

    /// Returns the format identifier as a borrowed string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for FormatId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl From<WireFormat> for FormatId {
    fn from(format: WireFormat) -> Self {
        Self::known(format)
    }
}

impl From<&WireFormat> for FormatId {
    fn from(format: &WireFormat) -> Self {
        Self::known(*format)
    }
}

impl From<&str> for FormatId {
    fn from(id: &str) -> Self {
        Self::new(id)
    }
}

impl From<String> for FormatId {
    fn from(id: String) -> Self {
        Self::new(id)
    }
}

impl From<&String> for FormatId {
    fn from(id: &String) -> Self {
        Self::new(id.clone())
    }
}

impl From<Cow<'_, str>> for FormatId {
    fn from(id: Cow<'_, str>) -> Self {
        Self::new(id.into_owned())
    }
}
