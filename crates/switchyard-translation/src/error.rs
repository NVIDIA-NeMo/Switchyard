// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Error types for request, response, and streaming translation.

use thiserror::Error;

use crate::format::FormatId;

/// Result alias for translation operations.
pub type Result<T> = std::result::Result<T, TranslationError>;

/// Failures that can occur while decoding, encoding, or enforcing policy.
#[derive(Debug, Error)]
pub enum TranslationError {
    #[error("invalid JSON: {0}")]
    InvalidJson(#[from] serde_json::Error),

    #[error("expected {expected} at {path}")]
    InvalidType {
        path: String,
        expected: &'static str,
    },

    #[error("translation from {from} to {to} is not supported")]
    UnsupportedTranslation { from: FormatId, to: FormatId },

    #[error("lossy conversion rejected: {0}")]
    LossyConversion(String),

    #[error("unknown field rejected at {path}")]
    UnknownField { path: String },

    #[error("invalid value at {path}: {message}")]
    InvalidValue { path: String, message: String },

    #[error("provider-owned {kind} cannot be translated without target affinity")]
    NonPortableReference { kind: &'static str },

    #[error("{0}")]
    Other(String),
}

impl TranslationError {
    /// Returns the stable variant name for FFI and language-binding errors.
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::InvalidJson(_) => "InvalidJson",
            Self::InvalidType { .. } => "InvalidType",
            Self::UnsupportedTranslation { .. } => "UnsupportedTranslation",
            Self::LossyConversion(_) => "LossyConversion",
            Self::UnknownField { .. } => "UnknownField",
            Self::InvalidValue { .. } => "InvalidValue",
            Self::NonPortableReference { .. } => "NonPortableReference",
            Self::Other(_) => "Other",
        }
    }

    /// Builds an [`TranslationError::InvalidValue`] for an unsupported message
    /// role. A transparent router must reject the same payloads the upstream
    /// provider would: an unknown role string (e.g. `"api"`) should surface as
    /// an `invalid_value`-style error rather than being silently coerced to
    /// `user`. `path` is a JSON-path pointer to the offending field so the
    /// error reads like a provider validation error.
    pub fn unsupported_role(path: impl Into<String>, value: &str) -> Self {
        Self::InvalidValue {
            path: path.into(),
            message: format!(
                "Invalid value: {value:?}. Supported message roles are \
                 system, developer, user, assistant, tool."
            ),
        }
    }
}
