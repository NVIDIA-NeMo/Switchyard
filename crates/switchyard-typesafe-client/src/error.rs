// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;

/// Errors constructing a [`crate::TypeSafeHttpClient`].
///
/// These are configuration-time failures (a missing or empty credential, an
/// invalid base URL) — never surfaced from `classify`, which folds every
/// runtime failure into `switchyard_libsy::TypeSafeProviderError` instead.
#[derive(Debug)]
pub enum TypeSafeClientError {
    /// The named environment variable was not set.
    MissingApiKey {
        /// The environment variable that was read.
        variable: String,
    },
    /// The named environment variable was set to an empty (or whitespace-only) value.
    EmptyApiKey {
        /// The environment variable that was read.
        variable: String,
    },
    /// The supplied base URL could not be parsed or did not use HTTPS.
    InvalidBaseUrl {
        /// The value that failed to parse.
        base_url: String,
        /// The underlying parse error, rendered to text.
        reason: String,
    },
}

impl fmt::Display for TypeSafeClientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingApiKey { variable } => {
                write!(f, "environment variable {variable} is not set")
            }
            Self::EmptyApiKey { variable } => {
                write!(f, "environment variable {variable} is empty")
            }
            Self::InvalidBaseUrl { base_url, reason } => {
                write!(f, "base_url {base_url:?} is not a valid URL: {reason}")
            }
        }
    }
}

impl std::error::Error for TypeSafeClientError {}
