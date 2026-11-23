// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Errors shared by configured routing and native execution.

use std::error::Error;

use libsy::LibsyError;
use switchyard_protocol::LlmClientError;
use thiserror::Error;

/// Caller credential family required by a forwarded-auth route.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CallerAuthKind {
    Anthropic,
    OpenAi,
}

impl CallerAuthKind {
    /// Stable provider name used by the server compatibility API.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::OpenAi => "openai",
        }
    }

    #[cfg(not(target_os = "wasi"))]
    pub(crate) const fn accepts(self, wire_format: switchyard_protocol::WireFormat) -> bool {
        use switchyard_protocol::WireFormat;

        matches!(
            (self, wire_format),
            (Self::Anthropic, WireFormat::AnthropicMessages)
                | (
                    Self::OpenAi,
                    WireFormat::OpenAiChat | WireFormat::OpenAiResponses
                )
        )
    }
}

/// Error returned while loading or executing configured routes.
#[derive(Debug, Error)]
pub enum RunnerError {
    #[error("{message}")]
    Configuration {
        message: String,
        #[source]
        source: Option<Box<dyn Error + Send + Sync>>,
    },
    #[error("unknown route model {0:?}")]
    UnknownRouteModel(String),
    #[error("caller format is incompatible with {} credentials", .0.as_str())]
    IncompatibleCallerFormat(CallerAuthKind),
    #[error("route has no compatible target for the auxiliary operation")]
    AuxiliaryUnsupported,
    #[error(transparent)]
    Algorithm(#[from] LibsyError),
    #[error(transparent)]
    Client(#[from] LlmClientError),
}

impl RunnerError {
    pub(crate) fn configuration(message: impl Into<String>) -> Self {
        Self::Configuration {
            message: message.into(),
            source: None,
        }
    }

    pub(crate) fn configuration_source(
        message: impl Into<String>,
        source: impl Error + Send + Sync + 'static,
    ) -> Self {
        Self::Configuration {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }
}
