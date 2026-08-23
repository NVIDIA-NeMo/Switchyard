// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use serde::Deserialize;
use switchyard_protocol::WireFormat;
use switchyard_runner::Runner;

pub(crate) fn protocol_from_call(name: &str) -> Option<WireFormat> {
    match name {
        "openai.chat_completions" => Some(WireFormat::OpenAiChat),
        "openai.responses" => Some(WireFormat::OpenAiResponses),
        "anthropic.messages" => Some(WireFormat::AnthropicMessages),
        _ => None,
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SwitchyardConfig {
    #[serde(default)]
    pub(crate) priority: i32,
    pub(crate) deployment_path: PathBuf,
}

impl SwitchyardConfig {
    pub(crate) fn load_runner(&self) -> Result<Runner, String> {
        Runner::load(&self.deployment_path).map_err(|error| error.to_string())
    }
}
