// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Algorithm construction from deployment TOML without provider clients.

use std::collections::BTreeMap;
use std::sync::Arc;

use libsy::{Algorithm, RuntimeModels};

use crate::RunnerError;
use crate::config::DeploymentConfig;

/// An algorithm and the model groups needed to drive its configured routes.
pub struct ConfiguredAlgorithm {
    /// The routing algorithm; reuse it to retain algorithm-owned session state.
    pub algorithm: Arc<dyn Algorithm>,
    /// Ordered model groups for parent and subagent calls.
    pub models: RuntimeModels,
}

/// Parsed deployment configuration for building algorithms with host-owned execution.
pub struct AlgorithmFactory {
    pub(crate) config: DeploymentConfig,
}

impl AlgorithmFactory {
    /// Parses deployment TOML without resolving credentials or constructing provider clients.
    /// Invalid TOML returns a configuration error; routing settings are checked by [`Self::build`].
    pub fn from_toml(source: &str) -> Result<Self, RunnerError> {
        let config = toml::from_str(source).map_err(|error| {
            RunnerError::configuration_source(format!("failed to parse TOML: {error}"), error)
        })?;
        Ok(Self { config })
    }

    /// Builds algorithms and their model groups, keyed by configured route `id`.
    ///
    /// The host serves model calls and owns target prompts, provider settings and transport.
    /// Reuse the returned algorithms across requests to retain algorithm-owned state.
    pub fn build(&self) -> Result<BTreeMap<String, ConfiguredAlgorithm>, RunnerError> {
        self.config.validate()?;
        self.config
            .build_algorithms()
            .map(|entry| {
                let (_, config, algorithm) = entry?;
                Ok((config.id.to_string(), algorithm))
            })
            .collect()
    }
}
