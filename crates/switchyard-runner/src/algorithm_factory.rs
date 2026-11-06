// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Algorithm construction from deployment TOML without provider clients.

use std::collections::BTreeMap;
use std::sync::Arc;

use libsy::{Algorithm, RuntimeModels, State, StepStream};
use parking_lot::Mutex;
use switchyard_protocol::{LlmClientError, Request};

use crate::RunnerError;
use crate::config::{DeploymentConfig, MAX_ELIGIBLE_LANES, TargetBindings};
use crate::eligibility::{EligibilityPolicy, has_target_requirements};

pub(crate) type AlgorithmAndModels = (Arc<dyn Algorithm>, RuntimeModels);

/// Parsed deployment configuration for building algorithms with host-owned execution.
pub struct AlgorithmFactory {
    pub(crate) config: Arc<DeploymentConfig>,
}

impl AlgorithmFactory {
    /// Parses deployment TOML without resolving credentials or constructing provider clients.
    /// Invalid TOML returns a configuration error; routing settings are checked when building.
    pub fn from_toml(source: &str) -> Result<Self, RunnerError> {
        let mut config: DeploymentConfig = toml::from_str(source).map_err(|error| {
            RunnerError::configuration_source(format!("failed to parse TOML: {error}"), error)
        })?;
        config.infer_decision_judges();
        Ok(Self {
            config: Arc::new(config),
        })
    }

    /// Builds algorithms and their model groups, keyed by configured route `id`.
    ///
    /// LLM calls and outcomes use target names; the host resolves their provider IDs and
    /// owns target prompts, provider settings and transport. Decision calls use provider IDs.
    /// Reuse the returned algorithms across requests to retain algorithm-owned state.
    pub fn build(&self) -> Result<BTreeMap<String, AlgorithmAndModels>, RunnerError> {
        self.config.validate()?;
        if self.config.requires_target_admission() {
            return Err(RunnerError::configuration(
                "target eligibility requires Runner admission; use Runner instead of AlgorithmFactory",
            ));
        }
        self.config
            .build_algorithms()
            .map(|entry| {
                let (_, config, algorithm) = entry?;
                Ok((config.id.to_string(), algorithm))
            })
            .collect()
    }

    /// Builds reusable routes that check target requirements before yielding model calls.
    ///
    /// Keys are configured route IDs. The host executes all yielded calls and final answer
    /// candidates, including their prompts, provider settings, credentials and transport.
    /// No provider clients are constructed and no credential values are read.
    pub fn build_configured(&self) -> Result<BTreeMap<String, ConfiguredAlgorithm>, RunnerError> {
        self.config.validate()?;
        self.config.validate_target_approvals()?;
        self.config
            .build_algorithms()
            .map(|entry| {
                let (name, route, (algorithm, mut models)) = entry?;
                let policy = self.config.eligibility_policy(name);
                if policy.is_some() {
                    self.config.validate_decision_models(name)?;
                    models = models.with_target_restriction();
                }
                Ok((
                    route.id.to_string(),
                    ConfiguredAlgorithm {
                        standard: AlgorithmSetup {
                            algorithm,
                            models: Arc::new(models),
                        },
                        eligibility: policy.map(|policy| ConfiguredEligibility {
                            policy,
                            deployment: Arc::clone(&self.config),
                            route_name: name.to_owned(),
                            setups: Mutex::default(),
                        }),
                    },
                ))
            })
            .collect()
    }
}

#[derive(Clone)]
struct AlgorithmSetup {
    algorithm: Arc<dyn Algorithm>,
    models: Arc<RuntimeModels>,
}

struct ConfiguredEligibility {
    policy: EligibilityPolicy,
    deployment: Arc<DeploymentConfig>,
    route_name: String,
    setups: Mutex<BTreeMap<TargetBindings, AlgorithmSetup>>,
}

/// A reusable algorithm that checks target requirements before host-owned model calls.
pub struct ConfiguredAlgorithm {
    standard: AlgorithmSetup,
    eligibility: Option<ConfiguredEligibility>,
}

impl ConfiguredAlgorithm {
    /// Checks requirements before returning model or Decision work for the host to execute.
    ///
    /// Invalid requirements or unavailable approved targets return an error without starting
    /// the algorithm. Requires a Tokio runtime. Reuse this handle to retain task requirements
    /// and algorithm-owned state; `state` supplies libsy's separate per-run routing state.
    /// The host must keep final answer attempts within the returned candidates.
    pub fn run_stream(&self, request: Request, state: State) -> Result<StepStream, RunnerError> {
        let setup = match &self.eligibility {
            Some(eligibility) => {
                let requirements = eligibility.policy.admit(&request)?;
                match eligibility.deployment.resolve_target_bindings(
                    &eligibility.route_name,
                    requirements.required_labels(),
                )? {
                    None => self.standard.clone(),
                    Some(bindings) => {
                        // Building a setup is synchronous; no model I/O occurs under the lock.
                        let mut setups = eligibility.setups.lock();
                        if let Some(setup) = setups.get(&bindings) {
                            setup.clone()
                        } else {
                            if setups.len() >= MAX_ELIGIBLE_LANES {
                                return Err(LlmClientError::TemporarilyUnavailable.into());
                            }
                            let (algorithm, models) = eligibility
                                .deployment
                                .build_bound_algorithm(&eligibility.route_name, &bindings)?;
                            let setup = AlgorithmSetup {
                                algorithm,
                                models: Arc::new(models),
                            };
                            setups.insert(bindings, setup.clone());
                            setup
                        }
                    }
                }
            }
            None if has_target_requirements(&request) => {
                return Err(LlmClientError::InvalidRequest {
                    message: "route does not accept target requirements".into(),
                }
                .into());
            }
            None => self.standard.clone(),
        };
        Ok(setup.algorithm.run_stream(request, setup.models, state))
    }
}
