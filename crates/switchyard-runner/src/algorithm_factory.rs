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
                let setup = entry?;
                Ok((setup.config.id.to_string(), (setup.algorithm, setup.models)))
            })
            .collect()
    }

    /// Builds reusable algorithms keyed by route ID that check target requirements.
    /// The host handles model/Decision calls, approved answer/fallback targets, prompts,
    /// provider settings, credentials and transport. No provider clients are built or credentials read.
    pub fn build_configured(&self) -> Result<BTreeMap<String, ConfiguredAlgorithm>, RunnerError> {
        self.config.validate()?;
        self.config.validate_target_approvals()?;
        self.config
            .build_algorithms()
            .map(|entry| {
                let mut setup = entry?;
                let policy = self.config.eligibility_policy(setup.route_name);
                if policy.is_some() {
                    self.config.validate_decision_models(setup.route_name)?;
                    setup.models = setup.models.with_target_restriction();
                }
                Ok((
                    setup.config.id.to_string(),
                    ConfiguredAlgorithm {
                        standard: AlgorithmSetup {
                            algorithm: setup.algorithm,
                            models: Arc::new(setup.models),
                        },
                        eligibility: policy.map(|policy| ConfiguredEligibility {
                            policy,
                            deployment: Arc::clone(&self.config),
                            route_name: setup.route_name.to_owned(),
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

impl ConfiguredEligibility {
    fn select(
        &self,
        request: &Request,
        standard: &AlgorithmSetup,
    ) -> Result<AlgorithmSetup, RunnerError> {
        let requirements = self.policy.admit(request)?;
        let Some(bindings) = self
            .deployment
            .select_eligible_targets(&self.route_name, requirements.required_labels())?
        else {
            return Ok(standard.clone());
        };
        // No model I/O under this lock.
        let mut setups = self.setups.lock();
        if let Some(setup) = setups.get(&bindings) {
            return Ok(setup.clone());
        }
        if setups.len() >= MAX_ELIGIBLE_LANES {
            return Err(LlmClientError::TemporarilyUnavailable.into());
        }
        let (algorithm, models) = self
            .deployment
            .build_bound_algorithm(&self.route_name, &bindings)?;
        let setup = AlgorithmSetup {
            algorithm,
            models: Arc::new(models),
        };
        setups.insert(bindings, setup.clone());
        Ok(setup)
    }
}

/// A reusable algorithm with target requirement checks.
pub struct ConfiguredAlgorithm {
    standard: AlgorithmSetup,
    eligibility: Option<ConfiguredEligibility>,
}

impl ConfiguredAlgorithm {
    /// Checks requirements before yielding model or Decision calls.
    /// Invalid requirements or unavailable approved targets fail before the algorithm starts.
    /// Requires Tokio. Reuse the handle to retain task requirements and algorithm-owned state;
    /// `state` is per-run routing state. Answer/fallback attempts must stay within returned candidates.
    pub fn run_stream(&self, request: Request, state: State) -> Result<StepStream, RunnerError> {
        let setup = match &self.eligibility {
            Some(eligibility) => eligibility.select(&request, &self.standard)?,
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
