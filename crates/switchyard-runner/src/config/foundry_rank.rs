// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Foundry Rank validation that depends on the deployment's routes and targets.

use std::collections::BTreeSet;

use switchyard_protocol::Category;

use super::{DecisionClientConfig, DeploymentConfig, RouteConfig, RunnerResult};
use crate::{AlgorithmSpec, ClassifierMode, RunnerError};

impl DeploymentConfig {
    pub(super) fn validate_foundry_rank(
        &self,
        name: &str,
        route: &RouteConfig,
    ) -> RunnerResult<()> {
        let judge_names = route.algorithm.decision_target_names();
        let foundry_versions = |target: &str| {
            let target = self.decision_targets.get(target)?;
            match self.decision_clients.get(&target.decision_client)? {
                DecisionClientConfig::FoundryRank { model_versions, .. } => Some(model_versions),
                _ => None,
            }
        };
        let invalid = |message: &str| {
            RunnerError::configuration(format!("route {name}: Foundry Rank {message}"))
        };
        if let Some(eligibility) = &route.eligibility
            && eligibility
                .target_alternatives
                .values()
                .flatten()
                .any(|target| foundry_versions(target).is_some())
        {
            return Err(invalid("does not support decision target alternatives"));
        }
        let Some(versions) = judge_names.iter().find_map(|judge| foundry_versions(judge)) else {
            return Ok(());
        };
        let AlgorithmSpec::LlmClassifier { config } = &route.algorithm else {
            return Err(invalid(
                "requires a standalone llm_classifier in direct mode",
            ));
        };
        let Some(models) = config
            .models
            .as_ref()
            .filter(|_| matches!(config.mode, Some(ClassifierMode::Direct)))
        else {
            return Err(invalid(
                "requires a standalone llm_classifier in direct mode",
            ));
        };
        if judge_names.len() != 1 || route.eligibility.is_some() {
            return Err(invalid(
                "requires one judge and does not support route eligibility",
            ));
        }
        let mut mapped_targets = BTreeSet::new();
        let mut mapped_bindings = BTreeSet::new();
        for (category, targets) in models.groups() {
            if targets.is_empty() || matches!(category, Category::Judge | Category::Any) {
                continue;
            }
            let id = category.as_str();
            if id.trim() != id
                || id.to_ascii_lowercase() != id
                || versions
                    .get(id)
                    .is_none_or(|v| v.is_empty() || v.trim() != v)
            {
                return Err(invalid(
                    "requires lowercase canonical model group IDs with configured versions",
                ));
            }
            let [target] = targets.as_slice() else {
                return Err(invalid("requires one execution target per model group"));
            };
            // Different target aliases can still point to the same upstream deployment.
            let binding = self.targets.get(target).ok_or_else(|| {
                RunnerError::configuration(format!(
                    "route {name} references unknown target {target}"
                ))
            })?;
            if !mapped_targets.insert(target.clone())
                || !mapped_bindings.insert((&binding.llm_client, &binding.id))
            {
                return Err(invalid(
                    "requires a distinct execution target per model group",
                ));
            }
        }
        if models
            .get(&Category::Any)
            .iter()
            .any(|target| !mapped_targets.contains(target))
        {
            return Err(invalid(
                "requires every models.any target to have a canonical model group",
            ));
        }
        Ok(())
    }
}
