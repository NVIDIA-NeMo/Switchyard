// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Profile registry used by the components-v2 Rust server.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use switchyard_components_v2::{
    DecisionContext, PassthroughProfileConfig, Profile, ProfileConfig, ProfileConfigPlan,
    RoutingDecision,
};
use switchyard_core::{ModelId, ProfileId, Result, SwitchyardError};

/// Public model entry advertised by `/v1/models`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ServedModel {
    /// Public model or route ID accepted in inbound request bodies.
    pub id: ModelId,
    /// Human-readable label shown by CLI startup logs and `/v1/models`.
    pub display_name: String,
}

#[derive(Clone)]
struct RegistryEntry {
    model: ServedModel,
    profile: Arc<dyn Profile>,
}

#[derive(Clone)]
struct DecisionProfileEntry {
    id: ProfileId,
    profile: Arc<dyn Profile>,
}

/// Shared registry for model dispatch and profile-ID decision lookup.
///
/// Configured profile runtimes are built once and the same [`Arc`] is used by
/// both paths so stateful routing policy cannot diverge between endpoints.
#[derive(Clone, Default)]
pub struct ProfileRegistry {
    entries: Vec<RegistryEntry>,
    decision_profiles: Vec<DecisionProfileEntry>,
}

impl ProfileRegistry {
    /// Builds a registry from already-built exact-match profile runtimes.
    pub fn from_profiles(
        entries: impl IntoIterator<Item = (ModelId, Arc<dyn Profile>, String)>,
    ) -> Result<Self> {
        let mut registry = Self::default();
        for (model_id, profile, display_name) in entries {
            registry.insert_decision_profile(
                ProfileId::new(model_id.as_str())?,
                Arc::clone(&profile),
            )?;
            registry.insert(model_id, profile, display_name)?;
        }
        Ok(registry)
    }

    /// Builds a registry from a resolved profile config plan.
    pub fn from_plan(plan: &ProfileConfigPlan) -> Result<Self> {
        let mut registry = Self::default();

        for profile_id in plan.profile_ids() {
            let model_id = ModelId::new(profile_id.as_str())?;
            let profile: Arc<dyn Profile> = Arc::from(plan.build_profile(profile_id)?);
            let display_name = plan.profile_type(profile_id).unwrap_or("profile");
            registry.insert_decision_profile(profile_id.clone(), Arc::clone(&profile))?;
            registry.insert(model_id, profile, display_name)?;
        }

        for (_target_id, target) in plan.targets() {
            let profile: Arc<dyn Profile> = Arc::from(
                PassthroughProfileConfig {
                    target: target.clone(),
                }
                .build_boxed()?,
            );
            registry.insert(
                ModelId::new(target.id.as_str())?,
                Arc::clone(&profile),
                target.model.as_str(),
            )?;
            if target.id.as_str() != target.model.as_str() {
                registry.insert(
                    target.model.clone(),
                    profile,
                    format!("target {}", target.id.as_str()),
                )?;
            }
        }

        Ok(registry)
    }

    /// Returns the profile for an inbound request model.
    pub fn lookup(&self, model: Option<&str>) -> Result<Arc<dyn Profile>> {
        let Some(model) = model else {
            return Err(SwitchyardError::InvalidRequest(
                "request body must include a non-empty string `model`".to_string(),
            ));
        };
        let model = ModelId::new(model)?;
        self.entries
            .iter()
            .find(|entry| entry.model.id == model)
            .map(|entry| Arc::clone(&entry.profile))
            .ok_or(SwitchyardError::ModelNotFound { model })
    }

    /// Produces a decision through the configured profile runtime with the requested ID.
    pub async fn decide(&self, context: DecisionContext) -> Result<RoutingDecision> {
        context.request().validate()?;
        let profile_id = context.request().decision_profile.profile_id.clone();
        let profile = self
            .decision_profiles
            .iter()
            .find(|entry| entry.id == profile_id)
            .map(|entry| Arc::clone(&entry.profile))
            .ok_or(SwitchyardError::DecisionProfileNotFound { profile_id })?;
        profile.decide(context).await
    }

    /// Returns model entries in deterministic registration order.
    pub fn served_models(&self) -> Vec<ServedModel> {
        self.entries
            .iter()
            .map(|entry| entry.model.clone())
            .collect()
    }

    fn insert(
        &mut self,
        model_id: ModelId,
        profile: Arc<dyn Profile>,
        display_name: impl Into<String>,
    ) -> Result<()> {
        if self.entries.iter().any(|entry| entry.model.id == model_id) {
            return Err(SwitchyardError::DuplicateRegistration {
                kind: "model",
                id: model_id.to_string(),
            });
        }
        self.entries.push(RegistryEntry {
            model: ServedModel {
                id: model_id,
                display_name: display_name.into(),
            },
            profile,
        });
        Ok(())
    }

    fn insert_decision_profile(
        &mut self,
        profile_id: ProfileId,
        profile: Arc<dyn Profile>,
    ) -> Result<()> {
        if self
            .decision_profiles
            .iter()
            .any(|entry| entry.id == profile_id)
        {
            return Err(SwitchyardError::DuplicateRegistration {
                kind: "profile",
                id: profile_id.to_string(),
            });
        }
        self.decision_profiles.push(DecisionProfileEntry {
            id: profile_id,
            profile,
        });
        Ok(())
    }
}
