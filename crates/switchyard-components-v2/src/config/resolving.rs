// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Resolution of parsed profile configs into runtime-ready objects.

use std::{collections::BTreeMap, fmt};

use serde_json::Value;
use switchyard_core::{
    EndpointConfig, EndpointId, LlmTarget, LlmTargetId, ProfileId, Result, SwitchyardError,
};

use super::parsing::{ProfileConfigDocument, SerializedProfileConfig, TargetConfig};
use crate::profiles::{ProfileConfigEntry, SubagentOverrideProfile};
use crate::{Profile, ProfileHooks};

impl ProfileConfigDocument {
    /// Resolves endpoints and validates every profile through its owning profile type.
    pub fn resolve(&self) -> Result<ProfileConfigPlan> {
        let targets = self.resolve_targets()?;
        let (profiles, profile_subagent_targets) = self.resolve_profiles(&targets)?;
        Ok(ProfileConfigPlan {
            targets,
            profiles,
            profile_subagent_targets,
        })
    }

    // Resolves all file-facing targets before profile-specific config expansion.
    fn resolve_targets(&self) -> Result<BTreeMap<LlmTargetId, LlmTarget>> {
        let mut targets = BTreeMap::new();
        for (target_id, target) in &self.targets {
            targets.insert(
                target_id.clone(),
                target.resolve(target_id, &self.endpoints)?,
            );
        }
        Ok(targets)
    }

    // Parses profile-specific config bodies into the generated typed profile enum.
    // Also collects resolved sub-agent targets for profiles that declare one.
    fn resolve_profiles(
        &self,
        targets: &BTreeMap<LlmTargetId, LlmTarget>,
    ) -> Result<(
        BTreeMap<ProfileId, ProfileConfigEntry>,
        BTreeMap<ProfileId, LlmTarget>,
    )> {
        let env = ProfileBuildEnv::new(targets);
        let mut profiles = BTreeMap::new();
        let mut subagent_targets: BTreeMap<ProfileId, LlmTarget> = BTreeMap::new();
        for (profile_id, profile) in &self.profiles {
            // The envelope-level sub-agent override must name a resolvable target;
            // an unknown reference is a startup configuration error.
            if let Some(target_id) = profile.subagent_target() {
                let target = targets.get(target_id).ok_or_else(|| {
                    SwitchyardError::InvalidConfig(format!(
                        "profile {profile_id}: subagent_target references unknown target {}",
                        target_id.as_str()
                    ))
                })?;
                subagent_targets.insert(profile_id.clone(), target.clone());
            }
            profiles.insert(
                profile_id.clone(),
                resolve_profile(profile_id, profile, &env)?,
            );
        }
        Ok((profiles, subagent_targets))
    }
}

fn resolve_profile(
    profile_id: &ProfileId,
    profile: &SerializedProfileConfig,
    env: &ProfileBuildEnv<'_>,
) -> Result<ProfileConfigEntry> {
    crate::profiles::parse_profile_config(profile.profile_type(), profile.body().clone(), env)
        .map_err(|error| SwitchyardError::InvalidConfig(format!("profile {profile_id}: {error}")))
}

/// Target lookup environment used while profile-owned configs are parsed.
pub struct ProfileBuildEnv<'a> {
    targets: &'a BTreeMap<LlmTargetId, LlmTarget>,
}

impl<'a> ProfileBuildEnv<'a> {
    /// Creates a build environment over already-resolved targets.
    pub fn new(targets: &'a BTreeMap<LlmTargetId, LlmTarget>) -> Self {
        Self { targets }
    }

    /// Resolves a profile-local target ID into a concrete runtime target.
    pub fn target(&self, target_id: &LlmTargetId) -> Result<&'a LlmTarget> {
        self.targets.get(target_id).ok_or_else(|| {
            SwitchyardError::InvalidConfig(format!("profile references unknown target {target_id}"))
        })
    }
}

/// Trait implemented by the profile config macro for profile-owned parsing.
pub trait ProfileConfigDefinition: Sized {
    /// Stable discriminator used in the file-facing `type` field.
    const PROFILE_TYPE: &'static str;

    /// Parses profile-owned fields and resolves any `#[profile_target]` references.
    fn parse_profile_config(value: Value, env: &ProfileBuildEnv<'_>) -> Result<Self>;
}

/// Public contract implemented by every profile config type.
///
/// The `profile_config` attribute macro generates parsing metadata, but the
/// runtime build remains explicit here so profile authors can construct
/// backends, pollers, stats handles, and validation in normal Rust.
pub trait ProfileConfig: ProfileConfigDefinition {
    /// Runtime profile produced by this config.
    type Runtime: Profile + ProfileHooks + 'static;

    /// Builds this config into its runtime profile.
    fn build(&self) -> Result<Self::Runtime>;

    /// Builds this config into the object-safe profile used by config plans.
    fn build_boxed(&self) -> Result<Box<dyn Profile>> {
        Ok(Box::new(self.build()?))
    }
}

impl TargetConfig {
    // Resolves endpoint inheritance while preserving the map key as the target ID.
    fn resolve(
        &self,
        target_id: &LlmTargetId,
        endpoints: &BTreeMap<EndpointId, EndpointConfig>,
    ) -> Result<LlmTarget> {
        let inherited = match &self.endpoint {
            Some(endpoint_id) => endpoints.get(endpoint_id).ok_or_else(|| {
                SwitchyardError::InvalidConfig(format!(
                    "target {target_id} references unknown endpoint {endpoint_id}"
                ))
            })?,
            None => &EndpointConfig::default(),
        };
        let overrides = EndpointConfig {
            base_url: self.base_url.clone(),
            api_key: self.api_key.clone(),
            timeout_secs: self.timeout_secs,
        };
        Ok(LlmTarget {
            id: target_id.clone(),
            model: self.model.clone(),
            format: self.format,
            endpoint: inherited.with_overrides(&overrides),
            extra_body: self.extra_body.clone(),
            extra_headers: self.extra_headers.clone(),
        })
    }
}

/// Validated profile config plan ready to build runtime profiles.
///
/// This type contains resolved targets and typed profile config bodies. It
/// deliberately does not contain built `Profile` runtimes; call
/// `build_profile` or `build_profiles` when runtime ownership is needed.
#[derive(Clone, PartialEq)]
pub struct ProfileConfigPlan {
    /// Resolved targets keyed by the IDs used in the parsed config.
    targets: BTreeMap<LlmTargetId, LlmTarget>,
    /// Typed profile configs keyed by the user-facing profile ID.
    profiles: BTreeMap<ProfileId, ProfileConfigEntry>,
    /// Resolved sub-agent override targets for profiles that declare one.
    profile_subagent_targets: BTreeMap<ProfileId, LlmTarget>,
}

impl fmt::Debug for ProfileConfigPlan {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Avoid printing resolved target/profile configs because they can carry API keys.
        let profile_types = self
            .profiles
            .iter()
            .map(|(profile_id, profile)| (profile_id, profile.profile_type()))
            .collect::<BTreeMap<_, _>>();
        formatter
            .debug_struct("ProfileConfigPlan")
            .field("targets", &self.targets.keys().collect::<Vec<_>>())
            .field("profiles", &profile_types)
            .finish()
    }
}

impl ProfileConfigPlan {
    /// Returns the number of resolved targets in the plan.
    pub fn target_count(&self) -> usize {
        self.targets.len()
    }

    /// Returns the number of validated profile configs in the plan.
    pub fn profile_count(&self) -> usize {
        self.profiles.len()
    }

    /// Returns a resolved target by ID.
    pub fn target(&self, target_id: &LlmTargetId) -> Option<&LlmTarget> {
        self.targets.get(target_id)
    }

    /// Iterates resolved targets in deterministic target-ID order.
    pub fn targets(&self) -> impl Iterator<Item = (&LlmTargetId, &LlmTarget)> {
        self.targets.iter()
    }

    /// Iterates profile IDs in deterministic profile-ID order.
    pub fn profile_ids(&self) -> impl Iterator<Item = &ProfileId> {
        self.profiles.keys()
    }

    /// Returns the serialized type discriminator for one profile config.
    pub fn profile_type(&self, profile_id: &ProfileId) -> Option<&str> {
        self.profiles
            .get(profile_id)
            .map(ProfileConfigEntry::profile_type)
    }

    /// Builds one profile runtime by profile ID.
    pub fn build_profile(&self, profile_id: &ProfileId) -> Result<Box<dyn Profile>> {
        let profile = self.profiles.get(profile_id).ok_or_else(|| {
            SwitchyardError::InvalidConfig(format!("unknown profile {profile_id}"))
        })?;
        self.build_typed_profile(profile_id, profile)
    }

    /// Builds every profile in the document into an object-safe runtime profile.
    pub fn build_profiles(&self) -> Result<BTreeMap<ProfileId, Box<dyn Profile>>> {
        let mut profiles = BTreeMap::new();
        for (profile_id, profile) in &self.profiles {
            profiles.insert(
                profile_id.clone(),
                self.build_typed_profile(profile_id, profile)?,
            );
        }
        Ok(profiles)
    }

    /// Returns every resolved target as a directly addressable serving target.
    pub fn exposed_targets(&self) -> impl Iterator<Item = &LlmTarget> {
        self.targets.values()
    }

    // Builds one typed profile config with profile-ID context in errors.
    // Wraps with SubagentOverrideProfile when the profile declares a subagent_target.
    fn build_typed_profile(
        &self,
        profile_id: &ProfileId,
        profile: &ProfileConfigEntry,
    ) -> Result<Box<dyn Profile>> {
        let inner = profile.build_boxed().map_err(|error| {
            SwitchyardError::InvalidConfig(format!("profile {profile_id}: {error}"))
        })?;
        if let Some(target) = self.profile_subagent_targets.get(profile_id) {
            SubagentOverrideProfile::new(inner, target.clone())
                .map(|p| Box::new(p) as Box<dyn Profile>)
                .map_err(|error| {
                    SwitchyardError::InvalidConfig(format!(
                        "profile {profile_id}: subagent override: {error}"
                    ))
                })
        } else {
            Ok(inner)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use async_trait::async_trait;
    use serde_json::json;
    use switchyard_core::{BackendFormat, ChatResponse, ModelId};

    use super::*;

    #[switchyard_components_v2_macros::profile_config("optional-target-test")]
    struct OptionalTargetConfig {
        /// Optional target used to test macro-generated `Option<LlmTarget>` resolution.
        #[profile_target]
        pub target: Option<LlmTarget>,
    }

    struct OptionalTargetProfile;

    impl ProfileConfig for OptionalTargetConfig {
        type Runtime = OptionalTargetProfile;

        fn build(&self) -> Result<Self::Runtime> {
            Ok(OptionalTargetProfile)
        }
    }

    #[async_trait]
    impl crate::ProfileHooks for OptionalTargetProfile {
        type ProcessedRequest = crate::ProfileInput;

        async fn process(&self, input: crate::ProfileInput) -> Result<Self::ProcessedRequest> {
            Ok(input)
        }

        async fn rprocess(
            &self,
            _processed: &Self::ProcessedRequest,
            response: ChatResponse,
        ) -> Result<ChatResponse> {
            Ok(response)
        }
    }

    #[async_trait]
    impl crate::Profile for OptionalTargetProfile {
        async fn run(&self, input: crate::ProfileInput) -> Result<crate::ProfileResponse> {
            Ok(ChatResponse::openai_completion(json!({
                "model": input.request.model(),
                "choices": []
            }))
            .into())
        }
    }

    fn target(id: &str, model: &str) -> Result<LlmTarget> {
        let mut target = LlmTarget::new(LlmTargetId::new(id)?, ModelId::new(model)?);
        target.format = BackendFormat::OpenAi;
        Ok(target)
    }

    #[test]
    fn optional_target_fields_are_resolved_by_the_profile_config_macro() -> Result<()> {
        let weak = target("weak", "weak/model")?;
        let targets = BTreeMap::from([(weak.id.clone(), weak)]);
        let env = ProfileBuildEnv::new(&targets);

        let with_target =
            OptionalTargetConfig::parse_profile_config(json!({"target": "weak"}), &env)?;
        assert_eq!(
            with_target.target.map(|target| target.id),
            Some(LlmTargetId::new("weak")?)
        );

        let without_target = OptionalTargetConfig::parse_profile_config(json!({}), &env)?;
        assert_eq!(without_target.target, None);
        Ok(())
    }
}
