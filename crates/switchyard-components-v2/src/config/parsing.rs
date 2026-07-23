// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! String parsing and file-facing schema for profile configs.

use std::collections::{BTreeMap, BTreeSet};

use serde::{de, ser, Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use switchyard_core::{
    BackendFormat, EndpointConfig, EndpointId, LlmTargetId, ModelId, ProfileId, Result,
    SwitchyardError,
};

use super::loading::ProfileConfigFormat;

/// User-facing profile config document produced by parsing.
///
/// This is intentionally not runtime-ready yet. Endpoint inheritance, target
/// references, and profile-owned validation happen in the resolving stage.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileConfigDocument {
    /// Shared endpoint definitions referenced by targets.
    #[serde(default)]
    pub(super) endpoints: BTreeMap<EndpointId, EndpointConfig>,
    /// Concrete targets keyed by their stable target IDs.
    #[serde(default)]
    pub(super) targets: BTreeMap<LlmTargetId, TargetConfig>,
    /// Named serialized profile bodies keyed by the model/profile ID exposed by serving.
    #[serde(default)]
    pub(super) profiles: BTreeMap<ProfileId, SerializedProfileConfig>,
}

impl ProfileConfigDocument {
    /// Parses a profile config string and interpolates `${VAR}` environment values.
    pub fn from_str(input: &str, format: ProfileConfigFormat) -> Result<Self> {
        parse_profile_config_str_with_env_lookup(input, format, |name| std::env::var(name).ok())
    }

    /// Iterates parsed profile IDs before profile-owned validation.
    pub fn profile_ids(&self) -> impl Iterator<Item = &ProfileId> {
        self.profiles.keys()
    }

    /// Returns the serialized type discriminator for one parsed profile.
    pub fn profile_type(&self, profile_id: &ProfileId) -> Option<&str> {
        self.profiles
            .get(profile_id)
            .map(SerializedProfileConfig::profile_type)
    }

    /// Returns the profile-owned config body without the `type` discriminator.
    pub fn profile_body(&self, profile_id: &ProfileId) -> Option<&Value> {
        self.profiles
            .get(profile_id)
            .map(SerializedProfileConfig::body)
    }

    /// Returns a profile's envelope `subagent_target` reference, if configured.
    pub fn profile_subagent_target(&self, profile_id: &ProfileId) -> Option<&LlmTargetId> {
        self.profiles
            .get(profile_id)
            .and_then(SerializedProfileConfig::subagent_target)
    }

    /// Returns a copy of this document with the selected profiles removed.
    pub fn without_profiles(&self, profile_ids: &[ProfileId]) -> Self {
        let omitted = profile_ids.iter().collect::<BTreeSet<_>>();
        let mut document = self.clone();
        document
            .profiles
            .retain(|profile_id, _profile| !omitted.contains(profile_id));
        document
    }
}

/// Serialized profile body split into a type discriminator, common envelope
/// fields, and profile-owned fields.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct SerializedProfileConfig {
    profile_type: String,
    subagent_target: Option<LlmTargetId>,
    body: Value,
}

impl SerializedProfileConfig {
    /// Creates a serialized profile config after verifying the body is an object.
    fn new(
        profile_type: impl Into<String>,
        subagent_target: Option<LlmTargetId>,
        body: Value,
    ) -> Result<Self> {
        let profile_type = profile_type.into();
        if profile_type.trim().is_empty() {
            return Err(SwitchyardError::InvalidConfig(
                "profile config `type` must not be empty".to_string(),
            ));
        }
        if !body.is_object() {
            return Err(SwitchyardError::InvalidConfig(
                "profile config body must be an object".to_string(),
            ));
        }
        Ok(Self {
            profile_type,
            subagent_target,
            body,
        })
    }

    /// Returns the profile type discriminator from the file's `type` field.
    pub(super) fn profile_type(&self) -> &str {
        &self.profile_type
    }

    /// Returns the envelope `subagent_target` reference from the file, if any.
    pub(super) fn subagent_target(&self) -> Option<&LlmTargetId> {
        self.subagent_target.as_ref()
    }

    /// Returns the profile-owned config fields without the envelope fields.
    pub(super) fn body(&self) -> &Value {
        &self.body
    }
}

impl<'de> Deserialize<'de> for SerializedProfileConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let mut value = Value::deserialize(deserializer)?;
        let Value::Object(map) = &mut value else {
            return Err(de::Error::custom("profile config must be an object"));
        };
        let profile_type = map
            .remove("type")
            .ok_or_else(|| de::Error::custom("profile config requires a `type` field"))?;
        let Value::String(profile_type) = profile_type else {
            return Err(de::Error::custom("profile config `type` must be a string"));
        };
        // `subagent_target` is a common envelope field consumed here, like `type`,
        // so profile-owned bodies (which deny unknown fields) never see it.
        let subagent_target = match map.remove("subagent_target") {
            None => None,
            Some(Value::String(target_id)) => {
                Some(LlmTargetId::new(target_id).map_err(de::Error::custom)?)
            }
            Some(_) => {
                return Err(de::Error::custom(
                    "profile config `subagent_target` must be a target id string",
                ));
            }
        };
        SerializedProfileConfig::new(profile_type, subagent_target, value)
            .map_err(de::Error::custom)
    }
}

impl Serialize for SerializedProfileConfig {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let Value::Object(mut map) = self.body.clone() else {
            return Err(ser::Error::custom("profile config body must be an object"));
        };
        map.insert("type".to_string(), Value::String(self.profile_type.clone()));
        if let Some(subagent_target) = &self.subagent_target {
            map.insert(
                "subagent_target".to_string(),
                Value::String(subagent_target.as_str().to_owned()),
            );
        }
        Value::Object(map).serialize(serializer)
    }
}

/// File-facing target config that can inherit from a shared endpoint.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetConfig {
    /// Optional shared endpoint ID to inherit connection settings from.
    #[serde(default)]
    pub endpoint: Option<EndpointId>,
    /// Upstream model name sent to the provider.
    pub model: ModelId,
    /// Wire format expected by the upstream target.
    pub format: BackendFormat,
    /// Target-local base URL override.
    #[serde(default)]
    pub base_url: Option<String>,
    /// Target-local API key override.
    #[serde(default)]
    pub api_key: Option<String>,
    /// Target-local timeout override in seconds.
    #[serde(default)]
    pub timeout_secs: Option<f64>,
    /// Per-target outbound request body extensions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_body: Option<Value>,
    /// Per-target outbound request header extensions.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra_headers: BTreeMap<String, String>,
}

/// Parses a profile config string with the requested format.
pub fn parse_profile_config_str(
    input: &str,
    format: ProfileConfigFormat,
) -> Result<ProfileConfigDocument> {
    ProfileConfigDocument::from_str(input, format)
}

/// Parses a profile config string using a caller-provided environment lookup.
///
/// This is useful for embedders and tests that need deterministic interpolation
/// without mutating process-global environment variables.
pub fn parse_profile_config_str_with_env_lookup(
    input: &str,
    format: ProfileConfigFormat,
    lookup: impl Fn(&str) -> Option<String>,
) -> Result<ProfileConfigDocument> {
    parse_profile_config_str_with_lookup(input, format, lookup)
}

fn parse_profile_config_str_with_lookup(
    input: &str,
    format: ProfileConfigFormat,
    lookup: impl Fn(&str) -> Option<String>,
) -> Result<ProfileConfigDocument> {
    let mut value = parse_value(input, format)?;
    interpolate_value(&mut value, &lookup)?;
    serde_json::from_value(value).map_err(|error| {
        SwitchyardError::InvalidConfig(format!("failed to parse profile config: {error}"))
    })
}

fn parse_value(input: &str, format: ProfileConfigFormat) -> Result<Value> {
    match format {
        ProfileConfigFormat::Json => serde_json::from_str(input).map_err(|error| {
            SwitchyardError::InvalidConfig(format!("failed to parse JSON profile config: {error}"))
        }),
        ProfileConfigFormat::Toml => {
            let value = toml::from_str::<toml::Value>(input).map_err(|error| {
                SwitchyardError::InvalidConfig(format!(
                    "failed to parse TOML profile config: {error}"
                ))
            })?;
            serde_json::to_value(value).map_err(|error| {
                SwitchyardError::InvalidConfig(format!(
                    "failed to normalize TOML profile config: {error}"
                ))
            })
        }
        ProfileConfigFormat::Yaml => yaml_serde::from_str(input).map_err(|error| {
            SwitchyardError::InvalidConfig(format!("failed to parse YAML profile config: {error}"))
        }),
    }
}

fn interpolate_value(value: &mut Value, lookup: &impl Fn(&str) -> Option<String>) -> Result<()> {
    match value {
        Value::String(text) => {
            *text = interpolate_string(text, lookup)?;
        }
        Value::Array(items) => {
            for item in items {
                interpolate_value(item, lookup)?;
            }
        }
        Value::Object(map) => {
            for item in map.values_mut() {
                interpolate_value(item, lookup)?;
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
    Ok(())
}

fn interpolate_string(input: &str, lookup: &impl Fn(&str) -> Option<String>) -> Result<String> {
    let mut output = String::new();
    let mut rest = input;

    while let Some(start) = rest.find("${") {
        output.push_str(&rest[..start]);
        let after_start = &rest[start + 2..];
        let Some(end) = after_start.find('}') else {
            return Err(SwitchyardError::InvalidConfig(format!(
                "unterminated environment variable in `{input}`"
            )));
        };
        let name = &after_start[..end];
        if name.is_empty() {
            return Err(SwitchyardError::InvalidConfig(
                "empty environment variable reference".to_string(),
            ));
        }
        let value = lookup(name).ok_or_else(|| {
            SwitchyardError::InvalidConfig(format!(
                "environment variable {name} is not set for profile config interpolation"
            ))
        })?;
        output.push_str(&value);
        rest = &after_start[end + 1..];
    }

    output.push_str(rest);
    Ok(output)
}
