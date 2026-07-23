// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Adversarial tests for profile config parsing and profile-owned resolution.

use serde_json::json;
use switchyard_components_v2::{
    parse_profile_config_str_with_env_lookup, NoopProfileConfig, ProfileConfig,
    ProfileConfigDocument, ProfileConfigFormat, ProfileHooks, ProfileInput, RequestMetadata,
};
use switchyard_core::{ChatRequest, ChatResponse, LlmTargetId, ProfileId, Result, SwitchyardError};

const YAML_CONFIG: &str = r#"
endpoints:
  nvidia:
    api_key: ${NVIDIA_API_KEY}
    base_url: https://inference-api.nvidia.com/v1
    timeout_secs: 120.0

targets:
  strong:
    endpoint: nvidia
    model: nvidia/moonshotai/kimi-k2.5
    format: openai
  weak:
    endpoint: nvidia
    model: nvidia/nvidia/nemotron-nano-9b-v2
    format: openai
  direct-weak:
    endpoint: nvidia
    model: nvidia/nvidia/nemotron-nano-9b-v2
    format: openai
  classifier:
    endpoint: nvidia
    model: nvidia/nvidia/nemotron-nano-9b-v2
    format: openai

profiles:
  direct:
    type: passthrough
    target: direct-weak
  smart-stage-router:
    type: stage_router
    capable: strong
    efficient: weak
    fallback_target_on_evict: strong
    picker: capable_first
    confidence_threshold: 0.7
  health-aware:
    type: latency-service
    latency_service_url: http://latency.local
    targets: [strong, weak]
  llm:
    type: llm-routing
    strong: strong
    weak: weak
    classifier: classifier
    profile_name: coding_agent
  stage_router:
    type: stage_router
    capable: strong
    efficient: weak
    fallback_target_on_evict: strong
    picker: capable_first
    classifier:
      model: nvidia/nvidia/nemotron-nano-9b-v2
      api_key: ${NVIDIA_API_KEY}
      base_url: https://inference-api.nvidia.com/v1
  bench:
    type: noop
"#;

// Parses YAML profile config using the test environment lookup.
fn parse_yaml(input: &str) -> Result<ProfileConfigDocument> {
    parse_profile_config_str_with_env_lookup(input, ProfileConfigFormat::Yaml, test_env)
}

// Parses a profile config in the requested format using the test environment lookup.
fn parse_with_format(input: &str, format: ProfileConfigFormat) -> Result<ProfileConfigDocument> {
    parse_profile_config_str_with_env_lookup(input, format, test_env)
}

// Supplies deterministic values for environment interpolation tests.
fn test_env(name: &str) -> Option<String> {
    match name {
        "NVIDIA_API_KEY" => Some("nvapi-test".to_string()),
        "TENANT" => Some("team-a".to_string()),
        _ => None,
    }
}

// Resolving a full YAML config should inherit endpoints and validate profile-owned configs.
#[test]
fn yaml_config_resolves_endpoints_targets_and_profile_owned_configs() -> Result<()> {
    let config = parse_yaml(YAML_CONFIG)?;
    let plan = config.resolve()?;

    assert_eq!(plan.target_count(), 4);
    let direct_weak = plan
        .target(&LlmTargetId::new("direct-weak")?)
        .ok_or_else(|| SwitchyardError::Other("direct-weak target missing".to_string()))?;
    assert_eq!(direct_weak.endpoint.api_key.as_deref(), Some("nvapi-test"));
    assert_eq!(
        direct_weak.endpoint.base_url.as_deref(),
        Some("https://inference-api.nvidia.com/v1")
    );

    assert_eq!(
        plan.profile_type(&ProfileId::new("direct")?),
        Some("passthrough")
    );
    assert_eq!(
        plan.profile_type(&ProfileId::new("smart-stage-router")?),
        Some("stage_router")
    );
    assert_eq!(
        plan.profile_type(&ProfileId::new("health-aware")?),
        Some("latency-service")
    );
    assert_eq!(
        plan.profile_type(&ProfileId::new("llm")?),
        Some("llm-routing")
    );
    assert_eq!(
        plan.profile_type(&ProfileId::new("stage_router")?),
        Some("stage_router")
    );

    let profiles = plan.build_profiles()?;
    assert_eq!(profiles.len(), 6);

    let targets = plan
        .targets()
        .map(|(target_id, _target)| target_id.clone())
        .collect::<Vec<_>>();
    assert_eq!(
        targets,
        vec![
            LlmTargetId::new("classifier")?,
            LlmTargetId::new("direct-weak")?,
            LlmTargetId::new("strong")?,
            LlmTargetId::new("weak")?,
        ]
    );
    Ok(())
}

// JSON and TOML should normalize into equivalent resolved config plans.
#[test]
fn json_and_toml_parse_to_the_same_document_shape() -> Result<()> {
    let json_config = r#"
{
  "endpoints": {
    "nvidia": {
      "api_key": "${NVIDIA_API_KEY}",
      "base_url": "https://inference-api.nvidia.com/v1",
      "timeout_secs": 120.0
    }
  },
  "targets": {
    "weak": {
      "endpoint": "nvidia",
      "model": "nvidia/nvidia/nemotron-nano-9b-v2",
      "format": "openai"
    }
  },
  "profiles": {
    "direct": {
      "type": "passthrough",
      "target": "weak"
    }
  }
}
"#;
    let toml_config = r#"
[endpoints.nvidia]
api_key = "${NVIDIA_API_KEY}"
base_url = "https://inference-api.nvidia.com/v1"
timeout_secs = 120.0

[targets.weak]
endpoint = "nvidia"
model = "nvidia/nvidia/nemotron-nano-9b-v2"
format = "openai"

[profiles.direct]
type = "passthrough"
target = "weak"
"#;

    let json = parse_with_format(json_config, ProfileConfigFormat::Json)?.resolve()?;
    let toml = parse_with_format(toml_config, ProfileConfigFormat::Toml)?.resolve()?;

    assert_eq!(json, toml);
    Ok(())
}

// Unknown fields inside a profile body should be rejected by that profile's config parser.
#[test]
fn unknown_profile_fields_are_rejected_by_owning_profile_config() -> Result<()> {
    let input = r#"
targets:
  strong:
    model: strong/model
    format: openai
  weak:
    model: weak/model
    format: openai
profiles:
  bad:
    type: random-routing
    strong: strong
    weak: weak
    strong_probability: 0.3
    tina: codes well
"#;

    let error = parse_yaml(input)?
        .resolve()
        .err()
        .map(|error| error.to_string())
        .unwrap_or_else(|| "expected resolve failure".to_string());
    assert!(error.contains("unknown field"));
    assert!(error.contains("tina"));
    assert!(error.contains("profile bad"));
    Ok(())
}

// Target-level `expose` stays rejected because every v2 target is addressable.
#[test]
fn target_expose_field_is_rejected_because_all_targets_are_exposed() {
    let input = r#"
targets:
  weak:
    model: weak/model
    format: openai
    expose: true
profiles:
  direct:
    type: passthrough
    target: weak
"#;

    let error = parse_yaml(input)
        .err()
        .map(|error| error.to_string())
        .unwrap_or_else(|| "expected parse failure".to_string());
    assert!(error.contains("unknown field"));
    assert!(error.contains("expose"));
}

// Target IDs should come only from map keys, not duplicated `id` fields.
#[test]
fn target_id_field_is_rejected_because_map_key_is_the_id() {
    let input = r#"
targets:
  weak:
    id: duplicate
    model: weak/model
    format: openai
profiles:
  direct:
    type: passthrough
    target: weak
"#;

    let error = parse_yaml(input)
        .err()
        .map(|error| error.to_string())
        .unwrap_or_else(|| "expected parse failure".to_string());
    assert!(error.contains("unknown field"));
    assert!(error.contains("id"));
}

// The top-level schema should reject stale route/table-era sections.
#[test]
fn unknown_top_level_fields_are_rejected() {
    let input = r#"
endpoints: {}
targets: {}
profiles: {}
routes: {}
"#;

    let error = parse_yaml(input)
        .err()
        .map(|error| error.to_string())
        .unwrap_or_else(|| "expected parse failure".to_string());
    assert!(error.contains("unknown field"));
    assert!(error.contains("routes"));
}

// Unknown profile types should parse as documents but fail during profile resolution.
#[test]
fn unknown_profile_type_is_rejected_during_resolution() -> Result<()> {
    let input = r#"
profiles:
  bad:
    type: handmade
"#;

    let error = parse_yaml(input)?
        .resolve()
        .err()
        .map(|error| error.to_string())
        .unwrap_or_else(|| "expected resolve failure".to_string());
    assert!(error.contains("handmade"));
    assert!(error.contains("profile bad"));
    Ok(())
}

// `-` and `_` are equivalent in the `type` discriminator, so a name written with
// either separator resolves to the same v2 profile: a legacy `random_routing`
// spelling, and `stage-router` for the underscore-canonical stage router.
#[test]
fn profile_type_separators_are_normalized_during_resolution() -> Result<()> {
    let input = r#"
endpoints:
  nvidia:
    api_key: ${NVIDIA_API_KEY}
    base_url: https://inference-api.nvidia.com/v1

targets:
  strong:
    endpoint: nvidia
    model: nvidia/moonshotai/kimi-k2.5
    format: openai
  weak:
    endpoint: nvidia
    model: nvidia/nvidia/nemotron-nano-9b-v2
    format: openai
  classifier:
    endpoint: nvidia
    model: nvidia/nvidia/nemotron-nano-9b-v2
    format: openai

profiles:
  snake-llm:
    type: llm_routing
    strong: strong
    weak: weak
    classifier: classifier
    profile_name: coding_agent
  hyphen-stage:
    type: stage-router
    capable: strong
    efficient: weak
    fallback_target_on_evict: strong
    picker: capable_first
    confidence_threshold: 0.7
  snake-latency:
    type: latency_service
    latency_service_url: http://latency.local
    targets: [strong, weak]
"#;

    let plan = parse_yaml(input)?.resolve()?;

    // Resolved profiles report the canonical type name regardless of the
    // separator spelled in the config.
    assert_eq!(
        plan.profile_type(&ProfileId::new("snake-llm")?),
        Some("llm-routing")
    );
    assert_eq!(
        plan.profile_type(&ProfileId::new("hyphen-stage")?),
        Some("stage_router")
    );
    assert_eq!(
        plan.profile_type(&ProfileId::new("snake-latency")?),
        Some("latency-service")
    );

    // The profiles build, not just parse.
    assert_eq!(plan.build_profiles()?.len(), 3);
    Ok(())
}

// Missing profile type discriminators should fail while parsing the document.
#[test]
fn missing_profile_type_is_rejected_during_document_parse() {
    let input = r#"
profiles:
  bad:
    target: weak
"#;

    let error = parse_yaml(input)
        .err()
        .map(|error| error.to_string())
        .unwrap_or_else(|| "expected parse failure".to_string());
    assert!(error.contains("type"));
}

// Empty profile type discriminators should fail while parsing the document.
#[test]
fn empty_profile_type_is_rejected_during_document_parse() {
    let input = r#"
profiles:
  bad:
    type: " "
"#;

    let error = parse_yaml(input)
        .err()
        .map(|error| error.to_string())
        .unwrap_or_else(|| "expected parse failure".to_string());
    assert!(error.contains("must not be empty"));
}

// Missing environment variables should fail before document deserialization completes.
#[test]
fn missing_env_var_is_rejected() {
    let var_name = format!(
        "SWITCHYARD_COMPONENTS_V2_MISSING_ENV_{}",
        std::process::id()
    );
    let input = format!(
        r#"
endpoints:
  missing:
    api_key: ${{{var_name}}}
"#
    );

    let error =
        parse_profile_config_str_with_env_lookup(&input, ProfileConfigFormat::Yaml, test_env)
            .err()
            .map(|error| error.to_string())
            .unwrap_or_else(|| "expected parse failure".to_string());

    assert!(error.contains(&var_name));
    assert!(error.contains("not set"));
}

// Targets that reference a missing endpoint should fail during target resolution.
#[test]
fn missing_endpoint_reference_is_rejected_during_resolution() -> Result<()> {
    let input = r#"
targets:
  weak:
    endpoint: missing
    model: weak/model
    format: openai
profiles:
  direct:
    type: passthrough
    target: weak
"#;

    let error = parse_yaml(input)?
        .resolve()
        .err()
        .map(|error| error.to_string())
        .unwrap_or_else(|| "expected resolve failure".to_string());
    assert!(error.contains("unknown endpoint missing"));
    Ok(())
}

// Macro-resolved profile target fields should fail when they name an unknown target.
#[test]
fn missing_profile_target_reference_is_rejected_by_macro_generated_resolver() -> Result<()> {
    let input = r#"
targets:
  weak:
    model: weak/model
    format: openai
profiles:
  direct:
    type: passthrough
    target: typo
"#;

    let error = parse_yaml(input)?
        .resolve()
        .err()
        .map(|error| error.to_string())
        .unwrap_or_else(|| "expected resolve failure".to_string());
    assert!(error.contains("unknown target typo"));
    assert!(error.contains("profile direct"));
    Ok(())
}

// Target-local endpoint fields should override shared endpoints and interpolate nested strings.
#[test]
fn target_overrides_endpoint_fields_and_interpolates_nested_strings() -> Result<()> {
    let input = r#"
endpoints:
  nvidia:
    api_key: ${NVIDIA_API_KEY}
    base_url: https://inference-api.nvidia.com/v1
    timeout_secs: 120.0
targets:
  weak:
    endpoint: nvidia
    model: weak/model
    format: openai
    base_url: https://override.example/v1
    timeout_secs: 30.0
    extra_body:
      metadata:
        tenant: ${TENANT}
profiles:
  direct:
    type: passthrough
    target: weak
"#;
    let plan = parse_yaml(input)?.resolve()?;
    let weak = plan
        .target(&LlmTargetId::new("weak")?)
        .ok_or_else(|| SwitchyardError::Other("weak target missing".to_string()))?;

    assert_eq!(
        weak.endpoint.base_url.as_deref(),
        Some("https://override.example/v1")
    );
    assert_eq!(weak.endpoint.timeout_secs, Some(30.0));
    assert_eq!(weak.endpoint.api_key.as_deref(), Some("nvapi-test"));
    assert_eq!(
        weak.extra_body.as_ref().and_then(|body| {
            body.get("metadata")
                .and_then(|metadata| metadata.get("tenant"))
                .and_then(serde_json::Value::as_str)
        }),
        Some("team-a")
    );
    Ok(())
}

// Debug output for resolved plans should not include inherited endpoint API keys.
#[test]
fn profile_config_plan_debug_redacts_target_api_keys() -> Result<()> {
    let plan = parse_yaml(YAML_CONFIG)?.resolve()?;
    let debug = format!("{plan:?}");

    assert!(!debug.contains("nvapi-test"));
    assert!(debug.contains("direct-weak"));
    assert!(debug.contains("passthrough"));
    Ok(())
}

// Resolved plans should build one profile or all profiles into runtime objects.
#[test]
fn profile_config_plan_builds_runtime_profiles() -> Result<()> {
    let plan = parse_yaml(YAML_CONFIG)?.resolve()?;

    let direct = plan.build_profile(&ProfileId::new("direct")?)?;
    let profiles = plan.build_profiles()?;

    drop(direct);
    assert_eq!(profiles.len(), 6);
    assert!(profiles.contains_key(&ProfileId::new("direct")?));
    assert!(profiles.contains_key(&ProfileId::new("smart-stage-router")?));
    assert!(profiles.contains_key(&ProfileId::new("health-aware")?));
    assert!(profiles.contains_key(&ProfileId::new("llm")?));
    assert!(profiles.contains_key(&ProfileId::new("stage_router")?));
    assert!(profiles.contains_key(&ProfileId::new("bench")?));
    Ok(())
}

// Components-v2 stage_router config should reject route-era/top-level classifier
// knobs instead of carrying compatibility behavior forward.
#[test]
fn stage_router_rejects_top_level_classifier_knobs() -> Result<()> {
    let input = r#"
targets:
  strong:
    model: strong/model
    format: openai
  weak:
    model: weak/model
    format: openai
profiles:
  stale:
    type: stage_router
    capable: strong
    efficient: weak
    fallback_target_on_evict: strong
    classifier_max_tokens: 64
"#;

    let error = parse_yaml(input)?
        .resolve()
        .err()
        .map(|error| error.to_string())
        .unwrap_or_else(|| "expected resolve failure".to_string());
    assert!(error.contains("unknown field"));
    assert!(error.contains("classifier_max_tokens"));
    Ok(())
}

// Invalid stage_router picker names fail during profile config resolution.
#[test]
fn stage_router_resolve_rejects_unknown_picker() -> Result<()> {
    let input = r#"
targets:
  strong:
    model: strong/model
    format: openai
  weak:
    model: weak/model
    format: openai
profiles:
  bad:
    type: stage_router
    capable: strong
    efficient: weak
    fallback_target_on_evict: strong
    picker: not-a-picker
"#;

    let error = parse_yaml(input)?
        .resolve()
        .err()
        .map(|error| error.to_string())
        .unwrap_or_else(|| "expected resolve failure".to_string());
    assert!(error.contains("unknown variant"));
    assert!(error.contains("not-a-picker"));
    Ok(())
}

// Strict OpenAI classifier calls require OpenAI-format classifier targets.
#[test]
fn llm_routing_build_rejects_non_openai_classifier_target() -> Result<()> {
    let input = r#"
targets:
  strong:
    model: strong/model
    format: openai
  weak:
    model: weak/model
    format: openai
  classifier:
    model: classifier/model
    format: anthropic
profiles:
  bad:
    type: llm-routing
    strong: strong
    weak: weak
    classifier: classifier
"#;

    let error = parse_yaml(input)?
        .resolve()?
        .build_profile(&ProfileId::new("bad")?)
        .err()
        .map(|error| error.to_string())
        .unwrap_or_else(|| "expected build failure".to_string());
    assert!(error.contains("classifier target must use format: openai"));
    Ok(())
}

// Config-owned `build()` should return a concrete profile runtime with hook methods.
#[tokio::test]
async fn profile_config_build_returns_profile_with_request_and_response_hooks() -> Result<()> {
    let profile = NoopProfileConfig {}.build()?;
    let input = ProfileInput {
        request: ChatRequest::openai_chat(json!({
            "model": "unit/noop",
            "messages": [],
        })),
        metadata: RequestMetadata::default(),
    };

    let processed = profile.process(input).await?;
    let response = profile
        .rprocess(
            &processed,
            ChatResponse::openai_completion(json!({
                "id": "unit-response",
                "object": "chat.completion",
                "model": "unit/noop",
                "choices": [],
            })),
        )
        .await?;

    assert_eq!(processed.request.model(), Some("unit/noop"));
    assert_eq!(
        response
            .body()
            .and_then(|body| body.get("id"))
            .and_then(serde_json::Value::as_str),
        Some("unit-response")
    );
    Ok(())
}

// File format detection should reject unsupported config file extensions.
#[test]
fn profile_config_format_rejects_unknown_extension() {
    let error = ProfileConfigFormat::from_path("profiles.ini")
        .err()
        .map(|error| error.to_string())
        .unwrap_or_else(|| "expected extension failure".to_string());
    assert!(error.contains("unsupported profile config extension"));
}

// The envelope-level `subagent_target` is consumed before profile-owned parsing,
// so profile types that deny unknown fields still resolve, and the reference
// stays readable from the parsed document.
#[test]
fn subagent_target_envelope_field_is_stripped_and_exposed() -> Result<()> {
    let input = r#"
targets:
  weak:
    model: weak/model
    format: openai
profiles:
  direct:
    type: passthrough
    target: weak
    subagent_target: weak
"#;
    let config = parse_yaml(input)?;
    let profile_id = ProfileId::new("direct")?;
    assert_eq!(
        config.profile_subagent_target(&profile_id),
        Some(&LlmTargetId::new("weak")?)
    );
    let body_has_field = config
        .profile_body(&profile_id)
        .and_then(|body| body.get("subagent_target"))
        .is_some();
    assert!(!body_has_field);
    config.resolve()?;
    Ok(())
}

// A profile without the envelope field reports no sub-agent target.
#[test]
fn subagent_target_is_absent_by_default() -> Result<()> {
    let input = r#"
targets:
  weak:
    model: weak/model
    format: openai
profiles:
  direct:
    type: passthrough
    target: weak
"#;
    let config = parse_yaml(input)?;
    assert_eq!(
        config.profile_subagent_target(&ProfileId::new("direct")?),
        None
    );
    Ok(())
}

// An unknown `subagent_target` reference is a startup configuration error.
#[test]
fn subagent_target_referencing_unknown_target_is_rejected_at_resolve() -> Result<()> {
    let input = r#"
targets:
  weak:
    model: weak/model
    format: openai
profiles:
  direct:
    type: passthrough
    target: weak
    subagent_target: ghost
"#;
    let error = parse_yaml(input)?
        .resolve()
        .err()
        .map(|error| error.to_string())
        .unwrap_or_else(|| "expected resolve failure".to_string());
    assert!(error.contains("profile direct"));
    assert!(error.contains("subagent_target references unknown target ghost"));
    Ok(())
}

// The envelope field must be a target id string, not a structured value.
#[test]
fn subagent_target_must_be_a_target_id_string() {
    let input = r#"
targets:
  weak:
    model: weak/model
    format: openai
profiles:
  direct:
    type: passthrough
    target: weak
    subagent_target: [weak]
"#;
    let error = parse_yaml(input)
        .err()
        .map(|error| error.to_string())
        .unwrap_or_else(|| "expected parse failure".to_string());
    assert!(error.contains("`subagent_target` must be a target id string"));
}
