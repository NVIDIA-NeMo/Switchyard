// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public configuration and startup-validation tests for the learned prefill-probe profile.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use serde_json::{json, Value};
use switchyard_components_v2::{
    parse_profile_config_str, PrefillProbeProfileConfig, PrefillProbeRoutingPolicyConfig,
    ProfileConfig, ProfileConfigFormat,
};
use switchyard_core::{
    BackendFormat, LlmTarget, LlmTargetId, ModelId, ProfileId, Result, SwitchyardError,
};

static NEXT_TEST_DIRECTORY: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy)]
enum ArtifactFault {
    None,
    EncoderMismatch,
    MissingTensor,
    WrongShape,
    MalformedTensorFile,
}

struct TestTensor {
    name: String,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

struct TestArtifactDirectory {
    path: PathBuf,
}

impl TestArtifactDirectory {
    fn create(fault: ArtifactFault) -> Result<Self> {
        let sequence = NEXT_TEST_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "switchyard-prefill-profile-{}-{sequence}",
            std::process::id(),
        ));
        std::fs::create_dir(&path).map_err(|error| {
            SwitchyardError::Other(format!(
                "failed to create test directory {}: {error}",
                path.display()
            ))
        })?;
        let directory = Self { path };
        directory.write_artifact(fault)?;
        Ok(directory)
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn write_artifact(&self, fault: ArtifactFault) -> Result<()> {
        let encoder = if matches!(fault, ArtifactFault::EncoderMismatch) {
            "different/probe"
        } else {
            "probe/model"
        };
        let metadata = json!({
            "format_version": 1,
            "training_mode": "single_pca_block",
            "encoder": encoder,
            "representation": "token_mean_per_layer_concat",
            "extraction_layer_ids": [0, 1],
            "hidden_size": 2,
            "raw_feature_dim": 4,
            "feature_block_count": 1,
            "pca_dim": 200,
            "pca_whiten": false,
            "output_names": ["qwen-122b", "nemotron-3-super", "opus-4.7", "gpt-5.5"],
            "trunk_hidden": [256, 128],
            "ensemble_size": 5,
            "probability_link": "independent_sigmoid",
            "ensemble_reduction": "probability_mean",
            "tensor_file": "router.safetensors",
        });
        let metadata_bytes = serde_json::to_vec(&metadata)
            .map_err(|error| SwitchyardError::Other(format!("metadata encode failed: {error}")))?;
        std::fs::write(self.path.join("router.json"), metadata_bytes)
            .map_err(|error| SwitchyardError::Other(format!("metadata write failed: {error}")))?;

        if matches!(fault, ArtifactFault::MalformedTensorFile) {
            std::fs::write(self.path.join("router.safetensors"), b"not safetensors").map_err(
                |error| SwitchyardError::Other(format!("malformed tensor write failed: {error}")),
            )?;
            return Ok(());
        }

        let mut tensors = test_artifact_tensors()?;
        if matches!(fault, ArtifactFault::MissingTensor) {
            tensors.retain(|tensor| tensor.name != "transform.scaler_mean");
        } else if matches!(fault, ArtifactFault::WrongShape) {
            tensors[0] = test_tensor("transform.scaler_mean", vec![3], 0.0)?;
        }
        let mut views = Vec::with_capacity(tensors.len());
        for tensor in &tensors {
            let view = TensorView::new(Dtype::F32, tensor.shape.clone(), &tensor.bytes).map_err(
                |error| {
                    SwitchyardError::Other(format!("tensor {} is invalid: {error}", tensor.name))
                },
            )?;
            views.push((tensor.name.as_str(), view));
        }
        serialize_to_file(
            views,
            &Some(HashMap::new()),
            &self.path.join("router.safetensors"),
        )
        .map_err(|error| SwitchyardError::Other(format!("artifact encode failed: {error}")))?;
        Ok(())
    }
}

impl Drop for TestArtifactDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn test_tensor(name: impl Into<String>, shape: Vec<usize>, value: f32) -> Result<TestTensor> {
    let value_count = shape
        .iter()
        .try_fold(1usize, |count, size| count.checked_mul(*size))
        .ok_or_else(|| SwitchyardError::Other("test tensor shape overflow".to_string()))?;
    let bytes = (0..value_count).flat_map(|_| value.to_le_bytes()).collect();
    Ok(TestTensor {
        name: name.into(),
        shape,
        bytes,
    })
}

fn test_artifact_tensors() -> Result<Vec<TestTensor>> {
    let mut tensors = vec![
        test_tensor("transform.scaler_mean", vec![4], 0.0)?,
        test_tensor("transform.scaler_scale", vec![4], 1.0)?,
        test_tensor("transform.pca_mean", vec![4], 0.0)?,
        test_tensor("transform.pca_components", vec![200, 4], 0.0)?,
    ];
    for index in 0..5 {
        let prefix = format!("ensemble.{index}");
        tensors.extend([
            test_tensor(format!("{prefix}.linear1.weight"), vec![256, 200], 0.0)?,
            test_tensor(format!("{prefix}.linear1.bias"), vec![256], 0.0)?,
            test_tensor(format!("{prefix}.linear2.weight"), vec![128, 256], 0.0)?,
            test_tensor(format!("{prefix}.linear2.bias"), vec![128], 0.0)?,
            test_tensor(format!("{prefix}.output.weight"), vec![4, 128], 0.0)?,
            test_tensor(format!("{prefix}.output.bias"), vec![4], 0.0)?,
        ]);
    }
    Ok(tensors)
}

fn target(id: &str, model: &str) -> Result<LlmTarget> {
    let mut target = LlmTarget::new(LlmTargetId::new(id)?, ModelId::new(model)?);
    target.format = BackendFormat::OpenAi;
    Ok(target)
}

fn routing_policy() -> PrefillProbeRoutingPolicyConfig {
    PrefillProbeRoutingPolicyConfig::CostAware {
        lambda: 0.5,
        weak_cost: 0.01,
        strong_cost: 0.10,
    }
}

fn config_with_artifact(artifact_dir: &Path) -> Result<PrefillProbeProfileConfig> {
    Ok(PrefillProbeProfileConfig {
        probe: target("probe", "probe/model")?,
        strong: target("strong", "frontier/model")?,
        strong_checkpoint_head: "opus-4.7".to_string(),
        weak: target("weak", "cheap/model")?,
        weak_checkpoint_head: "nemotron-3-super".to_string(),
        hidden_states_dir: "/dev/shm/hidden_states".to_string(),
        inference_artifact_dir: artifact_dir.to_string_lossy().into_owned(),
        routing_policy: routing_policy(),
    })
}

fn assert_build_error(fault: ArtifactFault, expected: &str) -> Result<()> {
    let artifact = TestArtifactDirectory::create(fault)?;
    let error = config_with_artifact(artifact.path())?
        .build()
        .err()
        .ok_or_else(|| SwitchyardError::Other("invalid artifact should fail".to_string()))?;
    assert!(format!("{error}").contains(expected));
    Ok(())
}

#[test]
fn registered_yaml_resolves_and_builds_the_profile() -> Result<()> {
    let artifact = TestArtifactDirectory::create(ArtifactFault::None)?;
    let yaml = format!(
        r#"
targets:
  probe:
    model: probe/model
    format: openai
    base_url: http://localhost:8000/v1
  strong:
    model: frontier/model
    format: openai
  weak:
    model: cheap/model
    format: openai
profiles:
  router:
    type: prefill-probe
    probe: probe
    strong: strong
    strong_checkpoint_head: opus-4.7
    weak: weak
    weak_checkpoint_head: nemotron-3-super
    hidden_states_dir: /dev/shm/hidden_states
    checkpoint_dir: {}
    routing_policy:
      type: cost-aware
      lambda: 0.5
      weak_cost: 0.01
      strong_cost: 0.10
"#,
        artifact.path().display(),
    );

    let plan = parse_profile_config_str(&yaml, ProfileConfigFormat::Yaml)?.resolve()?;
    let profile_id = ProfileId::new("router")?;
    assert_eq!(plan.profile_type(&profile_id), Some("prefill-probe"));
    assert_eq!(plan.target_count(), 3);
    let probe = plan
        .target(&LlmTargetId::new("probe")?)
        .ok_or_else(|| SwitchyardError::Other("probe target should resolve".to_string()))?;
    assert_eq!(probe.model.as_str(), "probe/model");
    let _profile = plan.build_profile(&profile_id)?;
    Ok(())
}

#[test]
fn probe_can_differ_from_both_completion_targets() -> Result<()> {
    let artifact = TestArtifactDirectory::create(ArtifactFault::None)?;
    let config = config_with_artifact(artifact.path())?;

    assert_eq!(config.probe.model.as_str(), "probe/model");
    assert_eq!(config.strong.model.as_str(), "frontier/model");
    assert_eq!(config.weak.model.as_str(), "cheap/model");
    let _profile = config.build()?;
    Ok(())
}

#[test]
fn config_schema_requires_every_field_and_rejects_unknown_fields() -> Result<()> {
    let artifact = TestArtifactDirectory::create(ArtifactFault::None)?;
    let config = config_with_artifact(artifact.path())?;
    assert_eq!(PrefillProbeProfileConfig::PROFILE_TYPE, "prefill-probe");
    assert_eq!(config.profile_type(), "prefill-probe");
    let base = serde_json::to_value(&config)
        .map_err(|error| SwitchyardError::Other(format!("config encode failed: {error}")))?;

    for field in [
        "probe",
        "strong",
        "strong_checkpoint_head",
        "weak",
        "weak_checkpoint_head",
        "hidden_states_dir",
        "checkpoint_dir",
        "routing_policy",
    ] {
        let mut missing = base.clone();
        missing
            .as_object_mut()
            .ok_or_else(|| SwitchyardError::Other("config should encode as an object".to_string()))?
            .remove(field);
        let error = serde_json::from_value::<PrefillProbeProfileConfig>(missing)
            .err()
            .ok_or_else(|| SwitchyardError::Other(format!("missing {field} should fail")))?;
        assert!(error
            .to_string()
            .contains(&format!("missing field `{field}`")));
    }

    let mut unknown = base;
    unknown
        .as_object_mut()
        .ok_or_else(|| SwitchyardError::Other("config should encode as an object".to_string()))?
        .insert("unexpected_field".to_string(), Value::Bool(true));
    let error = serde_json::from_value::<PrefillProbeProfileConfig>(unknown)
        .err()
        .ok_or_else(|| SwitchyardError::Other("unknown profile field should fail".to_string()))?;
    assert!(error
        .to_string()
        .contains("unknown field `unexpected_field`"));
    Ok(())
}

#[test]
fn legacy_inference_artifact_dir_alias_is_accepted() -> Result<()> {
    let artifact = TestArtifactDirectory::create(ArtifactFault::None)?;
    let config = config_with_artifact(artifact.path())?;
    let mut value = serde_json::to_value(&config)
        .map_err(|error| SwitchyardError::Other(format!("config encode failed: {error}")))?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| SwitchyardError::Other("config should encode as an object".to_string()))?;
    let checkpoint_dir = object.remove("checkpoint_dir").ok_or_else(|| {
        SwitchyardError::Other("serialized config should contain checkpoint_dir".to_string())
    })?;
    object.insert("inference_artifact_dir".to_string(), checkpoint_dir);

    let parsed = serde_json::from_value::<PrefillProbeProfileConfig>(value)
        .map_err(|error| SwitchyardError::Other(format!("legacy config parse failed: {error}")))?;
    assert_eq!(parsed, config);
    Ok(())
}

#[test]
fn policy_schema_is_tagged_and_strict() -> Result<()> {
    let valid = serde_json::to_value(routing_policy())
        .map_err(|error| SwitchyardError::Other(format!("policy encode failed: {error}")))?;
    let parsed = serde_json::from_value::<PrefillProbeRoutingPolicyConfig>(valid.clone())
        .map_err(|error| SwitchyardError::Other(format!("policy parse failed: {error}")))?;
    assert_eq!(parsed, routing_policy());

    let mut missing_tag = valid.clone();
    missing_tag
        .as_object_mut()
        .ok_or_else(|| SwitchyardError::Other("policy should encode as an object".to_string()))?
        .remove("type");
    let missing_tag_error = serde_json::from_value::<PrefillProbeRoutingPolicyConfig>(missing_tag)
        .err()
        .ok_or_else(|| SwitchyardError::Other("missing policy type should fail".to_string()))?;
    assert!(missing_tag_error.to_string().contains("type"));

    let mut unknown = valid;
    unknown
        .as_object_mut()
        .ok_or_else(|| SwitchyardError::Other("policy should encode as an object".to_string()))?
        .insert("threshold".to_string(), json!(0.5));
    let unknown_error = serde_json::from_value::<PrefillProbeRoutingPolicyConfig>(unknown)
        .err()
        .ok_or_else(|| SwitchyardError::Other("unknown policy field should fail".to_string()))?;
    assert!(unknown_error
        .to_string()
        .contains("unknown field `threshold`"));
    Ok(())
}

#[test]
fn legacy_routing_knobs_are_rejected() -> Result<()> {
    let artifact = TestArtifactDirectory::create(ArtifactFault::None)?;
    let base = serde_json::to_value(config_with_artifact(artifact.path())?)
        .map_err(|error| SwitchyardError::Other(format!("config encode failed: {error}")))?;

    for (field, value) in [
        ("confidence_threshold", json!(0.5)),
        ("probe_signal", json!("entropy")),
    ] {
        let mut config = base.clone();
        config
            .as_object_mut()
            .ok_or_else(|| SwitchyardError::Other("config should encode as an object".to_string()))?
            .insert(field.to_string(), value);
        let error = serde_json::from_value::<PrefillProbeProfileConfig>(config)
            .err()
            .ok_or_else(|| SwitchyardError::Other(format!("legacy {field} should fail")))?;
        assert!(error.to_string().contains("unknown field"));
        assert!(error.to_string().contains(field));
    }
    Ok(())
}

#[test]
fn artifact_encoder_and_tensor_failures_are_reported_at_build() -> Result<()> {
    assert_build_error(ArtifactFault::EncoderMismatch, "does not match probe model")?;
    assert_build_error(
        ArtifactFault::MissingTensor,
        "missing tensor transform.scaler_mean",
    )?;
    assert_build_error(ArtifactFault::WrongShape, "has shape [3]; expected [4]")?;
    assert_build_error(ArtifactFault::MalformedTensorFile, "failed to parse")?;
    Ok(())
}

#[test]
fn missing_artifact_directory_is_reported_at_build() -> Result<()> {
    let missing = std::env::temp_dir().join(format!(
        "switchyard-missing-prefill-artifact-{}",
        NEXT_TEST_DIRECTORY.fetch_add(1, Ordering::Relaxed),
    ));
    let error = config_with_artifact(&missing)?
        .build()
        .err()
        .ok_or_else(|| SwitchyardError::Other("missing artifact should fail".to_string()))?;
    assert!(format!("{error}").contains("failed to read"));
    assert!(format!("{error}").contains("router.json"));
    Ok(())
}

#[test]
fn checkpoint_head_mappings_are_validated_at_build() -> Result<()> {
    let artifact = TestArtifactDirectory::create(ArtifactFault::None)?;
    let mut unknown = config_with_artifact(artifact.path())?;
    unknown.weak_checkpoint_head = "unknown-head".to_string();
    let unknown_error = unknown
        .build()
        .err()
        .ok_or_else(|| SwitchyardError::Other("unknown head should fail".to_string()))?;
    assert!(format!("{unknown_error}").contains("weak_checkpoint_head `unknown-head`"));

    let mut duplicate = config_with_artifact(artifact.path())?;
    duplicate.weak_checkpoint_head = duplicate.strong_checkpoint_head.clone();
    let duplicate_error = duplicate
        .build()
        .err()
        .ok_or_else(|| SwitchyardError::Other("duplicate head should fail".to_string()))?;
    assert!(format!("{duplicate_error}").contains("must map to distinct outputs"));
    Ok(())
}

#[test]
fn invalid_policy_values_are_reported_before_artifact_loading() -> Result<()> {
    let cases = [
        (
            PrefillProbeRoutingPolicyConfig::CostAware {
                lambda: 1.5,
                weak_cost: 0.01,
                strong_cost: 0.10,
            },
            "lambda",
        ),
        (
            PrefillProbeRoutingPolicyConfig::CostAware {
                lambda: 0.5,
                weak_cost: -0.01,
                strong_cost: 0.10,
            },
            "weak_cost",
        ),
        (
            PrefillProbeRoutingPolicyConfig::CostAware {
                lambda: 0.5,
                weak_cost: 0.01,
                strong_cost: f64::INFINITY,
            },
            "strong_cost",
        ),
    ];

    for (routing_policy, expected) in cases {
        let config = PrefillProbeProfileConfig {
            probe: target("probe", "probe/model")?,
            strong: target("strong", "frontier/model")?,
            strong_checkpoint_head: "opus-4.7".to_string(),
            weak: target("weak", "cheap/model")?,
            weak_checkpoint_head: "nemotron-3-super".to_string(),
            hidden_states_dir: "/dev/shm/hidden_states".to_string(),
            inference_artifact_dir: "/missing/artifact".to_string(),
            routing_policy,
        };
        let error = config
            .build()
            .err()
            .ok_or_else(|| SwitchyardError::Other(format!("invalid {expected} should fail")))?;
        assert!(format!("{error}").contains(expected));
    }
    Ok(())
}

#[test]
fn loaded_artifact_is_owned_after_source_files_are_removed() -> Result<()> {
    let artifact = TestArtifactDirectory::create(ArtifactFault::None)?;
    let _profile = config_with_artifact(artifact.path())?.build()?;
    std::fs::remove_dir_all(artifact.path()).map_err(|error| {
        SwitchyardError::Other(format!("failed to remove loaded artifact: {error}"))
    })?;
    Ok(())
}
