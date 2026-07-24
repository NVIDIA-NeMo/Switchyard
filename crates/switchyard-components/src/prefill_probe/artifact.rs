// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Loading, validation, and CPU inference for learned prefill-router artifacts.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use switchyard_core::{Result, SwitchyardError};

const METADATA_FILE: &str = "router.json";
const TENSOR_FILE: &str = "router.safetensors";
const FORMAT_VERSION: u64 = 1;
const TRAINING_MODE: &str = "single_pca_block";
const REPRESENTATION: &str = "token_mean_per_layer_concat";
const PCA_DIM: usize = 200;
const TRUNK_HIDDEN: [usize; 2] = [256, 128];
const ENSEMBLE_SIZE: usize = 5;
const OUTPUT_NAMES: [&str; 4] = ["qwen-122b", "nemotron-3-super", "opus-4.7", "gpt-5.5"];
const PROBABILITY_LINK: &str = "independent_sigmoid";
const ENSEMBLE_REDUCTION: &str = "probability_mean";

/// Immutable learned-router metadata and decoded tensors owned by a profile runtime.
pub(crate) struct InferenceArtifact {
    metadata: ArtifactMetadata,
    tensors: BTreeMap<String, Vec<f32>>,
}

impl InferenceArtifact {
    /// Loads and validates an exported artifact against the configured probe model.
    pub(crate) fn load(directory: impl AsRef<Path>, probe_model: &str) -> Result<Self> {
        let directory = directory.as_ref();
        let metadata_path = directory.join(METADATA_FILE);
        let metadata_json = std::fs::read_to_string(&metadata_path).map_err(|error| {
            invalid_artifact(format!(
                "failed to read {}: {error}",
                metadata_path.display()
            ))
        })?;
        let metadata: ArtifactMetadata = serde_json::from_str(&metadata_json).map_err(|error| {
            invalid_artifact(format!(
                "failed to parse {}: {error}",
                metadata_path.display()
            ))
        })?;
        metadata.validate(probe_model)?;

        let tensor_path = directory.join(&metadata.tensor_file);
        let tensor_bytes = std::fs::read(&tensor_path).map_err(|error| {
            invalid_artifact(format!("failed to read {}: {error}", tensor_path.display()))
        })?;
        let tensors = {
            let tensors = SafeTensors::deserialize(&tensor_bytes).map_err(|error| {
                invalid_artifact(format!(
                    "failed to parse {}: {error}",
                    tensor_path.display()
                ))
            })?;
            validate_tensors(&tensors, &metadata)?;
            decode_tensors(&tensors)?
        };

        Ok(Self { metadata, tensors })
    }

    /// Returns the encoder checkpoint named by the artifact.
    #[cfg(test)]
    pub(crate) fn encoder(&self) -> &str {
        &self.metadata.encoder
    }

    /// Returns the number of independently trained trunk members.
    #[cfg(test)]
    pub(crate) fn ensemble_size(&self) -> usize {
        self.metadata.ensemble_size
    }

    /// Returns the number of output heads in checkpoint order.
    #[cfg(test)]
    pub(crate) fn output_count(&self) -> usize {
        self.metadata.output_names.len()
    }

    /// Returns checkpoint output names in the order used by learned probabilities.
    pub(crate) fn output_names(&self) -> &[String] {
        &self.metadata.output_names
    }

    /// Returns the number of hidden-state layers expected from the probe server.
    pub(crate) fn layer_count(&self) -> usize {
        self.metadata.extraction_layer_ids.len()
    }

    /// Returns the hidden width expected for each extracted layer.
    pub(crate) fn hidden_size(&self) -> usize {
        self.metadata.hidden_size
    }

    /// Returns the flattened token-mean feature dimension.
    pub(crate) fn raw_feature_dim(&self) -> usize {
        self.metadata.raw_feature_dim
    }

    /// Returns the number of decoded artifact tensors.
    #[cfg(test)]
    pub(crate) fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Returns the number of decoded F32 values across all artifact tensors.
    #[cfg(test)]
    pub(crate) fn tensor_value_count(&self) -> usize {
        self.tensors.values().map(Vec::len).sum()
    }

    /// Applies the artifact's fitted scaler followed by its PCA-200 projection.
    pub(crate) fn project(&self, raw_features: &[f32]) -> Result<Vec<f32>> {
        let standardized = standardize(
            raw_features,
            self.tensor("transform.scaler_mean")?,
            self.tensor("transform.scaler_scale")?,
        )?;
        project_pca(
            &standardized,
            self.tensor("transform.pca_mean")?,
            self.tensor("transform.pca_components")?,
            self.metadata.pca_dim,
        )
    }

    /// Runs all learned shared-trunk members and returns logits in artifact output order.
    pub(crate) fn ensemble_logits(&self, pca_features: &[f32]) -> Result<Vec<Vec<f32>>> {
        if pca_features.len() != self.metadata.pca_dim {
            return Err(trunk_error(format!(
                "input dimension {} does not match pca_dim {}",
                pca_features.len(),
                self.metadata.pca_dim,
            )));
        }

        let mut ensemble_logits = Vec::with_capacity(self.metadata.ensemble_size);
        for index in 0..self.metadata.ensemble_size {
            let prefix = format!("ensemble.{index}");
            let hidden1 = dense_layer(
                pca_features,
                self.tensor(&format!("{prefix}.linear1.weight"))?,
                self.tensor(&format!("{prefix}.linear1.bias"))?,
                TRUNK_HIDDEN[0],
                true,
            )?;
            let hidden2 = dense_layer(
                &hidden1,
                self.tensor(&format!("{prefix}.linear2.weight"))?,
                self.tensor(&format!("{prefix}.linear2.bias"))?,
                TRUNK_HIDDEN[1],
                true,
            )?;
            let logits = dense_layer(
                &hidden2,
                self.tensor(&format!("{prefix}.output.weight"))?,
                self.tensor(&format!("{prefix}.output.bias"))?,
                self.metadata.output_names.len(),
                false,
            )?;
            ensemble_logits.push(logits);
        }
        Ok(ensemble_logits)
    }

    /// Applies independent sigmoid links and averages probabilities across members.
    pub(crate) fn ensemble_probabilities(&self, ensemble_logits: &[Vec<f32>]) -> Result<Vec<f32>> {
        if self.metadata.ensemble_size == 0 || ensemble_logits.len() != self.metadata.ensemble_size
        {
            return Err(trunk_error(format!(
                "logit member count {} does not match ensemble_size {}",
                ensemble_logits.len(),
                self.metadata.ensemble_size,
            )));
        }

        let output_count = self.metadata.output_names.len();
        let mut probability_sums = vec![0.0f32; output_count];
        for (member, logits) in ensemble_logits.iter().enumerate() {
            if logits.len() != output_count {
                return Err(trunk_error(format!(
                    "member {member} logit count {} does not match output count {output_count}",
                    logits.len(),
                )));
            }
            for (sum, logit) in probability_sums.iter_mut().zip(logits) {
                *sum += sigmoid_probability(*logit)?;
            }
        }

        let member_count = self.metadata.ensemble_size as f32;
        probability_sums
            .iter_mut()
            .for_each(|probability| *probability /= member_count);
        Ok(probability_sums)
    }

    fn tensor(&self, name: &str) -> Result<&[f32]> {
        self.tensors
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| invalid_artifact(format!("missing decoded tensor {name}")))
    }
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(serde::Serialize))]
struct ArtifactMetadata {
    format_version: u64,
    training_mode: String,
    encoder: String,
    representation: String,
    extraction_layer_ids: Vec<usize>,
    hidden_size: usize,
    raw_feature_dim: usize,
    feature_block_count: usize,
    pca_dim: usize,
    pca_whiten: bool,
    output_names: Vec<String>,
    trunk_hidden: Vec<usize>,
    ensemble_size: usize,
    probability_link: String,
    ensemble_reduction: String,
    tensor_file: String,
}

impl ArtifactMetadata {
    fn validate(&self, probe_model: &str) -> Result<()> {
        require(
            self.format_version == FORMAT_VERSION,
            format!(
                "unsupported format_version {}; expected {FORMAT_VERSION}",
                self.format_version
            ),
        )?;
        require(
            self.training_mode == TRAINING_MODE,
            format!(
                "training_mode must be {TRAINING_MODE}; got {}",
                self.training_mode
            ),
        )?;
        require(
            self.encoder == probe_model,
            format!(
                "artifact encoder {} does not match probe model {probe_model}",
                self.encoder
            ),
        )?;
        require(
            self.representation == REPRESENTATION,
            format!(
                "representation must be {REPRESENTATION}; got {}",
                self.representation
            ),
        )?;
        require(
            !self.extraction_layer_ids.is_empty(),
            "extraction_layer_ids must not be empty",
        )?;
        let expected_layer_ids = (0..self.extraction_layer_ids.len()).collect::<Vec<_>>();
        require(
            self.extraction_layer_ids == expected_layer_ids,
            "extraction_layer_ids must be contiguous and ordered from zero",
        )?;
        require(self.hidden_size > 0, "hidden_size must be positive")?;
        let expected_raw_dim = self
            .extraction_layer_ids
            .len()
            .checked_mul(self.hidden_size)
            .ok_or_else(|| invalid_artifact("raw feature dimension overflow"))?;
        require(
            self.raw_feature_dim == expected_raw_dim,
            format!(
                "raw_feature_dim {} does not equal layer count {} * hidden_size {}",
                self.raw_feature_dim,
                self.extraction_layer_ids.len(),
                self.hidden_size
            ),
        )?;
        require(
            self.feature_block_count == 1,
            format!(
                "feature_block_count must be 1; got {}",
                self.feature_block_count
            ),
        )?;
        require(
            self.pca_dim == PCA_DIM,
            format!("pca_dim must be {PCA_DIM}; got {}", self.pca_dim),
        )?;
        require(!self.pca_whiten, "pca_whiten must be false")?;
        require(
            self.output_names
                .iter()
                .map(String::as_str)
                .eq(OUTPUT_NAMES),
            format!("output_names must be ordered as {OUTPUT_NAMES:?}"),
        )?;
        require(
            self.trunk_hidden == TRUNK_HIDDEN,
            format!("trunk_hidden must be {TRUNK_HIDDEN:?}"),
        )?;
        require(
            self.ensemble_size == ENSEMBLE_SIZE,
            format!(
                "ensemble_size must be {ENSEMBLE_SIZE}; got {}",
                self.ensemble_size
            ),
        )?;
        require(
            self.probability_link == PROBABILITY_LINK,
            format!(
                "probability_link must be {PROBABILITY_LINK}; got {}",
                self.probability_link
            ),
        )?;
        require(
            self.ensemble_reduction == ENSEMBLE_REDUCTION,
            format!(
                "ensemble_reduction must be {ENSEMBLE_REDUCTION}; got {}",
                self.ensemble_reduction
            ),
        )?;
        require(
            self.tensor_file == TENSOR_FILE,
            format!(
                "tensor_file must be {TENSOR_FILE}; got {}",
                self.tensor_file
            ),
        )?;
        Ok(())
    }
}

struct TensorSpec {
    name: String,
    shape: Vec<usize>,
}

fn expected_tensors(metadata: &ArtifactMetadata) -> Vec<TensorSpec> {
    let mut expected = vec![
        TensorSpec {
            name: "transform.scaler_mean".into(),
            shape: vec![metadata.raw_feature_dim],
        },
        TensorSpec {
            name: "transform.scaler_scale".into(),
            shape: vec![metadata.raw_feature_dim],
        },
        TensorSpec {
            name: "transform.pca_mean".into(),
            shape: vec![metadata.raw_feature_dim],
        },
        TensorSpec {
            name: "transform.pca_components".into(),
            shape: vec![metadata.pca_dim, metadata.raw_feature_dim],
        },
    ];
    for index in 0..metadata.ensemble_size {
        let prefix = format!("ensemble.{index}");
        expected.extend([
            TensorSpec {
                name: format!("{prefix}.linear1.weight"),
                shape: vec![TRUNK_HIDDEN[0], metadata.pca_dim],
            },
            TensorSpec {
                name: format!("{prefix}.linear1.bias"),
                shape: vec![TRUNK_HIDDEN[0]],
            },
            TensorSpec {
                name: format!("{prefix}.linear2.weight"),
                shape: vec![TRUNK_HIDDEN[1], TRUNK_HIDDEN[0]],
            },
            TensorSpec {
                name: format!("{prefix}.linear2.bias"),
                shape: vec![TRUNK_HIDDEN[1]],
            },
            TensorSpec {
                name: format!("{prefix}.output.weight"),
                shape: vec![metadata.output_names.len(), TRUNK_HIDDEN[1]],
            },
            TensorSpec {
                name: format!("{prefix}.output.bias"),
                shape: vec![metadata.output_names.len()],
            },
        ]);
    }
    expected
}

fn validate_tensors(tensors: &SafeTensors<'_>, metadata: &ArtifactMetadata) -> Result<()> {
    let expected = expected_tensors(metadata);
    for spec in &expected {
        let tensor = tensors
            .tensor(&spec.name)
            .map_err(|_| invalid_artifact(format!("missing tensor {}", spec.name)))?;
        require(
            tensor.dtype() == Dtype::F32,
            format!("tensor {} must use F32", spec.name),
        )?;
        require(
            tensor.shape() == spec.shape,
            format!(
                "tensor {} has shape {:?}; expected {:?}",
                spec.name,
                tensor.shape(),
                spec.shape
            ),
        )?;
        validate_finite_f32(&spec.name, tensor.data())?;
        if spec.name == "transform.scaler_scale" {
            validate_positive_f32(&spec.name, tensor.data())?;
        }
    }

    let expected_names = expected
        .iter()
        .map(|spec| spec.name.as_str())
        .collect::<BTreeSet<_>>();
    let actual_names = tensors
        .names()
        .into_iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    require(
        actual_names == expected_names,
        "artifact contains unexpected tensors",
    )?;
    Ok(())
}

/// Decodes artifact tensors once so requests never reparse the safetensors file.
fn decode_tensors(tensors: &SafeTensors<'_>) -> Result<BTreeMap<String, Vec<f32>>> {
    let mut decoded = BTreeMap::new();
    for name in tensors.names() {
        let tensor = tensors
            .tensor(name)
            .map_err(|error| invalid_artifact(format!("failed to read tensor {name}: {error}")))?;
        let values = tensor
            .data()
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect();
        decoded.insert(name.clone(), values);
    }
    Ok(decoded)
}

/// Applies the fitted `StandardScaler` elementwise in F32.
fn standardize(raw: &[f32], mean: &[f32], scale: &[f32]) -> Result<Vec<f32>> {
    if raw.len() != mean.len() || raw.len() != scale.len() {
        return Err(transform_error(format!(
            "scaler dimensions do not match: raw={}, mean={}, scale={}",
            raw.len(),
            mean.len(),
            scale.len(),
        )));
    }

    raw.iter()
        .zip(mean)
        .zip(scale)
        .map(|((value, mean), scale)| {
            if !value.is_finite() {
                return Err(transform_error("raw features contain non-finite values"));
            }
            if !scale.is_finite() || *scale <= 0.0 {
                return Err(transform_error("scaler scale must be finite and positive"));
            }
            let standardized = (*value - *mean) / *scale;
            if !standardized.is_finite() {
                return Err(transform_error(
                    "standardization produced a non-finite value",
                ));
            }
            Ok(standardized)
        })
        .collect()
}

/// Projects standardized features using row-major sklearn PCA components.
fn project_pca(
    standardized: &[f32],
    mean: &[f32],
    components: &[f32],
    output_dim: usize,
) -> Result<Vec<f32>> {
    if standardized.is_empty() || standardized.len() != mean.len() {
        return Err(transform_error(format!(
            "PCA input dimensions do not match: standardized={}, mean={}",
            standardized.len(),
            mean.len(),
        )));
    }
    let expected_components = output_dim
        .checked_mul(standardized.len())
        .ok_or_else(|| transform_error("PCA component dimensions overflow"))?;
    if output_dim == 0 || components.len() != expected_components {
        return Err(transform_error(format!(
            "PCA component dimensions do not match: values={}, expected={expected_components}",
            components.len(),
        )));
    }

    let centered = standardized
        .iter()
        .zip(mean)
        .map(|(value, mean)| {
            let centered = *value - *mean;
            if centered.is_finite() {
                Ok(centered)
            } else {
                Err(transform_error("PCA centering produced a non-finite value"))
            }
        })
        .collect::<Result<Vec<_>>>()?;

    components
        .chunks_exact(standardized.len())
        .map(|component| {
            // Wide F32 dot products drift beyond the exporter's golden tolerance.
            let projected = component
                .iter()
                .zip(&centered)
                .map(|(weight, value)| f64::from(*weight) * f64::from(*value))
                .sum::<f64>() as f32;
            if projected.is_finite() {
                Ok(projected)
            } else {
                Err(transform_error(
                    "PCA projection produced a non-finite value",
                ))
            }
        })
        .collect()
}

/// Applies a row-major dense layer and an optional ReLU activation.
fn dense_layer(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output_dim: usize,
    relu: bool,
) -> Result<Vec<f32>> {
    if input.is_empty() || output_dim == 0 || bias.len() != output_dim {
        return Err(trunk_error(format!(
            "dense dimensions do not match: input={}, output={output_dim}, bias={}",
            input.len(),
            bias.len(),
        )));
    }
    let expected_weights = output_dim
        .checked_mul(input.len())
        .ok_or_else(|| trunk_error("dense weight dimensions overflow"))?;
    if weights.len() != expected_weights {
        return Err(trunk_error(format!(
            "dense weight dimensions do not match: values={}, expected={expected_weights}",
            weights.len(),
        )));
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(trunk_error("dense input contains non-finite values"));
    }

    weights
        .chunks_exact(input.len())
        .zip(bias)
        .map(|(row, bias)| {
            let value = row
                .iter()
                .zip(input)
                .map(|(weight, input)| *weight * *input)
                .sum::<f32>()
                + *bias;
            if !value.is_finite() {
                return Err(trunk_error("dense layer produced a non-finite value"));
            }
            Ok(if relu { value.max(0.0) } else { value })
        })
        .collect()
}

/// Converts one finite logit to a probability without exponential overflow.
fn sigmoid_probability(logit: f32) -> Result<f32> {
    if !logit.is_finite() {
        return Err(trunk_error("sigmoid input contains a non-finite value"));
    }
    let probability = if logit >= 0.0 {
        1.0 / (1.0 + (-logit).exp())
    } else {
        let exp = logit.exp();
        exp / (1.0 + exp)
    };
    if probability.is_finite() {
        Ok(probability)
    } else {
        Err(trunk_error("sigmoid produced a non-finite value"))
    }
}

fn validate_finite_f32(name: &str, data: &[u8]) -> Result<()> {
    for bytes in data.chunks_exact(4) {
        let value = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        require(
            value.is_finite(),
            format!("tensor {name} contains non-finite values"),
        )?;
    }
    Ok(())
}

fn validate_positive_f32(name: &str, data: &[u8]) -> Result<()> {
    for bytes in data.chunks_exact(4) {
        let value = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        require(
            value > 0.0,
            format!("tensor {name} contains non-positive values"),
        )?;
    }
    Ok(())
}

fn require(condition: bool, message: impl Into<String>) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(invalid_artifact(message))
    }
}

fn invalid_artifact(message: impl Into<String>) -> SwitchyardError {
    SwitchyardError::InvalidConfig(format!(
        "invalid prefill-router artifact: {}",
        message.into()
    ))
}

fn transform_error(message: impl Into<String>) -> SwitchyardError {
    SwitchyardError::Other(format!(
        "prefill-router feature transform error: {}",
        message.into()
    ))
}

fn trunk_error(message: impl Into<String>) -> SwitchyardError {
    SwitchyardError::Other(format!(
        "prefill-router trunk inference error: {}",
        message.into()
    ))
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    use safetensors::tensor::{serialize, TensorView};

    use super::*;

    static NEXT_TEST_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    struct TestArtifactDirectory {
        path: PathBuf,
    }

    impl TestArtifactDirectory {
        fn create() -> Result<Self> {
            let sequence = NEXT_TEST_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "switchyard-prefill-artifact-{}-{sequence}",
                std::process::id()
            ));
            std::fs::create_dir(&path).map_err(|error| {
                invalid_artifact(format!(
                    "failed to create test directory {}: {error}",
                    path.display()
                ))
            })?;
            Ok(Self { path })
        }

        fn write_metadata(&self, metadata: &ArtifactMetadata) -> Result<()> {
            let bytes = serde_json::to_vec(metadata).map_err(|error| {
                invalid_artifact(format!("failed to serialize test metadata: {error}"))
            })?;
            self.write(METADATA_FILE, &bytes)
        }

        fn write(&self, name: &str, bytes: &[u8]) -> Result<()> {
            let path = self.path.join(name);
            std::fs::write(&path, bytes).map_err(|error| {
                invalid_artifact(format!("failed to write {}: {error}", path.display()))
            })
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TestArtifactDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn test_metadata() -> ArtifactMetadata {
        ArtifactMetadata {
            format_version: FORMAT_VERSION,
            training_mode: TRAINING_MODE.into(),
            encoder: "probe/model".into(),
            representation: REPRESENTATION.into(),
            extraction_layer_ids: vec![0, 1],
            hidden_size: 2,
            raw_feature_dim: 4,
            feature_block_count: 1,
            pca_dim: PCA_DIM,
            pca_whiten: false,
            output_names: OUTPUT_NAMES.iter().map(|name| (*name).into()).collect(),
            trunk_hidden: TRUNK_HIDDEN.to_vec(),
            ensemble_size: ENSEMBLE_SIZE,
            probability_link: PROBABILITY_LINK.into(),
            ensemble_reduction: ENSEMBLE_REDUCTION.into(),
            tensor_file: TENSOR_FILE.into(),
        }
    }

    fn repeated_f32_bytes(value: f32, count: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(count * size_of::<f32>());
        for _ in 0..count {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn serialize_single_tensor(name: &str, shape: Vec<usize>, values: &[f32]) -> Result<Vec<u8>> {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let tensor = TensorView::new(Dtype::F32, shape, &bytes).map_err(|error| {
            invalid_artifact(format!("failed to create test tensor {name}: {error}"))
        })?;
        serialize([(name, tensor)], &None).map_err(|error| {
            invalid_artifact(format!("failed to serialize test tensor {name}: {error}"))
        })
    }

    fn serialize_valid_artifact(metadata: &ArtifactMetadata) -> Result<Vec<u8>> {
        let storage = expected_tensors(metadata)
            .into_iter()
            .map(|spec| {
                let value_count = spec.shape.iter().product();
                let fill = if spec.name == "transform.scaler_scale" {
                    1.0
                } else {
                    0.0
                };
                (spec.name, spec.shape, repeated_f32_bytes(fill, value_count))
            })
            .collect::<Vec<_>>();
        let tensors = storage
            .iter()
            .map(|(name, shape, bytes)| {
                TensorView::new(Dtype::F32, shape.clone(), bytes)
                    .map(|tensor| (name.as_str(), tensor))
                    .map_err(|error| {
                        invalid_artifact(format!("failed to create test tensor {name}: {error}"))
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        serialize(tensors, &None).map_err(|error| {
            invalid_artifact(format!("failed to serialize test artifact: {error}"))
        })
    }

    fn trunk_test_artifact() -> InferenceArtifact {
        let metadata = test_metadata();
        let mut tensors = BTreeMap::new();
        for index in 0..ENSEMBLE_SIZE {
            let prefix = format!("ensemble.{index}");
            let mut linear1_bias = vec![0.0; TRUNK_HIDDEN[0]];
            linear1_bias[0] = (index + 1) as f32;
            let mut linear2_weight = vec![0.0; TRUNK_HIDDEN[1] * TRUNK_HIDDEN[0]];
            linear2_weight[0] = 1.0;
            let mut output_weight = vec![0.0; OUTPUT_NAMES.len() * TRUNK_HIDDEN[1]];
            for (output, factor) in [1.0, -2.0, 3.0, -4.0].into_iter().enumerate() {
                output_weight[output * TRUNK_HIDDEN[1]] = factor;
            }
            tensors.insert(
                format!("{prefix}.linear1.weight"),
                vec![0.0; TRUNK_HIDDEN[0] * PCA_DIM],
            );
            tensors.insert(format!("{prefix}.linear1.bias"), linear1_bias);
            tensors.insert(format!("{prefix}.linear2.weight"), linear2_weight);
            tensors.insert(format!("{prefix}.linear2.bias"), vec![0.0; TRUNK_HIDDEN[1]]);
            tensors.insert(format!("{prefix}.output.weight"), output_weight);
            tensors.insert(
                format!("{prefix}.output.bias"),
                vec![0.0; OUTPUT_NAMES.len()],
            );
        }
        InferenceArtifact { metadata, tensors }
    }

    #[test]
    fn artifact_loads_decodes_and_executes_the_exported_shape() -> Result<()> {
        let directory = TestArtifactDirectory::create()?;
        let metadata = test_metadata();
        directory.write_metadata(&metadata)?;
        directory.write(TENSOR_FILE, &serialize_valid_artifact(&metadata)?)?;

        let artifact = InferenceArtifact::load(directory.path(), "probe/model")?;
        assert_eq!(artifact.encoder(), "probe/model");
        assert_eq!(artifact.layer_count(), 2);
        assert_eq!(artifact.hidden_size(), 2);
        assert_eq!(artifact.raw_feature_dim(), 4);
        assert_eq!(artifact.ensemble_size(), ENSEMBLE_SIZE);
        assert_eq!(artifact.output_count(), OUTPUT_NAMES.len());
        assert!(artifact
            .output_names()
            .iter()
            .map(String::as_str)
            .eq(OUTPUT_NAMES));
        assert_eq!(artifact.tensor_count(), 4 + ENSEMBLE_SIZE * 6);
        assert!(artifact.tensor_value_count() > 0);

        let projected = artifact.project(&vec![0.0; artifact.raw_feature_dim()])?;
        assert_eq!(projected, vec![0.0; PCA_DIM]);
        let logits = artifact.ensemble_logits(&projected)?;
        assert_eq!(logits, vec![vec![0.0; OUTPUT_NAMES.len()]; ENSEMBLE_SIZE]);
        let probabilities = artifact.ensemble_probabilities(&logits)?;
        assert_eq!(probabilities, vec![0.5; OUTPUT_NAMES.len()]);
        Ok(())
    }

    #[test]
    fn metadata_rejects_encoder_layer_and_raw_dimension_mismatches() -> Result<()> {
        let metadata = test_metadata();
        let encoder_error = metadata
            .validate("different/probe")
            .err()
            .ok_or_else(|| invalid_artifact("encoder mismatch should fail"))?;
        assert!(format!("{encoder_error}").contains("does not match probe model"));

        let mut metadata = test_metadata();
        metadata.extraction_layer_ids = vec![0, 2];
        let layer_error = metadata
            .validate("probe/model")
            .err()
            .ok_or_else(|| invalid_artifact("layer ordering mismatch should fail"))?;
        assert!(format!("{layer_error}").contains("contiguous and ordered"));

        let mut metadata = test_metadata();
        metadata.raw_feature_dim += 1;
        let dimension_error = metadata
            .validate("probe/model")
            .err()
            .ok_or_else(|| invalid_artifact("raw dimension mismatch should fail"))?;
        assert!(format!("{dimension_error}").contains("does not equal layer count"));
        Ok(())
    }

    #[test]
    fn tensor_validation_rejects_missing_and_malformed_tensors() -> Result<()> {
        let metadata = test_metadata();
        let unexpected = serialize_single_tensor("unexpected", vec![1], &[0.0])?;
        let tensors = SafeTensors::deserialize(&unexpected).map_err(|error| {
            invalid_artifact(format!("failed to deserialize test tensor: {error}"))
        })?;
        let missing_error = validate_tensors(&tensors, &metadata)
            .err()
            .ok_or_else(|| invalid_artifact("missing tensor should fail"))?;
        assert!(format!("{missing_error}").contains("missing tensor transform.scaler_mean"));

        let wrong_shape = serialize_single_tensor(
            "transform.scaler_mean",
            vec![metadata.raw_feature_dim - 1],
            &vec![0.0; metadata.raw_feature_dim - 1],
        )?;
        let tensors = SafeTensors::deserialize(&wrong_shape).map_err(|error| {
            invalid_artifact(format!("failed to deserialize test tensor: {error}"))
        })?;
        let shape_error = validate_tensors(&tensors, &metadata)
            .err()
            .ok_or_else(|| invalid_artifact("malformed tensor shape should fail"))?;
        assert!(format!("{shape_error}").contains("has shape"));
        Ok(())
    }

    #[test]
    fn tensor_validation_rejects_non_finite_values() -> Result<()> {
        let metadata = test_metadata();
        let mut values = vec![0.0; metadata.raw_feature_dim];
        values[0] = f32::NAN;
        let encoded = serialize_single_tensor(
            "transform.scaler_mean",
            vec![metadata.raw_feature_dim],
            &values,
        )?;
        let tensors = SafeTensors::deserialize(&encoded).map_err(|error| {
            invalid_artifact(format!("failed to deserialize test tensor: {error}"))
        })?;
        let error = validate_tensors(&tensors, &metadata)
            .err()
            .ok_or_else(|| invalid_artifact("non-finite tensor should fail"))?;
        assert!(format!("{error}").contains("contains non-finite values"));
        Ok(())
    }

    #[test]
    fn artifact_load_rejects_malformed_tensor_file() -> Result<()> {
        let directory = TestArtifactDirectory::create()?;
        directory.write_metadata(&test_metadata())?;
        directory.write(TENSOR_FILE, b"not a safetensors file")?;

        let error = InferenceArtifact::load(directory.path(), "probe/model")
            .err()
            .ok_or_else(|| invalid_artifact("malformed tensor file should fail"))?;
        assert!(format!("{error}").contains("failed to parse"));
        Ok(())
    }

    #[test]
    fn scaler_and_pca_match_exported_row_major_math() -> Result<()> {
        let standardized = standardize(&[3.0, 6.0, 11.0], &[1.0, 2.0, 3.0], &[2.0, 2.0, 4.0])?;
        assert_eq!(standardized, vec![1.0, 2.0, 2.0]);

        let projected = project_pca(
            &standardized,
            &[0.5, 1.0, 1.5],
            &[
                1.0, 10.0, 100.0, // PCA component 0
                -2.0, 0.5, 4.0, // PCA component 1
            ],
            2,
        )?;
        assert_eq!(projected, vec![60.5, 1.5]);
        Ok(())
    }

    #[test]
    fn scaler_rejects_dimension_mismatch() -> Result<()> {
        let error = standardize(&[1.0, 2.0], &[0.0], &[1.0, 1.0])
            .err()
            .ok_or_else(|| transform_error("dimension mismatch should fail"))?;
        assert!(format!("{error}").contains("scaler dimensions do not match"));
        Ok(())
    }

    #[test]
    fn scaler_rejects_non_finite_output() -> Result<()> {
        let error = standardize(&[f32::MAX], &[-f32::MAX], &[1.0])
            .err()
            .ok_or_else(|| transform_error("non-finite output should fail"))?;
        assert!(format!("{error}").contains("standardization produced"));
        Ok(())
    }

    #[test]
    fn pca_rejects_component_shape_mismatch() -> Result<()> {
        let error = project_pca(&[1.0, 2.0], &[0.0, 0.0], &[1.0, 2.0, 3.0], 2)
            .err()
            .ok_or_else(|| transform_error("component mismatch should fail"))?;
        assert!(format!("{error}").contains("PCA component dimensions do not match"));
        Ok(())
    }

    #[test]
    fn pca_rejects_non_finite_output() -> Result<()> {
        let error = project_pca(&[f32::MAX, f32::MAX], &[0.0, 0.0], &[2.0, 2.0], 1)
            .err()
            .ok_or_else(|| transform_error("non-finite projection should fail"))?;
        assert!(format!("{error}").contains("PCA projection produced"));
        Ok(())
    }

    #[test]
    fn dense_layer_applies_row_major_weights_bias_and_relu() -> Result<()> {
        let output = dense_layer(
            &[2.0, -1.0],
            &[
                1.0, 2.0, // output row 0
                -3.0, 4.0, // output row 1
            ],
            &[1.0, 1.0],
            2,
            true,
        )?;
        assert_eq!(output, vec![1.0, 0.0]);
        Ok(())
    }

    #[test]
    fn dense_layer_rejects_weight_shape_mismatch() -> Result<()> {
        let error = dense_layer(&[1.0, 2.0], &[1.0, 2.0, 3.0], &[0.0, 0.0], 2, false)
            .err()
            .ok_or_else(|| trunk_error("weight mismatch should fail"))?;
        assert!(format!("{error}").contains("dense weight dimensions do not match"));
        Ok(())
    }

    #[test]
    fn dense_layer_rejects_non_finite_output() -> Result<()> {
        let error = dense_layer(&[f32::MAX], &[2.0], &[0.0], 1, false)
            .err()
            .ok_or_else(|| trunk_error("non-finite output should fail"))?;
        assert!(format!("{error}").contains("dense layer produced"));
        Ok(())
    }

    #[test]
    fn ensemble_trunk_preserves_member_and_output_order() -> Result<()> {
        let logits = trunk_test_artifact().ensemble_logits(&vec![0.0; PCA_DIM])?;
        assert_eq!(logits.len(), ENSEMBLE_SIZE);
        for (member, member_logits) in logits.iter().enumerate() {
            let scale = (member + 1) as f32;
            assert_eq!(
                member_logits,
                &[scale, -2.0 * scale, 3.0 * scale, -4.0 * scale]
            );
        }
        Ok(())
    }

    #[test]
    fn ensemble_trunk_rejects_wrong_pca_dimension() -> Result<()> {
        let error = trunk_test_artifact()
            .ensemble_logits(&[0.0; PCA_DIM - 1])
            .err()
            .ok_or_else(|| trunk_error("PCA dimension mismatch should fail"))?;
        assert!(format!("{error}").contains("does not match pca_dim"));
        Ok(())
    }

    #[test]
    fn sigmoid_is_stable_for_extreme_logits() -> Result<()> {
        assert_eq!(sigmoid_probability(-f32::MAX)?, 0.0);
        assert_eq!(sigmoid_probability(0.0)?, 0.5);
        assert_eq!(sigmoid_probability(f32::MAX)?, 1.0);
        Ok(())
    }

    #[test]
    fn ensemble_probabilities_average_members_in_output_order() -> Result<()> {
        let log_three = 3.0f32.ln();
        let mut logits = vec![vec![0.0; OUTPUT_NAMES.len()]; ENSEMBLE_SIZE];
        logits[0][0] = log_three;
        for member in &mut logits {
            member[2] = log_three;
            member[3] = -log_three;
        }

        let probabilities = trunk_test_artifact().ensemble_probabilities(&logits)?;
        let expected = [0.55, 0.5, 0.75, 0.25];
        for (actual, expected) in probabilities.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
        assert!((probabilities.iter().sum::<f32>() - 1.0).abs() > 0.5);
        Ok(())
    }

    #[test]
    fn ensemble_probabilities_reject_wrong_member_count() -> Result<()> {
        let logits = vec![vec![0.0; OUTPUT_NAMES.len()]; ENSEMBLE_SIZE - 1];
        let error = trunk_test_artifact()
            .ensemble_probabilities(&logits)
            .err()
            .ok_or_else(|| trunk_error("member count mismatch should fail"))?;
        assert!(format!("{error}").contains("does not match ensemble_size"));
        Ok(())
    }

    #[test]
    fn ensemble_probabilities_reject_wrong_output_count() -> Result<()> {
        let logits = vec![vec![0.0; OUTPUT_NAMES.len() - 1]; ENSEMBLE_SIZE];
        let error = trunk_test_artifact()
            .ensemble_probabilities(&logits)
            .err()
            .ok_or_else(|| trunk_error("output count mismatch should fail"))?;
        assert!(format!("{error}").contains("does not match output count"));
        Ok(())
    }

    #[test]
    fn ensemble_probabilities_reject_non_finite_logits() -> Result<()> {
        let mut logits = vec![vec![0.0; OUTPUT_NAMES.len()]; ENSEMBLE_SIZE];
        logits[0][0] = f32::NAN;
        let error = trunk_test_artifact()
            .ensemble_probabilities(&logits)
            .err()
            .ok_or_else(|| trunk_error("non-finite logit should fail"))?;
        assert!(format!("{error}").contains("sigmoid input contains a non-finite value"));
        Ok(())
    }
}
