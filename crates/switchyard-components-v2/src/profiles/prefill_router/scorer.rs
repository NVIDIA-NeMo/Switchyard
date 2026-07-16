// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Validation and token-mean reduction for vLLM hidden-state artifacts.

use std::fs::{File, OpenOptions};
use std::io::Read;
use std::path::{Path, PathBuf};

use safetensors::{Dtype, SafeTensors};
use switchyard_core::{Result, SwitchyardError};

use super::artifact::InferenceArtifact;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HiddenStateLayout {
    layer_count: usize,
    hidden_size: usize,
    raw_feature_dim: usize,
}

impl HiddenStateLayout {
    fn from_artifact(artifact: &InferenceArtifact) -> Self {
        Self {
            layer_count: artifact.layer_count(),
            hidden_size: artifact.hidden_size(),
            raw_feature_dim: artifact.raw_feature_dim(),
        }
    }
}

/// Decodes `[tokens, layers, hidden]` data into one layer-major token-mean vector.
fn token_mean_per_layer(
    data: &[u8],
    dtype: Dtype,
    shape: &[usize],
    expected: HiddenStateLayout,
) -> Result<Vec<f32>> {
    if shape.len() != 3 {
        return Err(hidden_state_error(
            "expected hidden_states shape [prompt_tokens, layers, hidden_size]",
        ));
    }
    let (prompt_tokens, layer_count, hidden_size) = (shape[0], shape[1], shape[2]);
    if prompt_tokens == 0 {
        return Err(hidden_state_error(
            "hidden_states token dimension must be non-zero",
        ));
    }
    if layer_count == 0 || hidden_size == 0 {
        return Err(hidden_state_error(
            "hidden_states layer and hidden dimensions must be non-zero",
        ));
    }
    if layer_count != expected.layer_count {
        return Err(hidden_state_error(format!(
            "hidden_states layer count {layer_count} does not match artifact layer count {}",
            expected.layer_count,
        )));
    }
    if hidden_size != expected.hidden_size {
        return Err(hidden_state_error(format!(
            "hidden_states hidden size {hidden_size} does not match artifact hidden size {}",
            expected.hidden_size,
        )));
    }

    let element_size = match dtype {
        Dtype::F32 => size_of::<f32>(),
        Dtype::BF16 => size_of::<u16>(),
        other => {
            return Err(hidden_state_error(format!(
                "unsupported hidden_states dtype: {other:?}"
            )))
        }
    };
    let features_per_token = layer_count
        .checked_mul(hidden_size)
        .ok_or_else(|| hidden_state_error("hidden_states shape is too large"))?;
    let bytes_per_token = features_per_token
        .checked_mul(element_size)
        .ok_or_else(|| hidden_state_error("hidden_states byte length is too large"))?;
    let expected_bytes = prompt_tokens
        .checked_mul(bytes_per_token)
        .ok_or_else(|| hidden_state_error("hidden_states byte length is too large"))?;
    if data.len() != expected_bytes {
        return Err(hidden_state_error(format!(
            "hidden_states byte length {} does not match shape byte length {expected_bytes}",
            data.len(),
        )));
    }

    let mut pooled = vec![0.0f32; features_per_token];
    match dtype {
        Dtype::F32 => {
            for token in data.chunks_exact(bytes_per_token) {
                for (index, bytes) in token.chunks_exact(size_of::<f32>()).enumerate() {
                    let value = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                    accumulate(&mut pooled[index], value)?;
                }
            }
        }
        Dtype::BF16 => {
            for token in data.chunks_exact(bytes_per_token) {
                for (index, bytes) in token.chunks_exact(size_of::<u16>()).enumerate() {
                    let value = half::bf16::from_le_bytes([bytes[0], bytes[1]]).to_f32();
                    accumulate(&mut pooled[index], value)?;
                }
            }
        }
        other => {
            return Err(hidden_state_error(format!(
                "unsupported hidden_states dtype: {other:?}"
            )))
        }
    }

    let token_count = prompt_tokens as f32;
    for value in &mut pooled {
        *value /= token_count;
        if !value.is_finite() {
            return Err(hidden_state_error(
                "hidden-state token mean produced a non-finite value",
            ));
        }
    }
    if pooled.len() != expected.raw_feature_dim {
        return Err(hidden_state_error(format!(
            "hidden-state feature length {} does not match artifact raw_feature_dim {}",
            pooled.len(),
            expected.raw_feature_dim,
        )));
    }
    Ok(pooled)
}

fn accumulate(sum: &mut f32, value: f32) -> Result<()> {
    if !value.is_finite() {
        return Err(hidden_state_error(
            "hidden_states contains non-finite values",
        ));
    }
    *sum += value;
    if !sum.is_finite() {
        return Err(hidden_state_error(
            "hidden-state token accumulation produced a non-finite value",
        ));
    }
    Ok(())
}

fn validate_token_ids(tensors: &SafeTensors<'_>, prompt_tokens: usize) -> Result<()> {
    if !tensors
        .names()
        .iter()
        .any(|name| name.as_str() == "token_ids")
    {
        return Ok(());
    }
    let token_ids = tensors
        .tensor("token_ids")
        .map_err(|error| hidden_state_error(format!("token_ids tensor error: {error}")))?;
    if token_ids.dtype() != Dtype::I64 {
        return Err(hidden_state_error(format!(
            "token_ids must use I64; got {:?}",
            token_ids.dtype(),
        )));
    }
    if token_ids.shape() != [prompt_tokens] {
        return Err(hidden_state_error(format!(
            "token_ids shape {:?} does not match hidden_states token count {prompt_tokens}",
            token_ids.shape(),
        )));
    }
    for bytes in token_ids.data().chunks_exact(size_of::<i64>()) {
        let token_id = i64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        if token_id < 0 {
            return Err(hidden_state_error("token_ids contains a negative token ID"));
        }
    }
    Ok(())
}

fn parse_hidden_state_features(bytes: &[u8], expected: HiddenStateLayout) -> Result<Vec<f32>> {
    let tensors = SafeTensors::deserialize(bytes)
        .map_err(|error| hidden_state_error(format!("safetensors parse error: {error}")))?;
    let hidden_states = tensors
        .tensor("hidden_states")
        .map_err(|error| hidden_state_error(format!("hidden_states tensor not found: {error}")))?;
    let prompt_tokens = match hidden_states.shape() {
        [prompt_tokens, _, _] => *prompt_tokens,
        _ => {
            return Err(hidden_state_error(
                "expected hidden_states shape [prompt_tokens, layers, hidden_size]",
            ))
        }
    };
    validate_token_ids(&tensors, prompt_tokens)?;
    token_mean_per_layer(
        hidden_states.data(),
        hidden_states.dtype(),
        hidden_states.shape(),
        expected,
    )
}

fn validate_hidden_states_path(root: &Path, path: &Path) -> Result<PathBuf> {
    if !has_safetensors_extension(path) {
        return Err(hidden_state_error(format!(
            "hidden-state artifact must be a .safetensors file: {}",
            path.display(),
        )));
    }

    let root = root.canonicalize().map_err(|error| {
        hidden_state_error(format!(
            "hidden-states directory {} is not accessible: {error}",
            root.display(),
        ))
    })?;
    if !root.is_dir() {
        return Err(hidden_state_error(format!(
            "hidden-states root is not a directory: {}",
            root.display(),
        )));
    }
    let actual = path.canonicalize().map_err(|error| {
        hidden_state_error(format!(
            "hidden-state artifact {} is not accessible: {error}",
            path.display(),
        ))
    })?;
    if !actual.starts_with(&root) {
        return Err(hidden_state_error(format!(
            "hidden-state artifact {} is outside configured directory {}",
            actual.display(),
            root.display(),
        )));
    }
    if !has_safetensors_extension(&actual) {
        return Err(hidden_state_error(format!(
            "canonical hidden-state artifact must be a .safetensors file: {}",
            actual.display(),
        )));
    }
    let metadata = actual.metadata().map_err(|error| {
        hidden_state_error(format!(
            "hidden-state artifact metadata error for {}: {error}",
            actual.display(),
        ))
    })?;
    if !metadata.is_file() {
        return Err(hidden_state_error(format!(
            "hidden-state artifact is not a regular file: {}",
            actual.display(),
        )));
    }
    Ok(actual)
}

fn has_safetensors_extension(path: &Path) -> bool {
    path.extension().and_then(|extension| extension.to_str()) == Some("safetensors")
}

fn open_locked_artifact(path: &Path) -> Result<File> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(|error| {
            hidden_state_error(format!(
                "hidden-state artifact open error for {}: {error}",
                path.display(),
            ))
        })?;
    file.lock().map_err(|error| {
        hidden_state_error(format!(
            "hidden-state artifact lock error for {}: {error}",
            path.display(),
        ))
    })?;
    Ok(file)
}

/// Reads and removes one validated artifact while holding its exclusive lock.
fn read_and_cleanup_hidden_states(
    root: &Path,
    path: &Path,
    expected: HiddenStateLayout,
) -> Result<Vec<f32>> {
    let artifact_path = validate_hidden_states_path(root, path)?;
    let mut artifact = open_locked_artifact(&artifact_path)?;
    let mut bytes = Vec::new();
    artifact.read_to_end(&mut bytes).map_err(|error| {
        hidden_state_error(format!(
            "hidden-state artifact read error for {}: {error}",
            artifact_path.display(),
        ))
    })?;
    let features = parse_hidden_state_features(&bytes, expected)?;
    std::fs::remove_file(&artifact_path).map_err(|error| {
        hidden_state_error(format!(
            "hidden-state artifact cleanup error for {}: {error}",
            artifact_path.display(),
        ))
    })?;
    Ok(features)
}

fn hidden_state_error(message: impl Into<String>) -> SwitchyardError {
    SwitchyardError::Other(format!(
        "prefill-router hidden-state error: {}",
        message.into()
    ))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use safetensors::tensor::{serialize, TensorView};

    use super::*;

    static NEXT_TEST_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory {
        path: PathBuf,
    }

    impl TestDirectory {
        fn create() -> Result<Self> {
            let sequence = NEXT_TEST_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "switchyard-hidden-state-{}-{sequence}",
                std::process::id()
            ));
            std::fs::create_dir(&path).map_err(|error| {
                hidden_state_error(format!(
                    "failed to create test directory {}: {error}",
                    path.display()
                ))
            })?;
            Ok(Self { path })
        }

        fn path(&self) -> &Path {
            &self.path
        }

        fn create_subdirectory(&self, name: &str) -> Result<PathBuf> {
            let path = self.path.join(name);
            std::fs::create_dir(&path).map_err(|error| {
                hidden_state_error(format!(
                    "failed to create test directory {}: {error}",
                    path.display()
                ))
            })?;
            Ok(path)
        }

        fn write(&self, name: &str, bytes: &[u8]) -> Result<PathBuf> {
            let path = self.path.join(name);
            std::fs::write(&path, bytes).map_err(|error| {
                hidden_state_error(format!("failed to write {}: {error}", path.display()))
            })?;
            Ok(path)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn layout(layer_count: usize, hidden_size: usize) -> HiddenStateLayout {
        HiddenStateLayout {
            layer_count,
            hidden_size,
            raw_feature_dim: layer_count * hidden_size,
        }
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| half::bf16::from_f32(*value).to_le_bytes())
            .collect()
    }

    fn i64_bytes(values: &[i64]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn serialize_f32_hidden_states(
        shape: Vec<usize>,
        values: &[f32],
        token_ids: Option<(Dtype, Vec<usize>, Vec<u8>)>,
    ) -> Result<Vec<u8>> {
        let hidden_data = f32_bytes(values);
        let hidden_view = TensorView::new(Dtype::F32, shape, &hidden_data).map_err(|error| {
            hidden_state_error(format!(
                "failed to create hidden-state test tensor: {error}"
            ))
        })?;
        let serialized = if let Some((dtype, shape, token_data)) = token_ids.as_ref() {
            let token_view =
                TensorView::new(*dtype, shape.clone(), token_data).map_err(|error| {
                    hidden_state_error(format!("failed to create token ID test tensor: {error}"))
                })?;
            serialize(
                [("hidden_states", hidden_view), ("token_ids", token_view)],
                &None,
            )
        } else {
            serialize([("hidden_states", hidden_view)], &None)
        };
        serialized.map_err(|error| {
            hidden_state_error(format!(
                "failed to serialize hidden-state test tensor: {error}"
            ))
        })
    }

    fn valid_artifact_bytes() -> Result<Vec<u8>> {
        serialize_f32_hidden_states(
            vec![2, 2, 2],
            &[
                1.0, 3.0, 5.0, 7.0, // token 0, layers 0 and 1
                2.0, 4.0, 6.0, 8.0, // token 1, layers 0 and 1
            ],
            None,
        )
    }

    #[test]
    fn token_mean_pooling_produces_layer_major_features() -> Result<()> {
        let pooled = token_mean_per_layer(
            &f32_bytes(&[
                1.0, 3.0, 5.0, 7.0, // token 0, layers 0 and 1
                2.0, 4.0, 6.0, 8.0, // token 1, layers 0 and 1
            ]),
            Dtype::F32,
            &[2, 2, 2],
            layout(2, 2),
        )?;
        assert_eq!(pooled, vec![1.5, 3.5, 5.5, 7.5]);
        Ok(())
    }

    #[test]
    fn one_token_is_unchanged_by_token_mean_pooling() -> Result<()> {
        let values = [1.0, 3.0, 5.0, 7.0];
        let pooled =
            token_mean_per_layer(&f32_bytes(&values), Dtype::F32, &[1, 2, 2], layout(2, 2))?;
        assert_eq!(pooled, values);
        Ok(())
    }

    #[test]
    fn token_and_layer_axes_cannot_be_swapped() -> Result<()> {
        let error =
            token_mean_per_layer(&f32_bytes(&[0.0; 12]), Dtype::F32, &[2, 3, 2], layout(2, 3))
                .err()
                .ok_or_else(|| hidden_state_error("swapped axes should fail"))?;
        assert!(format!("{error}").contains("layer count 3"));
        Ok(())
    }

    #[test]
    fn bf16_and_f32_token_means_agree() -> Result<()> {
        let values = [1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0];
        let f32_pooled =
            token_mean_per_layer(&f32_bytes(&values), Dtype::F32, &[2, 2, 2], layout(2, 2))?;
        let bf16_pooled =
            token_mean_per_layer(&bf16_bytes(&values), Dtype::BF16, &[2, 2, 2], layout(2, 2))?;
        for (left, right) in f32_pooled.iter().zip(&bf16_pooled) {
            assert!((left - right).abs() < 1e-3);
        }
        Ok(())
    }

    #[test]
    fn token_mean_pooling_rejects_invalid_shape_and_byte_length() -> Result<()> {
        let malformed =
            token_mean_per_layer(&f32_bytes(&[1.0, 2.0]), Dtype::F32, &[2, 1], layout(1, 2))
                .err()
                .ok_or_else(|| hidden_state_error("malformed shape should fail"))?;
        assert!(format!("{malformed}").contains("expected hidden_states shape"));

        let empty = token_mean_per_layer(&[], Dtype::F32, &[0, 2, 2], layout(2, 2))
            .err()
            .ok_or_else(|| hidden_state_error("empty token axis should fail"))?;
        assert!(format!("{empty}").contains("token dimension must be non-zero"));

        let overflow = token_mean_per_layer(
            &[],
            Dtype::F32,
            &[1, usize::MAX, 2],
            HiddenStateLayout {
                layer_count: usize::MAX,
                hidden_size: 2,
                raw_feature_dim: 0,
            },
        )
        .err()
        .ok_or_else(|| hidden_state_error("shape overflow should fail"))?;
        assert!(format!("{overflow}").contains("shape is too large"));

        let wrong_length =
            token_mean_per_layer(&f32_bytes(&[1.0]), Dtype::F32, &[2, 2, 2], layout(2, 2))
                .err()
                .ok_or_else(|| hidden_state_error("wrong byte length should fail"))?;
        assert!(format!("{wrong_length}").contains("byte length"));
        Ok(())
    }

    #[test]
    fn token_mean_pooling_rejects_invalid_values_and_dtype() -> Result<()> {
        let non_finite = token_mean_per_layer(
            &f32_bytes(&[1.0, f32::NAN]),
            Dtype::F32,
            &[1, 1, 2],
            layout(1, 2),
        )
        .err()
        .ok_or_else(|| hidden_state_error("non-finite value should fail"))?;
        assert!(format!("{non_finite}").contains("contains non-finite"));

        let accumulation_overflow = token_mean_per_layer(
            &f32_bytes(&[f32::MAX, f32::MAX]),
            Dtype::F32,
            &[2, 1, 1],
            layout(1, 1),
        )
        .err()
        .ok_or_else(|| hidden_state_error("accumulation overflow should fail"))?;
        assert!(format!("{accumulation_overflow}").contains("accumulation produced"));

        let unsupported =
            token_mean_per_layer(&i64_bytes(&[1]), Dtype::I64, &[1, 1, 1], layout(1, 1))
                .err()
                .ok_or_else(|| hidden_state_error("unsupported dtype should fail"))?;
        assert!(format!("{unsupported}").contains("unsupported hidden_states dtype"));
        Ok(())
    }

    #[test]
    fn final_feature_length_must_match_artifact() -> Result<()> {
        let error = token_mean_per_layer(
            &f32_bytes(&[1.0, 2.0]),
            Dtype::F32,
            &[1, 1, 2],
            HiddenStateLayout {
                layer_count: 1,
                hidden_size: 2,
                raw_feature_dim: 3,
            },
        )
        .err()
        .ok_or_else(|| hidden_state_error("raw feature dimension mismatch should fail"))?;
        assert!(format!("{error}").contains("raw_feature_dim 3"));
        Ok(())
    }

    #[test]
    fn token_ids_are_validated_when_present() -> Result<()> {
        let valid = serialize_f32_hidden_states(
            vec![2, 1, 2],
            &[1.0, 2.0, 3.0, 4.0],
            Some((Dtype::I64, vec![2], i64_bytes(&[101, 102]))),
        )?;
        assert_eq!(
            parse_hidden_state_features(&valid, layout(1, 2))?,
            vec![2.0, 3.0]
        );

        let wrong_shape = serialize_f32_hidden_states(
            vec![2, 1, 2],
            &[1.0, 2.0, 3.0, 4.0],
            Some((Dtype::I64, vec![1], i64_bytes(&[101]))),
        )?;
        let error = parse_hidden_state_features(&wrong_shape, layout(1, 2))
            .err()
            .ok_or_else(|| hidden_state_error("token count mismatch should fail"))?;
        assert!(format!("{error}").contains("token_ids shape [1]"));

        let negative = serialize_f32_hidden_states(
            vec![2, 1, 2],
            &[1.0, 2.0, 3.0, 4.0],
            Some((Dtype::I64, vec![2], i64_bytes(&[101, -1]))),
        )?;
        let error = parse_hidden_state_features(&negative, layout(1, 2))
            .err()
            .ok_or_else(|| hidden_state_error("negative token ID should fail"))?;
        assert!(format!("{error}").contains("negative token ID"));

        let wrong_dtype = serialize_f32_hidden_states(
            vec![2, 1, 2],
            &[1.0, 2.0, 3.0, 4.0],
            Some((Dtype::F32, vec![2], f32_bytes(&[101.0, 102.0]))),
        )?;
        let error = parse_hidden_state_features(&wrong_dtype, layout(1, 2))
            .err()
            .ok_or_else(|| hidden_state_error("token ID dtype mismatch should fail"))?;
        assert!(format!("{error}").contains("token_ids must use I64"));
        Ok(())
    }

    #[test]
    fn hidden_state_path_must_stay_under_configured_directory() -> Result<()> {
        let directory = TestDirectory::create()?;
        let root = directory.create_subdirectory("root")?;
        let outside = directory.write("outside.safetensors", b"not a tensor")?;
        let traversing_path = root.join("..").join("outside.safetensors");

        let error = validate_hidden_states_path(&root, &traversing_path)
            .err()
            .ok_or_else(|| hidden_state_error("outside path should fail"))?;
        assert!(format!("{error}").contains("outside configured directory"));
        assert_eq!(
            outside
                .canonicalize()
                .map_err(|error| hidden_state_error(error.to_string()))?,
            traversing_path
                .canonicalize()
                .map_err(|error| hidden_state_error(error.to_string()))?
        );
        Ok(())
    }

    #[test]
    fn hidden_state_path_requires_extension_and_regular_file() -> Result<()> {
        let directory = TestDirectory::create()?;
        let wrong_extension = directory.write("hidden.bin", b"not a tensor")?;
        let error = validate_hidden_states_path(directory.path(), &wrong_extension)
            .err()
            .ok_or_else(|| hidden_state_error("wrong extension should fail"))?;
        assert!(format!("{error}").contains(".safetensors"));

        let not_a_file = directory.create_subdirectory("directory.safetensors")?;
        let error = validate_hidden_states_path(directory.path(), &not_a_file)
            .err()
            .ok_or_else(|| hidden_state_error("directory artifact should fail"))?;
        assert!(format!("{error}").contains("not a regular file"));
        Ok(())
    }

    #[test]
    fn artifact_file_is_exclusively_locked() -> Result<()> {
        let directory = TestDirectory::create()?;
        let path = directory.write("hidden.safetensors", b"locked")?;
        let first = open_locked_artifact(&path)?;
        let second = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|error| hidden_state_error(format!("second test open failed: {error}")))?;

        let error = second
            .try_lock()
            .err()
            .ok_or_else(|| hidden_state_error("second exclusive lock should fail"))?;
        assert!(matches!(error, std::fs::TryLockError::WouldBlock));
        drop(first);
        second.try_lock().map_err(|error| {
            hidden_state_error(format!("released lock was not reusable: {error}"))
        })?;
        Ok(())
    }

    #[test]
    fn successful_locked_read_removes_artifact() -> Result<()> {
        let directory = TestDirectory::create()?;
        let path = directory.write("hidden.safetensors", &valid_artifact_bytes()?)?;

        let features = read_and_cleanup_hidden_states(directory.path(), &path, layout(2, 2))?;

        assert_eq!(features, vec![1.5, 3.5, 5.5, 7.5]);
        assert!(!path.exists());
        Ok(())
    }

    #[test]
    fn failed_locked_read_preserves_artifact() -> Result<()> {
        let directory = TestDirectory::create()?;
        let path = directory.write("hidden.safetensors", b"not a tensor")?;

        let error = read_and_cleanup_hidden_states(directory.path(), &path, layout(2, 2))
            .err()
            .ok_or_else(|| hidden_state_error("malformed artifact should fail"))?;

        assert!(format!("{error}").contains("safetensors parse error"));
        assert!(path.exists());
        Ok(())
    }
}
