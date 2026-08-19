// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! JSON encoding for saved and replayed scaling runs.

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::{Result, ScalingError, ScalingRun};

/// Encodes a completed run as readable JSON.
pub fn encode_run<O>(run: &ScalingRun<O>) -> Result<Vec<u8>>
where
    O: Serialize,
{
    serde_json::to_vec_pretty(run).map_err(|error| ScalingError::Record(error.to_string()))
}

/// Decodes a previously saved run.
pub fn decode_run<O>(bytes: &[u8]) -> Result<ScalingRun<O>>
where
    O: DeserializeOwned,
{
    serde_json::from_slice(bytes).map_err(|error| ScalingError::Record(error.to_string()))
}
