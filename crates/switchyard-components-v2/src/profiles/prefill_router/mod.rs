// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Learned prefill-router artifact inference and routing internals.

#[allow(dead_code)]
mod artifact;
mod policy;
mod profile;
mod scorer;

pub use profile::{
    PrefillProbeDecision, PrefillProbeProcessedRequest, PrefillProbeProfile,
    PrefillProbeProfileConfig, PrefillProbeRoutingPolicyConfig,
};
