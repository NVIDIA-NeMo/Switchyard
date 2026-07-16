// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Central profile config registry generated from concrete profile config types.

use super::macros::profile_types;
use super::{
    LatencyServiceProfileConfig, LlmRoutingProfileConfig, NoopProfileConfig,
    PassthroughProfileConfig, PrefillProbeProfileConfig, RandomRoutingProfileConfig,
    StageRouterProfileConfig,
};

profile_types! {
    StageRouterProfileConfig,
    PassthroughProfileConfig,
    RandomRoutingProfileConfig,
    LatencyServiceProfileConfig,
    LlmRoutingProfileConfig,
    PrefillProbeProfileConfig,
    NoopProfileConfig,
}
