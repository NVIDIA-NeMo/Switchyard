// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stage-router tier selection — thin re-export of libsy's decision core.
//!
//! The scoring and tier-selection logic lives in [`switchyard_libsy::stage_router`].
//! Its input, a [`ToolResultSignal`][crate::dimension_collector::ToolResultSignal],
//! is already libsy's `ToolSignals`, so no request adaptation is needed here — this
//! module simply re-exports the API so the crate's processors decide a turn's tier
//! through the same implementation the libsy profile uses.

pub use switchyard_libsy::stage_router::{
    dimensions_from_signal, pick_tier, score_signal, CodingAgentDimensions, DecisionSource,
    PickOutcome, PickerMode, ScoreResult, Tier,
};
