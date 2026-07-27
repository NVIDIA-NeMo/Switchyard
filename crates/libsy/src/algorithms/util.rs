// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod affinity;
pub(crate) mod handoff_notes;
pub(crate) mod stage_router;
mod subagent;
pub(crate) mod tool_signals;

pub use affinity::AffinityRouter;
pub use subagent::SubagentOverride;
