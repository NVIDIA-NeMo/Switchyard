// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-neutral LLM protocol types shared across Switchyard crates.

pub mod format;
pub mod ir;

pub use format::*;
pub use ir::*;
