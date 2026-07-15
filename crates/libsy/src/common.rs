// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// Per-request state threaded to an algorithm alongside its [`Driver`]. A placeholder
/// for cross-cutting state (correlation ids, budgets, deadlines) an algorithm will
/// read; empty today. It does not carry the offload driver, so it is safe to share.
#[derive(Clone, Default)]
pub struct Context {}

impl Context {
    /// Build an empty context.
    pub fn new() -> Self {
        Self {}
    }
}
