// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core orchestration: the [`Algorithm`](algorithm::Algorithm) trait and its typed
//! step-stream [`Driver`](algorithm::Driver). Algorithm implementations live in
//! [`crate::algorithms`].

#[cfg(test)]
pub(crate) mod testing;

pub mod algorithm;
pub mod classifier;
pub mod processor;
pub mod state;
