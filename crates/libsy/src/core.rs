// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core orchestration: the [`Algorithm`](algorithm::Algorithm) trait and its
//! [`Driver`](algorithm::Driver), built on the type-erased promise-over-a-stream
//! pump in [`driver`]. Algorithm implementations live in [`crate::algorithms`].

mod driver;
#[cfg(test)]
pub(crate) mod testing;

pub mod algorithm;
pub mod classifier;
pub mod processor;
pub mod state;
