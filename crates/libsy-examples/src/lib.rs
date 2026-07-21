// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reference algorithms for [`libsy`], kept out of the core crate but compiled and
//! tested here so they stay current. Each is a worked example of the
//! [`Algorithm`](libsy::Algorithm) trait; the `examples/` directory has runnable agents
//! that drive them. The core crate provides [`libsy::algorithms::Random`] for uniform random
//! routing.
//!
//! - [`ensemble::EnsembleOrchAlgo`] — fan out to several models, judge, and commit
//!   (stateful).

pub mod ensemble;
