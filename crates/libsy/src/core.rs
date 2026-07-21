// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core orchestration: the [`Algorithm`] trait and its [`Driver`], built on the
//! type-erased promise-over-a-stream pump in [`driver`]. Its public items are
//! re-exported at the crate root; algorithm implementations live in
//! [`crate::algorithms`].

mod driver;

pub mod algorithm;
pub mod classifier;
pub mod processor;
pub mod state;

pub use algorithm::*;
pub use classifier::*;
pub use processor::*;
pub use state::*;
