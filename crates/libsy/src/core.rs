// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core orchestration: the [`Algorithm`] trait and its [`Driver`], built on the
//! type-erased promise-over-a-stream pump in [`driver`]. Its public items are
//! re-exported at the crate root; algorithm implementations live in
//! [`crate::algorithms`].

pub mod algorithm;
mod driver;

pub use algorithm::*;
