// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A step that runs before routing and decides nothing.
//!
//! Sits between [`Processor`](crate::core::processor::Processor), which cannot
//! reach a model, and [`Classifier`](crate::core::classifier::Classifier), which
//! is expected to pick a target. A prelude gets the [`Driver`] so it can consult
//! one, and writes what it learns to state for the cascade behind it to use.

use async_trait::async_trait;

use crate::Result;
use crate::core::algorithm::Driver;
use switchyard_protocol::Request;

/// Runs ahead of a composition's classifiers, with the request and a driver.
#[async_trait]
pub trait Prelude<S = ()>: Send + Sync {
    /// Observes the request and records what it finds. Errors abort the turn.
    async fn run(&self, state: &mut S, request: &mut Request, driver: &Driver) -> Result<()>;
}
