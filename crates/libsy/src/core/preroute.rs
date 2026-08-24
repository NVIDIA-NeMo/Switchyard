// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A step that runs before routing and picks no target.
//!
//! Unlike a [`Processor`](crate::core::processor::Processor) it holds the
//! [`Driver`], so it can consult a model before the cascade runs.

use async_trait::async_trait;

use crate::Result;
use crate::core::algorithm::Driver;
use switchyard_protocol::Request;

/// Runs ahead of a composition's classifiers, with the request and a driver.
#[async_trait]
pub trait Preroute<S = ()>: Send + Sync {
    /// Observes the request and records what it finds. Errors abort the turn.
    async fn run(&self, state: &mut S, request: &mut Request, driver: &Driver) -> Result<()>;
}
