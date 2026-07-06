// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core contracts for Switchyard's Rust implementation.
//!
//! This crate owns provider-agnostic values, IDs, errors, context storage, and
//! the backend trait used by concrete LLM callers. Request/response processor
//! logic now lives on concrete components and profile runtimes instead of core
//! role traits.

pub mod backend;
pub mod context;
pub mod error;
pub mod ids;
pub mod roles;
pub mod routing_trace;
pub mod session;
pub mod types;

pub use backend::*;
pub use context::*;
pub use error::*;
pub use ids::*;
pub use roles::*;
pub use routing_trace::*;
pub use session::*;
pub use types::*;
