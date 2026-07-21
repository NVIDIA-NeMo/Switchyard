// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # libsy — multi-LLM agent optimization (routing first)
//!
//! `libsy` decides, per request, *how* to serve an LLM call: which model(s) to
//! invoke, in what order, and how to combine the results. Routing is the first
//! and simplest case; the same interfaces also express classifier routing,
//! ensembles, cascades, and other optimizations. The library owns no HTTP client
//! and no provider SDK — it decides, and the host makes (or is asked to make) the
//! actual calls — so it embeds cleanly in a proxy, gateway, or agent runtime.
//!
//! ## The model
//!
//! - An [`Algorithm`] is the optimization *algorithm*. Its
//!   [`create_run_task`](Algorithm::create_run_task) runs once per request
//!   and makes as many model calls as it needs — via [`Driver::call_llm_target`], which look
//!   like ordinary calls — publishes its [`Decision`]s with [`Driver::info`], and
//!   returns the final [`Response`]. The provided
//!   [`run_stream`](Algorithm::run_stream) drives that on its own task and hands
//!   back a stream of [`Step`]s; [`run`](Algorithm::run) runs
//!   it to completion with the targets' default clients.
//! - An [`LlmTarget`] names a routing target by its [`semantic_name`](LlmTarget::semantic_name).
//!   Every call is *offloaded* to the request's stream as a [`Step::CallLlm`]; the
//!   target's [`RoutedLlmClient`], if any, rides along as
//!   [`RoutedRequest::default_client`] so the host can serve it by default or
//!   override it (see below).
//!
//! ## Running a request
//!
//! Hold the algorithm as `Arc<dyn Algorithm>` and call one of two provided methods:
//!
//! - [`run`](Algorithm::run) — run to completion, serving each
//!   offloaded call via its [`RoutedRequest::default_client`], and return the decision
//!   trace plus the final [`Response`]. The simplest integration; use it when the
//!   algorithm holds the model clients (it errors if a routed target has no client).
//! - [`run_stream`](Algorithm::run_stream) — return a stream of [`Step`]s. Each
//!   model call is offloaded: the stream yields a [`Step::CallLlm`] carrying a promise;
//!   the host performs the real model call (optionally via the promise's
//!   `default_client`) and fulfills it with [`CallLlmRequest::respond`]. Decisions
//!   arrive as [`Step::Decision`] as the algorithm makes them, and the run ends with a
//!   [`Step::ReturnToAgent`] carrying the final response. The step stream is bounded,
//!   so pulling it paces the algorithm one step at a time — an "ask, don't call" mode
//!   that lets a host that owns its transport keep control of every call.
//!
//! ## Concurrency
//!
//! [`Algorithm::create_run_task`] takes `self: Arc<Self>`, so one shared
//! `Arc<dyn Algorithm>` (no lock) serves many requests in parallel. Each
//! [`run_stream`](Algorithm::run_stream) call builds its own [`Driver`], so
//! offloaded calls never cross between concurrent requests. An algorithm is
//! responsible for its own thread-safety — stateless (like the reference routers) or
//! interior mutability over just its own state.
//!
//! ## Algorithms
//!
//! Concrete algorithms live in [`algorithms`]:
//!
//! [`algorithms::Random`](crate::algorithms::Random) provides uniform random routing.
//!
//! [`algorithms::LlmClassifierOrch`](crate::algorithms::LlmClassifierOrch) classifies
//! with one model, then routes to a strong/weak model depending on the classifier's choice.

mod core;
pub use core::*;

pub mod algorithms;
