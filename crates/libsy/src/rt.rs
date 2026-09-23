// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime abstraction over task spawning and monotonic clocks, so the crate
//! runs on native Tokio hosts and on single-threaded wasm32 hosts (browsers,
//! Cloudflare Workers) alike.

use futures::future::{AbortHandle, Abortable};

/// Monotonic instant. On wasm32 `std::time::Instant::now()` aborts, so a
/// JS-clock-backed drop-in replacement is used there.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use std::time::Instant;
#[cfg(target_arch = "wasm32")]
pub(crate) use web_time::Instant;

/// Spawns a future on the host runtime and returns a handle that aborts it.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn spawn_abortable<F>(future: F) -> AbortHandle
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    let (handle, registration) = AbortHandle::new_pair();
    drop(tokio::spawn(async move {
        let _ = Abortable::new(future, registration).await;
    }));
    handle
}

/// Spawns a future on the JS microtask queue and returns a handle that aborts it.
#[cfg(target_arch = "wasm32")]
pub(crate) fn spawn_abortable<F>(future: F) -> AbortHandle
where
    F: std::future::Future<Output = ()> + 'static,
{
    let (handle, registration) = AbortHandle::new_pair();
    wasm_bindgen_futures::spawn_local(async move {
        let _ = Abortable::new(future, registration).await;
    });
    handle
}
