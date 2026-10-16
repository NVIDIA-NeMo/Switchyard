// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime abstraction over task spawning and monotonic clocks, so the crate
//! runs on native Tokio hosts and on single-threaded wasm32 hosts (browsers,
//! Cloudflare Workers) alike.

/// Monotonic instant. On wasm32 `std::time::Instant::now()` aborts, so a
/// JS-clock-backed drop-in replacement is used there.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub(crate) use std::time::Instant;
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub(crate) use web_time::Instant;

/// Handle that aborts the task returned by [`spawn_abortable`].
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub(crate) use futures::future::AbortHandle;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub(crate) use tokio::task::AbortHandle;

/// Spawns a future on the host runtime and returns a handle that aborts it.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub(crate) fn spawn_abortable<F>(future: F) -> AbortHandle
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    tokio::spawn(future).abort_handle()
}

/// Spawns a future on the JS microtask queue and returns a handle that aborts it.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub(crate) fn spawn_abortable<F>(future: F) -> AbortHandle
where
    F: std::future::Future<Output = ()> + 'static,
{
    let (handle, registration) = AbortHandle::new_pair();
    wasm_bindgen_futures::spawn_local(async move {
        let _ = futures::future::Abortable::new(future, registration).await;
    });
    handle
}
