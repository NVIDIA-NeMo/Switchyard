// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Helpers shared by this crate's unit tests.

use std::sync::Once;

/// Runs `run` with `subscriber` as this thread's default subscriber, so the test
/// sees the events and spans that `run` creates.
///
/// For each call site, such as one `warn!` line, `tracing` caches whether any
/// subscriber wants its events. While `subscriber` is the only registered
/// subscriber, `tracing` fills that cache by asking the default subscriber of
/// the first thread that reaches the call site. If that thread belongs to a
/// parallel test with no subscriber, the cache says "never", and this test
/// captures nothing from that call site. To prevent this, the first call to
/// this helper in a test binary also sets a global `Registry` with no layers.
/// That `Registry` wants every call site and writes no output, so the cache can
/// no longer say "never".
pub(crate) fn with_subscriber<T>(
    subscriber: impl tracing::Subscriber + Send + Sync + 'static,
    run: impl FnOnce() -> T,
) -> T {
    static GLOBAL_SUBSCRIBER: Once = Once::new();
    GLOBAL_SUBSCRIBER.call_once(|| {
        // An error means that another global subscriber is already set. That
        // subscriber also keeps the cache from saying "never".
        let _ = tracing::subscriber::set_global_default(tracing_subscriber::registry());
    });
    tracing::subscriber::with_default(subscriber, run)
}
