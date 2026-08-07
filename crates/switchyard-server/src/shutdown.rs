// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Platform-specific process shutdown signals.

/// Waits for the platform's normal process termination signal.
pub(crate) async fn signal() {
    platform::signal().await;
}

async fn ctrl_c() {
    if let Err(error) = tokio::signal::ctrl_c().await {
        tracing::warn!(
            error = %error,
            "ctrl-c shutdown signal unavailable; continuing without shutdown trigger"
        );
        std::future::pending::<()>().await;
    }
}

#[cfg(unix)]
mod platform {
    pub(super) async fn signal() {
        tokio::select! {
            _ = super::ctrl_c() => {},
            _ = terminate() => {},
        }
    }

    async fn terminate() {
        let mut signal =
            match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                Ok(signal) => signal,
                Err(error) => {
                    tracing::warn!(
                        error = %error,
                        "SIGTERM shutdown signal unavailable; continuing without SIGTERM trigger"
                    );
                    std::future::pending::<()>().await;
                    return;
                }
            };
        signal.recv().await;
    }
}

#[cfg(not(unix))]
mod platform {
    pub(super) async fn signal() {
        super::ctrl_c().await;
    }
}
