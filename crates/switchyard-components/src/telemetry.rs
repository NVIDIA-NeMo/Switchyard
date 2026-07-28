// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared version helper for component telemetry payloads.

use std::env;

const SWITCHYARD_VERSION_ENV: &str = "SWITCHYARD_VERSION";

pub(crate) fn switchyard_version() -> String {
    env::var(SWITCHYARD_VERSION_ENV)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string())
}
