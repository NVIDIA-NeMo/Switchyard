// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared version and telemetry helpers for Rust-owned components.

use std::env;

pub(crate) const SWITCHYARD_VERSION_HEADER: &str = "X-Switchyard-Version";

const SWITCHYARD_VERSION_ENV: &str = "SWITCHYARD_VERSION";
const SWITCHYARD_TELEMETRY_OPT_OUT_ENV: &str = "SWITCHYARD_TELEMETRY_OPT_OUT";
const NEMO_SWITCHYARD_TELEMETRY_OPT_OUT_ENV: &str = "NEMO_SWITCHYARD_TELEMETRY_OPT_OUT";

pub(crate) fn telemetry_header_value() -> Option<String> {
    telemetry_header_value_from_values(
        env::var(SWITCHYARD_TELEMETRY_OPT_OUT_ENV).ok().as_deref(),
        env::var(NEMO_SWITCHYARD_TELEMETRY_OPT_OUT_ENV)
            .ok()
            .as_deref(),
        env::var(SWITCHYARD_VERSION_ENV).ok().as_deref(),
    )
}

fn telemetry_header_value_from_values(
    opt_out: Option<&str>,
    legacy_opt_out: Option<&str>,
    version: Option<&str>,
) -> Option<String> {
    if env_value_opts_out(opt_out) || env_value_opts_out(legacy_opt_out) {
        return None;
    }
    Some(configured_version(version).unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string()))
}

fn configured_version(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn env_value_opts_out(value: Option<&str>) -> bool {
    let Some(value) = value.map(str::trim) else {
        return false;
    };
    !matches!(
        value.to_ascii_lowercase().as_str(),
        "" | "0" | "false" | "no"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn telemetry_header_uses_configured_version_when_not_opted_out() {
        assert_eq!(
            telemetry_header_value_from_values(None, None, Some(" 1.2.3 ")),
            Some("1.2.3".to_string())
        );
    }

    #[test]
    fn telemetry_header_falls_back_to_crate_version() {
        assert_eq!(
            telemetry_header_value_from_values(None, None, None),
            Some(env!("CARGO_PKG_VERSION").to_string())
        );
        assert_eq!(
            telemetry_header_value_from_values(None, None, Some(" ")),
            Some(env!("CARGO_PKG_VERSION").to_string())
        );
    }

    #[test]
    fn telemetry_header_respects_current_and_legacy_opt_out_env_values() {
        assert_eq!(
            telemetry_header_value_from_values(Some("true"), None, Some("1.2.3")),
            None
        );
        assert_eq!(
            telemetry_header_value_from_values(None, Some("yes"), Some("1.2.3")),
            None
        );
    }

    #[test]
    fn telemetry_header_ignores_falsey_opt_out_values() {
        for value in ["", "0", "false", "FALSE", "no", " No "] {
            assert_eq!(
                telemetry_header_value_from_values(Some(value), None, Some("1.2.3")),
                Some("1.2.3".to_string())
            );
        }
    }
}
