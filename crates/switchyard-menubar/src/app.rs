// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ties the pieces together: read the log, probe the server, build the rows.

use chrono::Local;

use crate::config::Config;
use crate::health::probe;
use crate::rollup;
use crate::summary::{Row, build};

/// Reads new log entries, probes the server, and builds the menu body.
pub fn refresh(config: &Config, log: &mut rollup::Reader) -> Vec<Row> {
    let usage = match log.read(&config.routing_log, Local::now().date_naive()) {
        Ok(usage) => usage,
        Err(error) => return vec![Row::Label(format!("Cannot read routing log: {error}"))],
    };
    build(probe(&config.server_url), &usage, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_configured_log() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("routing.jsonl");
        let ts = Local::now().to_rfc3339();
        std::fs::write(
            &path,
            format!(
                r#"{{"ts":"{ts}","model":"luna","tier":"","prompt_tokens":1000,"cached_tokens":0,"completion_tokens":100,"reasoning_tokens":0,"cost":null}}"#
            ) + "\n",
        )
        .expect("write log");

        let rows = refresh(
            &Config {
                routing_log: path,
                ..Config::default()
            },
            &mut rollup::Reader::default(),
        );

        assert!(
            rows.contains(&Row::Label("Today — 1 request · 1100 tokens".to_string())),
            "unexpected rows: {rows:?}"
        );
    }
}
