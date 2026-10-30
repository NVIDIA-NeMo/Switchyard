// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Capability-only keys on an escalation route: rejected when the mode is spelled
//! out, accepted (and ignored, with a warning) when the escalation table selects
//! the mode on its own.

use switchyard_runner::Runner;

const CONFIG: &str = r#"
schema_version = 1
[llm_clients.local]
format = "openai_chat"
base_url = "http://127.0.0.1:1/v1"
[targets.fast]
id = "fast-model"
llm_client = "local"
[targets.strong]
id = "strong-model"
llm_client = "local"
[targets.judge]
id = "judge-model"
llm_client = "local"
[routes.agent]
id = "agent"
type = "llm_classifier"
classifier_target = "judge"
strong_target = "strong"
weak_target = "fast"
escalation = { confirmations = 2 }
"#;

const CAPABILITY_KEYS: [&str; 5] = [
    "base_threshold = 0.5",
    "threshold_step = 0.1",
    "fail_open = false",
    "recent_turn_window = 4",
    "message_hash_fallback = true",
];

#[test]
fn implicit_escalation_mode_keeps_loading_with_capability_keys() {
    for setting in CAPABILITY_KEYS {
        let config = format!("{CONFIG}{setting}\n");
        assert!(Runner::from_toml(&config).is_ok(), "{setting}");
    }
}

#[test]
fn explicit_escalation_mode_rejects_capability_keys() {
    for setting in CAPABILITY_KEYS {
        let config = format!("{CONFIG}mode = \"escalation\"\n{setting}\n");
        let error = match Runner::from_toml(&config) {
            Ok(_) => panic!("expected a load error for {setting}"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("mode escalation cannot use capability routing settings"),
            "{setting}: {error}"
        );
    }
}
