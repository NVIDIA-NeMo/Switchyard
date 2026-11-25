// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The assessment example question files must stay valid decision questions.

use std::{collections::BTreeMap, fs};

use serde_json::Value;
use switchyard_protocol::DecisionQuestion;

fn load(name: &str) -> Value {
    let path = format!("{}/examples/{name}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

#[test]
fn question_files_parse() {
    for (name, count) in [("task_questions.json", 8), ("turn_questions.json", 11)] {
        let mut rubric = load(name);
        assert!(rubric["preamble"].as_str().is_some_and(|p| !p.is_empty()));
        let questions: BTreeMap<String, DecisionQuestion> =
            serde_json::from_value(rubric["questions"].take()).unwrap();
        assert_eq!(questions.len(), count, "{name}");
        for (id, q) in &questions {
            assert!(q.instructions.is_string(), "{name}: {id}");
        }
    }
}

#[test]
fn context_files_have_messages() {
    for name in ["task_context.json", "turn_context.json"] {
        assert!(load(name)["messages"].is_array(), "{name}");
    }
}
