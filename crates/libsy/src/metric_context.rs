// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request labels shared across routing, model calls, and response streams.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::OnceLock;

use opentelemetry::{Context, KeyValue};
use parking_lot::Mutex;
use switchyard_protocol::Request;

use crate::algorithms::util::affinity::has_new_user_turn;

#[derive(Clone, Default)]
struct Labels(Vec<KeyValue>);

#[derive(Default)]
struct Turns {
    agent: i64,
    user: i64,
    turn_id: Option<String>,
    input: Option<u64>,
}

#[derive(Default)]
struct Tracker(HashMap<(String, String), Turns>);

impl Tracker {
    fn labels(&mut self, request: &Request) -> Labels {
        let Some(metadata) = &request.metadata else {
            return Labels::default();
        };
        let mut labels = Vec::new();
        let session = metadata.session_id.as_deref().filter(|id| !id.is_empty());
        let agent = metadata.agent_id.as_deref().filter(|id| !id.is_empty());
        if let Some(agent) = agent {
            labels.push(KeyValue::new("agent_id", agent.to_owned()));
        }
        let Some(session) = session else {
            return Labels(labels);
        };
        labels.push(KeyValue::new("session_id", session.to_owned()));
        let agent = agent.unwrap_or("root");
        if metadata.agent_id.as_deref().is_none_or(str::is_empty) {
            labels.push(KeyValue::new("agent_id", agent.to_owned()));
        }
        let input = has_new_user_turn(&request.llm_request.messages).then(|| {
            let mut hash = DefaultHasher::new();
            request.llm_request.messages.len().hash(&mut hash);
            // Message contains only JSON-serializable protocol values.
            serde_json::to_vec(request.llm_request.messages.last().unwrap())
                .expect("serialize user message")
                .hash(&mut hash);
            hash.finish()
        });
        let turn_id = metadata.turn_id.as_ref().filter(|id| !id.is_empty());
        let key = (session.to_owned(), agent.to_owned());
        let turns = match self.0.entry(key) {
            std::collections::hash_map::Entry::Vacant(entry) => entry.insert(Turns {
                turn_id: turn_id.cloned(),
                input,
                ..Turns::default()
            }),
            std::collections::hash_map::Entry::Occupied(entry) => {
                let turns = entry.into_mut();
                turns.agent += 1;
                if turn_id.is_some_and(|id| turns.turn_id.as_ref() != Some(id))
                    || input.is_some_and(|input| turns.input != Some(input))
                {
                    turns.user += 1;
                }
                if let Some(turn_id) = turn_id {
                    turns.turn_id = Some(turn_id.clone());
                }
                if input.is_some() {
                    turns.input = input;
                }
                turns
            }
        };
        labels.push(KeyValue::new("agent_turn_idx", turns.agent));
        labels.push(KeyValue::new("user_turn_idx", turns.user));
        if metadata.session_final == Some(true) {
            self.0.retain(|(id, _), _| id != session);
        }
        Labels(labels)
    }
}

/// Creates the telemetry context for one inbound inference request.
///
/// Nested Switchyard calls reuse the current labels, so routing calls and fallback
/// candidates do not advance the indices. Indices are zero-based and process-local,
/// keyed by session and agent. An absent agent ID uses `root`. Without a session ID,
/// no indices are assigned. A final-session marker releases that session's counters.
/// Attach this context while polling the request future, never across an await guard.
pub fn for_request(request: &Request) -> Context {
    let current = Context::current();
    if current.get::<Labels>().is_some() {
        return current;
    }
    static TRACKER: OnceLock<Mutex<Tracker>> = OnceLock::new();
    let labels = TRACKER.get_or_init(Default::default).lock().labels(request);
    current.with_value(labels)
}

/// Appends the current request's identity and turn indices to metric attributes.
pub fn attributes(base: &[KeyValue]) -> Vec<KeyValue> {
    let mut attributes = base.to_vec();
    Context::map_current(|context| {
        if let Some(labels) = context.get::<Labels>() {
            attributes.extend_from_slice(&labels.0);
        }
    });
    attributes
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentelemetry::context::FutureExt as _;
    use switchyard_protocol::{ContentBlock, Message, Metadata, Role, ToolResult, text_request};

    fn request(session: &str, agent: Option<&str>) -> Request {
        Request {
            llm_request: text_request(None, "hello"),
            metadata: Some(Metadata {
                session_id: Some(session.into()),
                agent_id: agent.map(str::to_owned),
                ..Metadata::default()
            }),
            raw_request: None,
        }
    }

    fn indices(labels: &Labels) -> (i64, i64) {
        let get = |name| {
            labels
                .0
                .iter()
                .find_map(|kv| match &kv.value {
                    opentelemetry::Value::I64(value) if kv.key.as_str() == name => Some(*value),
                    _ => None,
                })
                .expect("index label")
        };
        (get("agent_turn_idx"), get("user_turn_idx"))
    }

    #[test]
    fn tracks_user_input_across_openai_and_anthropic_tool_continuations() {
        let mut tracker = Tracker::default();
        let mut request = request("session", Some("agent"));
        assert_eq!(indices(&tracker.labels(&request)), (0, 0));
        // The same prompt can be retried without introducing another user turn.
        assert_eq!(indices(&tracker.labels(&request)), (1, 0));
        request
            .llm_request
            .messages
            .push(Message::text(Role::Assistant, "calling tool"));
        request
            .llm_request
            .messages
            .push(Message::text(Role::Tool, "result"));
        assert_eq!(indices(&tracker.labels(&request)), (2, 0));
        request.llm_request.messages.push(Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult(ToolResult {
                tool_call_id: "call".into(),
                content: vec![],
                is_error: None,
            })],
        });
        assert_eq!(indices(&tracker.labels(&request)), (3, 0));
        // Repeating the same text later in the history is a new user input.
        request
            .llm_request
            .messages
            .push(Message::text(Role::User, "hello"));
        assert_eq!(indices(&tracker.labels(&request)), (4, 1));
    }

    #[test]
    fn isolates_agents_and_sessions_and_releases_final_sessions() {
        let mut tracker = Tracker::default();
        let mut first = request("one", None);
        assert_eq!(indices(&tracker.labels(&first)), (0, 0));
        assert_eq!(indices(&tracker.labels(&first)), (1, 0));
        assert_eq!(
            indices(&tracker.labels(&request("one", Some("child")))),
            (0, 0)
        );
        assert_eq!(indices(&tracker.labels(&request("two", None))), (0, 0));
        first.metadata.as_mut().unwrap().session_final = Some(true);
        assert_eq!(indices(&tracker.labels(&first)), (2, 0));
        assert_eq!(indices(&tracker.labels(&request("one", None))), (0, 0));
        assert_eq!(
            indices(&tracker.labels(&request("one", Some("child")))),
            (0, 0)
        );
        assert_eq!(indices(&tracker.labels(&request("two", None))), (1, 0));
    }

    #[test]
    fn supplied_turn_ids_detect_new_turns_with_identical_or_missing_history() {
        let mut tracker = Tracker::default();
        let mut request = request("session", Some("agent"));
        request.metadata.as_mut().unwrap().turn_id = Some("first".into());
        assert_eq!(indices(&tracker.labels(&request)), (0, 0));
        assert_eq!(indices(&tracker.labels(&request)), (1, 0));
        request.metadata.as_mut().unwrap().turn_id = Some("second".into());
        request.llm_request.messages.clear();
        assert_eq!(indices(&tracker.labels(&request)), (2, 1));
        assert_eq!(indices(&tracker.labels(&request)), (3, 1));
    }

    #[test]
    fn missing_session_does_not_share_counters() {
        let mut tracker = Tracker::default();
        let mut request = request("", Some("agent"));
        assert_eq!(tracker.labels(&request).0.len(), 1);
        request.metadata = None;
        assert!(tracker.labels(&request).0.is_empty());
        assert!(tracker.0.is_empty());
    }

    #[tokio::test]
    async fn nested_calls_reuse_indices_and_async_scopes_do_not_leak() {
        let request = request("metric-context-nesting", Some("agent"));
        let context = for_request(&request);
        async {
            assert_eq!(attributes(&[]).len(), 4);
            let nested = for_request(&request);
            assert_eq!(indices(nested.get::<Labels>().unwrap()), (0, 0));
            tokio::task::yield_now().await;
            assert_eq!(indices(Context::current().get::<Labels>().unwrap()), (0, 0));
        }
        .with_context(context)
        .await;
        assert!(attributes(&[]).is_empty());
        assert_eq!(
            indices(for_request(&request).get::<Labels>().unwrap()),
            (1, 0)
        );
    }
}
