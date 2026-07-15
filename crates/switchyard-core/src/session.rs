// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Session-affinity primitives: a stable per-conversation key derived from a
//! request body and a bounded, access-ordered LRU cache keyed by that string.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;

use lru::LruCache;
use serde_json::Value;

/// Derive a stable per-conversation key from a request body.
///
/// Hashes only the prefix a harness never rewrites — system prompt + first user
/// message — so every turn of a conversation shares a key while distinct
/// conversations differ. Returns a 16-char lowercase hex string.
pub fn session_key_from_body(body: &Value) -> String {
    hash_conversation_prefix(body, 0).0
}

/// Deep variant of [`session_key_from_body`]: extend the hashed prefix with
/// the first `depth` messages after the first user message, so repeated runs
/// of an identical task diverge via early model responses (repeated-trial
/// benchmarking).
///
/// Returns `None` until the conversation actually contains a first user
/// message plus `depth` later messages — a shorter prefix would hash to a key
/// that later turns of the same conversation no longer match. `depth == 0` is
/// exactly the base key and always yields `Some`.
pub fn session_key_from_body_with_depth(body: &Value, depth: usize) -> Option<String> {
    let (key, complete) = hash_conversation_prefix(body, depth);
    complete.then_some(key)
}

/// Shared hashing core for both key variants: anchors (top-level system,
/// in-list system/developer messages, first user message) plus the first
/// `depth` non-system messages after the first user. Returns the key and
/// whether the requested prefix was complete (always true at depth 0).
fn hash_conversation_prefix(body: &Value, depth: usize) -> (String, bool) {
    let mut hasher = DefaultHasher::new();

    // Anthropic carries the system prompt at the top level.
    flatten_text(body.get("system")).hash(&mut hasher);

    // OpenAI uses "messages"; the Responses API uses "input". A messages list
    // with no user message falls through to "input".
    let mut anchored = false;
    let mut tail_taken = 0usize;
    for seq_key in ["messages", "input"] {
        if let Some(Value::Array(items)) = body.get(seq_key) {
            for item in items {
                match item.get("role").and_then(Value::as_str) {
                    Some("system") | Some("developer") => {
                        flatten_text(item.get("content")).hash(&mut hasher);
                    }
                    Some("user") if !anchored => {
                        flatten_text(item.get("content")).hash(&mut hasher);
                        anchored = true;
                    }
                    // Post-anchor tail (any non-system item, roleless
                    // Responses items included): hash the full message —
                    // tool calls too, so assistant turns that differ only
                    // in tool calls still diverge the deep key.
                    _ if anchored && tail_taken < depth => {
                        flatten_message_text(item).hash(&mut hasher);
                        tail_taken += 1;
                    }
                    _ => {}
                }
                if anchored && tail_taken >= depth {
                    break;
                }
            }
        }
        if anchored {
            break;
        }
    }

    let complete = depth == 0 || (anchored && tail_taken >= depth);
    (format!("{:016x}", hasher.finish()), complete)
}

/// Flatten a whole message for deep-key hashing: `content` text plus tool-call
/// payloads (OpenAI chat `tool_calls`, Responses `function_call` items) and
/// tool `output`. Anchors keep hashing `content` only — this richer form is
/// applied to post-anchor tail messages, where the divergence signal between
/// repeated trials often lives entirely in the tool calls.
fn flatten_message_text(item: &Value) -> String {
    let mut parts: Vec<String> = Vec::new();
    let content = flatten_text(item.get("content"));
    if !content.is_empty() {
        parts.push(content);
    }
    if let Some(Value::Array(calls)) = item.get("tool_calls") {
        for call in calls {
            if let Some(function) = call.get("function").and_then(Value::as_object) {
                let name = function.get("name").and_then(Value::as_str).unwrap_or("");
                let arguments = function
                    .get("arguments")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                parts.push(format!("tool_call {name}({arguments})"));
            }
        }
    }
    if item.get("type").and_then(Value::as_str) == Some("function_call") {
        let name = item.get("name").and_then(Value::as_str).unwrap_or("");
        let arguments = item.get("arguments").and_then(Value::as_str).unwrap_or("");
        parts.push(format!("tool_call {name}({arguments})"));
    }
    let output = flatten_text(item.get("output"));
    if !output.is_empty() {
        parts.push(output);
    }
    parts.join(" ")
}

/// Flatten a message-content value into a single string for hashing: strings
/// pass through, content-block arrays concatenate their first non-empty
/// `text`/`content` field (or the raw block for non-objects), null/absent yield
/// empty, and other scalars stringify. Text-only by design (block metadata such
/// as `cache_control` is excluded so it can't perturb the key).
fn flatten_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(blocks)) => {
            let mut out = String::new();
            for block in blocks {
                if let Value::Object(map) = block {
                    let text = map
                        .get("text")
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty())
                        .or_else(|| {
                            map.get("content")
                                .and_then(Value::as_str)
                                .filter(|s| !s.is_empty())
                        });
                    if let Some(text) = text {
                        out.push_str(text);
                    }
                } else {
                    out.push_str(&block.to_string());
                }
            }
            out
        }
        None | Some(Value::Null) => String::new(),
        Some(other) => other.to_string(),
    }
}

/// A bounded, access-ordered LRU cache keyed by `String`.
///
/// Recency refreshes on both [`SessionCache::get`] and [`SessionCache::put`];
/// the least-recently-used entry is evicted when capacity is exceeded. A
/// capacity of 0 retains nothing.
///
/// NOT thread-safe — intended for single-event-loop use.
pub struct SessionCache<V> {
    cache: Option<LruCache<String, V>>,
}

impl<V> SessionCache<V> {
    /// Create a cache holding at most `max_sessions` entries (0 retains nothing).
    pub fn new(max_sessions: usize) -> Self {
        Self {
            cache: NonZeroUsize::new(max_sessions).map(LruCache::new),
        }
    }

    /// Look up `key`, refreshing its recency to most-recently-used.
    pub fn get(&mut self, key: &str) -> Option<&V> {
        self.cache.as_mut().and_then(|c| c.get(key))
    }

    /// Insert `value` as most-recently-used, evicting the LRU entry if over
    /// capacity. No-op when capacity is 0.
    pub fn put(&mut self, key: String, value: V) {
        if let Some(c) = self.cache.as_mut() {
            c.put(key, value);
        }
    }

    /// Number of entries currently retained.
    pub fn len(&self) -> usize {
        self.cache.as_ref().map_or(0, LruCache::len)
    }

    /// Whether the cache holds no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The configured maximum number of sessions.
    pub fn max_sessions(&self) -> usize {
        self.cache.as_ref().map_or(0, |c| c.cap().get())
    }

    /// Iterate over the retained values (order unspecified).
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.cache.iter().flat_map(|c| c.iter().map(|(_, v)| v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn get_refreshes_recency() {
        let mut cache: SessionCache<i32> = SessionCache::new(2);
        cache.put("a".to_string(), 1);
        cache.put("b".to_string(), 2);
        // Touch "a" so "b" becomes the LRU entry.
        assert!(cache.get("a").is_some());
        cache.put("c".to_string(), 3);
        assert!(cache.get("a").is_some());
        assert!(cache.get("b").is_none());
        assert!(cache.get("c").is_some());
    }

    #[test]
    fn eviction_bounds_len() {
        let mut cache: SessionCache<i32> = SessionCache::new(2);
        cache.put("a".to_string(), 1);
        cache.put("b".to_string(), 2);
        cache.put("c".to_string(), 3);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn zero_capacity_retains_nothing() {
        let mut cache: SessionCache<&str> = SessionCache::new(0);
        cache.put("a".to_string(), "x");
        assert!(cache.get("a").is_none());
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.max_sessions(), 0);
    }

    #[test]
    fn session_key_stable_across_appended_turns() {
        let base = json!({
            "system": "you are helpful",
            "messages": [
                {"role": "user", "content": "first question"},
            ],
        });
        let extended = json!({
            "system": "you are helpful",
            "messages": [
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "an answer"},
                {"role": "user", "content": "a follow up"},
            ],
        });
        assert_eq!(
            session_key_from_body(&base),
            session_key_from_body(&extended)
        );
    }

    #[test]
    fn session_key_distinct_on_system_or_first_user() {
        let base = json!({
            "system": "you are helpful",
            "messages": [{"role": "user", "content": "first question"}],
        });
        let diff_system = json!({
            "system": "you are terse",
            "messages": [{"role": "user", "content": "first question"}],
        });
        let diff_user = json!({
            "system": "you are helpful",
            "messages": [{"role": "user", "content": "another question"}],
        });
        assert_ne!(
            session_key_from_body(&base),
            session_key_from_body(&diff_system)
        );
        assert_ne!(
            session_key_from_body(&base),
            session_key_from_body(&diff_user)
        );
    }

    #[test]
    fn session_key_blocks_equal_plain() {
        let blocks = json!({
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        });
        let plain = json!({
            "messages": [{"role": "user", "content": "hi"}],
        });
        assert_eq!(
            session_key_from_body(&blocks),
            session_key_from_body(&plain)
        );
    }

    #[test]
    fn key_is_16_char_hex() {
        let key = session_key_from_body(&json!({"messages": [{"role": "user", "content": "x"}]}));
        assert_eq!(key.len(), 16);
        assert!(key.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn deep_key_depth_zero_matches_base() {
        let body = json!({
            "system": "you are helpful",
            "messages": [{"role": "user", "content": "first question"}],
        });
        assert_eq!(
            session_key_from_body_with_depth(&body, 0),
            Some(session_key_from_body(&body))
        );
    }

    #[test]
    fn deep_key_diverges_on_early_responses() {
        // Identical anchors, different first assistant response: the deep key
        // separates the trials while the base key does not.
        let trial = |first_response: &str| {
            json!({
                "messages": [
                    {"role": "user", "content": "identical task text"},
                    {"role": "assistant", "content": first_response},
                ],
            })
        };
        let a = trial("read the tests first");
        let b = trial("look at the repo layout");
        assert_eq!(session_key_from_body(&a), session_key_from_body(&b));
        assert_ne!(
            session_key_from_body_with_depth(&a, 1),
            session_key_from_body_with_depth(&b, 1)
        );
    }

    #[test]
    fn deep_key_stable_as_conversation_grows() {
        // The deep key hashes a fixed prefix, so it survives appended turns.
        let base = json!({
            "messages": [
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "first response"},
            ],
        });
        let grown = json!({
            "messages": [
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "first response"},
                {"role": "user", "content": "tool result"},
                {"role": "assistant", "content": "much later turn"},
            ],
        });
        assert_eq!(
            session_key_from_body_with_depth(&base, 1),
            session_key_from_body_with_depth(&grown, 1)
        );
    }

    #[test]
    fn deep_key_none_until_prefix_complete() {
        // One post-user message can't satisfy depth 2; no user at all can't
        // satisfy any positive depth.
        let short = json!({
            "messages": [
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "first response"},
            ],
        });
        assert_eq!(session_key_from_body_with_depth(&short, 2), None);
        let unanchored = json!({"messages": [{"role": "system", "content": "sys"}]});
        assert_eq!(session_key_from_body_with_depth(&unanchored, 1), None);
    }

    #[test]
    fn deep_key_sees_tool_calls() {
        // Assistant turns whose divergence lives entirely in tool calls
        // (empty content) must still separate the trials.
        let trial = |arguments: &str| {
            json!({
                "messages": [
                    {"role": "user", "content": "task"},
                    {"role": "assistant", "content": null, "tool_calls": [
                        {"function": {"name": "read_file", "arguments": arguments}},
                    ]},
                ],
            })
        };
        assert_ne!(
            session_key_from_body_with_depth(&trial("{\"path\": \"a.rs\"}"), 1),
            session_key_from_body_with_depth(&trial("{\"path\": \"b.rs\"}"), 1)
        );
    }

    #[test]
    fn session_key_uses_input_when_messages_has_no_user() {
        // A messages list with no user message falls through to `input`; the
        // first user there anchors the key (proven by it affecting the hash).
        let alpha = json!({
            "messages": [{"role": "system", "content": "sys"}],
            "input": [{"role": "user", "content": "alpha"}],
        });
        let beta = json!({
            "messages": [{"role": "system", "content": "sys"}],
            "input": [{"role": "user", "content": "beta"}],
        });
        assert_ne!(session_key_from_body(&alpha), session_key_from_body(&beta));
    }

    #[test]
    fn session_key_supports_responses_input_only() {
        // Responses API bodies carry turns under `input` with no `messages`.
        let a = json!({"input": [{"role": "user", "content": "x"}]});
        let b = json!({"input": [{"role": "user", "content": "y"}]});
        assert_ne!(session_key_from_body(&a), session_key_from_body(&b));
    }

    #[test]
    fn session_key_includes_system_message_in_messages() {
        // A system/developer message inside `messages` (OpenAI shape) contributes.
        let a = json!({
            "messages": [
                {"role": "system", "content": "sys A"},
                {"role": "user", "content": "q"},
            ],
        });
        let b = json!({
            "messages": [
                {"role": "system", "content": "sys B"},
                {"role": "user", "content": "q"},
            ],
        });
        assert_ne!(session_key_from_body(&a), session_key_from_body(&b));
    }

    #[test]
    fn get_missing_returns_none() {
        let mut cache: SessionCache<i32> = SessionCache::new(2);
        assert!(cache.get("nope").is_none());
        cache.put("a".to_string(), 1);
        assert_eq!(cache.get("a"), Some(&1));
        assert!(cache.get("b").is_none());
    }

    #[test]
    fn put_overwrites_existing_value() {
        // Re-pinning a session updates its value (mirrors pin-on-success).
        let mut cache: SessionCache<i32> = SessionCache::new(2);
        cache.put("a".to_string(), 1);
        cache.put("a".to_string(), 2);
        assert_eq!(cache.get("a"), Some(&2));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn max_sessions_reports_capacity() {
        let cache: SessionCache<i32> = SessionCache::new(5);
        assert_eq!(cache.max_sessions(), 5);
    }

    #[test]
    fn values_yields_all_retained() {
        let mut cache: SessionCache<i32> = SessionCache::new(3);
        cache.put("a".to_string(), 1);
        cache.put("b".to_string(), 2);
        let mut vals: Vec<i32> = cache.values().copied().collect();
        vals.sort();
        assert_eq!(vals, vec![1, 2]);
    }
}
