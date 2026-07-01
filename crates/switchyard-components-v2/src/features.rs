// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded, router-neutral snapshots reconstructed from Relay ATOF events.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use switchyard_core::{Result, SwitchyardError};

/// Default maximum number of routing identities retained in memory.
pub const DEFAULT_MAX_RELAY_IDENTITIES: usize = 10_000;
/// Default maximum reconstructed messages retained for one identity.
pub const DEFAULT_MAX_RELAY_HISTORY_PER_IDENTITY: usize = 256;
/// Default maximum number of event idempotency keys retained in memory.
pub const DEFAULT_MAX_RELAY_DEDUPE_ENTRIES: usize = 100_000;
/// Default maximum encoded/string bytes retained across all Relay state.
pub const DEFAULT_MAX_RELAY_RETAINED_BYTES: usize = 64 * 1024 * 1024;
/// Default maximum encoded size accepted for one ATOF event.
pub const DEFAULT_MAX_ATOF_EVENT_BYTES: usize = 256 * 1024;
/// Default maximum encoded size accepted for one ATOF request batch.
pub const DEFAULT_MAX_ATOF_BATCH_BYTES: usize = 4 * 1024 * 1024;

/// Readiness of an ATOF-derived feature snapshot consumed by a router.
///
/// Absence is represented as `None`; the initial typed state is fresh only.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureFreshness {
    /// The snapshot contains routing-ready reconstructed history.
    Fresh,
}

/// Exact identity key used for Relay snapshot accumulation and lookup.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct RelayIdentityKey {
    /// Stable session identifier.
    pub session_id: String,
    /// Optional owner identifier, usually a subagent or resolved scope owner.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub owner_id: Option<String>,
}

impl RelayIdentityKey {
    /// Builds a key scoped only by session.
    pub fn session_only(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            owner_id: None,
        }
    }

    /// Builds a key scoped by session and optional owner.
    pub fn new(session_id: impl Into<String>, owner_id: Option<String>) -> Self {
        Self {
            session_id: session_id.into(),
            owner_id,
        }
    }
}

/// Deterministic memory and request-size limits for Relay accumulation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct RelaySnapshotLimits {
    /// Maximum number of exact identity keys retained at once.
    pub max_identities: usize,
    /// Maximum reconstructed messages retained for each identity.
    pub max_history_per_identity: usize,
    /// Maximum global event idempotency keys retained at once.
    pub max_dedupe_entries: usize,
    /// Maximum encoded/string bytes retained across identities, history, and dedupe keys.
    pub max_retained_bytes: usize,
    /// Maximum serialized size of one event.
    pub max_event_bytes: usize,
    /// Maximum encoded size of one HTTP ingestion batch.
    pub max_batch_bytes: usize,
}

impl RelaySnapshotLimits {
    /// Validates that every fixed-size bound retains at least one entry or byte.
    pub fn validate(self) -> Result<Self> {
        for (name, value) in [
            ("max_identities", self.max_identities),
            ("max_history_per_identity", self.max_history_per_identity),
            ("max_dedupe_entries", self.max_dedupe_entries),
            ("max_retained_bytes", self.max_retained_bytes),
            ("max_event_bytes", self.max_event_bytes),
            ("max_batch_bytes", self.max_batch_bytes),
        ] {
            if value == 0 {
                return Err(SwitchyardError::InvalidConfig(format!(
                    "Relay snapshot limit {name} must be greater than zero"
                )));
            }
        }
        if self.max_event_bytes > self.max_batch_bytes {
            return Err(SwitchyardError::InvalidConfig(format!(
                "Relay snapshot max_event_bytes ({}) cannot exceed max_batch_bytes ({})",
                self.max_event_bytes, self.max_batch_bytes
            )));
        }
        Ok(self)
    }
}

impl Default for RelaySnapshotLimits {
    fn default() -> Self {
        Self {
            max_identities: DEFAULT_MAX_RELAY_IDENTITIES,
            max_history_per_identity: DEFAULT_MAX_RELAY_HISTORY_PER_IDENTITY,
            max_dedupe_entries: DEFAULT_MAX_RELAY_DEDUPE_ENTRIES,
            max_retained_bytes: DEFAULT_MAX_RELAY_RETAINED_BYTES,
            max_event_bytes: DEFAULT_MAX_ATOF_EVENT_BYTES,
            max_batch_bytes: DEFAULT_MAX_ATOF_BATCH_BYTES,
        }
    }
}

/// Router-neutral state reconstructed for one exact Relay identity.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RelaySnapshot {
    /// Identity that owns this snapshot.
    pub identity: RelayIdentityKey,
    /// OpenAI-shaped tool-call and tool-result messages in retained event order.
    pub messages: Vec<Value>,
    /// Number of observed Relay turn-start scopes for this identity.
    pub turn_depth: u64,
    /// Number of unique recognized events ingested for this identity.
    pub event_count: u64,
}

/// Per-batch ingestion outcome, including every non-mutating drop category.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct RelayIngestReport {
    /// Number of parsed event objects supplied to the accumulator.
    pub received_events: u64,
    /// Number of unique recognized events that mutated identity state.
    pub ingested_events: u64,
    /// Number of recognized events ignored because their idempotency key was retained.
    pub duplicate_events: u64,
    /// Total events dropped without mutating identity state, including duplicates.
    pub dropped_events: u64,
    /// Events that were not supported scope/phase combinations.
    pub dropped_unrecognized_events: u64,
    /// Recognized events for which no session identity could be resolved.
    pub dropped_missing_identity_events: u64,
    /// Identity snapshots pruned because the identity cap was reached.
    pub pruned_identities: u64,
    /// Reconstructed messages pruned by identity or per-identity history bounds.
    pub pruned_history_entries: u64,
    /// Old idempotency keys pruned because the dedupe cap was reached.
    pub pruned_dedupe_entries: u64,
    /// Encoded/string bytes removed while pruning retained state.
    pub pruned_state_bytes: u64,
    /// Encoded/string bytes retained after this batch.
    pub retained_state_bytes: u64,
}

impl RelayIngestReport {
    fn record_drop(&mut self, reason: RelayDropReason) {
        self.dropped_events = self.dropped_events.saturating_add(1);
        match reason {
            RelayDropReason::Unrecognized => {
                self.dropped_unrecognized_events =
                    self.dropped_unrecognized_events.saturating_add(1);
            }
            RelayDropReason::MissingIdentity => {
                self.dropped_missing_identity_events =
                    self.dropped_missing_identity_events.saturating_add(1);
            }
        }
    }
}

/// Lifetime counters for one Relay snapshot accumulator.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct RelayAccumulatorCounters {
    /// Number of successfully validated and applied batches.
    pub batches: u64,
    /// Number of parsed event objects supplied across successful batches.
    pub received_events: u64,
    /// Number of unique recognized events that mutated state.
    pub ingested_events: u64,
    /// Number of recognized duplicate events ignored.
    pub duplicate_events: u64,
    /// Total events dropped without mutating identity state.
    pub dropped_events: u64,
    /// Unsupported event kinds or scope/phase combinations dropped.
    pub dropped_unrecognized_events: u64,
    /// Recognized events lacking a resolvable session identity dropped.
    pub dropped_missing_identity_events: u64,
    /// Identity snapshots pruned by the configured cap.
    pub pruned_identities: u64,
    /// Reconstructed history entries pruned by configured caps.
    pub pruned_history_entries: u64,
    /// Idempotency keys pruned by the configured cap.
    pub pruned_dedupe_entries: u64,
    /// Encoded/string bytes removed while pruning retained state.
    pub pruned_state_bytes: u64,
    /// Current encoded/string bytes retained by the accumulator.
    pub retained_state_bytes: u64,
}

impl RelayAccumulatorCounters {
    fn record(&mut self, report: RelayIngestReport) {
        self.batches = self.batches.saturating_add(1);
        self.received_events = self.received_events.saturating_add(report.received_events);
        self.ingested_events = self.ingested_events.saturating_add(report.ingested_events);
        self.duplicate_events = self
            .duplicate_events
            .saturating_add(report.duplicate_events);
        self.dropped_events = self.dropped_events.saturating_add(report.dropped_events);
        self.dropped_unrecognized_events = self
            .dropped_unrecognized_events
            .saturating_add(report.dropped_unrecognized_events);
        self.dropped_missing_identity_events = self
            .dropped_missing_identity_events
            .saturating_add(report.dropped_missing_identity_events);
        self.pruned_identities = self
            .pruned_identities
            .saturating_add(report.pruned_identities);
        self.pruned_history_entries = self
            .pruned_history_entries
            .saturating_add(report.pruned_history_entries);
        self.pruned_dedupe_entries = self
            .pruned_dedupe_entries
            .saturating_add(report.pruned_dedupe_entries);
        self.pruned_state_bytes = self
            .pruned_state_bytes
            .saturating_add(report.pruned_state_bytes);
        self.retained_state_bytes = report.retained_state_bytes;
    }
}

/// In-process accumulator for bounded Relay ATOF-derived history snapshots.
#[derive(Debug)]
pub struct RelaySnapshotAccumulator {
    limits: RelaySnapshotLimits,
    inner: Mutex<RelayAccumulatorState>,
}

impl RelaySnapshotAccumulator {
    /// Builds an empty accumulator using validated limits.
    pub fn with_limits(limits: RelaySnapshotLimits) -> Result<Self> {
        Ok(Self {
            limits: limits.validate()?,
            inner: Mutex::new(RelayAccumulatorState::default()),
        })
    }

    /// Returns this accumulator's immutable limits.
    pub fn limits(&self) -> RelaySnapshotLimits {
        self.limits
    }

    /// Validates the full batch, then applies it under one state lock.
    ///
    /// Invalid objects or oversized events fail before any state is mutated.
    pub fn ingest_batch(&self, events: &[Value]) -> Result<RelayIngestReport> {
        let mut prepared = Vec::with_capacity(events.len());
        let mut encoded_batch_size = 0_usize;
        for (index, event) in events.iter().enumerate() {
            let encoded_size = serde_json::to_vec(event)
                .map_err(|error| SwitchyardError::InvalidRequest(error.to_string()))?
                .len();
            // Count the minimum NDJSON separator between canonical event values.
            encoded_batch_size = encoded_batch_size
                .checked_add(usize::from(index > 0))
                .and_then(|size| size.checked_add(encoded_size))
                .ok_or_else(|| {
                    SwitchyardError::InvalidRequest(
                        "ATOF encoded batch size overflowed usize".to_string(),
                    )
                })?;
            if encoded_batch_size > self.limits.max_batch_bytes {
                return Err(SwitchyardError::InvalidRequest(format!(
                    "ATOF batch is at least {encoded_batch_size} bytes; maximum is {} bytes",
                    self.limits.max_batch_bytes
                )));
            }
            prepared.push(self.prepare_event(index, event, encoded_size)?);
        }

        let mut report = RelayIngestReport {
            received_events: prepared.len() as u64,
            ..RelayIngestReport::default()
        };
        let mut inner = self.inner.lock();
        for event in prepared {
            match event {
                PreparedRelayEvent::Drop(reason) => report.record_drop(reason),
                PreparedRelayEvent::Recognized(event) => {
                    inner.ingest(event, self.limits, &mut report);
                }
            }
        }
        report.retained_state_bytes = inner.retained_state_bytes as u64;
        inner.counters.record(report);
        Ok(report)
    }

    /// Returns a snapshot only for the exact `(session_id, owner_id)` key.
    pub fn snapshot(&self, key: &RelayIdentityKey) -> Option<RelaySnapshot> {
        let inner = self.inner.lock();
        inner.identities.get(key).map(|state| RelaySnapshot {
            identity: key.clone(),
            messages: state
                .messages
                .iter()
                .map(|message| message.value.clone())
                .collect(),
            turn_depth: state.turn_depth,
            event_count: state.event_count,
        })
    }

    /// Returns lifetime ingestion and pruning counters.
    pub fn counters(&self) -> RelayAccumulatorCounters {
        self.inner.lock().counters
    }

    /// Returns the number of retained exact identities.
    pub fn identity_count(&self) -> usize {
        self.inner.lock().identities.len()
    }

    /// Returns exact encoded/string bytes counted against the retained-state cap.
    pub fn retained_state_bytes(&self) -> usize {
        self.inner.lock().retained_state_bytes
    }

    fn prepare_event(
        &self,
        index: usize,
        event: &Value,
        encoded_size: usize,
    ) -> Result<PreparedRelayEvent> {
        if !event.is_object() {
            return Err(SwitchyardError::InvalidRequest(format!(
                "ATOF event {} must be a JSON object",
                index + 1
            )));
        }
        if encoded_size > self.limits.max_event_bytes {
            return Err(SwitchyardError::InvalidRequest(format!(
                "ATOF event {} is {encoded_size} bytes; maximum is {} bytes",
                index + 1,
                self.limits.max_event_bytes
            )));
        }

        let Some((update, dedupe_key)) = relay_update(event, index)? else {
            return Ok(PreparedRelayEvent::Drop(RelayDropReason::Unrecognized));
        };
        let Some(identity) = relay_identity_key_from_atof_event(event) else {
            return Ok(PreparedRelayEvent::Drop(RelayDropReason::MissingIdentity));
        };
        let identity_retained_bytes = retained_identity_bytes(&identity)?;
        let dedupe_retained_bytes = retained_string_copies_bytes(&dedupe_key)?;
        let max_new_state_bytes = identity_retained_bytes
            .checked_add(dedupe_retained_bytes)
            .and_then(|bytes| bytes.checked_add(update.retained_bytes()))
            .ok_or_else(|| {
                SwitchyardError::InvalidRequest(format!(
                    "ATOF event {} retained-state footprint overflowed usize",
                    index + 1
                ))
            })?;
        if max_new_state_bytes > self.limits.max_retained_bytes {
            return Err(SwitchyardError::InvalidRequest(format!(
                "ATOF event {} can retain {max_new_state_bytes} bytes; maximum retained state is {} bytes",
                index + 1,
                self.limits.max_retained_bytes
            )));
        }
        Ok(PreparedRelayEvent::Recognized(RecognizedRelayEvent {
            identity,
            dedupe_key,
            identity_retained_bytes,
            dedupe_retained_bytes,
            update,
        }))
    }
}

impl Default for RelaySnapshotAccumulator {
    fn default() -> Self {
        Self {
            limits: RelaySnapshotLimits::default(),
            inner: Mutex::new(RelayAccumulatorState::default()),
        }
    }
}

#[derive(Debug, Default)]
struct RelayAccumulatorState {
    identities: BTreeMap<RelayIdentityKey, RelayIdentityState>,
    identity_order: VecDeque<RelayIdentityKey>,
    seen_events: BTreeSet<String>,
    dedupe_order: VecDeque<RetainedDedupeKey>,
    retained_state_bytes: usize,
    counters: RelayAccumulatorCounters,
}

impl RelayAccumulatorState {
    fn ingest(
        &mut self,
        event: RecognizedRelayEvent,
        limits: RelaySnapshotLimits,
        report: &mut RelayIngestReport,
    ) {
        if self.seen_events.contains(&event.dedupe_key) {
            report.duplicate_events = report.duplicate_events.saturating_add(1);
            report.dropped_events = report.dropped_events.saturating_add(1);
            return;
        }

        if self.dedupe_order.len() >= limits.max_dedupe_entries {
            self.prune_oldest_dedupe(report);
        }
        if !self.identities.contains_key(&event.identity)
            && self.identities.len() >= limits.max_identities
        {
            self.prune_oldest_identity(report);
        }
        self.prune_history_for_incoming(
            &event.identity,
            event.update.retained_bytes(),
            limits.max_history_per_identity,
            report,
        );
        self.make_room_for(&event, limits.max_retained_bytes, report);

        self.insert_dedupe(event.dedupe_key, event.dedupe_retained_bytes);
        self.ensure_identity(event.identity.clone(), event.identity_retained_bytes);
        self.apply_update(&event.identity, event.update);
        report.ingested_events = report.ingested_events.saturating_add(1);
    }

    fn make_room_for(
        &mut self,
        event: &RecognizedRelayEvent,
        max_retained_bytes: usize,
        report: &mut RelayIngestReport,
    ) {
        loop {
            let identity_bytes = if self.identities.contains_key(&event.identity) {
                0
            } else {
                event.identity_retained_bytes
            };
            let additional = identity_bytes
                .saturating_add(event.dedupe_retained_bytes)
                .saturating_add(event.update.retained_bytes());
            if self
                .retained_state_bytes
                .checked_add(additional)
                .is_some_and(|total| total <= max_retained_bytes)
            {
                break;
            }
            if self.prune_oldest_dedupe(report) || self.prune_oldest_identity(report) {
                continue;
            }
            // Preparation rejects any single event whose worst-case footprint
            // exceeds the cap, so empty state always has enough room.
            break;
        }
    }

    fn insert_dedupe(&mut self, key: String, retained_bytes: usize) {
        self.seen_events.insert(key.clone());
        self.dedupe_order.push_back(RetainedDedupeKey {
            key,
            retained_bytes,
        });
        self.retained_state_bytes = self.retained_state_bytes.saturating_add(retained_bytes);
    }

    fn ensure_identity(&mut self, key: RelayIdentityKey, retained_key_bytes: usize) {
        if self.identities.contains_key(&key) {
            return;
        }
        self.identity_order.push_back(key.clone());
        self.identities.insert(
            key,
            RelayIdentityState {
                retained_key_bytes,
                ..RelayIdentityState::default()
            },
        );
        self.retained_state_bytes = self.retained_state_bytes.saturating_add(retained_key_bytes);
    }

    fn apply_update(&mut self, key: &RelayIdentityKey, update: RelayUpdate) {
        let Some(state) = self.identities.get_mut(key) else {
            return;
        };
        state.event_count = state.event_count.saturating_add(1);
        match update {
            RelayUpdate::TurnStart => {
                state.turn_depth = state.turn_depth.saturating_add(1);
            }
            RelayUpdate::TurnEnd => {}
            RelayUpdate::Message(message) => {
                self.retained_state_bytes = self
                    .retained_state_bytes
                    .saturating_add(message.encoded_bytes);
                state.messages.push_back(message);
            }
        }
    }

    fn prune_history_for_incoming(
        &mut self,
        key: &RelayIdentityKey,
        incoming_message_bytes: usize,
        max_history: usize,
        report: &mut RelayIngestReport,
    ) {
        if incoming_message_bytes == 0 {
            return;
        }
        let Some(state) = self.identities.get_mut(key) else {
            return;
        };
        while state.messages.len() >= max_history {
            let Some(message) = state.messages.pop_front() else {
                break;
            };
            self.retained_state_bytes = self
                .retained_state_bytes
                .saturating_sub(message.encoded_bytes);
            record_pruned_bytes(report, message.encoded_bytes);
            report.pruned_history_entries = report.pruned_history_entries.saturating_add(1);
        }
    }

    fn prune_oldest_dedupe(&mut self, report: &mut RelayIngestReport) -> bool {
        let Some(oldest) = self.dedupe_order.pop_front() else {
            return false;
        };
        self.seen_events.remove(&oldest.key);
        self.retained_state_bytes = self
            .retained_state_bytes
            .saturating_sub(oldest.retained_bytes);
        record_pruned_bytes(report, oldest.retained_bytes);
        report.pruned_dedupe_entries = report.pruned_dedupe_entries.saturating_add(1);
        true
    }

    /// Identity eviction is FIFO by first insertion, making pruning repeatable.
    fn prune_oldest_identity(&mut self, report: &mut RelayIngestReport) -> bool {
        let Some(oldest) = self.identity_order.pop_front() else {
            return false;
        };
        let Some(removed) = self.identities.remove(&oldest) else {
            return false;
        };
        let message_bytes = removed
            .messages
            .iter()
            .map(|message| message.encoded_bytes)
            .sum::<usize>();
        let removed_bytes = removed.retained_key_bytes.saturating_add(message_bytes);
        self.retained_state_bytes = self.retained_state_bytes.saturating_sub(removed_bytes);
        record_pruned_bytes(report, removed_bytes);
        report.pruned_identities = report.pruned_identities.saturating_add(1);
        report.pruned_history_entries = report
            .pruned_history_entries
            .saturating_add(removed.messages.len() as u64);
        true
    }
}

#[derive(Debug, Default)]
struct RelayIdentityState {
    messages: VecDeque<RetainedMessage>,
    turn_depth: u64,
    event_count: u64,
    retained_key_bytes: usize,
}

#[derive(Debug)]
enum PreparedRelayEvent {
    Drop(RelayDropReason),
    Recognized(RecognizedRelayEvent),
}

#[derive(Clone, Copy, Debug)]
enum RelayDropReason {
    Unrecognized,
    MissingIdentity,
}

#[derive(Debug)]
struct RecognizedRelayEvent {
    identity: RelayIdentityKey,
    dedupe_key: String,
    identity_retained_bytes: usize,
    dedupe_retained_bytes: usize,
    update: RelayUpdate,
}

#[derive(Debug)]
enum RelayUpdate {
    TurnStart,
    TurnEnd,
    Message(RetainedMessage),
}

impl RelayUpdate {
    fn retained_bytes(&self) -> usize {
        match self {
            Self::TurnStart | Self::TurnEnd => 0,
            Self::Message(message) => message.encoded_bytes,
        }
    }
}

#[derive(Debug)]
struct RetainedMessage {
    value: Value,
    encoded_bytes: usize,
}

#[derive(Debug)]
struct RetainedDedupeKey {
    key: String,
    retained_bytes: usize,
}

/// Resolves an identity using the compatibility fallback order from Relay PR integration.
pub fn relay_identity_key_from_atof_event(event: &Value) -> Option<RelayIdentityKey> {
    let metadata = event.get("metadata").unwrap_or(&Value::Null);
    let data = event.get("data").unwrap_or(&Value::Null);
    let session_id = non_empty_json_string_at(metadata, &["session_id"])
        .or_else(|| non_empty_json_string_at(metadata, &["hermes_session_id"]))
        .or_else(|| non_empty_json_string_at(data, &["session_id"]))
        .or_else(|| {
            non_empty_json_string_at(data, &["request", "headers", "x-nemo-relay-session-id"])
        })?;
    let owner_id = non_empty_json_string_at(metadata, &["switchyard_owner_id"])
        .or_else(|| non_empty_json_string_at(metadata, &["llm_correlation_subagent_id"]))
        .or_else(|| non_empty_json_string_at(metadata, &["tool_correlation_subagent_id"]))
        .or_else(|| non_empty_json_string_at(metadata, &["subagent_id"]))
        .or_else(|| non_empty_json_string_at(metadata, &["subagent_session_id"]))
        .or_else(|| non_empty_json_string_at(metadata, &["hermes_subagent_session_id"]))
        .or_else(|| {
            non_empty_json_string_at(data, &["request", "headers", "x-nemo-relay-owner-id"])
        });
    Some(RelayIdentityKey::new(session_id, owner_id))
}

/// Returns a scope-event idempotency key when a UUID is available.
pub fn atof_event_dedupe_key(event: &Value) -> Option<String> {
    let uuid = non_empty_json_string_at(event, &["uuid"])?;
    let phase =
        non_empty_json_string_at(event, &["scope_category"]).unwrap_or_else(|| "mark".to_string());
    Some(format!("{uuid}:{phase}"))
}

/// Reads a string from a nested JSON object path.
pub fn json_string_at(value: &Value, path: &[&str]) -> Option<String> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    current.as_str().map(ToOwned::to_owned)
}

fn non_empty_json_string_at(value: &Value, path: &[&str]) -> Option<String> {
    json_string_at(value, path).and_then(|value| {
        let trimmed = value.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_string())
    })
}

fn retained_identity_bytes(identity: &RelayIdentityKey) -> Result<usize> {
    let one_copy = identity
        .session_id
        .len()
        .checked_add(identity.owner_id.as_deref().map_or(0, str::len))
        .ok_or_else(|| {
            SwitchyardError::InvalidRequest(
                "Relay identity retained-state footprint overflowed usize".to_string(),
            )
        })?;
    one_copy.checked_mul(2).ok_or_else(|| {
        SwitchyardError::InvalidRequest(
            "Relay identity retained-state footprint overflowed usize".to_string(),
        )
    })
}

fn retained_string_copies_bytes(value: &str) -> Result<usize> {
    value.len().checked_mul(2).ok_or_else(|| {
        SwitchyardError::InvalidRequest(
            "Relay dedupe retained-state footprint overflowed usize".to_string(),
        )
    })
}

fn retained_message(value: Value) -> Result<RetainedMessage> {
    let encoded_bytes = serde_json::to_vec(&value)
        .map_err(|error| SwitchyardError::InvalidRequest(error.to_string()))?
        .len();
    Ok(RetainedMessage {
        value,
        encoded_bytes,
    })
}

fn record_pruned_bytes(report: &mut RelayIngestReport, bytes: usize) {
    report.pruned_state_bytes = report.pruned_state_bytes.saturating_add(bytes as u64);
}

/// Semantically validates recognized scope events before accumulator locking.
fn relay_update(event: &Value, index: usize) -> Result<Option<(RelayUpdate, String)>> {
    if event.get("kind").and_then(Value::as_str) != Some("scope") {
        return Ok(None);
    }
    let turn_scope = is_turn_scope(event);
    let tool_scope = event.get("category").and_then(Value::as_str) == Some("tool");
    if !turn_scope && !tool_scope {
        return Ok(None);
    }

    let uuid = required_event_string(event, &["uuid"], "uuid", index)?;
    let phase = required_event_string(event, &["scope_category"], "scope_category", index)?;
    if !matches!(phase.as_str(), "start" | "end") {
        return Err(invalid_recognized_event(
            index,
            format!("scope_category must be start or end, got {phase:?}"),
        ));
    }
    let dedupe_key = format!("{uuid}:{phase}");

    if turn_scope {
        let update = if phase == "start" {
            RelayUpdate::TurnStart
        } else {
            RelayUpdate::TurnEnd
        };
        return Ok(Some((update, dedupe_key)));
    }

    let name = required_event_string(event, &["name"], "name", index)?;
    let tool_call_id = required_event_string(
        event,
        &["category_profile", "tool_call_id"],
        "category_profile.tool_call_id",
        index,
    )?;
    let data = event
        .get("data")
        .filter(|value| !value.is_null())
        .ok_or_else(|| invalid_recognized_event(index, "data must be present and non-null"))?;
    let message = if phase == "start" {
        tool_call_message(&name, &tool_call_id, data)
    } else {
        tool_result_message(&tool_call_id, data)
    };
    Ok(Some((
        RelayUpdate::Message(retained_message(message)?),
        dedupe_key,
    )))
}

fn is_turn_scope(event: &Value) -> bool {
    non_empty_json_string_at(event, &["metadata", "nemo_relay_scope_role"]).as_deref()
        == Some("turn")
}

fn required_event_string(
    event: &Value,
    path: &[&str],
    field: &str,
    index: usize,
) -> Result<String> {
    non_empty_json_string_at(event, path).ok_or_else(|| {
        invalid_recognized_event(index, format!("{field} must be a non-empty string"))
    })
}

fn invalid_recognized_event(index: usize, message: impl Into<String>) -> SwitchyardError {
    SwitchyardError::InvalidRequest(format!(
        "ATOF event {} is not a canonical recognized scope: {}",
        index + 1,
        message.into()
    ))
}

fn tool_call_message(name: &str, tool_call_id: &str, arguments: &Value) -> Value {
    json!({
        "role": "assistant",
        "tool_calls": [{
            "id": tool_call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments.to_string(),
            },
        }],
    })
}

fn tool_result_message(tool_call_id: &str, data: &Value) -> Value {
    let content = event_data_to_text(data);
    json!({
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content,
    })
}

fn event_data_to_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Object(object) => object
            .get("output")
            .or_else(|| object.get("result"))
            .or_else(|| object.get("content"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| value.to_string()),
        _ => value.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn limits(
        max_identities: usize,
        max_history_per_identity: usize,
        max_dedupe_entries: usize,
    ) -> RelaySnapshotLimits {
        RelaySnapshotLimits {
            max_identities,
            max_history_per_identity,
            max_dedupe_entries,
            max_retained_bytes: 64 * 1024,
            max_event_bytes: 4_096,
            max_batch_bytes: 16_384,
        }
    }

    fn tool_event(
        session_id: Option<&str>,
        owner_id: Option<&str>,
        uuid: &str,
        phase: &str,
        data: Value,
    ) -> Value {
        let mut metadata = serde_json::Map::new();
        if let Some(session_id) = session_id {
            metadata.insert("hermes_session_id".to_string(), json!(session_id));
        }
        if let Some(owner_id) = owner_id {
            metadata.insert("switchyard_owner_id".to_string(), json!(owner_id));
        }
        json!({
            "kind": "scope",
            "uuid": uuid,
            "scope_category": phase,
            "name": "Bash",
            "category": "tool",
            "category_profile": {"tool_call_id": uuid},
            "data": data,
            "metadata": metadata,
        })
    }

    fn turn_event(session_id: &str, uuid: &str, phase: &str) -> Value {
        json!({
            "kind": "scope",
            "uuid": uuid,
            "scope_category": phase,
            "name": "turn",
            "category": "agent",
            "metadata": {
                "hermes_session_id": session_id,
                "nemo_relay_scope_role": "turn",
            },
        })
    }

    fn retained_footprint(event: &Value) -> Result<usize> {
        let identity = relay_identity_key_from_atof_event(event)
            .ok_or_else(|| SwitchyardError::Other("test event identity missing".to_string()))?;
        let (update, dedupe_key) = relay_update(event, 0)?
            .ok_or_else(|| SwitchyardError::Other("test event was not recognized".to_string()))?;
        retained_identity_bytes(&identity)?
            .checked_add(retained_string_copies_bytes(&dedupe_key)?)
            .and_then(|bytes| bytes.checked_add(update.retained_bytes()))
            .ok_or_else(|| SwitchyardError::Other("test footprint overflow".to_string()))
    }

    #[test]
    fn default_limits_are_valid() -> Result<()> {
        let limits = RelaySnapshotLimits::default().validate()?;
        assert!(limits.max_batch_bytes >= limits.max_event_bytes);
        assert_eq!(limits.max_retained_bytes, 64 * 1024 * 1024);
        Ok(())
    }

    #[test]
    fn zero_and_inverted_limits_are_rejected() {
        let zero = RelaySnapshotLimits {
            max_identities: 0,
            ..RelaySnapshotLimits::default()
        };
        assert!(zero.validate().is_err());

        let inverted = RelaySnapshotLimits {
            max_event_bytes: 5,
            max_batch_bytes: 4,
            ..RelaySnapshotLimits::default()
        };
        assert!(inverted.validate().is_err());
    }

    #[test]
    fn identity_extraction_preserves_session_and_owner_fallback_order() {
        let mut session_event = json!({
            "metadata": {
                "session_id": "session-primary",
                "hermes_session_id": "session-secondary",
            },
            "data": {
                "session_id": "session-data",
                "request": {"headers": {
                    "x-nemo-relay-session-id": "session-header",
                }},
            },
        });
        for (expected, remove_from_metadata) in [
            ("session-primary", Some("session_id")),
            ("session-secondary", Some("hermes_session_id")),
            ("session-data", None),
        ] {
            assert_eq!(
                relay_identity_key_from_atof_event(&session_event),
                Some(RelayIdentityKey::session_only(expected))
            );
            if let Some(field) = remove_from_metadata {
                if let Some(metadata) = session_event["metadata"].as_object_mut() {
                    metadata.remove(field);
                }
            } else if let Some(data) = session_event["data"].as_object_mut() {
                data.remove("session_id");
            }
        }
        assert_eq!(
            relay_identity_key_from_atof_event(&session_event),
            Some(RelayIdentityKey::session_only("session-header"))
        );

        let mut owner_event = json!({
            "metadata": {
                "hermes_session_id": "s",
                "switchyard_owner_id": "owner-switchyard",
                "llm_correlation_subagent_id": "owner-llm",
                "tool_correlation_subagent_id": "owner-tool",
                "subagent_id": "owner-subagent",
                "subagent_session_id": "owner-session",
                "hermes_subagent_session_id": "owner-hermes",
            },
            "data": {"request": {"headers": {
                "x-nemo-relay-owner-id": "owner-header"
            }}},
        });
        for (field, expected) in [
            ("switchyard_owner_id", "owner-switchyard"),
            ("llm_correlation_subagent_id", "owner-llm"),
            ("tool_correlation_subagent_id", "owner-tool"),
            ("subagent_id", "owner-subagent"),
            ("subagent_session_id", "owner-session"),
            ("hermes_subagent_session_id", "owner-hermes"),
        ] {
            assert_eq!(
                relay_identity_key_from_atof_event(&owner_event),
                Some(RelayIdentityKey::new("s", Some(expected.to_string())))
            );
            if let Some(metadata) = owner_event["metadata"].as_object_mut() {
                metadata.remove(field);
            }
        }
        assert_eq!(
            relay_identity_key_from_atof_event(&owner_event),
            Some(RelayIdentityKey::new("s", Some("owner-header".to_string())))
        );
    }

    #[test]
    fn reconstructs_router_neutral_snapshot_and_uses_exact_lookup() -> Result<()> {
        let accumulator = RelaySnapshotAccumulator::with_limits(limits(4, 8, 16))?;
        let events = vec![
            turn_event("session-1", "turn-1", "start"),
            tool_event(
                Some("session-1"),
                Some("owner-a"),
                "tool-1",
                "start",
                json!({"command": "cargo test"}),
            ),
            tool_event(
                Some("session-1"),
                Some("owner-a"),
                "tool-1",
                "end",
                json!({"output": "ok"}),
            ),
        ];
        let report = accumulator.ingest_batch(&events)?;
        assert_eq!(report.ingested_events, 3);

        let session = accumulator
            .snapshot(&RelayIdentityKey::session_only("session-1"))
            .ok_or_else(|| SwitchyardError::Other("missing session snapshot".to_string()))?;
        assert_eq!(session.turn_depth, 1);
        assert!(session.messages.is_empty());
        assert_eq!(session.event_count, 1);

        let owner_key = RelayIdentityKey::new("session-1", Some("owner-a".to_string()));
        let owner = accumulator
            .snapshot(&owner_key)
            .ok_or_else(|| SwitchyardError::Other("missing owner snapshot".to_string()))?;
        assert_eq!(owner.turn_depth, 0);
        assert_eq!(owner.event_count, 2);
        assert_eq!(owner.messages.len(), 2);
        assert_eq!(owner.messages[0]["role"], "assistant");
        assert_eq!(owner.messages[1]["role"], "tool");
        assert_eq!(owner.messages[1]["content"], "ok");
        assert!(accumulator
            .snapshot(&RelayIdentityKey::new(
                "session-1",
                Some("owner-b".to_string())
            ))
            .is_none());
        Ok(())
    }

    #[test]
    fn invalid_batch_is_atomic() -> Result<()> {
        let accumulator = RelaySnapshotAccumulator::with_limits(limits(4, 8, 16))?;
        let events = vec![
            tool_event(Some("session-1"), None, "tool-1", "start", json!({})),
            json!(["not", "an", "object"]),
        ];

        assert!(accumulator.ingest_batch(&events).is_err());
        assert!(accumulator
            .snapshot(&RelayIdentityKey::session_only("session-1"))
            .is_none());
        assert_eq!(accumulator.counters(), RelayAccumulatorCounters::default());
        Ok(())
    }

    #[test]
    fn malformed_recognized_scopes_reject_entire_batch_before_mutation() -> Result<()> {
        let valid = tool_event(Some("session-1"), None, "valid-tool", "start", json!({}));
        let mut missing_tool_uuid = valid.clone();
        missing_tool_uuid
            .as_object_mut()
            .ok_or_else(|| SwitchyardError::Other("test tool event must be object".to_string()))?
            .remove("uuid");
        let mut missing_name = valid.clone();
        missing_name
            .as_object_mut()
            .ok_or_else(|| SwitchyardError::Other("test tool event must be object".to_string()))?
            .remove("name");
        let mut missing_profile = valid.clone();
        missing_profile
            .as_object_mut()
            .ok_or_else(|| SwitchyardError::Other("test tool event must be object".to_string()))?
            .remove("category_profile");
        let mut missing_data = valid.clone();
        missing_data
            .as_object_mut()
            .ok_or_else(|| SwitchyardError::Other("test tool event must be object".to_string()))?
            .remove("data");
        let mut missing_turn_uuid = turn_event("session-1", "turn-1", "start");
        missing_turn_uuid
            .as_object_mut()
            .ok_or_else(|| SwitchyardError::Other("test turn event must be object".to_string()))?
            .remove("uuid");
        let mut invalid_phase = valid.clone();
        invalid_phase["scope_category"] = json!("middle");

        for malformed in [
            missing_tool_uuid,
            missing_name,
            missing_profile,
            missing_data,
            missing_turn_uuid,
            invalid_phase,
        ] {
            let accumulator = RelaySnapshotAccumulator::with_limits(limits(4, 8, 16))?;
            let error = accumulator
                .ingest_batch(&[valid.clone(), malformed])
                .err()
                .ok_or_else(|| {
                    SwitchyardError::Other("expected canonical scope error".to_string())
                })?;
            assert!(error.to_string().contains("canonical recognized scope"));
            assert_eq!(accumulator.identity_count(), 0);
            assert_eq!(accumulator.retained_state_bytes(), 0);
            assert_eq!(accumulator.counters(), RelayAccumulatorCounters::default());
        }
        Ok(())
    }

    #[test]
    fn global_uuid_phase_dedupe_cannot_be_bypassed_by_identity_metadata() -> Result<()> {
        let accumulator = RelaySnapshotAccumulator::with_limits(limits(4, 8, 16))?;
        let first = tool_event(
            Some("session-a"),
            Some("owner-a"),
            "shared-uuid",
            "start",
            json!({"command": "first"}),
        );
        let replay = tool_event(
            Some("session-b"),
            Some("owner-b"),
            "shared-uuid",
            "start",
            json!({"command": "changed"}),
        );

        assert_eq!(accumulator.ingest_batch(&[first])?.ingested_events, 1);
        let report = accumulator.ingest_batch(&[replay])?;
        assert_eq!(report.ingested_events, 0);
        assert_eq!(report.duplicate_events, 1);
        assert!(accumulator
            .snapshot(&RelayIdentityKey::new(
                "session-b",
                Some("owner-b".to_string())
            ))
            .is_none());
        Ok(())
    }

    #[test]
    fn retained_byte_accounting_covers_identity_message_and_dedupe_copies() -> Result<()> {
        let event = tool_event(Some("s"), Some("o"), "u", "start", json!({"command": "x"}));
        let expected = retained_footprint(&event)?;
        let accumulator = RelaySnapshotAccumulator::with_limits(limits(4, 8, 16))?;

        let report = accumulator.ingest_batch(&[event])?;

        assert_eq!(report.retained_state_bytes, expected as u64);
        assert_eq!(accumulator.retained_state_bytes(), expected);
        assert_eq!(accumulator.counters().retained_state_bytes, expected as u64);
        Ok(())
    }

    #[test]
    fn total_retained_byte_cap_prunes_and_accounts_before_inserting() -> Result<()> {
        let first = tool_event(
            Some("session"),
            None,
            "tool-a",
            "start",
            json!({"command": "a"}),
        );
        let second = tool_event(
            Some("session"),
            None,
            "tool-b",
            "start",
            json!({"command": "b"}),
        );
        let cap = retained_footprint(&first)?;
        assert_eq!(cap, retained_footprint(&second)?);
        let accumulator = RelaySnapshotAccumulator::with_limits(RelaySnapshotLimits {
            max_retained_bytes: cap,
            ..limits(4, 8, 16)
        })?;

        accumulator.ingest_batch(&[first])?;
        let report = accumulator.ingest_batch(&[second])?;

        assert_eq!(report.pruned_dedupe_entries, 1);
        assert_eq!(report.pruned_identities, 1);
        assert_eq!(report.pruned_history_entries, 1);
        assert_eq!(report.pruned_state_bytes, cap as u64);
        assert_eq!(report.retained_state_bytes, cap as u64);
        assert_eq!(accumulator.retained_state_bytes(), cap);
        assert!(accumulator.retained_state_bytes() <= accumulator.limits().max_retained_bytes);
        Ok(())
    }

    #[test]
    fn oversized_retained_footprint_rejects_batch_before_lock() -> Result<()> {
        let event = tool_event(
            Some("session"),
            None,
            "tool-a",
            "start",
            json!({"command": "a"}),
        );
        let footprint = retained_footprint(&event)?;
        let accumulator = RelaySnapshotAccumulator::with_limits(RelaySnapshotLimits {
            max_retained_bytes: footprint.saturating_sub(1),
            ..limits(4, 8, 16)
        })?;

        let error = accumulator
            .ingest_batch(&[event])
            .err()
            .ok_or_else(|| SwitchyardError::Other("expected retained limit error".to_string()))?;
        assert!(error.to_string().contains("maximum retained state"));
        assert_eq!(accumulator.counters(), RelayAccumulatorCounters::default());
        assert_eq!(accumulator.retained_state_bytes(), 0);
        Ok(())
    }

    #[test]
    fn oversized_event_rejects_whole_batch() -> Result<()> {
        let accumulator = RelaySnapshotAccumulator::with_limits(RelaySnapshotLimits {
            max_event_bytes: 200,
            max_batch_bytes: 1_000,
            ..limits(4, 8, 16)
        })?;
        let events = vec![
            turn_event("session-1", "turn-1", "start"),
            tool_event(
                Some("session-1"),
                None,
                "tool-1",
                "end",
                json!({"output": "x".repeat(500)}),
            ),
        ];

        let error = accumulator
            .ingest_batch(&events)
            .err()
            .ok_or_else(|| SwitchyardError::Other("expected event limit error".to_string()))?;
        assert!(error.to_string().contains("ATOF event 2"));
        assert_eq!(accumulator.identity_count(), 0);
        Ok(())
    }

    #[test]
    fn encoded_batch_limit_is_enforced_before_mutation() -> Result<()> {
        let accumulator = RelaySnapshotAccumulator::with_limits(RelaySnapshotLimits {
            max_event_bytes: 300,
            max_batch_bytes: 300,
            ..limits(4, 8, 16)
        })?;
        let events = vec![
            tool_event(
                Some("session-1"),
                None,
                "tool-1",
                "start",
                json!({"command": "x".repeat(80)}),
            ),
            tool_event(
                Some("session-1"),
                None,
                "tool-2",
                "start",
                json!({"command": "y".repeat(80)}),
            ),
        ];

        let error = accumulator
            .ingest_batch(&events)
            .err()
            .ok_or_else(|| SwitchyardError::Other("expected batch limit error".to_string()))?;
        assert!(error.to_string().contains("ATOF batch"));
        assert_eq!(accumulator.identity_count(), 0);
        Ok(())
    }

    #[test]
    fn reports_duplicates_and_non_feature_drops() -> Result<()> {
        let accumulator = RelaySnapshotAccumulator::with_limits(limits(4, 8, 16))?;
        let event = tool_event(Some("session-1"), None, "tool-1", "start", json!({}));
        let missing_identity = tool_event(None, None, "tool-2", "start", json!({}));
        let report = accumulator.ingest_batch(&[
            event.clone(),
            event,
            missing_identity,
            json!({"kind": "mark", "uuid": "mark-1"}),
        ])?;

        assert_eq!(report.received_events, 4);
        assert_eq!(report.ingested_events, 1);
        assert_eq!(report.duplicate_events, 1);
        assert_eq!(report.dropped_events, 3);
        assert_eq!(report.dropped_missing_identity_events, 1);
        assert_eq!(report.dropped_unrecognized_events, 1);
        let counters = accumulator.counters();
        assert_eq!(counters.batches, 1);
        assert_eq!(counters.dropped_events, 3);
        Ok(())
    }

    #[test]
    fn fixed_caps_prune_fifo_identity_history_and_dedupe_state() -> Result<()> {
        let accumulator = RelaySnapshotAccumulator::with_limits(limits(2, 2, 2))?;
        let report = accumulator.ingest_batch(&[
            tool_event(Some("a"), None, "a-1", "start", json!({})),
            tool_event(Some("b"), None, "b-1", "start", json!({})),
            tool_event(Some("c"), None, "c-1", "start", json!({})),
            tool_event(Some("c"), None, "c-2", "end", json!("one")),
            tool_event(Some("c"), None, "c-3", "end", json!("two")),
        ])?;

        assert_eq!(accumulator.identity_count(), 2);
        assert!(accumulator
            .snapshot(&RelayIdentityKey::session_only("a"))
            .is_none());
        let c = accumulator
            .snapshot(&RelayIdentityKey::session_only("c"))
            .ok_or_else(|| SwitchyardError::Other("missing c snapshot".to_string()))?;
        assert_eq!(c.messages.len(), 2);
        assert_eq!(c.messages[0]["content"], "one");
        assert_eq!(c.messages[1]["content"], "two");
        assert_eq!(report.pruned_identities, 1);
        assert_eq!(report.pruned_history_entries, 2);
        assert_eq!(report.pruned_dedupe_entries, 3);

        // a-1's key was pruned from the bounded dedupe history, so replay is accepted.
        let replay =
            accumulator.ingest_batch(&[tool_event(Some("a"), None, "a-1", "start", json!({}))])?;
        assert_eq!(replay.ingested_events, 1);
        Ok(())
    }
}
