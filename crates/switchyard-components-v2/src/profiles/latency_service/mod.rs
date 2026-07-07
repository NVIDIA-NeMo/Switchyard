// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Latency-service profile implemented as a profile-owned runtime.

mod health;
mod polling;
mod selection;

#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use async_trait::async_trait;
use parking_lot::RwLock;
use switchyard_components::StatsAccumulator;
use switchyard_core::{ChatResponse, LlmTarget, LlmTargetId, Result, SwitchyardError};

pub use self::health::{EndpointHealth, EndpointHealthStatus};
use self::polling::{duration_from_secs, HealthPoller, HealthResponse};
use self::selection::select_target;
pub use self::selection::SelectedTarget;
use crate::backend::{native_target_backend, TargetBackend};
use crate::profile_stats_accumulator;
use crate::stats::{record_usage_or_tap_stream, UsageAttribution};
use crate::{
    profile_config, Profile, ProfileConfig, ProfileHooks, ProfileInput, ProfileResponse,
    RoutingMetadata,
};

const DEFAULT_POLL_TIMEOUT_SECS: f64 = 5.0;
const DEFAULT_MAX_RETRIES: usize = 2;

/// Default HTTP timeout for latency-service health polls.
fn default_poll_timeout_secs() -> f64 {
    DEFAULT_POLL_TIMEOUT_SECS
}

/// Default number of alternate targets to try after the first failed call.
fn default_max_retries() -> usize {
    DEFAULT_MAX_RETRIES
}

/// Config for the flatter latency-service profile.
///
/// The profile owns the health cache and exposes [`LatencyServiceProfile::poll_once`]
/// for whichever control plane embeds it. It does not spawn a background task;
/// `run()` and `process()` perform a guarded initial refresh for multi-target
/// profiles and then route from the profile-owned cache.
#[profile_config("latency-service")]
pub struct LatencyServiceProfileConfig {
    /// Base URL for the external latency service.
    pub latency_service_url: String,
    /// Concrete targets this profile may select at request time.
    #[profile_target]
    pub targets: Vec<LlmTarget>,
    /// Per-poll HTTP timeout in seconds.
    #[serde(default = "default_poll_timeout_secs")]
    pub poll_timeout_secs: f64,
    /// Number of alternate target attempts after the first failed call.
    #[serde(default = "default_max_retries")]
    pub max_retries: usize,
}

impl ProfileConfig for LatencyServiceProfileConfig {
    type Runtime = LatencyServiceProfile;

    /// Builds the runtime profile using existing native backend construction.
    fn build(&self) -> Result<Self::Runtime> {
        validate_config(self)?;
        let target_ids = self
            .targets
            .iter()
            .map(|target| target.id.clone())
            .collect::<Vec<_>>();
        let poll_timeout = duration_from_secs(self.poll_timeout_secs, "poll_timeout_secs")?;
        let poller =
            HealthPoller::new(&self.latency_service_url, target_ids.clone(), poll_timeout)?;

        let mut backends = BTreeMap::new();
        let mut health = BTreeMap::new();
        for target in &self.targets {
            backends.insert(target.id.clone(), native_target_backend(target.clone())?);
            health.insert(
                target.id.clone(),
                EndpointHealth::new(EndpointHealthStatus::Unknown),
            );
        }

        Ok(LatencyServiceProfile {
            poller,
            backends,
            health: RwLock::new(health),
            poll_count: AtomicU64::new(0),
            initial_refresh_in_flight: AtomicBool::new(false),
            max_retries: self.max_retries,
            stats: profile_stats_accumulator(),
        })
    }
}

/// Latency-service profile in the flatter design.
pub struct LatencyServiceProfile {
    poller: HealthPoller,
    backends: BTreeMap<LlmTargetId, TargetBackend>,
    health: RwLock<BTreeMap<LlmTargetId, EndpointHealth>>,
    poll_count: AtomicU64,
    initial_refresh_in_flight: AtomicBool,
    max_retries: usize,
    stats: StatsAccumulator,
}

/// Processed latency-service request with the target selected for response processing.
pub struct LatencyServiceProcessedRequest {
    /// Routed input prepared for the selected backend.
    pub profile_input: ProfileInput,
    /// Target associated with the backend response passed to `rprocess()`.
    pub selected: SelectedTarget,
}

impl LatencyServiceProfile {
    /// Returns true once at least one explicit health poll has completed successfully.
    pub fn is_ready(&self) -> bool {
        self.poll_count.load(Ordering::Relaxed) > 0
    }

    /// Returns a point-in-time copy of the cached health table.
    pub fn health_snapshot(&self) -> BTreeMap<LlmTargetId, EndpointHealth> {
        self.health.read().clone()
    }

    /// Applies externally supplied health for a configured target.
    ///
    /// This is useful for tests and control-plane integrations that already own health
    /// polling and want the profile hot path to remain network-free.
    pub fn update_health(&self, target_id: LlmTargetId, health: EndpointHealth) -> Result<()> {
        let mut cache = self.health.write();
        let Some(entry) = cache.get_mut(&target_id) else {
            return Err(SwitchyardError::InvalidConfig(format!(
                "latency_service target {target_id} is not configured"
            )));
        };
        *entry = health;
        Ok(())
    }

    /// Performs one latency-service health poll and updates the local cache.
    ///
    /// A failed poll resets known health to `unknown` so stale healthy readings do not
    /// keep biasing target selection after the control plane becomes unavailable.
    pub async fn poll_once(&self) -> Result<()> {
        match self
            .poller
            .fetch_health()
            .await
            .and_then(|payload| self.apply_health(payload))
        {
            Ok(()) => {
                self.poll_count.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Err(error) => {
                self.reset_to_unknown();
                Err(error)
            }
        }
    }

    async fn refresh_if_unready(&self) {
        if self.backends.len() <= 1 || self.is_ready() || self.has_health_signal() {
            return;
        }
        if self
            .initial_refresh_in_flight
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }

        if let Err(error) = self.poll_once().await {
            tracing::warn!(
                error = %error,
                "latency_service initial health refresh failed; routing with unknown health"
            );
        }
        self.initial_refresh_in_flight
            .store(false, Ordering::Release);
    }

    fn has_health_signal(&self) -> bool {
        self.health.read().values().any(|health| {
            health.status != EndpointHealthStatus::Unknown || health.last_latency_ms.is_some()
        })
    }

    // Selects the target and rewrites the request model without persisting state.
    fn route_request(&self, mut input: ProfileInput) -> Result<LatencyServiceProcessedRequest> {
        let selected = self.select(&BTreeSet::new())?;
        let backend = self.selected_backend(&selected.target_id)?;
        input.request.set_model(backend.target().model.as_str());
        Ok(LatencyServiceProcessedRequest {
            profile_input: input,
            selected,
        })
    }

    // Reads the health cache and applies the latency-service tiering policy.
    fn select(&self, excluded: &BTreeSet<LlmTargetId>) -> Result<SelectedTarget> {
        let snapshot = self.health.read();
        select_target(&snapshot, excluded)
    }

    // Finds the concrete backend for a selected target.
    fn selected_backend(&self, target_id: &LlmTargetId) -> Result<&TargetBackend> {
        self.backends.get(target_id).ok_or_else(|| {
            SwitchyardError::InvalidConfig(format!(
                "latency_service selected target {target_id} without a configured backend"
            ))
        })
    }

    // Resets preference signals while preserving the configured target set.
    fn reset_to_unknown(&self) {
        let mut cache = self.health.write();
        for health in cache.values_mut() {
            *health = EndpointHealth::new(EndpointHealthStatus::Unknown);
        }
    }

    // Applies health for known targets and resets omitted known targets to unknown.
    fn apply_health(&self, payload: HealthResponse) -> Result<()> {
        let known_ids = self
            .poller
            .target_ids()
            .iter()
            .map(LlmTargetId::as_str)
            .collect::<BTreeSet<_>>();
        let mut reported = BTreeSet::new();
        let mut cache = self.health.write();
        for (target_id, health) in payload.endpoint_health {
            if known_ids.contains(target_id.as_str()) {
                let target_id = LlmTargetId::new(target_id)?;
                if let Some(entry) = cache.get_mut(&target_id) {
                    *entry = health;
                    reported.insert(target_id);
                }
            }
        }
        for (target_id, health) in cache.iter_mut() {
            if !reported.contains(target_id) {
                *health = EndpointHealth::new(EndpointHealthStatus::Unknown);
            }
        }
        Ok(())
    }

    fn routing_metadata(&self, selected: &SelectedTarget, selected_model: &str) -> RoutingMetadata {
        RoutingMetadata {
            selected_model: Some(selected_model.to_string()),
            selected_tier: Some(selected.health_status.as_str().to_string()),
            confidence: None,
            router_version: Some("latency-service:v1".to_string()),
            tolerance: None,
            rationale: Some(format!(
                "latency service selected target {} from {} health tier",
                selected.target_id,
                selected.health_status.as_str()
            )),
        }
    }
}

#[async_trait]
impl ProfileHooks for LatencyServiceProfile {
    type ProcessedRequest = LatencyServiceProcessedRequest;

    /// Performs a standalone latency-service routing rewrite for hook-level inspection.
    async fn process(&self, input: ProfileInput) -> Result<Self::ProcessedRequest> {
        self.refresh_if_unready().await;
        self.route_request(input)
    }

    /// Leaves the backend response unchanged after latency-aware routing completes.
    async fn rprocess(
        &self,
        _processed: &Self::ProcessedRequest,
        response: ChatResponse,
    ) -> Result<ChatResponse> {
        Ok(response)
    }
}

#[async_trait]
impl Profile for LatencyServiceProfile {
    /// Executes latency-aware routing with retry state local to this call.
    async fn run(&self, mut input: ProfileInput) -> Result<ProfileResponse> {
        self.refresh_if_unready().await;
        let profile_started_at = Instant::now();
        let mut tried = BTreeSet::new();
        let mut last_error = None;

        for _ in 0..=self.max_retries {
            if tried.len() == self.backends.len() {
                break;
            }
            let selected = match self.select(&tried) {
                Ok(selected) => selected,
                Err(_) if last_error.is_some() => break,
                Err(error) => return Err(error),
            };
            tried.insert(selected.target_id.clone());
            let selected_backend = self.selected_backend(&selected.target_id)?;
            let stats_model = selected_backend.target().model.to_string();
            let routing_metadata = self.routing_metadata(&selected, &stats_model);
            input
                .request
                .set_model(selected_backend.target().model.as_str());
            let backend_started_at = Instant::now();

            match selected_backend.call(&input.request).await {
                Ok(response) => {
                    let backend_latency_ms = backend_started_at.elapsed().as_secs_f64() * 1000.0;
                    self.stats.record_success(
                        stats_model.clone(),
                        Some(backend_latency_ms),
                        None,
                    )?;
                    let response = record_usage_or_tap_stream(
                        response,
                        UsageAttribution::new(
                            self.stats.clone(),
                            stats_model,
                            None,
                            profile_started_at,
                            backend_latency_ms,
                        ),
                    );
                    let processed = LatencyServiceProcessedRequest {
                        profile_input: input,
                        selected,
                    };
                    let response = self.rprocess(&processed, response).await?;
                    return Ok(ProfileResponse::with_routing_metadata(
                        response,
                        routing_metadata,
                    ));
                }
                Err(error) => {
                    self.stats.record_error(stats_model, None)?;
                    last_error = Some(error);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            SwitchyardError::Backend(
                "latency_service profile exhausted retries without calling a target".to_string(),
            )
        }))
    }
}

fn validate_config(config: &LatencyServiceProfileConfig) -> Result<()> {
    if config.latency_service_url.trim().is_empty() {
        return Err(SwitchyardError::InvalidConfig(
            "latency_service profile requires latency_service_url".to_string(),
        ));
    }
    if config.targets.is_empty() {
        return Err(SwitchyardError::InvalidConfig(
            "latency_service profile requires at least one target".to_string(),
        ));
    }
    duration_from_secs(config.poll_timeout_secs, "poll_timeout_secs")?;

    let mut seen = BTreeSet::new();
    for target in &config.targets {
        if !seen.insert(target.id.clone()) {
            return Err(SwitchyardError::InvalidConfig(format!(
                "latency_service profile has duplicate target {}",
                target.id
            )));
        }
    }
    Ok(())
}
