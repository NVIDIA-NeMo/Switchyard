// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-wide Prometheus export for libsy's OpenTelemetry metrics.

use std::sync::OnceLock;

use opentelemetry::global;
use opentelemetry_sdk::metrics::SdkMeterProvider;
use prometheus::{Encoder, Registry, TextEncoder};

pub(crate) const CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

struct Metrics {
    registry: Registry,
    _provider: SdkMeterProvider,
}

static METRICS: OnceLock<Result<Metrics, String>> = OnceLock::new();

/// Returns the registry shared by every server router in this process.
pub(crate) fn registry() -> Result<Registry, String> {
    match METRICS.get_or_init(initialize) {
        Ok(metrics) => Ok(metrics.registry.clone()),
        Err(error) => Err(error.clone()),
    }
}

fn initialize() -> Result<Metrics, String> {
    let registry = Registry::new();
    let exporter = opentelemetry_prometheus::exporter()
        .with_registry(registry.clone())
        .build()
        .map_err(|error| format!("failed to initialize Prometheus metrics: {error}"))?;
    let provider = SdkMeterProvider::builder().with_reader(exporter).build();
    global::set_meter_provider(provider.clone());
    Ok(Metrics {
        registry,
        _provider: provider,
    })
}

/// Encodes the current cumulative metric values in Prometheus text format.
pub(crate) fn encode(registry: &Registry) -> Result<Vec<u8>, String> {
    let mut body = Vec::new();
    TextEncoder::new()
        .encode(&registry.gather(), &mut body)
        .map_err(|error| format!("failed to encode Prometheus metrics: {error}"))?;
    Ok(body)
}
