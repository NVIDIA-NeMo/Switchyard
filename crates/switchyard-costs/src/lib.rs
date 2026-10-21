// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module prices observed calls and keeps missing usage or rates unknown.

mod anthropic;

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use switchyard_protocol::Usage;

/// Pricing maps served model IDs to token rates in USD per million tokens.
#[derive(Clone, Debug, Default)]
pub struct Pricing(HashMap<String, Rates>);

impl<'de> Deserialize<'de> for Pricing {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // This visitor rejects duplicate IDs before a map can overwrite a rate table.
        struct UniqueModels;
        impl<'de> serde::de::Visitor<'de> for UniqueModels {
            type Value = Pricing;
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a map of unique model IDs to token rates")
            }
            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                mut map: M,
            ) -> Result<Pricing, M::Error> {
                let mut rates = HashMap::new();
                while let Some((model, rate)) = map.next_entry::<String, Rates>()? {
                    if rates.insert(model.clone(), rate).is_some() {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate pricing model {model:?}"
                        )));
                    }
                }
                Ok(Pricing(rates))
            }
        }
        deserializer.deserialize_map(UniqueModels)
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Rates {
    input: f64,
    output: f64,
    cache_read: f64,
    cache_write: f64,
    cache_write_1h: Option<f64>,
}

/// This value identifies how a saved monetary amount was obtained.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CostSource {
    /// The provider supplied this charge.
    ProviderReported,
    /// Configured token rates supplied this estimate.
    Estimated,
}

/// This record stores a charge or estimate for one call.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize, Serialize)]
pub struct RecordedCost {
    /// The amount is in USD.
    pub usd: f64,
    /// The source distinguishes charges from estimates.
    pub source: CostSource,
}

impl Pricing {
    /// This method loads the rates shipped with this version of Switchyard.
    pub fn bundled() -> Result<Self, String> {
        Self::parse_toml(include_str!("../prices.toml"))
    }

    /// This method loads TOML from `.toml` files and JSON from other filenames.
    /// The extension check ignores ASCII case.
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|error| format!("read pricing file {}: {error}", path.display()))?;
        let pricing = if path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("toml"))
        {
            std::str::from_utf8(&bytes)
                .map_err(|error| error.to_string())
                .and_then(Self::parse_toml)
        } else {
            Self::parse(&bytes)
        };
        pricing.map_err(|error| format!("invalid pricing file {}: {error}", path.display()))
    }

    /// This function reads a pricing file's JSON and checks every model ID and rate.
    pub fn parse(json: &[u8]) -> Result<Self, String> {
        let pricing: Self = serde_json::from_slice(json).map_err(|error| error.to_string())?;
        pricing.validate()
    }

    /// TOML rates use the same typed fields and validation as JSON rates.
    fn parse_toml(text: &str) -> Result<Self, String> {
        let pricing: Self = toml::from_str(text).map_err(|error| error.to_string())?;
        pricing.validate()
    }

    /// Both parsers use this check so model IDs and rate limits agree across formats.
    fn validate(self) -> Result<Self, String> {
        for (model, rate) in &self.0 {
            if model.trim().is_empty()
                || model.trim() != model
                || [
                    Some(rate.input),
                    Some(rate.output),
                    Some(rate.cache_read),
                    Some(rate.cache_write),
                    rate.cache_write_1h,
                ]
                .into_iter()
                .flatten()
                .any(|value| !value.is_finite() || !(0.0..=10000.0).contains(&value))
            {
                return Err(format!(
                    "invalid pricing for model {model:?}: use a nonempty model ID and rates from 0 to 10000 USD per million tokens"
                ));
            }
        }
        Ok(self)
    }

    /// Provider charges take precedence because token rates may omit provider surcharges.
    pub fn cost(&self, model: &str, usage: &Usage) -> Option<RecordedCost> {
        if let Some(usd) = usage
            .provider_cost
            .filter(|value| value.is_finite() && *value >= 0.0)
        {
            return Some(RecordedCost {
                usd,
                source: CostSource::ProviderReported,
            });
        }
        self.estimate(model, usage)
    }

    /// This method uses token rates alone so baseline estimates exclude provider charges.
    pub fn estimate(&self, model: &str, usage: &Usage) -> Option<RecordedCost> {
        let rate = self.0.get(model)?;
        let input = usage.input_tokens?;
        let output = usage.output_tokens?;
        let read = usage.cached_input_tokens().unwrap_or(0);
        let usd = (input as f64 * rate.input
            + output as f64 * rate.output
            + read as f64 * rate.cache_read
            + anthropic::cache_write_cost(usage, rate.cache_write, rate.cache_write_1h)?)
            / 1_000_000.0;
        Some(RecordedCost {
            usd,
            source: CostSource::Estimated,
        })
    }
}

/// The known subtotal remains visible when any call has unknown cost.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct CostTotals {
    /// This subtotal includes every known cost.
    pub known_usd: f64,
    /// This count includes calls whose cost is unknown.
    pub unknown_calls: u64,
    /// This count includes calls priced from configured rates.
    pub estimated_calls: u64,
    /// This count includes calls with a provider charge.
    pub provider_reported_calls: u64,
    /// The total is absent when any call has unknown cost.
    pub total_usd: Option<f64>,
}

impl Default for CostTotals {
    fn default() -> Self {
        Self {
            known_usd: 0.0,
            unknown_calls: 0,
            estimated_calls: 0,
            provider_reported_calls: 0,
            total_usd: Some(0.0),
        }
    }
}

impl CostTotals {
    /// This method adds one call and keeps the total unknown if any cost is unknown.
    pub fn add(&mut self, cost: Option<RecordedCost>) {
        match cost.filter(|cost| {
            cost.usd.is_finite() && cost.usd >= 0.0 && (self.known_usd + cost.usd).is_finite()
        }) {
            Some(cost) => {
                self.known_usd += cost.usd;
                match cost.source {
                    CostSource::Estimated => self.estimated_calls += 1,
                    CostSource::ProviderReported => self.provider_reported_calls += 1,
                }
            }
            None => self.unknown_calls += 1,
        }
        self.total_usd = (self.unknown_calls == 0).then_some(self.known_usd);
    }

    /// This method combines disjoint groups of calls without repricing them.
    pub fn merge(&mut self, other: &Self) {
        if (self.known_usd + other.known_usd).is_finite() {
            self.known_usd += other.known_usd;
        } else {
            self.unknown_calls += 1;
        }
        self.unknown_calls += other.unknown_calls;
        self.estimated_calls += other.estimated_calls;
        self.provider_reported_calls += other.provider_reported_calls;
        self.total_usd = (self.unknown_calls == 0).then_some(self.known_usd);
    }
}
