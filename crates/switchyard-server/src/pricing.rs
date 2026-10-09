// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module prices observed calls and keeps missing usage or rates unknown.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use switchyard_protocol::Usage;

use crate::{ServerError, ServerResult};

#[derive(Default)]
pub(crate) struct Pricing(HashMap<String, Rates>);

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

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Rates {
    input: f64,
    output: f64,
    cache_read: f64,
    cache_write: f64,
    cache_write_1h: Option<f64>,
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CostSource {
    ProviderReported,
    Estimated,
}

#[derive(Clone, Copy, Deserialize, Serialize)]
pub(crate) struct RecordedCost {
    pub usd: f64,
    pub source: CostSource,
}

impl Pricing {
    pub(crate) fn load(path: &Path) -> ServerResult<Self> {
        let bytes = std::fs::read(path).map_err(|error| {
            ServerError::new(format!(
                "failed to read pricing file {}: {error}",
                path.display()
            ))
        })?;
        let pricing: Self = serde_json::from_slice(&bytes).map_err(|error| {
            ServerError::new(format!("invalid pricing file {}: {error}", path.display()))
        })?;
        for (model, rate) in &pricing.0 {
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
                return Err(ServerError::new(format!(
                    "invalid pricing for model {model:?}: use a nonempty model ID and rates from 0 to 10000 USD per million tokens"
                )));
            }
        }
        Ok(pricing)
    }

    /// Provider charges take precedence because token rates may omit provider surcharges.
    pub(crate) fn cost(&self, model: &str, usage: &Usage) -> Option<RecordedCost> {
        if let Some(usd) = usage
            .provider_cost
            .filter(|value| value.is_finite() && *value >= 0.0)
        {
            return Some(RecordedCost {
                usd,
                source: CostSource::ProviderReported,
            });
        }
        let rate = self.0.get(model)?;
        let input = usage.input_tokens?;
        let output = usage.output_tokens?;
        let read = usage.cached_input_tokens().unwrap_or(0);
        let write = usage.cache_creation_input_tokens().unwrap_or(0);
        let long_write = usage
            .cache
            .as_ref()
            .and_then(|cache| cache.cache_creation_1h_input_tokens)
            .unwrap_or(0);
        let short_write = write.checked_sub(long_write)?;
        let long_rate = if long_write > 0 {
            rate.cache_write_1h?
        } else {
            0.0
        };
        let usd = (input as f64 * rate.input
            + output as f64 * rate.output
            + read as f64 * rate.cache_read
            + short_write as f64 * rate.cache_write
            + long_write as f64 * long_rate)
            / 1_000_000.0;
        Some(RecordedCost {
            usd,
            source: CostSource::Estimated,
        })
    }
}

/// The known subtotal remains visible when any call has unknown cost.
#[derive(Serialize)]
pub(crate) struct CostTotals {
    pub known_usd: f64,
    pub unknown_calls: u64,
    pub estimated_calls: u64,
    pub provider_reported_calls: u64,
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
    pub(crate) fn add(&mut self, cost: Option<RecordedCost>) {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_rate_tables() -> Result<(), Box<dyn std::error::Error>> {
        let valid = r#"{"input":1,"output":5,"cache_read":0.1,"cache_write":2}"#;
        for rates in [
            r#"{"input":-1,"output":5,"cache_read":0.1,"cache_write":2}"#.to_string(),
            r#"{"input":1,"output":5,"cache_read":0.1}"#.to_string(),
            r#"{"input":1,"output":5,"cache_read":0.1,"cache_write":2,"cache_wirte":2}"#
                .to_string(),
            r#"{"input":1e999,"output":5,"cache_read":0.1,"cache_write":2}"#.to_string(),
        ] {
            let file = tempfile::NamedTempFile::new()?;
            std::fs::write(file.path(), format!("{{\"model\":{rates}}}"))?;
            assert!(Pricing::load(file.path()).is_err(), "{rates}");
        }
        let file = tempfile::NamedTempFile::new()?;
        std::fs::write(
            file.path(),
            format!("{{\"model\":{valid},\"model\":{valid}}}"),
        )?;
        assert!(Pricing::load(file.path()).is_err());
        Ok(())
    }

    #[test]
    fn estimates_disjoint_buckets_and_prefers_reported_charges()
    -> Result<(), Box<dyn std::error::Error>> {
        let file = tempfile::NamedTempFile::new()?;
        std::fs::write(
            file.path(),
            r#"{"model":{"input":1,"output":5,"cache_read":0.1,"cache_write":2,"cache_write_1h":3},"free":{"input":0,"output":0,"cache_read":0,"cache_write":0}}"#,
        )?;
        let prices = Pricing::load(file.path())?;
        let mut usage = Usage {
            input_tokens: Some(50),
            output_tokens: Some(10),
            reasoning_tokens: Some(4),
            cache: Usage::cache_details(Some(20), Some(30)),
            ..Usage::default()
        };
        usage
            .cache
            .as_mut()
            .ok_or("missing cache detail")?
            .cache_creation_1h_input_tokens = Some(10);
        assert!(
            (prices.cost("model", &usage).ok_or("missing estimate")?.usd - 0.000172).abs() < 1e-12
        );
        assert_eq!(
            prices
                .cost(
                    "free",
                    &Usage {
                        cache: None,
                        ..usage.clone()
                    }
                )
                .ok_or("missing free price")?
                .usd,
            0.0
        );
        assert!(prices.cost("unknown", &usage).is_none());
        assert!(prices.cost("model", &Usage::default()).is_none());
        assert!(prices.cost("free", &usage).is_none());
        let mut invalid_duration = usage.clone();
        invalid_duration
            .cache
            .as_mut()
            .ok_or("missing cache detail")?
            .cache_creation_1h_input_tokens = Some(31);
        assert!(prices.cost("model", &invalid_duration).is_none());
        usage.provider_cost = Some(0.0);
        assert_eq!(
            prices
                .cost("unknown", &usage)
                .ok_or("missing reported cost")?
                .usd,
            0.0
        );
        usage.provider_cost = Some(0.25);
        assert_eq!(
            prices
                .cost("model", &usage)
                .ok_or("missing reported cost")?
                .usd,
            0.25
        );
        Ok(())
    }

    #[test]
    fn partial_and_overflowing_totals_remain_unknown() {
        let mut totals = CostTotals::default();
        totals.add(Some(RecordedCost {
            usd: 0.25,
            source: CostSource::Estimated,
        }));
        totals.add(None);
        assert_eq!(totals.known_usd, 0.25);
        assert_eq!(totals.unknown_calls, 1);
        assert_eq!(totals.total_usd, None);
        totals.add(Some(RecordedCost {
            usd: f64::MAX,
            source: CostSource::ProviderReported,
        }));
        totals.add(Some(RecordedCost {
            usd: f64::MAX,
            source: CostSource::ProviderReported,
        }));
        assert!(totals.known_usd.is_finite());
        assert_eq!(totals.unknown_calls, 2);
    }
}
