// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Group completed calls within a session without repricing them.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use switchyard_protocol::Usage;

use crate::{CostTotals, Pricing, RecordedCost, anthropic};

/// The cache portions use catalog rates. A provider may independently report the total charge.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
pub struct CallCost {
    pub cost: Option<RecordedCost>,
    pub cache_read_usd: Option<f64>,
    pub cache_write_usd: Option<f64>,
}

impl Pricing {
    pub fn call_cost(&self, model: &str, usage: &Usage) -> CallCost {
        let rate = self.0.get(model);
        CallCost {
            cost: self.cost(model, usage),
            cache_read_usd: rate
                .zip(usage.cached_input_tokens())
                .map(|(rate, tokens)| tokens as f64 * rate.cache_read / 1_000_000.0),
            cache_write_usd: rate
                .zip(usage.cache_creation_input_tokens())
                .and_then(|(rate, _)| {
                    anthropic::cache_write_cost(usage, rate.cache_write, rate.cache_write_1h)
                })
                .map(|amount| amount / 1_000_000.0),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GroupBy {
    Session,
    Model,
    Turn,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct CostKey {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub call_kind: String,
}

#[derive(Serialize)]
pub struct CostRow {
    #[serde(flatten)]
    pub key: CostKey,
    #[serde(flatten)]
    pub cost: CostTotals,
    pub cache_read_usd: Option<f64>,
    pub cache_write_usd: Option<f64>,
}

/// A report is scoped to one session. The first completed call for a turn assigns its display index.
pub struct CostReport {
    group_by: GroupBy,
    turns: BTreeMap<String, usize>,
    rows: BTreeMap<CostKey, CostRow>,
}

impl CostReport {
    pub fn new(group_by: GroupBy) -> Self {
        Self {
            group_by,
            turns: BTreeMap::new(),
            rows: BTreeMap::new(),
        }
    }

    pub fn record(&mut self, turn_id: Option<&str>, model: &str, call_kind: &str, call: CallCost) {
        let turn = turn_id.filter(|_| matches!(self.group_by, GroupBy::Turn));
        let next = self.turns.len();
        let key = CostKey {
            turn_index: turn.map(|id| *self.turns.entry(id.to_owned()).or_insert(next)),
            turn_id: turn.map(str::to_owned),
            model: (!matches!(self.group_by, GroupBy::Session)).then(|| model.to_owned()),
            call_kind: call_kind.to_owned(),
        };
        let row = self.rows.entry(key.clone()).or_insert_with(|| CostRow {
            key,
            cost: CostTotals::default(),
            cache_read_usd: Some(0.0),
            cache_write_usd: Some(0.0),
        });
        row.cost.add(call.cost);
        row.cache_read_usd = row
            .cache_read_usd
            .zip(call.cache_read_usd)
            .map(|(a, b)| a + b);
        row.cache_write_usd = row
            .cache_write_usd
            .zip(call.cache_write_usd)
            .map(|(a, b)| a + b);
    }

    pub fn into_rows(self) -> Vec<CostRow> {
        self.rows.into_values().collect()
    }
}
