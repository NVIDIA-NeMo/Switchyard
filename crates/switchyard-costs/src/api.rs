// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cost API views. Observed spend reports completed calls.
//! Upfront-task, next-call, and remaining-task predictions can become sibling modules when implemented.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use switchyard_protocol::Usage;

use crate::{CostSource, CostTotals, Pricing, RecordedCost};

/// The cache portions use catalog rates. A provider may independently report the total charge.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
pub struct CallCost {
    pub cost: Option<RecordedCost>,
    pub cache_read_usd: Option<f64>,
    pub cache_write_usd: Option<f64>,
}

impl Pricing {
    /// Prices the existing normalized usage for one served model, including available cache costs.
    pub fn observed_spend(&self, model: &str, usage: &Usage) -> CallCost {
        let mut call = self.estimate_call(model, usage);
        if let Some(usd) = usage
            .provider_cost
            .filter(|usd| usd.is_finite() && *usd >= 0.0)
        {
            call.cost = Some(RecordedCost {
                usd,
                source: CostSource::ProviderReported,
            });
        }
        call
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

/// Configured routing identities that contributed to a cost row.
#[derive(Clone, Debug, Default, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(default)]
pub struct CostRoute {
    pub route_id: String,
    pub algorithm: String,
    pub target: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
}

#[derive(Serialize)]
pub struct CostRow {
    #[serde(flatten)]
    pub key: CostKey,
    #[serde(skip_serializing_if = "BTreeSet::is_empty")]
    pub routes: BTreeSet<CostRoute>,
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

    pub fn record(
        &mut self,
        turn_id: Option<&str>,
        model: &str,
        call_kind: &str,
        call: CallCost,
        route: &CostRoute,
    ) {
        let turn = turn_id.filter(|_| matches!(self.group_by, GroupBy::Turn));
        let next = self.turns.len();
        let key = CostKey {
            turn_index: turn.map(|id| *self.turns.entry(id.to_owned()).or_insert(next)),
            turn_id: turn.map(str::to_owned),
            model: (!matches!(self.group_by, GroupBy::Session)).then(|| model.to_owned()),
            call_kind: call_kind.to_owned(),
        };
        let row = self.rows.entry(key).or_insert_with_key(|key| CostRow {
            key: key.clone(),
            routes: BTreeSet::new(),
            cost: CostTotals::default(),
            cache_read_usd: Some(0.0),
            cache_write_usd: Some(0.0),
        });
        if !route.route_id.is_empty() || !route.target.is_empty() || !route.algorithm.is_empty() {
            row.routes.insert(route.clone());
        }
        row.cost.add(call.cost);
        row.cache_read_usd = add_known(row.cache_read_usd, call.cache_read_usd);
        row.cache_write_usd = add_known(row.cache_write_usd, call.cache_write_usd);
    }

    pub fn into_rows(self) -> Vec<CostRow> {
        self.rows.into_values().collect()
    }
}

/// A missing component in any call leaves its group total unknown.
fn add_known(total: Option<f64>, amount: Option<f64>) -> Option<f64> {
    Some(total? + amount?)
}
