// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Normalize inline strategies and reusable profiles into the shared algorithm schema.

use toml::{Table, Value};

fn strategy(mut table: Table) -> Result<Table, String> {
    if let Some(kind) = table.remove("strategy")
        && table.insert("type".into(), kind).is_some()
    {
        return Err("configure strategy or type, not both".into());
    }
    if table.get("type").and_then(Value::as_str) == Some("profile_selector") {
        return Err("use profile and selector to configure profile selection".into());
    }
    Ok(table)
}

pub(crate) fn normalize(mut document: Table) -> Result<Table, String> {
    if document.get("schema_version").and_then(Value::as_integer) != Some(2) {
        return Ok(document);
    }
    let profiles = match document.remove("profiles") {
        None => Table::new(),
        Some(Value::Table(profiles)) => profiles,
        _ => return Err("profiles must be a table".into()),
    };
    let profiles = profiles
        .into_iter()
        .map(|(name, value)| {
            let Value::Table(table) = value else {
                return Err(format!("profile {name} must be a table"));
            };
            let table = strategy(table)?;
            let _: crate::AlgorithmSpec = Value::Table(table.clone())
                .try_into()
                .map_err(|error| format!("profile {name}: {error}"))?;
            Ok((name, Value::Table(table)))
        })
        .collect::<Result<Table, String>>()?;
    let Some(Value::Table(routes)) = document.get_mut("routes") else {
        return Err("routes must be a table".into());
    };
    for (name, value) in routes {
        let Value::Table(route) = value else {
            return Err(format!("route {name} must be a table"));
        };
        let Some(profile) = route.remove("profile") else {
            *route = strategy(std::mem::take(route))?;
            continue;
        };
        let profile = profile
            .as_str()
            .ok_or_else(|| format!("route {name}: profile must be a name"))?;
        for key in route.keys() {
            if ![
                "id",
                "context_window",
                "tool_calling",
                "reasoning",
                "vision",
                "infra",
                "selector",
            ]
            .contains(&key.as_str())
            {
                return Err(format!(
                    "route {name}: {key} must be configured in profile {profile}"
                ));
            }
        }
        let definition = profiles
            .get(profile)
            .ok_or_else(|| format!("route {name}: unknown profile {profile}"))?;
        if let Some(selector) = route.remove("selector") {
            let config: crate::ProfileSelectorConfig = selector
                .clone()
                .try_into()
                .map_err(|error| format!("route {name} selector: {error}"))?;
            let mut selected = Table::new();
            for profile in
                std::iter::once(profile).chain(config.choices.values().map(String::as_str))
            {
                let definition = profiles
                    .get(profile)
                    .ok_or_else(|| format!("route {name}: unknown profile {profile}"))?;
                selected.insert(profile.into(), definition.clone());
            }
            route.insert("type".into(), "profile_selector".into());
            route.insert("default_profile".into(), profile.into());
            route.insert("selector".into(), selector);
            route.insert("profiles".into(), Value::Table(selected));
        } else {
            route.extend(definition.as_table().unwrap().clone());
        }
    }
    Ok(document)
}
