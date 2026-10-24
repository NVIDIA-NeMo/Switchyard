// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This command migrates old settings and logs without changing the originals.

use std::io::{BufRead, Write};
use std::path::Path;

use serde::Deserialize;
use serde_json::{Value, json};
use switchyard_costs::{Pricing, RecordedCost};
use switchyard_protocol::Usage;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OldRate {
    input_per_mtok: f64,
    output_per_mtok: f64,
    cached_input_per_mtok: Option<f64>,
}

#[derive(Deserialize)]
struct Record {
    ts: String,
    #[serde(default, rename = "tier")]
    _tier: String,
    model: String,
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    cached_tokens: Option<u64>,
    cache_creation_tokens: Option<u64>,
    cache_creation_1h_tokens: Option<u64>,
    // The nested option distinguishes an absent cost from an explicit unknown cost.
    #[serde(default, deserialize_with = "present_cost")]
    cost: Option<Option<RecordedCost>>,
}

// An explicit null remains unknown instead of being reconstructed from token rates.
fn present_cost<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Option<RecordedCost>>, D::Error> {
    Option::deserialize(deserializer).map(Some)
}

// Only records without a cost field receive reconstructed estimates.
fn migrate(
    settings: &Path,
    log: &Path,
    destination: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut config: toml::Table = std::fs::read_to_string(settings)?.parse()?;
    let rates = match config.remove("prices") {
        Some(old) => {
            if config.contains_key("pricing_file") {
                return Err("settings contain both prices and pricing_file".into());
            }
            let old: std::collections::BTreeMap<String, OldRate> = old.try_into()?;
            Value::Object(old.into_iter().map(|(model, rate)| (model, json!({
                "input":rate.input_per_mtok, "output":rate.output_per_mtok,
                "cache_read":rate.cached_input_per_mtok.unwrap_or(rate.input_per_mtok),
                "cache_write":rate.input_per_mtok,
            }))).collect())
        }
        None => return Err("migration requires the old settings with a [prices] table".into()),
    };
    let rates = serde_json::to_vec_pretty(&rates)?;
    let pricing = Pricing::parse(&rates)?;
    let destination = std::path::absolute(destination)?;
    config.insert(
        "pricing_file".into(),
        toml::Value::String(
            destination
                .join("prices.json")
                .to_string_lossy()
                .into_owned(),
        ),
    );
    config.insert(
        "routing_log".into(),
        toml::Value::String(
            destination
                .join("routing.jsonl")
                .to_string_lossy()
                .into_owned(),
        ),
    );
    // Creating a fresh directory prevents overwriting inputs or earlier migration results.
    std::fs::create_dir(&destination)?;
    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        let mut output =
            std::io::BufWriter::new(std::fs::File::create(destination.join("routing.jsonl"))?);
        let mut source = std::io::BufReader::new(std::fs::File::open(log)?);
        let mut line = String::new();
        loop {
            line.clear();
            if source.read_line(&mut line)? == 0 {
                break;
            }
            if !line.ends_with('\n') {
                return Err("unfinished log record; stop the writer before migrating".into());
            }
            // Typed parsing catches duplicate known fields before Value could overwrite them.
            let record: Record = serde_json::from_str(&line)?;
            chrono::DateTime::parse_from_rfc3339(&record.ts)?;
            if record.model.trim().is_empty() {
                return Err("log record has an empty model".into());
            }
            if let Some(Some(cost)) = record.cost
                && (!cost.usd.is_finite() || cost.usd < 0.0)
            {
                return Err("log record has an invalid cost".into());
            }
            let mut value: Value = serde_json::from_str(&line)?;
            if !value.is_object() {
                return Err("log record must be a JSON object".into());
            }
            if record.cost.is_none() {
                let read = record.cached_tokens.unwrap_or(0);
                let written = record.cache_creation_tokens.unwrap_or(0);
                let mut cache = Usage::cache_details(Some(read), Some(written));
                if let Some(tokens) = record.cache_creation_1h_tokens {
                    cache
                        .get_or_insert_with(Default::default)
                        .cache_creation_1h_input_tokens = Some(tokens);
                }
                let usage = Usage {
                    input_tokens: record
                        .prompt_tokens
                        .and_then(|input| input.checked_sub(read)?.checked_sub(written)),
                    output_tokens: record.completion_tokens,
                    cache,
                    ..Usage::default()
                };
                value["cost"] = serde_json::to_value(pricing.estimate(&record.model, &usage))?;
            }
            serde_json::to_writer(&mut output, &value)?;
            output.write_all(b"\n")?;
        }
        output.flush()?;
        std::fs::write(destination.join("prices.json"), rates)?;
        std::fs::write(
            destination.join("menubar.toml"),
            toml::to_string_pretty(&config)?,
        )?;
        Ok(())
    })();
    if result.is_err() {
        std::fs::remove_dir_all(&destination)?;
    }
    result
}

fn main() -> std::process::ExitCode {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 3 {
        eprintln!(
            "Usage: switchyard-costs OLD_MENUBAR_TOML ROUTING_JSONL NEW_DIRECTORY\nStop the server and menubar first. Outputs use the rates in the old settings."
        );
        return std::process::ExitCode::FAILURE;
    }
    match migrate(
        Path::new(&args[0]),
        Path::new(&args[1]),
        Path::new(&args[2]),
    ) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("switchyard-costs: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
