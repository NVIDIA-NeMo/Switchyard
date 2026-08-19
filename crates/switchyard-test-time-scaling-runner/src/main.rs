// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod backend;
mod command;
mod config;
mod model_client;

use std::path::Path;

use backend::LiveBackend;
use command::call_json;
use config::RunConfig;
use serde::Serialize;
use switchyard_test_time_scaling::{
    ExperimentMetrics, RolloutEvaluation, ScalingController, TaskEvaluation, encode_run,
    evaluate_run, experiment_metrics,
};

#[derive(Serialize)]
struct EvaluationRecord {
    task: TaskEvaluation,
    metrics: ExperimentMetrics,
}

#[tokio::main]
async fn main() -> Result<(), String> {
    let mut arguments = std::env::args_os();
    let program = arguments.next().unwrap_or_default();
    let Some(config_path) = arguments.next() else {
        return Err(format!(
            "usage: {} CONFIG.json",
            Path::new(&program).display()
        ));
    };
    if arguments.next().is_some() {
        return Err("expected exactly one config path".to_string());
    }

    let bytes = tokio::fs::read(&config_path)
        .await
        .map_err(|error| error.to_string())?;
    let config: RunConfig = serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
    config.validate()?;
    config
        .manifest
        .validate()
        .map_err(|error| error.to_string())?;
    if tokio::fs::try_exists(&config.output_dir)
        .await
        .map_err(|error| error.to_string())?
    {
        return Err(format!(
            "output directory already exists: {}",
            config.output_dir.display()
        ));
    }
    let backend = LiveBackend::new(
        config.manifest.model_id.clone(),
        config.rollout_command.clone(),
        config.model.clone(),
        config.output_dir.join("model_calls.jsonl"),
    )?;
    tokio::fs::create_dir_all(&config.output_dir)
        .await
        .map_err(|error| error.to_string())?;
    let controller = ScalingController::new(backend, config.scaling, config.manifest)
        .map_err(|error| error.to_string())?;
    let run = controller
        .run(config.task)
        .await
        .map_err(|error| error.to_string())?;
    let run_path = config.output_dir.join("run.json");
    let run_bytes = encode_run(&run).map_err(|error| error.to_string())?;
    tokio::fs::write(&run_path, run_bytes)
        .await
        .map_err(|error| error.to_string())?;
    println!("saved method record: {}", run_path.display());

    if let Some(command) = &config.evaluation_command {
        let outcomes: Vec<RolloutEvaluation> = call_json(command, &()).await?;
        let task = evaluate_run(&run, outcomes).map_err(|error| error.to_string())?;
        let metrics =
            experiment_metrics(std::slice::from_ref(&task)).map_err(|error| error.to_string())?;
        let record = EvaluationRecord { task, metrics };
        let evaluation_path = config.output_dir.join("evaluation.json");
        let bytes = serde_json::to_vec_pretty(&record).map_err(|error| error.to_string())?;
        tokio::fs::write(&evaluation_path, bytes)
            .await
            .map_err(|error| error.to_string())?;
        println!(
            "saved post-selection evaluation: {}",
            evaluation_path.display()
        );
    }
    Ok(())
}
