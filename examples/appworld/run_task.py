# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run an unmodified AppWorld full-code agent against a Switchyard gateway."""

import argparse
import json
import os
import time
from uuid import uuid4

import _jsonnet
import appworld_agents.code.simplified.full_code_agent  # noqa: F401
from appworld.common.path_store import path_store
from appworld.evaluator import evaluate_task
from appworld_agents.code.simplified.agent import Agent
from appworld_agents.configs._generator.run import make_experiment_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_id")
    parser.add_argument("--gateway", default="http://127.0.0.1:4000/v1")
    parser.add_argument("--route", default="appworld-agent")
    args = parser.parse_args()
    os.environ.setdefault("MODEL_SERVER_URL", "")
    os.environ.setdefault("OPENAI_API_KEY", "local-gateway")

    template = make_experiment_config(
        model_name="gpt-4.1-2025-04-14",
        agent_name="simplified_full_code_agent",
        dataset_name="test_normal",
        save=False,
    )
    rendered = _jsonnet.evaluate_snippet(
        "full_code_agent.jsonnet",
        template["config"],
        ext_vars={
            "APPWORLD_EXPERIMENT_PROMPTS_PATH": path_store.experiment_prompts,
            "APPWORLD_EXPERIMENT_CONFIGS_PATH": path_store.experiment_configs,
            "APPWORLD_EXPERIMENT_CODE_PATH": path_store.experiment_code,
        },
    )
    config = json.loads(rendered)["config"]["agent"]
    for model in (config["model_config"], config["api_predictor_config"]["model_config"]):
        model.update(name=args.route, base_url=args.gateway, use_cache=False)
    config["appworld_config"]["load_ground_truth"] = False
    config["log_lm_calls"] = False
    experiment = f"switchyard_{uuid4().hex}"
    agent = Agent.from_dict(config)
    start = time.monotonic()
    agent.solve_tasks(task_ids=[args.task_id], experiment_name=experiment)
    seconds = time.monotonic() - start
    score = evaluate_task(args.task_id, experiment_name=experiment)
    print(json.dumps({
        "task": args.task_id,
        "experiment": experiment,
        "passed": score.success,
        "checks_passed": score.pass_count,
        "checks_total": score.num_tests,
        "seconds": round(seconds, 2),
    }))


if __name__ == "__main__":
    main()
