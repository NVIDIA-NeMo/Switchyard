# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write configs for one SWE-bench test-time scaling run."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="https://inference-api.nvidia.com/v1")
    parser.add_argument("--api-key-env", default="NVIDIA_API_KEY")
    parser.add_argument("--rollouts", type=int, default=16)
    parser.add_argument("--refinement-count", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--votes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=16)
    return parser


def _choice(value: str, source: str = "reconstructed") -> dict[str, str]:
    return {"source": source, "value": value}


def _revision(repo: Path) -> str:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    return f"{head}-dirty" if dirty else head


def _manifest(
    revision: str,
    model: str,
    dataset_label: str,
    base_url: str,
    rollouts: int,
    refinement_count: int,
    group_size: int,
    votes: int,
    seed: int,
    concurrency: int,
) -> dict[str, Any]:
    fields = {
        "exact_prompts": _choice(f"prompt constants in source revision {revision}"),
        "summary_schema": _choice("one JSON object; no fixed JSON Schema"),
        "malformed_summary_policy": _choice(
            "accept a bare JSON object or one JSON code fence; retry once, then stop"
        ),
        "model_ids_and_api_revisions": _choice(f"{model} through {base_url}"),
        "role_inference_settings": _choice(
            "same model for every role; mini-swe-agent action defaults; summary temperature 0.0; vote temperature 0.4; provider seed omitted"
        ),
        "scaffold_revisions_and_protocols": _choice(
            "Harbor from uv.lock; mini-swe-agent 2.4.6; native tool-calling"
        ),
        "agent_limits": _choice("task timeout with multiplier 1.0"),
        "summary_serialization": _choice("original task plus patch and native trajectory JSON"),
        "pairing_order": _choice("in order"),
        "display_order": _choice("in order"),
        "tie_break": _choice("first candidate in the group"),
        "invalid_vote_policy": _choice("stop the run"),
        "model_retry_policy": _choice(
            "Harbor retries a failed rollout twice; summary and vote calls use up to three HTTP attempts; summary JSON uses two content attempts"
        ),
        "experiment_seeds": _choice(f"root seed {seed}; provider seed omitted"),
        "benchmark_revisions": _choice(dataset_label, "repository"),
        "terminal_bench_task_list": _choice("not applicable to SWE-bench Verified"),
        "summary_input_contents": _choice(
            "task, patch, and trajectory only; official verification runs after selection"
        ),
        "refinement_summary_order": _choice("tournament survivor order"),
        "observation_truncation": _choice("keep the first and last 100000 characters"),
        "unfinished_rollout_policy": _choice("keep the returned patch and trajectory"),
        "concurrency_and_rate_limits": _choice(
            f"Harbor and model concurrency {concurrency}; gateway retries bounded"
        ),
        "ablation_fixed_settings": _choice(
            f"N={rollouts}, K={refinement_count}, G={group_size}, V={votes}",
            "paper"
            if (rollouts, refinement_count, group_size, votes) == (16, 4, 2, 8)
            else "reconstructed",
        ),
    }
    return {
        "schema_version": 1,
        "replication_mode": "conceptual",
        "code_revision": revision,
        "model_id": model,
        "fields": fields,
    }


def write_configs(args: argparse.Namespace, repo: Path) -> tuple[Path, Path]:
    """Write the Harbor and Rust runner configs and return their paths."""
    dataset = args.dataset.resolve()
    task_dir = dataset / args.task
    instruction = task_dir / "instruction.md"
    dataset_manifest = dataset / "switchyard_dataset_manifest.json"
    if not instruction.is_file() or not dataset_manifest.is_file():
        raise FileNotFoundError(
            "dataset must contain the task and switchyard_dataset_manifest.json"
        )
    if (
        min(
            args.rollouts,
            args.refinement_count,
            args.group_size,
            args.votes,
            args.concurrency,
        )
        <= 0
    ):
        raise ValueError("counts and concurrency must be positive")
    if args.refinement_count > args.rollouts:
        raise ValueError("refinement count must not exceed rollout count")
    if args.group_size < 2:
        raise ValueError("group size must be at least two")
    output = args.output.resolve()
    config_dir = output / "config"
    config_dir.mkdir(parents=True, exist_ok=False)
    harbor_path = config_dir / "harbor.json"
    runner_path = config_dir / "runner.json"
    grade_root = output / "private-grades"
    dataset_data = json.loads(dataset_manifest.read_text())
    dataset_digest = hashlib.sha256(dataset_manifest.read_bytes()).hexdigest()
    dataset_label = (
        f"{dataset_data.get('source_dataset', dataset.name)}; manifest_sha256={dataset_digest}"
    )
    revision = _revision(repo)

    harbor = {
        "repo_root": str(repo),
        "dataset_root": str(dataset),
        "run_baseline": str(repo / "benchmark" / "run-baseline.sh"),
        "method_root": str(output / "harbor"),
        "private_grade_root": str(grade_root),
        "run_record": str(output / "method" / "run.json"),
        "task_id": args.task,
        "model_id": args.model,
        "harbor_model": f"openai/{args.model}",
        "agent_import_path": (
            "benchmark.test_time_scaling_harbor_agent:TestTimeScalingMiniSweAgent"
        ),
        "n_concurrent": args.concurrency,
        "max_retries": 2,
        "agent_timeout_multiplier": 1.0,
        "upstream_base_url": args.base_url,
        "upstream_api_key_env": args.api_key_env,
        "harbor_command": ["uv", "run", "--no-sync", "harbor"],
    }
    runner = {
        "task": {
            "id": args.task,
            "benchmark": "swebench-verified",
            "prompt": instruction.read_text(),
        },
        "scaling": {
            "rollout_count": args.rollouts,
            "refinement_count": args.refinement_count,
            "group_size": args.group_size,
            "votes_per_group": args.votes,
            "seed": args.seed,
            "pairing_order": "in_order",
            "display_order": "in_order",
            "tie_policy": "first_in_group",
            "invalid_vote_policy": {"mode": "abort"},
        },
        "manifest": _manifest(
            revision,
            args.model,
            dataset_label,
            args.base_url,
            args.rollouts,
            args.refinement_count,
            args.group_size,
            args.votes,
            args.seed,
            args.concurrency,
        ),
        "output_dir": str(output / "method"),
        "rollout_command": {
            "argv": [
                "uv",
                "run",
                "--no-sync",
                "python",
                "benchmark/test_time_scaling_rollouts.py",
                "--config",
                str(harbor_path),
            ]
        },
        "evaluation_command": {
            "argv": [
                "uv",
                "run",
                "--no-sync",
                "python",
                "benchmark/test_time_scaling_grades.py",
                "--config",
                str(harbor_path),
            ]
        },
        "model": {
            "base_url": args.base_url,
            "api_key_env": args.api_key_env,
            "max_concurrency": args.concurrency,
            "summary_max_tokens": 4096,
            "comparison_max_tokens": 4096,
            "max_summary_input_chars": 200000,
            "summary_content_attempts": 2,
            "http_attempts": 3,
            "request_timeout_seconds": 180,
            "summary_temperature": 0.0,
            "comparison_temperature": 0.4,
            "send_seed": False,
        },
    }
    harbor_path.write_text(json.dumps(harbor, indent=2))
    runner_path.write_text(json.dumps(runner, indent=2))
    return runner_path, harbor_path


def main() -> int:
    args = _parser().parse_args()
    repo = Path(__file__).resolve().parents[1]
    runner_path, harbor_path = write_configs(args, repo)
    print(f"wrote runner config: {runner_path}")
    print(f"wrote Harbor config: {harbor_path}")
    print(f"run: cargo run -p switchyard-test-time-scaling-runner -- {runner_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
