# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one iteration of fresh Harbor rollouts and return method-only records."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _read_batch() -> dict[str, Any]:
    value = json.load(sys.stdin)
    if not isinstance(value, dict):
        raise ValueError("stdin must contain one JSON object")
    return value


def _single_iteration(requests: list[dict[str, Any]]) -> int:
    iterations: set[int] = set()
    indexes: list[int] = []
    for request in requests:
        iteration = request.get("iteration")
        rollout_index = request.get("rollout_index")
        if not isinstance(iteration, int) or not isinstance(rollout_index, int):
            raise ValueError("iteration and rollout_index must be integers")
        iterations.add(iteration)
        indexes.append(rollout_index)
    if len(iterations) != 1:
        raise ValueError("one Harbor batch must contain one iteration")
    iteration = iterations.pop()
    if iteration not in (0, 1):
        raise ValueError("iteration must be 0 or 1")
    expected = list(range(len(requests)))
    actual = sorted(indexes)
    if actual != expected:
        raise ValueError("rollout indexes must be unique and contiguous")
    return iteration


def _safe_task_id(task_id: str) -> str:
    if not task_id or task_id in (".", "..") or Path(task_id).name != task_id:
        raise ValueError("task id must be one path component")
    return task_id


def _prepare_dataset(
    source_root: Path,
    method_root: Path,
    task: dict[str, Any],
    requests: list[dict[str, Any]],
    iteration: int,
) -> Path:
    task_id = _safe_task_id(str(task.get("id", "")))
    source_task = source_root / task_id
    if not source_task.is_dir():
        raise FileNotFoundError(f"task not found: {source_task}")
    dataset_root = method_root / f"dataset-iteration-{iteration}"
    if dataset_root.exists():
        raise FileExistsError(f"iteration dataset already exists: {dataset_root}")
    dataset_root.mkdir(parents=True)
    shutil.copy2(source_root / "switchyard_dataset_manifest.json", dataset_root)
    copied_task = dataset_root / task_id
    shutil.copytree(source_task, copied_task)

    instruction_path = copied_task / "instruction.md"
    original = instruction_path.read_text()
    expected_prompt = str(task.get("prompt", ""))
    if original.strip() != expected_prompt.strip():
        raise ValueError("configured task prompt does not match instruction.md")
    refinement_prompts = {request.get("refinement_prompt") for request in requests}
    if iteration == 0 and refinement_prompts != {None}:
        raise ValueError("iteration zero must not contain refinement context")
    if iteration == 1:
        if len(refinement_prompts) != 1 or None in refinement_prompts:
            raise ValueError("every refined rollout must use the same context")
        refinement = refinement_prompts.pop()
        instruction_path.write_text(f"{refinement}\n\nORIGINAL TASK\n{original}")
    return dataset_root


def _harbor_command(
    config: dict[str, Any],
    dataset_root: Path,
    task_id: str,
    attempt_count: int,
    iteration: int,
) -> list[str]:
    run_root = Path(config["method_root"]).resolve() / f"harbor-iteration-{iteration}"
    return [
        "bash",
        str(Path(config["run_baseline"]).resolve()),
        "--output-dir",
        str(run_root),
        "--harbor-path",
        str(dataset_root),
        "--model",
        str(config["model_id"]),
        "--agent",
        "mini-swe-agent",
        "--agent-import-path",
        str(config["agent_import_path"]),
        "--harbor-model",
        str(config["harbor_model"]),
        "--n-concurrent",
        str(config["n_concurrent"]),
        "--max-retries",
        str(config["max_retries"]),
        "--agent-timeout-multiplier",
        str(config["agent_timeout_multiplier"]),
        "--task-id",
        task_id,
        "--upstream-base-url",
        str(config["upstream_base_url"]),
        "--upstream-api-key-env",
        str(config["upstream_api_key_env"]),
        "--harbor-extra",
        "-k",
        "--harbor-extra",
        str(attempt_count),
        "--harbor-extra",
        "--disable-verification",
        "--foreground",
    ]


def _run_harbor(
    config: dict[str, Any],
    dataset_root: Path,
    task_id: str,
    attempt_count: int,
    iteration: int,
) -> Path:
    repo_root = Path(config["repo_root"]).resolve()
    run_root = Path(config["method_root"]).resolve() / f"harbor-iteration-{iteration}"
    if run_root.exists():
        raise FileExistsError(f"Harbor output already exists: {run_root}")
    command = _harbor_command(config, dataset_root, task_id, attempt_count, iteration)
    subprocess.run(command, cwd=repo_root, check=True, stdout=sys.stderr, stderr=sys.stderr)
    runs = [path for path in run_root.iterdir() if path.is_dir()]
    if len(runs) != 1:
        raise RuntimeError(f"expected one Harbor run under {run_root}; found {len(runs)}")
    return runs[0]


def _trial_directories(run_dir: Path) -> list[Path]:
    job_roots = [path for path in (run_dir / "jobs").iterdir() if path.is_dir()]
    if len(job_roots) != 1:
        raise RuntimeError(f"expected one Harbor job under {run_dir}")
    return sorted(
        path for path in job_roots[0].iterdir() if path.is_dir() and (path / "agent").is_dir()
    )


def _read_trajectory(trial_dir: Path) -> Any:
    native = trial_dir / "agent" / "mini-swe-agent.trajectory.json"
    if native.is_file():
        return json.loads(native.read_text())
    converted = trial_dir / "agent" / "trajectory.json"
    if converted.is_file():
        return json.loads(converted.read_text())
    log = trial_dir / "agent" / "mini-swe-agent.txt"
    return {"agent_log": log.read_text() if log.is_file() else ""}


def _public_rollout(
    task_id: str,
    model_id: str,
    request: dict[str, Any],
    trial_dir: Path,
) -> dict[str, Any]:
    patch_path = trial_dir / "artifacts" / "patch.diff"
    patch = patch_path.read_text() if patch_path.is_file() else ""
    digest = hashlib.sha256(patch.encode()).hexdigest()
    iteration = int(request["iteration"])
    rollout_index = int(request["rollout_index"])
    rollout_id = f"{task_id}-iteration-{iteration}-rollout-{rollout_index}"
    return {
        "id": rollout_id,
        "iteration": iteration,
        "rollout_index": rollout_index,
        "model_id": model_id,
        "environment_id": trial_dir.name,
        "output_digest": f"sha256:{digest}",
        "output": {
            "patch": patch,
            "trajectory": _read_trajectory(trial_dir),
        },
    }


def _write_private_grade_map(
    grade_root: Path,
    iteration: int,
    pairs: list[tuple[dict[str, Any], Path]],
) -> None:
    grade_root.mkdir(parents=True, exist_ok=True)
    path = grade_root / f"iteration-{iteration}.json"
    if path.exists():
        raise FileExistsError(f"grade map already exists: {path}")
    patch_root = grade_root / "patches"
    patch_root.mkdir(exist_ok=True)
    records = []
    for rollout, _trial_dir in pairs:
        patch_path = patch_root / f"{rollout['id']}.diff"
        patch_path.write_text(str(rollout["output"]["patch"]))
        records.append({"rollout_id": rollout["id"], "patch_file": str(patch_path.resolve())})
    path.write_text(json.dumps(records, indent=2))


def main() -> int:
    args = _parser().parse_args()
    config = _read_object(args.config)
    expected_harbor_model = f"openai/{config['model_id']}"
    if config.get("harbor_model") != expected_harbor_model:
        raise ValueError(f"harbor_model must be {expected_harbor_model}")
    batch = _read_batch()
    task = batch.get("task")
    requests = batch.get("requests")
    if not isinstance(task, dict) or not isinstance(requests, list) or not requests:
        raise ValueError("batch must contain a task and at least one request")
    if any(not isinstance(request, dict) for request in requests):
        raise ValueError("every rollout request must be an object")
    iteration = _single_iteration(requests)
    method_root = Path(config["method_root"]).resolve()
    method_root.mkdir(parents=True, exist_ok=True)
    source_root = Path(config["dataset_root"]).resolve()
    dataset_root = _prepare_dataset(source_root, method_root, task, requests, iteration)
    task_id = _safe_task_id(str(task["id"]))
    run_dir = _run_harbor(
        config,
        dataset_root,
        task_id,
        len(requests),
        iteration,
    )
    trials = _trial_directories(run_dir)
    if len(trials) != len(requests):
        raise RuntimeError(f"Harbor returned {len(trials)} trials; expected {len(requests)}")
    ordered_requests = sorted(requests, key=lambda request: request["rollout_index"])
    pairs = [
        (
            _public_rollout(task_id, str(config["model_id"]), request, trial),
            trial,
        )
        for request, trial in zip(ordered_requests, trials, strict=True)
    ]
    _write_private_grade_map(
        Path(config["private_grade_root"]).resolve(),
        iteration,
        pairs,
    )
    json.dump([rollout for rollout, _trial in pairs], sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
