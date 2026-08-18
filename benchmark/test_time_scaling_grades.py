# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run official Harbor verification after test-time selection finishes."""

import argparse
import json
import os
import secrets
import subprocess
import sys
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _records(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text())
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{path} must contain a list of objects")
    return value


def _string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must contain one or more strings")
    return value


def _grade_records(grade_root: Path) -> list[dict[str, Any]]:
    mappings = sorted(grade_root.glob("iteration-*.json"))
    if len(mappings) != 2:
        raise RuntimeError(f"expected two patch maps; found {len(mappings)}")
    return [record for mapping in mappings for record in _records(mapping)]


def _passed(trial_dir: Path) -> bool:
    reward_path = trial_dir / "verifier" / "reward.txt"
    if not reward_path.is_file():
        raise FileNotFoundError(f"missing official reward: {reward_path}")
    return float(reward_path.read_text().strip()) > 0.0


def _run_trial(
    harbor_command: list[str],
    task_dir: Path,
    trials_dir: Path,
    rollout_id: str,
    patch_file: Path,
    verifier_proxy: str,
    repo_root: Path,
    env: dict[str, str],
) -> bool:
    command = [
        *harbor_command,
        "trial",
        "start",
        "--path",
        str(task_dir),
        "--trial-name",
        rollout_id,
        "--trials-dir",
        str(trials_dir),
        "--agent-import-path",
        "benchmark.test_time_scaling_patch_agent:TestTimeScalingPatchAgent",
        "--agent-kwarg",
        f"patch_file={patch_file}",
        "--ve",
        f"HTTP_PROXY={verifier_proxy}",
        "--ve",
        f"HTTPS_PROXY={verifier_proxy}",
        "--ve",
        f"http_proxy={verifier_proxy}",
        "--ve",
        f"https_proxy={verifier_proxy}",
        "--ve",
        "NO_PROXY=localhost,127.0.0.1,proxy",
        "--ve",
        "no_proxy=localhost,127.0.0.1,proxy",
    ]
    subprocess.run(
        command, cwd=repo_root, env=env, check=True, stdout=sys.stderr, stderr=sys.stderr
    )
    return _passed(trials_dir / rollout_id)


def main() -> int:
    args = _parser().parse_args()
    config = _object(args.config)
    run_record = Path(config["run_record"]).resolve()
    if not run_record.is_file():
        raise FileNotFoundError(f"method record must be saved before grading: {run_record}")
    grade_root = Path(config["private_grade_root"]).resolve()
    repo_root = Path(config["repo_root"]).resolve()
    task_dir = Path(config["dataset_root"]).resolve() / str(config["task_id"])
    trials_dir = grade_root / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    network = f"switchyard-grade-{secrets.token_hex(6)}"
    token = secrets.token_urlsafe(32)
    verifier_proxy = f"http://verifier:{token}@proxy:3129"
    env = dict(os.environ)
    env.update(
        {
            "ALLOWED_HOSTS": "",
            "CLOSED_BOOK_MODE": "1",
            "SWITCHYARD_DOCKER_NETWORK": network,
            "SWITCHYARD_VERIFIER_PROXY_TOKEN": token,
        }
    )
    subprocess.run(["docker", "network", "create", network], check=True, timeout=30)
    try:
        outcomes = [
            {
                "rollout_id": str(record["rollout_id"]),
                "passed": _run_trial(
                    _string_list(config.get("harbor_command"), "harbor_command"),
                    task_dir,
                    trials_dir,
                    str(record["rollout_id"]),
                    Path(record["patch_file"]),
                    verifier_proxy,
                    repo_root,
                    env,
                ),
            }
            for record in _grade_records(grade_root)
        ]
    finally:
        subprocess.run(["docker", "network", "rm", network], check=False, timeout=30)
    json.dump(outcomes, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
