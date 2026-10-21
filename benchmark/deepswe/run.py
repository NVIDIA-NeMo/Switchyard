# /// script
# requires-python = ">=3.14,<3.15"
# dependencies = ["datacurve-pier==0.3.1", "harbor==0.24.0"]
# ///
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run three DeepSWE v1.1 tasks with memory checks."""

import argparse
import asyncio
import json
import os
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from pier.agents.installed.codex import Codex
from pier.job import Job
from pier.models.job.config import JobConfig

REVISION = "0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea"
MODEL = "openai/openai/gpt-6-luna"
TASKS = {
    "scc": "scc-bounded-memory-spilling",
    "wazero": "wazero-multi-module-snapshots",
    "optique": "optique-conditional-option-dependencies",
}
REPO = Path(__file__).resolve().parents[2]
HUB_CONFIG = """model_provider = "hub"
model_catalog_json = "/tmp/hub-model-catalog.json"
web_search = "disabled"
[model_providers.hub]
name = "NVIDIA Inference Hub"
base_url = "https://inference-api.nvidia.com/v1"
env_key = "OPENAI_API_KEY"
wire_api = "responses"
"""


class HubCodex(Codex):
    async def run(self, instruction, environment, context):
        result = await environment.exec(
            "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; codex debug models --bundled"
        )
        if result.return_code:
            raise RuntimeError("Cannot read the Codex model catalog")
        model = next(m for m in json.loads(result.stdout)["models"] if m["slug"] == "gpt-6-luna")
        model["slug"] = MODEL
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "catalog.json"
            path.write_text(json.dumps({"models": [model]}))
            await environment.upload_file(path, "/tmp/hub-model-catalog.json")
        await super().run(instruction, environment, context)


def summarize(output, agents, tasks):
    rows = []
    for agent in agents:
        for path in sorted((output / agent).glob("*/result.json")):
            result = json.loads(path.read_text())
            phases = [json.loads(p.read_text()) for p in path.parent.glob("memory-*.json")]
            complete = bool(result.get("finished_at")) and not result.get("exception_info")
            row = {
                "agent": agent, "task": result["task_name"],
                "reward": ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward"),
                "memory_pass": complete and result.get("verifier_result") is not None
                and len(phases) == 2 and all(p["passed"] for p in phases),
                "peak_gib": max((p["peak_bytes"] for p in phases), default=0) / 2**30,
            }
            rows.append(row)
            print(f"{agent} {row['task']}: memory {'PASS' if row['memory_pass'] else 'FAIL'}, "
                  f"peak estimate {row['peak_gib']:.3f} GiB, reward {row['reward']}")
    passed = len(rows) == len(agents) * len(tasks) and all(r["memory_pass"] for r in rows)
    (output / "summary.json").write_text(json.dumps({"passed": passed, "trials": rows}, indent=2) + "\n")
    return passed


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=("oracle", "codex", "all"), default="oracle")
    parser.add_argument("--task", choices=TASKS, action="append", help="Default: all three tasks")
    parser.add_argument("--output", type=Path, default=REPO / "benchmark/tb_runs/deepswe" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--dry-run", action="store_true", help="Check inputs and write configs only")
    args = parser.parse_args()
    agents = ["oracle", "codex"] if args.agent == "all" else [args.agent]
    if not args.dry_run and "codex" in agents and not os.environ.get("INFERENCE_HUB_KEY"):
        parser.error("Set INFERENCE_HUB_KEY")
    dataset = REPO / "benchmark/datasets/deep-swe"
    if not dataset.exists():
        subprocess.run(["git", "clone", "--no-checkout", "https://github.com/datacurve-ai/deep-swe", str(dataset)], check=True)
        subprocess.run(["git", "-C", str(dataset), "checkout", "--detach", REVISION], check=True)
    revision = subprocess.check_output(["git", "-C", str(dataset), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(dataset), "status", "--porcelain"], text=True)
    if revision != REVISION or dirty:
        parser.error(f"Use a clean DeepSWE checkout at {REVISION}")
    output = args.output.resolve()
    output.mkdir(parents=True)
    tasks = [TASKS[name] for name in dict.fromkeys(args.task or TASKS)]
    print(f"Run directory: {output}", flush=True)
    # Use the Hub key instead of host Codex credentials.
    for key in ("CODEX_AUTH_JSON_PATH", "CODEX_FORCE_AUTH_JSON"):
        os.environ.pop(key, None)
    try:
        for agent in agents:
            agent_config = {"name": "oracle"} if agent == "oracle" else {
                "import_path": "run:HubCodex", "model_name": MODEL,
                "kwargs": {
                    "version": "0.163.0", "command_model_name": MODEL,
                    "reasoning_effort": "high", "config_toml": HUB_CONFIG,
                },
                "env": {"OPENAI_API_KEY": "${INFERENCE_HUB_KEY}", "OPENAI_BASE_URL": "https://inference-api.nvidia.com/v1"},
            }
            config = JobConfig.model_validate({
                "job_name": agent, "jobs_dir": str(output), "n_concurrent_trials": 1,
                "n_attempts": 1, "retry": {"max_retries": 0}, "quiet": True,
                "environment": {"import_path": "memory:MemoryDocker", "cpu_enforcement_policy": "limit", "memory_enforcement_policy": "limit"},
                "agents": [agent_config],
                "datasets": [{"path": str(dataset / "tasks"), "task_names": tasks}],
            })
            (output / f"{agent}-config.json").write_text(config.model_dump_json(indent=2) + "\n")
            if not args.dry_run:
                await (await Job.create(config)).run()
    finally:
        passed = True if args.dry_run else summarize(output, agents, tasks)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
