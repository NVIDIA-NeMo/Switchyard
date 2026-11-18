# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a fresh mini-swe-agent task through a local Switchyard proxy.

Observe calls with LightGBM B and export forecasts through OTLP.
The reference trajectory supplies only initial messages and harness settings.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

PIN = "f21165163390fa76c0d448daa384a45b7580331c"


def run(reference, repo, research_root, base_url, output, steps, checkpoint):
    if steps <= 0:
        raise ValueError("max-steps must be positive")
    if output.exists():
        raise FileExistsError("Choose a fresh run directory")
    u = urlsplit(base_url)
    if u.scheme != "http" or u.hostname not in ("localhost", "127.0.0.1"):
        raise ValueError("Expected a local Switchyard HTTP endpoint")
    if (
        subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        != PIN
    ):
        raise ValueError("Pinned replay-gap revision required")
    if version("mini-swe-agent") != "2.4.5":
        raise ValueError("mini-swe-agent 2.4.5 required")
    raw = json.loads(reference.read_text())
    task = raw["replay_gap"]["instance_id"]
    subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"], check=True, capture_output=True
    )
    with urlopen(base_url.rstrip("/") + "/models", timeout=10) as response:
        routes = json.load(response)
    config = deepcopy(raw["info"]["config"])
    recorded_model = config["model"]["model_name"]
    route = recorded_model.removeprefix("openai/")
    if not any(r["id"] == route for r in routes["data"]):
        raise ValueError("Configure a Switchyard route matching the recorded model ID")
    os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
    os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    sys.path.insert(0, str(repo / "src"))
    from minisweagent.environments.docker import DockerEnvironment
    from replay_gap.branching import BranchableAgent
    from replay_gap.pool import build_model

    model = config["model"]
    opts = model.get("model_kwargs", {})
    spec = {
        "model_name": recorded_model,
        "api_base": base_url,
        "api_key": "local-switchyard",
        "temperature": opts.get("temperature", 0.0),
        "max_tokens": opts.get("max_tokens", 20000),
        "set_cache_control": model.get("set_cache_control"),
        "model_kwargs": {
            k: opts[k] for k in ("top_p", "reasoning_effort", "max_completion_tokens") if k in opts
        },
    }
    spec.update(
        {
            k: model[k]
            for k in ("format_error_template", "observation_template", "multimodal_regex")
            if k in model
        }
    )
    client = build_model(spec, seed=opts.get("seed", 42))
    first = next(i for i, m in enumerate(raw["messages"]) if m["role"] == "assistant")
    initial = [
        {
            k: deepcopy(m[k])
            for k in ("role", "content", "tool_calls", "tool_call_id", "name")
            if k in m
        }
        for m in raw["messages"][:first]
    ]
    output.mkdir(parents=True)
    config["agent"].update(step_limit=steps, output_path=output / "live.traj.json")
    config["environment"]["forward_env"] = []
    # These frozen SWE-bench images are x86-64, including on ARM demo laptops.
    args = config["environment"].get("run_args", ["--rm"])
    if "--platform" not in args and not any(a.startswith("--platform=") for a in args):
        config["environment"]["run_args"] = [*args, "--platform", "linux/amd64"]
    status = {
        "task_id": task,
        "model": recorded_model,
        "mini_version": "2.4.5",
        "replay_gap_revision": PIN,
        "switchyard_base_url": base_url,
        "forecasting_enabled": checkpoint is not None,
        "state": "starting_container",
        "started_unix": time.time(),
        "completed_model_calls": 0,
    }

    def save():
        tmp = output / "status.pending"
        tmp.write_text(json.dumps(status, indent=2) + "\n")
        tmp.replace(output / "status.json")

    save()
    env = None
    agent = None
    observer = None
    try:
        if checkpoint is not None:
            from backend import ResearchBackend
            from telemetry import ForecastObserver

            observer = ForecastObserver(
                ResearchBackend(research_root, checkpoint), config, task, output
            )

        class ObservedAgent(BranchableAgent):
            def step(self):
                # Let the harness handle its limit without forecasting a nonexistent request.
                limited = (
                    0 < self.config.step_limit <= self.n_calls
                    or 0 < self.config.cost_limit <= self.cost
                    or 0
                    < self.config.wall_time_limit_seconds
                    <= int(time.time() - self._start_time)
                )
                if observer is None or limited:
                    return super().step()
                with observer.call(self, self.n_calls + 1) as headers:
                    kwargs = self.model.config.model_kwargs
                    previous = kwargs.get("extra_headers")
                    kwargs["extra_headers"] = {**(previous or {}), **headers}
                    try:
                        return super().step()
                    finally:
                        if previous is None:
                            kwargs.pop("extra_headers", None)
                        else:
                            kwargs["extra_headers"] = previous

        env = DockerEnvironment(**config["environment"])
        agent = ObservedAgent(client, env, **config["agent"])
        agent.replay_metadata = {
            "instance_id": task,
            "arm": "switchyard_forecast_demo",
            "forecasting_enabled": observer is not None,
        }

        def before_step(agent, step):
            status.update(state="running", completed_model_calls=step - 1, next_model_call=step)
            save()
            print(f"{task}: starting model call {step} through Switchyard", flush=True)

        outcome = agent.run_from_prefix(initial, before_step=before_step)
        status.update(
            state="terminated",
            exit_status=outcome.get("exit_status"),
            completed_model_calls=sum(m["role"] == "assistant" for m in agent.messages),
        )
    except KeyboardInterrupt:
        status.update(state="cancelled", exit_status="CancelledByUser")
        raise SystemExit(130) from None
    except Exception as error:
        # Avoid echoing provider exception bodies or credentials.
        status.update(state="failed", error_type=type(error).__name__)
        print("Run failed: " + type(error).__name__, flush=True)
        raise SystemExit(1) from None
    finally:
        if agent is not None:
            agent.save(config["agent"]["output_path"])
        if env is not None:
            env.cleanup()
        if observer is not None:
            observer.close()
        status["ended_unix"] = time.time()
        save()
    print(json.dumps(status), flush=True)


if __name__ == "__main__":

    def cancel(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, cancel)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-trace", type=Path, required=True)
    p.add_argument("--replay-repo", type=Path, required=True)
    p.add_argument("--research-root", type=Path, required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:4110/v1")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument(
        "--forecast-checkpoint",
        type=Path,
        required=True,
        help="B bundle; export predictions and IR to configured OTLP endpoint",
    )
    a = p.parse_args()
    run(
        a.reference_trace,
        a.replay_repo,
        a.research_root,
        a.base_url,
        a.output,
        a.max_steps,
        a.forecast_checkpoint,
    )
