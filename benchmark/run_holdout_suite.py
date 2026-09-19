# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the complete VGR hold-out suite from one Windows- or Linux-hosted command."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from benchmark.vgr_holdout_artifacts import finalize_artifacts, path_digest
except ModuleNotFoundError:
    from vgr_holdout_artifacts import finalize_artifacts, path_digest

REPO_ROOT = Path(__file__).resolve().parents[1]
AUTOMATIONBENCH_PIN = "4a8e1061254004d9dac807054eed33fad7d1ff14"
APPWORLD_PIN = "a072b7a86e7c1d5b1d7175659d750ebb9b79f10a"
APPWORLD_EXPERIMENT = (
    Path("simplified_react_code_agent") / "switchyard_vgr" / "switchyard-vgr" / "dev_easy"
)
APPWORLD_TASK_IDS = (
    "4ec8de5_1",
    "4ec8de5_2",
    "4ec8de5_3",
    "d4e9306_1",
    "d4e9306_2",
    "d4e9306_3",
    "3ab5b8b_1",
    "3ab5b8b_2",
    "3ab5b8b_3",
    "df61dc5_1",
    "df61dc5_2",
    "df61dc5_3",
    "383cbac_1",
    "383cbac_2",
    "383cbac_3",
    "23cf851_1",
    "23cf851_2",
    "23cf851_3",
    "57c3486_1",
    "57c3486_2",
    "57c3486_3",
    "68ee2c9_1",
    "68ee2c9_2",
    "68ee2c9_3",
    "6bdbc26_1",
    "6bdbc26_2",
    "6bdbc26_3",
    "396c5a2_1",
    "396c5a2_2",
    "396c5a2_3",
)


def _default_checkout(name: str) -> Path:
    """Find a sibling harness next to either the checkout or its worktree root."""
    for parent in tuple(REPO_ROOT.parents)[:3]:
        candidate = parent / name
        if candidate.is_dir():
            return candidate
    return REPO_ROOT.parent / name


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run full TB 2.1, AutomationBench simple, and AppWorld easy through VGR. "
            "The same Python and Docker path is used natively on Windows and Linux."
        )
    )
    parser.add_argument(
        "--server-config",
        type=Path,
        default=REPO_ROOT / "benchmark/server-configs/vgr-holdout-qwen36-opus5.toml",
    )
    parser.add_argument(
        "--automationbench-root",
        type=Path,
        default=Path(
            os.environ.get(
                "AUTOMATIONBENCH_ROOT",
                _default_checkout("AutomationBench"),
            )
        ),
    )
    parser.add_argument(
        "--appworld-root",
        type=Path,
        default=Path(
            os.environ.get(
                "APPWORLD_ROOT",
                _default_checkout("appworld-repo"),
            )
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "benchmark/holdout_runs",
    )
    parser.add_argument("--tb-agent", default="hermes")
    parser.add_argument("--tb-concurrency", type=int, default=1)
    parser.add_argument("--automation-concurrency", type=int, default=4)
    parser.add_argument("--appworld-processes", type=int, default=1)
    parser.add_argument("--server-port", type=int, default=4000)
    parser.add_argument("--session-proxy-port", type=int, default=4001)
    parser.add_argument(
        "--counterfactual-labels",
        type=Path,
        help=(
            "Frozen schema-version-1 request labels from paired controls. "
            "Without them the summary records that the FPR/FNR gate is blocked."
        ),
    )
    parser.add_argument(
        "--summarize-run",
        type=Path,
        help="Rebuild artifact-only summaries for an existing run without starting services.",
    )
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    env: dict[str, str] | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running: {' '.join(command)}")
    with log_path.open("wb") as log:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(
            f"command failed with exit status {result.returncode}; see {log_path}"
        )


def _git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _require_pin(name: str, path: Path, expected: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{name} checkout not found: {path}")
    actual = _git_head(path)
    if actual != expected:
        raise RuntimeError(f"{name} must be pinned to {expected}; found {actual}")


def _wait_for_health(url: str, process: subprocess.Popen[bytes] | None = None) -> None:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"service exited before becoming healthy: {url}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    raise TimeoutError(f"service did not become healthy within 180 seconds: {url}")


def _capture_json(url: str, path: Path) -> None:
    """Capture one bounded JSON endpoint response as a run artifact."""
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = response.read(16 * 1024 * 1024 + 1)
    if len(payload) > 16 * 1024 * 1024:
        raise ValueError(f"JSON response exceeds 16 MiB: {url}")
    value = json.loads(payload)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _capture_text(url: str, path: Path) -> None:
    """Capture one bounded UTF-8 endpoint response as a run artifact."""
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = response.read(16 * 1024 * 1024 + 1)
    if len(payload) > 16 * 1024 * 1024:
        raise ValueError(f"text response exceeds 16 MiB: {url}")
    path.write_text(payload.decode("utf-8"), encoding="utf-8")


def _server_command(
    config: Path,
    port: int,
    network: str,
    container: str,
    artifacts_dir: Path,
) -> list[str]:
    config_mount = (
        f"type=bind,src={config.resolve()},dst=/etc/switchyard/config.toml,readonly"
    )
    artifacts_mount = (
        f"type=bind,src={artifacts_dir.resolve()},dst=/artifacts"
    )
    return [
        "docker",
        "run",
        "--rm",
        "--detach",
        "--name",
        container,
        "--network",
        network,
        "--network-alias",
        "switchyard",
        "--add-host",
        "host.docker.internal:host-gateway",
        "--publish",
        f"{port}:4000",
        "--env",
        "NVIDIA_API_KEY",
        "--mount",
        config_mount,
        "--mount",
        artifacts_mount,
        "switchyard-baseline:local",
        "--config",
        "/etc/switchyard/config.toml",
        "--host",
        "0.0.0.0",
        "--port",
        "4000",
        "--routing-log-file",
        "/artifacts/routing_requests.jsonl",
    ]


@contextlib.contextmanager
def _switchyard_server(
    config: Path,
    port: int,
    network: str,
    log_path: Path,
    artifacts_dir: Path,
) -> Iterator[None]:
    container = f"switchyard-holdout-{os.getpid()}"
    _run(
        ["docker", "network", "create", network],
        cwd=REPO_ROOT,
        log_path=log_path.with_name("docker-network.log"),
    )
    command = _server_command(config, port, network, container, artifacts_dir)
    try:
        _run(command, cwd=REPO_ROOT, log_path=log_path)
        _wait_for_health(f"http://127.0.0.1:{port}/health")
        yield
    finally:
        with log_path.open("ab") as log:
            subprocess.run(
                ["docker", "logs", container],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        subprocess.run(
            ["docker", "rm", "--force", container],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        subprocess.run(
            ["docker", "network", "rm", network],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


@contextlib.contextmanager
def _session_proxy(
    namespace: str,
    listen_port: int,
    upstream_port: int,
    log_path: Path,
) -> Iterator[None]:
    command = [
        sys.executable,
        str(REPO_ROOT / "benchmark/holdout_session_proxy.py"),
        "--listen-port",
        str(listen_port),
        "--upstream",
        f"http://127.0.0.1:{upstream_port}",
        "--namespace",
        namespace,
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_for_health(f"http://127.0.0.1:{listen_port}/health", process)
            yield
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


@contextlib.contextmanager
def _temporary_text(path: Path, content: str) -> Iterator[None]:
    original = path.read_bytes() if path.exists() else None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    try:
        yield
    finally:
        if original is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(original)


def _appworld_model_module(base_url: str) -> str:
    return f'''MODEL_INFOS = [
    {{
        "model_name": "switchyard-vgr",
        "client_name": "openai",
        "model_id": "switchyard/vgr",
        "model_kwargs": {{
            "api_type": "chat_completions",
            "temperature": 0,
            "seed": 100,
            "api_key_env_name": "OPENAI_API_KEY",
            "base_url": "{base_url}",
            "max_completion_tokens": 8000,
            "tool_parser_name": None,
            "parallel_tool_calls": True,
            "cost_per_token": {{
                "input_cache_miss": 0.0,
                "input_cache_hit": 0.0,
                "input_cache_write": 0.0,
                "output": 0.0,
            }},
        }},
        "function_calling": True,
        "tool_choice": "auto",
        "function_calling_demos": False,
        "part_of": ["vn"],
        "provider": "vllm",
    }}
]
'''


def _tb_command(args: argparse.Namespace, run_dir: Path) -> list[str]:
    job_name = "vgr-holdout-tb21"
    return [
        "uv",
        "run",
        "--no-sync",
        "harbor",
        "run",
        "--agent",
        args.tb_agent,
        "--model",
        "openai/switchyard/vgr",
        "--jobs-dir",
        str(run_dir / "tb21/jobs"),
        "--job-name",
        job_name,
        "-n",
        str(args.tb_concurrency),
        "--max-retries",
        "0",
        "--agent-timeout-multiplier",
        "2.0",
        "--path",
        str(REPO_ROOT / "benchmark/datasets/terminal-bench-2-1-closed-book"),
        "--artifact",
        "/etc/proxy-ca/strip.jsonl",
        "--ve",
        "HTTP_PROXY=${SWITCHYARD_VERIFIER_HTTP_PROXY}",
        "--ve",
        "HTTPS_PROXY=${SWITCHYARD_VERIFIER_HTTP_PROXY}",
        "--ve",
        "http_proxy=${SWITCHYARD_VERIFIER_HTTP_PROXY}",
        "--ve",
        "https_proxy=${SWITCHYARD_VERIFIER_HTTP_PROXY}",
        "--ve",
        "NO_PROXY=localhost,127.0.0.1,proxy",
        "--ve",
        "no_proxy=localhost,127.0.0.1,proxy",
        "--environment-build-timeout-multiplier",
        "3.0",
    ]


def _tb_environment(network: str) -> dict[str, str]:
    environment = os.environ.copy()
    verifier_token = os.urandom(24).hex()
    environment.update(
        {
            "ALLOWED_HOSTS": "switchyard",
            "CLOSED_BOOK_MODE": "1",
            "OPENAI_API_KEY": "switchyard-local",
            "OPENAI_BASE_URL": "http://switchyard:4000/v1",
            "SWITCHYARD_BASE_URL": "http://switchyard:4000",
            "SWITCHYARD_DOCKER_NETWORK": network,
            "SWITCHYARD_VERIFIER_HTTP_PROXY": (
                f"http://verifier:{verifier_token}@proxy:3129"
            ),
            "SWITCHYARD_VERIFIER_PROXY_TOKEN": verifier_token,
        }
    )
    return environment


def _automation_command(args: argparse.Namespace, output: Path) -> list[str]:
    return [
        "uv",
        "run",
        "--no-sync",
        "auto-bench",
        "--model",
        "switchyard/vgr",
        "--base-url",
        f"http://127.0.0.1:{args.session_proxy_port}/v1",
        "--api-key-var",
        "OPENAI_API_KEY",
        "--domains",
        "simple",
        "--toolset",
        "api",
        "--num-examples",
        "-1",
        "--max-steps",
        "50",
        "--max-concurrent",
        str(args.automation_concurrency),
        "--export-json",
        str(output),
    ]


def _appworld_command(args: argparse.Namespace) -> list[str]:
    return [
        "uv",
        "run",
        "--with-editable",
        ".",
        "--with-editable",
        "experiments[simplified]",
        "appworld",
        "run",
        "auto",
        "--agent-name",
        "simplified_react_code_agent",
        "--model-name",
        "switchyard-vgr",
        "--dataset-name",
        "dev_easy",
        "--with-evaluation",
        "--clear-first",
        "--num-processes",
        str(args.appworld_processes),
        "--with-setup",
        "--root",
        str(args.appworld_root),
    ]


def _ensure_harbor_patch(run_dir: Path) -> None:
    purelib = subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            "-c",
            "import sysconfig; print(sysconfig.get_paths()['purelib'])",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    patch_file = REPO_ROOT / "benchmark/patches/harbor-agent-patches.diff"
    reverse_check = subprocess.run(
        ["git", "apply", "--reverse", "--check", "-p1", str(patch_file)],
        cwd=purelib,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if reverse_check.returncode == 0:
        return
    _run(
        ["git", "apply", "-p1", str(patch_file)],
        cwd=Path(purelib),
        log_path=run_dir / "setup-harbor-patch.log",
    )


def _prepare(args: argparse.Namespace, run_dir: Path) -> None:
    _run(
        ["uv", "sync", "--locked", "--group", "dev"],
        cwd=REPO_ROOT,
        log_path=run_dir / "setup-switchyard.log",
    )
    _ensure_harbor_patch(run_dir)
    _run(
        ["uv", "sync", "--locked"],
        cwd=args.automationbench_root,
        log_path=run_dir / "setup-automationbench.log",
    )
    dataset_manifest = (
        REPO_ROOT
        / "benchmark/datasets/terminal-bench-2-1-closed-book/switchyard_dataset_manifest.json"
    )
    if not dataset_manifest.is_file():
        _run(
            [
                "uv",
                "run",
                "python",
                "benchmark/prepare_harbor_dataset.py",
                "--source-dataset",
                "terminal-bench/terminal-bench-2-1",
                "--output-dir",
                "benchmark/datasets/terminal-bench-2-1-closed-book",
                "--overwrite",
            ],
            cwd=REPO_ROOT,
            log_path=run_dir / "setup-tb21-dataset.log",
        )


def _write_manifest(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _snapshot_inputs(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    inputs_dir = run_dir / "inputs"
    inputs_dir.mkdir()
    config_snapshot = inputs_dir / "server-config.toml"
    shutil.copyfile(args.server_config, config_snapshot)
    return {
        "server_config": {
            "path": config_snapshot.relative_to(run_dir).as_posix(),
            "digest": path_digest(config_snapshot),
        },
        "cohorts": {
            "tb21": {"task_count": 89},
            "automationbench_simple": {"task_count": 200},
            "appworld_easy": {
                "task_count": len(APPWORLD_TASK_IDS),
                "task_ids": list(APPWORLD_TASK_IDS),
            },
        },
        "commands": {
            "tb21": _tb_command(args, run_dir),
            "automationbench_simple": _automation_command(
                args, run_dir / "automationbench-simple.json"
            ),
            "appworld_easy": _appworld_command(args),
        },
    }


def _snapshot_appworld_output(args: argparse.Namespace, run_dir: Path) -> Path:
    source = args.appworld_root / "experiments/outputs" / APPWORLD_EXPERIMENT
    if not source.is_dir():
        raise FileNotFoundError(f"AppWorld output not found: {source}")
    destination = run_dir / "appworld-output"
    shutil.copytree(source, destination)
    return destination


def run_suite(args: argparse.Namespace) -> Path:
    for value, label in (
        (args.tb_concurrency, "--tb-concurrency"),
        (args.automation_concurrency, "--automation-concurrency"),
        (args.appworld_processes, "--appworld-processes"),
    ):
        if value < 1:
            raise ValueError(f"{label} must be positive")
    if not args.server_config.is_file():
        raise FileNotFoundError(f"server config not found: {args.server_config}")
    if not os.environ.get("NVIDIA_API_KEY"):
        raise RuntimeError("NVIDIA_API_KEY must be set")

    _require_pin("AutomationBench", args.automationbench_root, AUTOMATIONBENCH_PIN)
    _require_pin("AppWorld", args.appworld_root, APPWORLD_PIN)

    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = args.output_dir.resolve() / f"vgr-holdout-{timestamp}"
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "run_manifest.json"
    inputs = _snapshot_inputs(args, run_dir)
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "started_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "suites": {},
        "pins": {
            "switchyard": _git_head(REPO_ROOT),
            "automationbench": AUTOMATIONBENCH_PIN,
            "appworld": APPWORLD_PIN,
        },
        "inputs": inputs,
        "artifacts": {
            "routing_records": "routing_requests.jsonl",
            "routing_stats": "routing_stats_final.json",
            "server_metrics": "server_metrics_final.prom",
            "machine_readable_summary": "machine_readable_summary.json",
            "per_task_routing": "per_task_routing.jsonl",
            "artifact_index": "artifact_index.json",
        },
    }
    _write_manifest(manifest_path, manifest)

    try:
        if not args.skip_setup:
            _prepare(args, run_dir)

        _run(
            ["docker", "build", "--tag", "switchyard-baseline:local", "."],
            cwd=REPO_ROOT,
            log_path=run_dir / "switchyard-image.log",
        )
        common_env = os.environ.copy()
        common_env["OPENAI_API_KEY"] = "switchyard-local"
        network = f"switchyard-holdout-{os.getpid()}-{timestamp.replace('_', '-')}"
        with _switchyard_server(
            args.server_config,
            args.server_port,
            network,
            run_dir / "server.log",
            run_dir,
        ):
            _run(
                _tb_command(args, run_dir),
                cwd=REPO_ROOT,
                log_path=run_dir / "tb21.log",
                env=_tb_environment(network),
            )
            manifest["suites"]["tb21"] = {
                "status": "completed",
                "output": "tb21/jobs",
            }
            _write_manifest(manifest_path, manifest)

            with _session_proxy(
                "automationbench",
                args.session_proxy_port,
                args.server_port,
                run_dir / "automationbench-proxy.log",
            ):
                automation_output = run_dir / "automationbench-simple.json"
                _run(
                    _automation_command(args, automation_output),
                    cwd=args.automationbench_root,
                    log_path=run_dir / "automationbench-simple.log",
                    env=common_env,
                )
                manifest["suites"]["automationbench_simple"] = {
                    "status": "completed",
                    "output": automation_output.relative_to(run_dir).as_posix(),
                }
                _write_manifest(manifest_path, manifest)

            dataset_file = args.appworld_root / "data/datasets/dev_easy.txt"
            model_file = (
                args.appworld_root
                / "experiments/configs/_generator/models/switchyard_vgr.py"
            )
            tasks = "\n".join(APPWORLD_TASK_IDS) + "\n"
            model = _appworld_model_module(
                f"http://127.0.0.1:{args.session_proxy_port}/v1"
            )
            with (
                _temporary_text(dataset_file, tasks),
                _temporary_text(model_file, model),
                _session_proxy(
                    "appworld",
                    args.session_proxy_port,
                    args.server_port,
                    run_dir / "appworld-proxy.log",
                ),
            ):
                _run(
                    _appworld_command(args),
                    cwd=args.appworld_root,
                    log_path=run_dir / "appworld-easy.log",
                    env=common_env,
                )
                manifest["suites"]["appworld_easy"] = {
                    "status": "completed",
                    "output": _snapshot_appworld_output(args, run_dir)
                    .relative_to(run_dir)
                    .as_posix(),
                }
                _write_manifest(manifest_path, manifest)
            _capture_json(
                f"http://127.0.0.1:{args.server_port}/v1/stats",
                run_dir / "routing_stats_final.json",
            )
            _capture_text(
                f"http://127.0.0.1:{args.server_port}/metrics",
                run_dir / "server_metrics_final.prom",
            )
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest["finished_at"] = datetime.now(UTC).isoformat()
        try:
            manifest["artifact_summary"] = finalize_artifacts(
                run_dir, args.counterfactual_labels
            )
        except (OSError, ValueError, RuntimeError) as artifact_error:
            manifest["artifact_error"] = f"{type(artifact_error).__name__}: {artifact_error}"
        _write_manifest(manifest_path, manifest)
        raise

    manifest["artifact_summary"] = finalize_artifacts(
        run_dir,
        args.counterfactual_labels,
        require_complete=True,
    )
    manifest["status"] = "completed"
    manifest["finished_at"] = datetime.now(UTC).isoformat()
    _write_manifest(manifest_path, manifest)
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(argv if argv is not None else sys.argv[1:])
    args = _parser().parse_args(raw_args)
    if args.summarize_run:
        manifest_path = args.summarize_run.resolve() / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("run manifest must be a JSON object")
        manifest["artifact_summary"] = finalize_artifacts(
            args.summarize_run,
            args.counterfactual_labels,
        )
        _write_manifest(manifest_path, manifest)
        print(f"Hold-out artifacts summarized: {args.summarize_run.resolve()}")
        return 0
    if args.dry_run:
        preview_dir = args.output_dir / "preview"
        preview = {
            "tb21": _tb_command(args, preview_dir),
            "automationbench_simple": _automation_command(
                args, args.output_dir / "automationbench-simple.json"
            ),
            "appworld_easy": _appworld_command(args),
            "artifact_contract": {
                "routing_records": "routing_requests.jsonl",
                "routing_stats": "routing_stats_final.json",
                "server_metrics": "server_metrics_final.prom",
                "per_task_routing": "per_task_routing.jsonl",
                "summary": "machine_readable_summary.json",
                "counterfactual_gate": (
                    "computed from --counterfactual-labels"
                    if args.counterfactual_labels
                    else "blocked until frozen labels are supplied"
                ),
            },
        }
        print(json.dumps(preview, indent=2))
        return 0

    run_dir = run_suite(args)
    print(f"Hold-out suite completed: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
