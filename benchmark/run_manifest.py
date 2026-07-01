# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-run reproducibility manifest for Switchyard Harbor baselines.

The manifest is written before Harbor starts and finalized after Harbor exits.
It intentionally stays stdlib-only so ``benchmark/run-baseline.sh`` can call it
with ``python3`` without forcing an editable Switchyard build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def _iso_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str] | None:
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.returncode, result.stdout.strip()


def _repo_root(path: Path) -> Path:
    cwd = path if path.is_dir() else path.parent
    result = _run(["git", "rev-parse", "--show-toplevel"], cwd=cwd)
    if result is None or result[0] != 0 or not result[1]:
        return cwd
    return Path(result[1])


def _git_value(args: list[str], cwd: Path) -> str | None:
    result = _run(["git", *args], cwd=cwd)
    if result is None or result[0] != 0:
        return None
    return result[1] or None


def _git_dirty(cwd: Path) -> bool | None:
    result = _run(["git", "status", "--porcelain"], cwd=cwd)
    if result is None or result[0] != 0:
        return None
    return bool(result[1])


def _tree_kind(dirty: bool | None) -> str:
    if dirty is None:
        return "not-git"
    return "git-dirty" if dirty else "git-clean"


def _harbor_version(harbor_command: list[str] | None = None) -> str | None:
    command = harbor_command or ["harbor"]
    result = _run([*command, "--version"])
    if result is None or result[0] != 0:
        return None
    return result[1].splitlines()[0] if result[1] else None


def path_digest(path: Path) -> str:
    """Return a deterministic sha256 digest for a file or directory."""
    try:
        resolved = path.resolve()
        hasher = hashlib.sha256()
        if resolved.is_file():
            hasher.update(resolved.name.encode())
            with resolved.open("rb") as fh:
                hasher.update(hashlib.file_digest(fh, "sha256").digest())
            return f"sha256:{hasher.hexdigest()}"
        if resolved.is_dir():
            for item in sorted(p for p in resolved.rglob("*") if p.is_file()):
                rel = item.relative_to(resolved).as_posix()
                with item.open("rb") as fh:
                    file_hash = hashlib.file_digest(fh, "sha256").hexdigest()
                hasher.update(f"{rel}\n{file_hash}\n".encode())
            return f"sha256:{hasher.hexdigest()}"
    except OSError:
        return "sha256:unknown"
    return "sha256:missing"


def dataset_fingerprint(
    dataset: str | None,
    task_list_file: Path | None = None,
    harbor_path: Path | None = None,
) -> str:
    """Fingerprint the Harbor dataset selector plus optional task-list file/path."""
    payload: dict[str, Any] = {"dataset": dataset}
    if harbor_path is not None:
        payload["harbor_path"] = str(harbor_path.resolve())
        payload["harbor_path_digest"] = path_digest(harbor_path)
    if task_list_file is not None:
        payload["task_list_file"] = str(task_list_file.resolve())
        payload["task_list_digest"] = path_digest(task_list_file)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def snapshot_routing_profiles(source: Path | None, run_dir: Path) -> Path | None:
    """Copy the route bundle used for the run into the run directory."""
    if source is None:
        return None
    if not source.is_file():
        raise FileNotFoundError(source)

    dest = run_dir.resolve() / "routing_profiles" / source.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
    return dest


def snapshot_dataset_manifest(source: Path | None, run_dir: Path) -> Path | None:
    """Copy a generated local dataset manifest into the run directory."""
    if source is None:
        return None
    manifest = source / "switchyard_dataset_manifest.json"
    if not manifest.is_file():
        return None

    dest = run_dir.resolve() / "dataset" / manifest.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(manifest, dest)
    return dest


def _json_arg(value: str, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _opt(value: str | None) -> str | None:
    return value if value else None


def _opt_int(value: int | None) -> int | None:
    return value if value is not None and value >= 0 else None


def _bool_arg(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def write_manifest(
    output_path: Path,
    harbor_command: list[str] | None = None,
    **fields: Any,
) -> None:
    repo = _repo_root(Path(__file__).resolve())
    dirty = _git_dirty(repo)
    run_meta: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": _iso_timestamp(),
        "hostname": socket.gethostname(),
        "git_branch": _git_value(["branch", "--show-current"], cwd=repo),
        "git_sha": _git_value(["rev-parse", "HEAD"], cwd=repo),
        "git_dirty": dirty,
        "git_tree_kind": _tree_kind(dirty),
        "harbor_command": harbor_command or ["harbor"],
        "harbor_version": _harbor_version(harbor_command),
        "launcher_argv": fields.pop("launcher_argv", []),
    }
    manifest = {"run": run_meta, **fields}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")


def _find_harbor_result(job_dir: Path | None) -> Path | None:
    if job_dir is None:
        return None
    direct = job_dir / "result.json"
    if direct.is_file():
        return direct
    candidates = sorted(job_dir.glob("*/result.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _find_artifact(job_dir: Path | None, artifact_name: str) -> Path | None:
    if job_dir is None or not artifact_name:
        return None
    direct = job_dir / artifact_name
    if direct.is_file():
        return direct
    candidates = sorted(job_dir.rglob(artifact_name), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _copy_if_present(source: Path | None, dest: Path | None) -> str:
    if source is None or dest is None or not source.is_file():
        return "missing"
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != dest.resolve():
            shutil.copyfile(source, dest)
    except OSError:
        return "missing"
    return "present"


def _finalize_routing_trace(manifest: dict[str, Any]) -> None:
    trace = manifest.get("routing_trace")
    if not isinstance(trace, dict) or trace.get("status") == "not-requested":
        return

    statuses: list[str] = []
    for key in ("raw_jsonl", "joined_jsonl", "report_json"):
        raw_path = trace.get(key)
        artifact = Path(raw_path) if isinstance(raw_path, str) and raw_path else None
        status = "present" if artifact is not None and artifact.is_file() else "missing"
        trace[f"{key}_status"] = status
        statuses.append(status)
        trace[f"{key}_digest"] = path_digest(artifact) if artifact is not None else None

    report_status = "missing"
    report_path = trace.get("report_json")
    if isinstance(report_path, str) and Path(report_path).is_file():
        try:
            report = json.loads(Path(report_path).read_text())
        except (OSError, UnicodeError, json.JSONDecodeError):
            report_status = "invalid"
        else:
            if isinstance(report, dict) and report.get("status") in {"complete", "incomplete"}:
                report_status = str(report["status"])
                counts = report.get("counts")
                trace["counts"] = counts if isinstance(counts, dict) else None
            else:
                report_status = "invalid"

    trace["status"] = (
        "complete"
        if report_status == "complete" and all(status == "present" for status in statuses)
        else "incomplete"
    )
    trace["completeness_status"] = report_status


def finalize_manifest(
    path: Path,
    *,
    harbor_rc: int | None,
    harbor_job_dir: Path | None = None,
    routing_stats: Path | None = None,
) -> int:
    if not path.is_file():
        print(f"ERROR: manifest not found: {path}")
        return 1

    manifest = json.loads(path.read_text())
    outcomes = manifest.setdefault("outcomes", {})

    harbor_dest = Path(outcomes["harbor_result_json"]) if outcomes.get("harbor_result_json") else None
    harbor_source = _find_harbor_result(harbor_job_dir)
    outcomes["harbor_result_json_status"] = _copy_if_present(harbor_source, harbor_dest)
    if harbor_job_dir is not None:
        outcomes["harbor_job_dir"] = str(harbor_job_dir.resolve())

    stats_dest = (
        Path(outcomes["routing_stats_json"]) if outcomes.get("routing_stats_json") else None
    )
    if outcomes.get("routing_stats_json_status") != "not-requested":
        if routing_stats is None and stats_dest is not None:
            routing_stats = stats_dest
        outcomes["routing_stats_json_status"] = _copy_if_present(routing_stats, stats_dest)

    closed_book = manifest.setdefault("closed_book", {})
    if isinstance(closed_book, dict) and closed_book.get("proxy_strip_log"):
        strip_dest = Path(closed_book["proxy_strip_log"])
        artifact_path = closed_book.get("proxy_strip_artifact") or strip_dest.name
        strip_source = None
        for artifact_name in dict.fromkeys(
            [Path(str(artifact_path)).name, strip_dest.name, "strip.jsonl"]
        ):
            strip_source = _find_artifact(harbor_job_dir, artifact_name)
            if strip_source is not None:
                break
        closed_book["proxy_strip_log_status"] = _copy_if_present(strip_source, strip_dest)

    _finalize_routing_trace(manifest)

    outcomes["harbor_rc"] = harbor_rc
    outcomes["completed_at"] = _iso_timestamp()
    path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    return 0


def _apply_extra(body: dict[str, Any], entries: list[str]) -> int:
    for entry in entries:
        if "=" not in entry or "." not in entry.split("=", 1)[0]:
            print(f"ERROR: --extra must be section.key=value, got {entry!r}")
            return 2
        lhs, raw = entry.split("=", 1)
        section, key = lhs.split(".", 1)
        try:
            parsed: Any = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw
        target = body.setdefault(section, {})
        if not isinstance(target, dict):
            print(f"ERROR: --extra target {section!r} is not an object")
            return 2
        target[key] = parsed
    return 0


def _cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_manifest")
    sub = parser.add_subparsers(dest="command")

    write = sub.add_parser("write")
    write.add_argument("--output", type=Path, required=True)
    write.add_argument("--launcher-argv-json", default="[]")
    write.add_argument("--server-preset", required=True)
    write.add_argument("--server-mode", default="")
    write.add_argument("--server-url", default="")
    write.add_argument("--server-port", type=int, default=0)
    write.add_argument("--server-argv-json", default="[]")
    write.add_argument("--server-config-json", default="{}")
    write.add_argument("--classifier-prompts-json", default="{}")
    write.add_argument("--harbor-server-url", default="")
    write.add_argument("--harbor-base-url", default="")
    write.add_argument("--upstream-base-url", default="")
    write.add_argument("--upstream-api-key-env", default="")
    write.add_argument("--routing-profiles", type=Path, default=None)
    write.add_argument("--route-model", default="")
    write.add_argument("--harbor-command-json", default="[]")
    write.add_argument("--dataset-label", default="")
    write.add_argument("--harbor-path", type=Path, default=None)
    write.add_argument("--codex-model-catalog", type=Path, default=None)
    write.add_argument("--agent", required=True)
    write.add_argument("--harbor-model", required=True)
    write.add_argument("--reasoning-effort", default="")
    write.add_argument("--n-concurrent", type=int, required=True)
    write.add_argument("--max-retries", type=int, required=True)
    write.add_argument("--agent-timeout-multiplier", required=True)
    write.add_argument("--n-tasks", type=int, default=-1)
    write.add_argument("--task-id", default="")
    write.add_argument("--task-list-file", type=Path, default=None)
    write.add_argument("--harbor-extra-json", default="[]")
    write.add_argument("--closed-book-mode", default="closed")
    write.add_argument("--closed-book-gateway-enforced", default="0")
    write.add_argument("--closed-book-hosted-tools-disabled", default="0")
    write.add_argument("--closed-book-proxy-strip-artifact", default="")
    write.add_argument("--harbor-patch-json", default="{}")
    write.add_argument("--agent-versions-json", default="{}")
    write.add_argument("--run-dir", type=Path, required=True)
    write.add_argument("--log-path", type=Path, required=True)
    write.add_argument("--harbor-result-json", type=Path, required=True)
    write.add_argument("--routing-stats-json", type=Path, required=True)
    write.add_argument("--routing-stats-status", default="predicted")
    write.add_argument("--routing-trace-jsonl", type=Path, default=None)
    write.add_argument("--routing-trace-joined-jsonl", type=Path, default=None)
    write.add_argument("--routing-trace-report-json", type=Path, default=None)
    write.add_argument("--routing-trace-status", default="not-requested")
    write.add_argument("--routing-trace-capture-content", default="0")
    write.add_argument("--extra", action="append", default=[])

    finalize = sub.add_parser("finalize")
    finalize.add_argument("--manifest", type=Path, required=True)
    finalize.add_argument("--harbor-rc", type=int, default=None)
    finalize.add_argument("--harbor-job-dir", type=Path, default=None)
    finalize.add_argument("--routing-stats", type=Path, default=None)

    ns = parser.parse_args(argv)
    if ns.command == "finalize":
        return finalize_manifest(
            ns.manifest,
            harbor_rc=ns.harbor_rc,
            harbor_job_dir=ns.harbor_job_dir,
            routing_stats=ns.routing_stats,
        )
    if ns.command != "write":
        parser.print_help()
        return 2

    task_list = ns.task_list_file.resolve() if ns.task_list_file else None
    routing_profiles = ns.routing_profiles.resolve() if ns.routing_profiles else None
    harbor_path = ns.harbor_path.resolve() if ns.harbor_path else None
    codex_model_catalog = (
        ns.codex_model_catalog.resolve() if ns.codex_model_catalog else None
    )
    run_dir = ns.run_dir.resolve()
    classifier_prompts = _json_arg(ns.classifier_prompts_json, {})
    if not isinstance(classifier_prompts, dict):
        print("ERROR: --classifier-prompts-json must decode to a JSON object")
        return 2
    try:
        routing_profiles_snapshot = snapshot_routing_profiles(routing_profiles, run_dir)
        dataset_manifest_snapshot = snapshot_dataset_manifest(harbor_path, run_dir)
    except OSError as exc:
        print(f"ERROR: failed to snapshot run inputs: {exc}")
        return 2
    harbor: dict[str, Any] = {
        "dataset": _opt(ns.dataset_label),
        "path": str(harbor_path) if harbor_path else None,
        "path_digest": path_digest(harbor_path) if harbor_path else None,
        "dataset_fingerprint": dataset_fingerprint(ns.dataset_label, task_list, harbor_path),
        "agent": ns.agent,
        "model": ns.harbor_model,
        "reasoning_effort": _opt(ns.reasoning_effort),
        "n_concurrent": ns.n_concurrent,
        "max_retries": ns.max_retries,
        "agent_timeout_multiplier": ns.agent_timeout_multiplier,
        "n_tasks": _opt_int(ns.n_tasks),
        "task_id": _opt(ns.task_id),
        "task_list_file": str(task_list) if task_list else None,
        "task_list_digest": path_digest(task_list) if task_list else None,
        "codex_model_catalog": (
            str(codex_model_catalog) if codex_model_catalog else None
        ),
        "codex_model_catalog_digest": (
            path_digest(codex_model_catalog) if codex_model_catalog else None
        ),
        "extra_args": _json_arg(ns.harbor_extra_json, []),
    }
    closed_book_mode = ns.closed_book_mode
    proxy_strip_log = run_dir / "proxy_strip_log.jsonl"
    closed_book: dict[str, Any] = {
        "mode": closed_book_mode,
        "gateway_enforced": _bool_arg(ns.closed_book_gateway_enforced),
        "hosted_tools_disabled": _bool_arg(ns.closed_book_hosted_tools_disabled),
        "proxy_strip_artifact": _opt(ns.closed_book_proxy_strip_artifact),
        "proxy_strip_log": str(proxy_strip_log),
        "proxy_strip_log_status": (
            "predicted" if ns.closed_book_proxy_strip_artifact else "not-requested"
        ),
        "verifier_egress": (
            "open-via-authenticated-proxy"
            if closed_book_mode == "closed"
            else "open-via-proxy"
        ),
        "agent_versions": _json_arg(ns.agent_versions_json, {}),
        "dataset_manifest_snapshot": (
            str(dataset_manifest_snapshot) if dataset_manifest_snapshot else None
        ),
        "dataset_manifest_snapshot_digest": (
            path_digest(dataset_manifest_snapshot) if dataset_manifest_snapshot else None
        ),
    }
    routing_trace_status = ns.routing_trace_status
    routing_trace: dict[str, Any] = {
        "status": routing_trace_status,
        "capture_content": _bool_arg(ns.routing_trace_capture_content),
    }
    for key, value in (
        ("raw_jsonl", ns.routing_trace_jsonl),
        ("joined_jsonl", ns.routing_trace_joined_jsonl),
        ("report_json", ns.routing_trace_report_json),
    ):
        resolved = value.resolve() if value is not None else None
        routing_trace[key] = str(resolved) if resolved is not None else None
        routing_trace[f"{key}_status"] = (
            "predicted" if resolved is not None and routing_trace_status != "not-requested"
            else "not-requested"
        )
    body: dict[str, Any] = {
        "server": {
            "preset": ns.server_preset,
            "mode": _opt(ns.server_mode),
            "url": _opt(ns.server_url),
            "port": ns.server_port or None,
            "argv": _json_arg(ns.server_argv_json, []),
            "config": _json_arg(ns.server_config_json, {}),
            "classifier_prompts": classifier_prompts,
            "harbor_server_url": _opt(ns.harbor_server_url),
            "harbor_base_url": _opt(ns.harbor_base_url),
            "upstream_base_url": _opt(ns.upstream_base_url),
            "upstream_api_key_env": _opt(ns.upstream_api_key_env),
            "routing_profiles": str(routing_profiles) if routing_profiles else None,
            "routing_profiles_digest": (
                path_digest(routing_profiles) if routing_profiles else None
            ),
            "routing_profiles_snapshot": (
                str(routing_profiles_snapshot) if routing_profiles_snapshot else None
            ),
            "routing_profiles_snapshot_digest": (
                path_digest(routing_profiles_snapshot) if routing_profiles_snapshot else None
            ),
            "route_model": _opt(ns.route_model),
        },
        "harbor": harbor,
        "harbor_patch": _json_arg(ns.harbor_patch_json, {}),
        "closed_book": closed_book,
        "routing_trace": routing_trace,
        "determinism": {
            "PYTHONHASHSEED": "0",
            "LC_ALL": "C.UTF-8",
        },
        "outcomes": {
            "run_dir": str(run_dir),
            "log_path": str(ns.log_path.resolve()),
            "harbor_result_json": str(ns.harbor_result_json.resolve()),
            "harbor_result_json_status": "predicted",
            "routing_stats_json": str(ns.routing_stats_json.resolve()),
            "routing_stats_json_status": ns.routing_stats_status,
            "harbor_rc": None,
        },
    }
    extra_rc = _apply_extra(body, ns.extra)
    if extra_rc:
        return extra_rc
    write_manifest(
        ns.output,
        launcher_argv=_json_arg(ns.launcher_argv_json, []),
        harbor_command=_json_arg(ns.harbor_command_json, ["harbor"]),
        **body,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli_main())
