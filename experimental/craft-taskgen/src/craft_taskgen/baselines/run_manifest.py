# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write a per-job reproducibility manifest.

One JSON file per harbor-run invocation, dropped next to harbor's own
`result.json`. Captures every knob the launcher resolves plus the
environment it resolved them in. A reviewer who wants to reproduce a
number opens this file + harbor's result.json and the per-trial
trajectory dirs and has everything they need.

Schema versioned via `run.schema_version`. Bump the integer when the
shape changes in a backwards-incompatible way.

Caller responsibilities:
- Pass every agent/backend/reasoning/output_cap/sampling/harness field
  as a kwarg. The launcher knows those values at manifest-write time;
  this module doesn't re-resolve them.

Self-resolved fields (filled in by `write_manifest` regardless of what
the caller passed):
- `run.timestamp` — UTC ISO-8601 at manifest-write time
- `run.hostname` — `socket.gethostname()`
- `run.craft_taskgen_sha` — `git rev-parse HEAD` of the launcher repo
- `run.craft_taskgen_dirty` — tri-state: True if uncommitted changes,
  False if clean, None if not in a git repo
- `run.craft_taskgen_tree_kind` — "git-clean" | "git-dirty" | "not-git"
  (string projection of the above for ease of jq/grep)
- `run.node_version` — `node --version` on the host; `None` if absent
- `run.harbor_version` — `uv run harbor --version`; `None` on failure

Task dataset provenance is captured via `harness.task_dir_digest` (a
content-hash over every file under `tasks_dir`) rather than a separate
repo SHA — the digest is what matters for reproduction and is
independent of whether the tasks directory happens to be versioned.

Kept zero-dependency (stdlib only) so the launcher's
`uv run python -m` invocation stays fast. Deliberately does NOT
enumerate or validate tasks — that's harbor's job, and harbor writes
its authoritative task-selection record into `result.json`. The
manifest captures `harness.tasks_dir` + `harness.task_dir_digest` so a
reviewer can confirm dataset identity; trial identity lives in
`result.json` next to this manifest.
"""

from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def _iso_timestamp() -> str:
    """UTC ISO-8601 with seconds precision, 'Z' suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    """Run a command and return its stripped stdout, or None on failure."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _run_rc(cmd: list[str], cwd: Path | None = None) -> tuple[int, str] | None:
    """Like `_run` but returns (returncode, stdout). None on OS/subprocess error.

    Lets callers distinguish "command ran, produced no output" (rc=0, "")
    from "command failed" (rc!=0) — the plain `_run` collapses both to None.
    """
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return result.returncode, result.stdout


def _git_sha(path: Path) -> str | None:
    """Resolve HEAD sha for the repo containing `path`, or None."""
    return _run(["git", "rev-parse", "HEAD"], cwd=path)


def _git_dirty(path: Path) -> bool | None:
    """Tri-state: True if uncommitted changes in the repo containing `path`;
    False if the tree is clean; None if `path` isn't inside a git repo.

    Must not collapse "clean repo" (rc=0, empty stdout) into None — that
    would make a clean tree indistinguishable from "not a git repo" in
    the manifest.
    """
    result = _run_rc(["git", "status", "--porcelain"], cwd=path)
    if result is None:
        return None
    rc, out = result
    if rc != 0:
        # Non-zero rc from git status typically means "not a git repo".
        return None
    return bool(out.strip())


def _tree_kind(dirty: bool | None) -> str:
    """Classify a directory for the manifest reader.

    `_git_dirty` is tri-state (True=dirty, False=clean, None=not a repo);
    projecting that into a single string field makes grep/jq downstream
    simpler than checking two related fields for consistency.
    """
    if dirty is None:
        return "not-git"
    return "git-dirty" if dirty else "git-clean"


def _node_version() -> str | None:
    return _run(["node", "--version"])


def _harbor_version() -> str | None:
    # `uv run harbor --version` is the canonical check the launcher uses.
    out = _run(["uv", "run", "harbor", "--version"])
    if out:
        return out.splitlines()[0]
    return None


def task_dir_digest(tasks_dir: Path) -> str:
    """Deterministic sha256 over `<relpath>\\n<sha256>\\n` lines for every
    regular file under `tasks_dir`, sorted. Suitable for "is your copy
    the same as ours" reviewer checks.

    Returns `sha256:...` or the string `sha256:unknown` on error.
    """
    try:
        hasher = hashlib.sha256()
        root = tasks_dir.resolve()
        entries: list[tuple[str, str]] = []
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(root).as_posix()
            with p.open("rb") as fh:
                file_hash = hashlib.file_digest(fh, "sha256").hexdigest()
            entries.append((rel, file_hash))
        for rel, file_hash in entries:
            hasher.update(f"{rel}\n{file_hash}\n".encode())
        return f"sha256:{hasher.hexdigest()}"
    except OSError:
        return "sha256:unknown"


def probe_vllm_models(base_url: str, api_key: str | None = None) -> dict[str, Any] | None:
    """GET <base_url>/models and return a small summary.

    Returns None on any error (network, non-200, malformed body). Shape:
        {"served_model_name": str, "served_model_root": str | None,
         "max_model_len": int | None, "owned_by": str | None}
    """
    try:
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key or 'EMPTY'}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.load(resp)
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
        return None
    data = payload.get("data") or []
    if not data:
        return None
    first = data[0]
    return {
        "served_model_name": first.get("id"),
        "served_model_root": first.get("root"),
        "max_model_len": first.get("max_model_len"),
        "owned_by": first.get("owned_by"),
    }


def write_manifest(
    output_path: Path,
    *,
    tasks_dir: Path | None = None,
    launcher_argv: list[str] | None = None,
    **fields: Any,
) -> None:
    """Write the run manifest JSON to `output_path`.

    `tasks_dir` is used to compute `harness.task_dir_digest` in
    `_cli_main` (a content-hash over every file under the dataset so a
    reviewer can confirm identity without needing the same filesystem
    layout). The rest of the manifest comes from `fields`; pass a dict
    per top-level section (agent=..., backend=..., reasoning=..., etc.).
    Unrecognized top-level keys are written through verbatim.
    """
    launcher_root = Path(__file__).resolve().parents[3]
    launcher_dirty = _git_dirty(launcher_root)
    run_meta: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": _iso_timestamp(),
        "hostname": socket.gethostname(),
        "craft_taskgen_sha": _git_sha(launcher_root),
        "craft_taskgen_dirty": launcher_dirty,
        "craft_taskgen_tree_kind": _tree_kind(launcher_dirty),
        "harbor_version": _harbor_version(),
        "node_version": _node_version(),
        "launcher_argv": launcher_argv or [],
    }

    manifest: dict[str, Any] = {"run": run_meta}

    # Caller-supplied top-level sections pass through as-is. The launcher
    # is expected to populate agent/backend/reasoning/output_cap/sampling/
    # determinism/compaction/harness/outcomes, but we don't require any
    # specific set — reviewers can add sections without a module change.
    manifest.update({k: v for k, v in fields.items() if k != "run"})

    # If the caller passed `run` fields, merge them on top of our
    # self-resolved ones (explicit-caller wins).
    if "run" in fields and isinstance(fields["run"], dict):
        manifest["run"] = {**run_meta, **fields["run"]}

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")


def _finalize_manifest(path: Path, harbor_rc: int | None) -> int:
    """Post-run update: flip outcomes.harbor_result_json_status and record rc.

    Called by the launcher after harbor exits so a reader can tell from
    the manifest alone whether the promised result.json actually
    materialized.
    """
    if not path.exists():
        print(f"ERROR: manifest not found at {path}", file=__import__("sys").stderr)
        return 1
    manifest = json.loads(path.read_text())
    outcomes = manifest.setdefault("outcomes", {})
    result_path = outcomes.get("harbor_result_json")
    if result_path:
        outcomes["harbor_result_json_status"] = "present" if Path(result_path).exists() else "missing"
    if harbor_rc is not None:
        outcomes["harbor_rc"] = harbor_rc
    path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    return 0


# Small helper that a bash caller can invoke with positional args;
# saves the launcher from a brittle `python -c "..."` heredoc.
def _cli_main(argv: list[str]) -> int:
    """CLI entry point with per-field flags.

    Build a manifest from named flags (one per field the launcher
    populates) rather than JSON-over-stdin. This keeps the launcher's
    shell-out cleanly readable and trivial to extend: add a new
    --foo-bar flag here, pass it from bash, and it flows into the
    manifest.

    Empty-string values are treated as `None` (JSON null) so bash can
    pass `"$VAR"` unconditionally without juggling quoting.

    If `--tasks-dir` is given, `harness.task_dir_digest` is
    auto-populated from the tasks_dir contents.
    """
    import argparse
    import shlex
    import sys

    # Sub-mode: --finalize <path> --harbor-rc <N>. Updates an existing
    # manifest in place. Done with a hand-rolled sniff rather than an
    # argparse subparser so the launcher invocation stays trivial.
    if "--finalize" in argv:
        fp = argparse.ArgumentParser(prog="run_manifest --finalize")
        fp.add_argument("--finalize", type=Path, required=True)
        fp.add_argument("--harbor-rc", type=int, default=None)
        ns = fp.parse_args(argv)
        return _finalize_manifest(ns.finalize, ns.harbor_rc)

    p = argparse.ArgumentParser(prog="run_manifest")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tasks-dir", type=Path, default=None)

    # Agent section
    p.add_argument("--agent", default="")
    p.add_argument("--agent-version", default="")
    p.add_argument("--model", default="")

    # Backend section
    p.add_argument("--backend", default="")
    p.add_argument("--base-url", default="")

    # Reasoning section
    p.add_argument("--effort", default="")
    p.add_argument("--reasoning-source", default="")
    p.add_argument("--reasoning-notes", default="")

    # Output cap section
    p.add_argument("--output-cap", type=int, default=0)
    p.add_argument("--output-cap-applied", default="")

    # Harness section
    p.add_argument("--n-tasks", type=int, default=0)
    p.add_argument("--n-concurrent", type=int, default=1)
    p.add_argument("--task-name", default="")
    p.add_argument("--exclude-task-name", default="")

    # Run section extras
    p.add_argument(
        "--launcher-argv",
        default="",
        help='single string — launcher passes "$0 $@" quoted; we shlex.split.',
    )

    # vLLM serving snapshot. When --vllm-probe is set, issue one GET to
    # <base_url>/v1/models and record the result in backend.vllm_snapshot.
    # Intended for --backend vllm runs; cheap (one HTTP call) and only
    # captures what the server is willing to share.
    p.add_argument(
        "--vllm-probe",
        action="store_true",
        help="probe base_url/models and record the result in backend.vllm_snapshot",
    )
    p.add_argument("--vllm-api-key", default="", help="defaults to EMPTY")

    # Any additional `--extra section.key=value` pairs for forward-
    # compat. Values are strings; numeric/boolean coercion is done by
    # the caller if needed.
    p.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Append a field to the manifest (repeatable).",
    )

    ns = p.parse_args(argv)

    def _opt(v: str) -> str | None:
        return v if v else None

    # Resolve paths to absolutes so the manifest is location-independent
    # (reviewers can share the file across machines and still trace
    # which directory was actually used).
    tasks_dir_abs = ns.tasks_dir.resolve() if ns.tasks_dir else None

    # launcher_argv: first element is the script path — resolve to abs
    # if it points at a real file. Rest (the flags/values) pass through.
    argv_list = shlex.split(ns.launcher_argv) if ns.launcher_argv else []
    if argv_list:
        script_path = Path(argv_list[0])
        if script_path.exists():
            argv_list[0] = str(script_path.resolve())

    body: dict[str, Any] = {
        "agent": {
            "name": _opt(ns.agent),
            "version": _opt(ns.agent_version),
            "model": _opt(ns.model),
        },
        "backend": {
            "kind": _opt(ns.backend),
            "base_url": _opt(ns.base_url),
        },
        "reasoning": {
            "effort": _opt(ns.effort),
            "source": _opt(ns.reasoning_source),
            "notes": _opt(ns.reasoning_notes),
        },
        "output_cap": {
            # tokens is null when the cap is unset (sanity agents pass
            # --output-cap 0 because they don't generate output) or the
            # agent's cap knob is a no-op (codex; see openai/codex#4138).
            # A reader shouldn't mistake a declared-but-ignored value
            # for a real cap.
            "tokens": (
                None
                if ns.output_cap == 0 or (ns.output_cap_applied and "UNCAPPED" in ns.output_cap_applied)
                else ns.output_cap
            ),
            "source": "output_cap.py",
            "applied": _opt(ns.output_cap_applied),
        },
        "harness": {
            "tasks_dir": str(tasks_dir_abs) if tasks_dir_abs else None,
            "n_tasks": ns.n_tasks,
            "n_concurrent": ns.n_concurrent,
            "task_name": _opt(ns.task_name),
            "exclude_task_name": _opt(ns.exclude_task_name),
        },
        "outcomes": {
            # Resolve relative to --output's parent so it's an absolute
            # path the reader can follow from the manifest alone.
            "harbor_result_json": str((ns.output.parent / "result.json").resolve()),
            # Initially predicted (manifest is written BEFORE harbor
            # starts). Flipped to "present"/"missing" by the --finalize
            # subcommand after harbor exits. A reader seeing "predicted"
            # in a finished run knows the finalize step was skipped or
            # failed, so they should verify existence themselves.
            "harbor_result_json_status": "predicted",
            "harbor_rc": None,
        },
        "run": {
            "launcher_argv": argv_list,
        },
    }

    # Merge --extra section.key=value entries. Value is JSON-coerced so
    # numeric, boolean, and null values land with the right JSON type
    # (e.g. `--extra foo.bar=95` → int, `--extra foo.bar=false` → bool,
    # `--extra foo.bar=null` → JSON null). A string that isn't valid JSON
    # (e.g. `C.UTF-8`, human-readable notes) falls through to string.
    for entry in ns.extra:
        if "=" not in entry or "." not in entry.split("=", 1)[0]:
            print(f"ERROR: --extra must be section.key=value, got {entry!r}", file=sys.stderr)
            return 2
        lhs, raw = entry.split("=", 1)
        section, key = lhs.split(".", 1)
        try:
            parsed: Any = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw
        body.setdefault(section, {})[key] = parsed

    # Auto-populate harness.task_dir_digest when tasks_dir is given.
    # Task enumeration and per-task config inspection are intentionally
    # NOT done here — that's harbor's job, and harbor's result.json (next
    # to this manifest) is the authoritative record of which tasks
    # actually ran with which timeouts. The digest lets a reviewer
    # confirm dataset identity; result.json tells them what happened.
    if ns.tasks_dir is not None and ns.tasks_dir.is_dir():
        body["harness"]["task_dir_digest"] = task_dir_digest(ns.tasks_dir)

    # vLLM serving snapshot (opt-in via --vllm-probe). Records what the
    # server is willing to share over /v1/models. None on any failure so
    # the manifest still writes.
    if ns.vllm_probe and ns.base_url:
        body["backend"]["vllm_snapshot"] = probe_vllm_models(ns.base_url, api_key=_opt(ns.vllm_api_key))

    write_manifest(ns.output, tasks_dir=ns.tasks_dir, **body)
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(_cli_main(sys.argv[1:]))
