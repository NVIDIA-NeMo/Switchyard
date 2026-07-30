# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-flight checks for long-running pipeline invocations.

Run `craft-taskgen-preflight --candidates 'candidates/*.json'` before a long,
unattended run to catch missing repos, broken auth, or low disk space before
wasting wall time.

Exits 0 if all checks pass; 1 otherwise. Each check prints PASS/FAIL with a short hint.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _print(label: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    line = f"  [{mark}] {label}"
    if detail:
        line += f" — {detail}"
    print(line)


def _print_warn(label: str, detail: str) -> None:
    """Non-fatal warning — prints but doesn't affect pass/fail counts."""
    print(f"  [WARN] {label} — {detail}")


def _print_info(label: str, detail: str) -> None:
    """Informational line — prints but doesn't affect pass/fail counts."""
    print(f"  [INFO] {label} — {detail}")


def check_claude_cli() -> bool:
    """Claude CLI is installed and returns a valid response to a trivial prompt.

    Uses `_cfg.CLAUDE_CMD` (the pinned binary if installed, else system claude)
    so preflight reports the version the pipeline will actually call.
    """
    import craft_taskgen.config as _cfg

    cmd = _cfg.CLAUDE_CMD
    if cmd == "claude" and not shutil.which("claude"):
        _print("claude CLI on PATH", False, "install Claude Code CLI")
        return False
    try:
        result = subprocess.run(
            [cmd, "--version"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        _print("claude --version", False, f"{e}")
        return False
    if result.returncode != 0:
        _print("claude --version", False, result.stderr.strip()[:120])
        return False
    version_line = result.stdout.strip().splitlines()[0] if result.stdout else ""
    suffix = f" (pinned at {cmd})" if cmd != "claude" else " (system)"
    _print("claude CLI", True, version_line + suffix)
    return True


def check_gh_cli() -> bool:
    """gh CLI present + authenticated (needed only for mining, not pipeline runs)."""
    if not shutil.which("gh"):
        _print("gh CLI on PATH", False, "needed for craft-taskgen-mine; skip if candidates already mined")
        return False
    try:
        result = subprocess.run(
            ["gh", "auth", "status"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        _print("gh auth", False, f"{e}")
        return False
    ok = result.returncode == 0
    _print("gh auth", ok, "Logged in" if ok else "run `gh auth login`")
    return ok


def check_docker() -> bool:
    """Docker daemon reachable."""
    if not shutil.which("docker"):
        _print("docker on PATH", False, "install Docker")
        return False
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        _print("docker info", False, f"{e}")
        return False
    ok = result.returncode == 0
    _print(
        "docker daemon",
        ok,
        f"server {result.stdout.strip()}" if ok else result.stderr.strip()[:120],
    )
    return ok


def check_harbor() -> bool:
    """Harbor venv exists with runnable executable (smoke-test step needs this)."""
    harbor_bin = Path(".venv/bin/harbor")
    if not harbor_bin.is_file():
        _print("Harbor executable", False, "expected .venv/bin/harbor — run `uv sync`")
        return False
    try:
        result = subprocess.run(
            [str(harbor_bin), "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        _print("Harbor --help", False, f"{e}")
        return False
    ok = result.returncode == 0
    _print("Harbor runnable", ok, "ready" if ok else result.stderr.strip()[:120])
    return ok


def check_harbor_lab() -> bool:
    """harbor-lab binary resolvable and runnable. Deep-dive step shells out to it
    to pull `errors`, `edits`, `tool-sequence`, and `metrics` from a trial. The
    resolver honors $HARBOR_LAB, then PATH, then the pinned pydev venv."""
    try:
        from craft_taskgen.steps import _resolve_harbor_lab_bin
    except ImportError as e:
        _print("harbor-lab resolver", False, f"import error: {e}")
        return False

    try:
        bin_path = _resolve_harbor_lab_bin()
    except RuntimeError as e:
        _print("harbor-lab binary", False, str(e)[:160])
        return False

    try:
        result = subprocess.run(
            [bin_path, "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        _print("harbor-lab --help", False, f"{e}")
        return False
    ok = result.returncode == 0
    _print("harbor-lab runnable", ok, f"at {bin_path}" if ok else result.stderr.strip()[:120])
    return ok


def check_gh_rate_limit(needed: int) -> bool:
    """Ensure GitHub core API has enough remaining calls for a mining run.

    The miner issues roughly 3 calls per repo (list merged PRs paged + per-PR diff stats).
    Pass `needed` as an estimate (e.g. 3 * number of repos to mine) so we fail early
    if the user is about to stall the miner on a rate-limit wall.
    """
    if not shutil.which("gh"):
        _print("gh API rate limit", False, "gh CLI not on PATH")
        return False
    try:
        result = subprocess.run(
            ["gh", "api", "rate_limit", "--jq", ".rate.remaining"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        _print("gh API rate limit", False, f"{e}")
        return False
    if result.returncode != 0:
        _print("gh API rate limit", False, result.stderr.strip()[:120])
        return False
    try:
        remaining = int(result.stdout.strip())
    except ValueError:
        _print("gh API rate limit", False, f"unparseable: {result.stdout!r}")
        return False
    ok = remaining >= needed
    _print(
        f"gh API rate limit (need ~{needed})",
        ok,
        f"{remaining} calls remaining this hour"
        + ("" if ok else "; wait or authenticate a higher-limit token"),
    )
    return ok


def check_docker_memory(min_gb: int = 8) -> None:
    """Warn (not fail) if Docker daemon has <min_gb of memory allocated.

    At concurrency 6 the pipeline can have 6 Docker containers running at once
    during F2P/P2P classify; low-memory Docker Desktop configs will OOM.
    """
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.MemTotal}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return  # docker already failed its own check; stay quiet
    if result.returncode != 0:
        return
    try:
        mem_bytes = int(result.stdout.strip())
    except ValueError:
        return
    mem_gb = mem_bytes / (1024**3)
    if mem_gb < min_gb:
        _print_warn(
            "Docker memory",
            f"{mem_gb:.1f}GB allocated; concurrency 6 may OOM. "
            f"Raise to {min_gb}GB+ in Docker Desktop → Settings → Resources.",
        )
    else:
        _print("Docker memory", True, f"{mem_gb:.1f}GB allocated")


def check_repos_csv(csv_path: str) -> bool:
    """Validate a repos CSV used by `craft-taskgen-mine --repos-csv`."""
    path = Path(csv_path)
    if not path.is_file():
        _print(f"repos CSV {csv_path}", False, "file not found")
        return False
    try:
        with open(path) as f:
            header = f.readline().strip().split(",")
            n_rows = sum(1 for _ in f)
    except OSError as e:
        _print(f"repos CSV {csv_path}", False, f"{e}")
        return False
    required = {"short_name", "github_repo"}
    missing = required - set(header)
    if missing:
        _print(f"repos CSV {csv_path}", False, f"missing columns: {', '.join(sorted(missing))}")
        return False
    _print(f"repos CSV {csv_path}", True, f"{n_rows} repos")
    return True


def check_claude_auth(model: str | None = None) -> bool:
    """End-to-end validation of the `claude -p` path through the gateway.

    Builds the same env `gateway.build_gateway_env` builds, shells out to
    `claude -p`, then asserts the response's `modelUsage` key looks gateway-shaped
    (e.g. `aws/anthropic/...`) — NOT a bare Anthropic-direct name like `opus`
    or `claude-opus-4-6`. A mismatch means the request leaked to OAuth/direct.
    """
    import craft_taskgen.config as _cfg
    from craft_taskgen.gateway import build_gateway_env

    if not model:
        _print("claude -p auth", False, "no model provided (pass --profile with llm_step_model set)")
        return False
    try:
        env = build_gateway_env(model)
    except RuntimeError as e:
        _print("claude -p auth", False, f"gateway env: {e}")
        return False

    cmd = [_cfg.CLAUDE_CMD, "-p", "hi", "--output-format", "json", "--max-turns", "1"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=env)
    except (subprocess.TimeoutExpired, OSError) as e:
        _print("claude -p auth", False, f"{e}")
        return False
    if result.returncode != 0:
        hint = result.stderr.strip().splitlines()[0] if result.stderr.strip() else "no stderr"
        _print("claude -p auth", False, hint[:120])
        return False
    try:
        out = json.loads(result.stdout)
    except json.JSONDecodeError:
        _print("claude -p auth", False, "response was not JSON")
        return False
    if out.get("is_error"):
        _print("claude -p auth", False, f"is_error=True subtype={out.get('subtype', '?')}")
        return False
    model_usage = out.get("modelUsage") or {}
    model_used = next(iter(model_usage.keys()), "") if model_usage else ""
    if not model_used:
        _print("claude -p auth", False, "response had no modelUsage — cannot verify gateway routing")
        return False
    # Gateway-shaped model IDs start with a provider prefix like `aws/`, `gcp/`, `azure/`.
    # Anthropic-direct names look like `claude-opus-4-6` or short aliases like `opus`.
    if "/" not in model_used:
        _print(
            "claude -p auth",
            False,
            f"model {model_used!r} is NOT gateway-shaped — request routed via OAuth/direct. "
            "Check ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL in .env.",
        )
        return False
    _print("claude -p auth", True, f"gateway ok ({model_used})")
    return True


def _post_openai_compat(
    label: str,
    base_url: str,
    api_key: str,
    path: str,
    payload: dict,
) -> bool:
    """POST to an OpenAI-compatible endpoint and report PASS/FAIL via _print.

    Shared body for check_litellm_endpoint (/chat/completions) and
    check_responses_endpoint (/responses). The two differ only in path +
    payload shape.
    """
    import urllib.error
    import urllib.request

    try:
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}{path}",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            resp.read()
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode()[:200]
        except Exception:
            pass
        _print(label, False, f"HTTP {e.code} — {body or e.reason}")
        return False
    except (urllib.error.URLError, OSError) as e:
        _print(label, False, f"{e}")
        return False
    _print(label, True, f"{payload['model']} reachable")
    return True


def check_litellm_endpoint(
    test_model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> bool:
    """POST to /chat/completions to catch "key exists but can't access this model" —
    a per-key policy failure LiteLLM gateways return only when you actually call.

    Reads OPENAI_BASE_URL (preferred, matches gateway.py / llm_judge) or
    OPENAI_API_BASE (legacy) when base_url / api_key are None.
    """
    api_key = (api_key or os.environ.get("OPENAI_API_KEY", "")).strip()
    base_url = (
        base_url or os.environ.get("OPENAI_BASE_URL", "") or os.environ.get("OPENAI_API_BASE", "")
    ).strip()
    if not api_key or not base_url:
        _print("LiteLLM endpoint", False, "OPENAI_API_KEY or OPENAI_BASE_URL missing in .env")
        return False
    if not test_model:
        _print("LiteLLM endpoint", False, "no test model provided (pass --profile)")
        return False
    return _post_openai_compat(
        "LiteLLM endpoint",
        base_url,
        api_key,
        "/chat/completions",
        {"model": test_model, "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]},
    )


def check_anthropic_endpoint(test_model: str | None = None) -> bool:
    """Validate ANTHROPIC_API_KEY against ANTHROPIC_BASE_URL with a minimal request.

    This is the path Harbor uses during smoke tests (claude_code.py reads these vars).
    `test_model` should be the profile's haiku_model so we exercise a model the
    key actually has policy access to.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "").strip().rstrip("/")
    if not api_key:
        _print("Anthropic endpoint", False, "ANTHROPIC_API_KEY missing in .env")
        return False
    if not base_url:
        _print("Anthropic endpoint", False, "ANTHROPIC_BASE_URL missing in .env")
        return False
    if not test_model:
        _print("Anthropic endpoint", False, "no test model provided (pass --profile)")
        return False
    try:
        import urllib.error
        import urllib.request

        payload = json.dumps(
            {
                "model": test_model,
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "hi"}],
            }
        ).encode()
        # Send both auth headers. LiteLLM-style gateways (NVIDIA inference-api)
        # expect Authorization: Bearer; real Anthropic expects x-api-key.
        # Sending both covers both cases without a fallback round-trip.
        req = urllib.request.Request(
            f"{base_url}/v1/messages",
            data=payload,
            headers={
                "x-api-key": api_key,
                "Authorization": f"Bearer {api_key}",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            resp.read()
    except urllib.error.HTTPError as e:
        # 401/403 = auth failure. 404 = bad base_url or path. 400 = model/body
        # quirk but auth+routing work (accept as PASS with note).
        body = ""
        try:
            body = e.read().decode()[:200]
        except Exception:
            pass
        if e.code in (401, 403):
            _print("Anthropic endpoint", False, f"HTTP {e.code} — {body or 'key rejected'}")
            return False
        if e.code == 404:
            _print("Anthropic endpoint", False, f"HTTP 404 — check ANTHROPIC_BASE_URL: {base_url}")
            return False
        _print("Anthropic endpoint", True, f"HTTP {e.code} (auth + routing ok)")
        return True
    except (urllib.error.URLError, OSError) as e:
        _print("Anthropic endpoint", False, f"{e}")
        return False
    _print("Anthropic endpoint", True, "ok")
    return True


def check_env_file() -> bool:
    """`.env` exists and the gateway-only keys are set to non-placeholder values.

    Hard-fails if ANTHROPIC_API_KEY or ANTHROPIC_BASE_URL are missing — OAuth
    fallback on this machine would charge the user's personal budget.
    """
    if not Path(".env").is_file():
        _print(".env file", False, "copy .env.example to .env and fill in API keys")
        return False
    values: dict[str, str] = {}
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip("\"'")

    placeholders = {"", "your-api-key-here", "sk-your-key-here", "changeme"}

    # OPENAI_* are required by `llm_judge` for every direct-API dispatch (all
    # models normalize to an `openai/` litellm route). Missing either key
    # passes preflight today but crashes on the first evaluate/build call.
    openai_key = values.get("OPENAI_API_KEY", "")
    openai_base = values.get("OPENAI_BASE_URL", "")
    if openai_key in placeholders:
        _print(".env OPENAI_API_KEY", False, "missing or placeholder; llm_judge will fail")
        return False
    if openai_base in placeholders:
        _print(".env OPENAI_BASE_URL", False, "missing or placeholder; llm_judge will fail")
        return False

    # Gateway-only policy: both of these must be set. Missing either means the
    # claude CLI will silently fall back to the OAuth token in ~/.claude and
    # charge the user's personal budget.
    anthropic_key = values.get("ANTHROPIC_API_KEY", "") or values.get("ANTHROPIC_AUTH_TOKEN", "")
    anthropic_base = values.get("ANTHROPIC_BASE_URL", "")
    if anthropic_key in placeholders:
        _print(
            ".env ANTHROPIC_API_KEY",
            False,
            "missing or placeholder. OAuth fallback is forbidden on this machine.",
        )
        return False
    if anthropic_base in placeholders:
        _print(
            ".env ANTHROPIC_BASE_URL",
            False,
            "missing or placeholder. OAuth fallback is forbidden on this machine.",
        )
        return False
    _print(".env file", True, "gateway keys present")
    return True


def check_disk_space(min_gb: int = 30) -> bool:
    """Free disk space on the current filesystem."""
    try:
        usage = shutil.disk_usage(".")
    except OSError as e:
        _print("disk usage", False, f"{e}")
        return False
    free_gb = usage.free / (1024**3)
    ok = free_gb >= min_gb
    _print(f"disk space >= {min_gb}GB", ok, f"{free_gb:.1f}GB free")
    return ok


def check_harbor_patches_applied() -> bool:
    """Verify patches/harbor-agent-patches.diff is applied to the installed harbor.

    `uv sync --reinstall-package harbor` silently resets the source tree,
    dropping our codex/opencode/claude-code patches. Unpatched codex routes
    to api.openai.com — only caught after a failed baseline.
    """
    import sysconfig

    patch_file = Path("patches/harbor-agent-patches.diff")
    site_packages = Path(sysconfig.get_paths()["purelib"])
    if not patch_file.is_file():
        _print("Harbor patches", False, f"{patch_file} not found")
        return False
    if not site_packages.is_dir():
        _print("Harbor patches", False, f"{site_packages} not found — run `uv sync`")
        return False

    added_by_file: dict[str, list[str]] = {}
    current_target: str | None = None
    with open(patch_file) as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("+++ b/"):
                current_target = line[len("+++ b/") :]
                added_by_file.setdefault(current_target, [])
            elif line.startswith("+") and not line.startswith("+++") and current_target:
                added = line[1:]
                if added.strip():
                    added_by_file[current_target].append(added)

    if not added_by_file:
        _print("Harbor patches", False, "could not parse any hunks from diff")
        return False

    missing: list[str] = []
    for rel_path, added_lines in added_by_file.items():
        target = site_packages / rel_path
        if not target.is_file():
            missing.append(f"{rel_path} (file not found)")
            continue
        # Exact-line match so an added line "foo" can't match "foo.bar()" via substring.
        target_lines = set(target.read_text().splitlines())
        for added in added_lines:
            if added not in target_lines:
                missing.append(f"{rel_path}: missing {added.strip()[:60]!r}")
                break

    if missing:
        _print(
            "Harbor patches",
            False,
            f"not applied ({len(missing)}/{len(added_by_file)} files) — "
            f"first: {missing[0]}. Run: patch -d {site_packages} "
            "-p1 < patches/harbor-agent-patches.diff",
        )
        return False
    _print("Harbor patches", True, f"applied across {len(added_by_file)} files")
    return True


def check_responses_endpoint(test_model: str | None = None) -> bool:
    """POST to /responses — the endpoint codex-family models (gpt-5.x-codex) require.

    Those models don't accept /chat/completions, so a LiteLLM check alone
    misses codex-specific auth/model-access failures.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    base_url = (os.environ.get("OPENAI_BASE_URL", "") or os.environ.get("OPENAI_API_BASE", "")).strip()
    if not api_key or not base_url:
        _print("Responses endpoint", False, "OPENAI_API_KEY or OPENAI_BASE_URL missing in .env")
        return False
    if not test_model:
        _print("Responses endpoint", False, "no test model provided")
        return False
    return _post_openai_compat(
        "Responses endpoint",
        base_url,
        api_key,
        "/responses",
        {"model": test_model, "input": "hi", "max_output_tokens": 16},
    )


def _is_sanity_only(agents: list[str]) -> bool:
    """True if every agent is a sanity-only (non-LLM) agent."""
    from craft_taskgen.baselines import SANITY_AGENT_NAMES

    return bool(agents) and all(a in SANITY_AGENT_NAMES for a in agents)


def check_baseline_readiness(backend: str, model: str | None, agents: list[str]) -> list[bool]:
    """Compose infrastructure + per-agent-endpoint checks for the baseline launcher."""
    # Report the resolved reasoning_effort for each agent we'll launch. Not a
    # check (can't fail), just visibility so the run's settings are in the log.
    from craft_taskgen.baselines import OUTPUT_TOKEN_CAP, SANITY_AGENT_NAMES, effort_for

    # Sanity agents (oracle, nop) don't talk to an LLM, so the reasoning
    # / cap / endpoint checks below don't apply.
    llm_agents = [a for a in agents if a not in SANITY_AGENT_NAMES]
    sanity_only = _is_sanity_only(agents)

    for agent in llm_agents:
        override = os.environ.get("REASONING_EFFORT_OVERRIDE")
        if override is not None:
            label = "set" if override else "disabled"
            _print_info(f"{agent} reasoning_effort override", f"{override!r} ({label})")
        else:
            effort = effort_for(agent, model)
            if effort:
                _print_info(f"{agent} reasoning_effort", f"{effort} (from reasoning_defaults)")
            else:
                _print_info(
                    f"{agent} reasoning_effort",
                    f"no row for ({agent}, {model}); agent-native default will apply",
                )

    # Report the output-token cap each agent will receive. Codex has no
    # working knob (openai/codex#4138) and runs uncapped; qwen-coder has
    # a settings.json knob but harbor doesn't plumb it.
    for agent in llm_agents:
        if agent == "codex":
            _print_info(f"{agent} output_cap", "UNCAPPED (openai/codex#4138)")
        elif agent == "qwen-coder":
            _print_info(f"{agent} output_cap", "UNCAPPED (no harbor plumbing for ~/.qwen/settings.json)")
        elif agent == "pi":
            _print_info(f"{agent} output_cap", "UNCAPPED (no env var for max_tokens)")
        elif agent in ("claude-code", "opencode"):
            _print_info(f"{agent} output_cap", f"{OUTPUT_TOKEN_CAP} tokens")
        else:
            _print_info(f"{agent} output_cap", f"{OUTPUT_TOKEN_CAP} (unknown agent; may not be honored)")

    for agent in agents:
        if agent in SANITY_AGENT_NAMES:
            _print_info(f"{agent} sanity agent", "no LLM endpoint check (oracle/nop)")

    results: list[bool] = [
        check_harbor(),
        check_harbor_patches_applied(),
        check_docker(),
        check_disk_space(min_gb=30),
    ]

    # Sanity-only runs need infra checks (harbor/docker/disk/patches) but
    # none of the per-agent LLM endpoint reachability — they don't talk
    # to an LLM. Return early; .env is irrelevant when no key is needed.
    if sanity_only:
        return results

    if backend == "gateway":
        results.append(check_env_file())
        # Short-circuit on the known aws/+claude-code breakage — otherwise we'd
        # still hit the gateway and record a second, redundant failure.
        if "claude-code" in agents and model.startswith("aws/"):
            _print(
                "claude-code + aws/ gateway model",
                False,
                "claude-code is temporarily broken against aws/ gateway endpoints. "
                "Use an azure/ model (e.g. azure/anthropic/claude-opus-4-6).",
            )
            results.append(False)
        elif "claude-code" in agents:
            results.append(check_anthropic_endpoint(test_model=model))
        if "codex" in agents:
            results.append(check_responses_endpoint(test_model=model))
        if "opencode" in agents:
            results.append(check_litellm_endpoint(test_model=model))
        if "qwen-coder" in agents:
            # Qwen Code CLI hits /v1/chat/completions via the OpenAI Node SDK
            # against OPENAI_BASE_URL. Same reachability check as opencode.
            results.append(check_litellm_endpoint(test_model=model))
        if "pi" in agents:
            # Pi (nvidia provider) hits /v1/chat/completions via the openai-
            # completions adapter against the baseUrl planted in models.json
            # (which the launcher seeds from OPENAI_BASE_URL).
            results.append(check_litellm_endpoint(test_model=model))
        return results

    # backend == "vllm" (argparse enforces the choice set)
    base_url = os.environ.get("VLLM_BASE_URL", "").strip()
    if not base_url:
        _print("vLLM backend", False, "VLLM_BASE_URL not set in env")
        results.append(False)
        return results
    _print("vLLM backend", True, base_url)
    # Note: `localhost`/`127.0.0.1` are rewritten to `host.docker.internal`
    # before forwarding to the container (see scripts/run-baselines.sh), which
    # also passes --extra-docker-compose patches/compose-overrides/host-gateway.yaml
    # to map host.docker.internal → host-gateway so the container can reach it.
    if "claude-code" in agents:
        _print_warn(
            "claude-code + vLLM",
            "vLLM does not speak the Anthropic protocol; claude-code will not work "
            "against this backend unless you've put an Anthropic-compatible shim in front.",
        )
    # opencode requires `provider/model_id` format (split at first "/"); the
    # `provider/` prefix is consumed by opencode's config machinery and only
    # model_id is forwarded to the upstream server. Strip it before preflight
    # so we hit vLLM with the name vLLM knows.
    test_model = model.split("/", 1)[1] if "opencode" in agents and "/" in model else model
    results.append(
        check_litellm_endpoint(
            test_model=test_model,
            base_url=base_url,
            api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
        )
    )
    return results


def check_repos_exist(candidate_files: list[str]) -> bool:
    """Every candidate file's repo directory exists under repos/."""
    if not candidate_files:
        _print("candidate repos", False, "no candidate files provided (use --candidates)")
        return False
    missing: list[str] = []
    shas_to_check: list[tuple[str, str]] = []  # (repo, base_sha)
    for fpath in candidate_files:
        repo = Path(fpath).stem
        repo_path = Path("repos") / repo
        if not repo_path.is_dir():
            missing.append(repo)
            continue
        # Sample one base_sha from the candidates to verify it's reachable
        try:
            with open(fpath) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        cands = data.get("candidates", [])
        if cands:
            base_sha = cands[0].get("merge_base_sha") or cands[0].get("base_sha")
            if base_sha:
                shas_to_check.append((str(repo_path), base_sha))

    if missing:
        _print("repos/ directories", False, f"missing: {', '.join(missing[:5])}")
        return False
    _print("repos/ directories", True, f"{len(candidate_files)} repos present")

    # Verify base SHAs resolve in their clones
    unreachable: list[str] = []
    for repo_path, sha in shas_to_check:
        try:
            result = subprocess.run(
                ["git", "-C", repo_path, "cat-file", "-t", sha],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (subprocess.TimeoutExpired, OSError):
            unreachable.append(f"{repo_path}@{sha[:8]}")
            continue
        if result.returncode != 0:
            unreachable.append(f"{repo_path}@{sha[:8]}")

    if unreachable:
        _print(
            "base_sha reachable in clones",
            False,
            f"missing: {', '.join(unreachable[:3])} (run `git fetch --all` in those repos)",
        )
        return False
    if shas_to_check:
        _print("base_sha reachable", True, f"verified {len(shas_to_check)} sample SHAs")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-flight checks for long-running pipeline invocations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--candidates",
        nargs="*",
        default=[],
        help="Candidate JSON glob(s) to verify repos/ for (e.g. 'candidates/*.json')",
    )
    parser.add_argument(
        "--repos-csv",
        type=str,
        default=None,
        help="Validate a repos CSV and check GH rate limit for mining it (e.g. references/repo_list.csv)",
    )
    parser.add_argument("--min-disk-gb", type=int, default=30, help="Minimum free disk in GB")
    parser.add_argument(
        "--min-docker-gb",
        type=int,
        default=8,
        help="Docker memory warning threshold in GB (non-fatal)",
    )
    parser.add_argument("--skip-gh", action="store_true", help="Skip gh CLI check")
    parser.add_argument("--skip-harbor", action="store_true", help="Skip Harbor check")
    parser.add_argument("--skip-harbor-lab", action="store_true", help="Skip harbor-lab check")
    parser.add_argument(
        "--check-endpoints",
        action="store_true",
        help="Make live API calls to validate auth for Claude CLI, LiteLLM, and Anthropic endpoints "
        "(recommended before long unattended runs; costs fractions of a cent)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="TOML profile to load (used with --check-endpoints to verify the profile's "
        "opus/haiku/llm_step_model names resolve at the LiteLLM endpoint)",
    )
    parser.add_argument(
        "--check-baselines",
        action="store_true",
        help="Preflight for scripts/run-baselines.sh (Harbor, Docker, disk, and endpoint reachability "
        "for the selected backend/model/agents). Incompatible with --candidates/--repos-csv.",
    )
    parser.add_argument(
        "--backend",
        choices=["gateway", "vllm"],
        default="gateway",
        help="For --check-baselines: which inference backend to verify.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="For --check-baselines: the model ID the baselines will run against.",
    )
    parser.add_argument(
        "--agents",
        type=str,
        default="claude-code",
        help=(
            "For --check-baselines: comma-separated agent names "
            "(claude-code, codex, opencode, qwen-coder, pi)."
        ),
    )
    args = parser.parse_args()

    if args.check_baselines and (args.candidates or args.repos_csv):
        print("--check-baselines is incompatible with --candidates/--repos-csv")
        return 1
    # Sanity agents (oracle, nop) don't talk to an LLM; --model is
    # only required when at least one LLM-driven agent is in the list.
    if args.check_baselines and not args.model:
        parsed_agents = [a.strip() for a in args.agents.split(",") if a.strip()]
        if not _is_sanity_only(parsed_agents):
            print("--check-baselines requires --model (unless --agents is sanity-only: oracle/nop)")
            return 1

    candidate_files: list[str] = []
    for pattern in args.candidates:
        candidate_files.extend(glob.glob(pattern))

    # Load .env so endpoint checks can read OPENAI_API_KEY etc.
    from craft_taskgen.config import _load_env

    _load_env()

    if args.check_baselines:
        agents = [a.strip() for a in args.agents.split(",") if a.strip()]
        print("Baseline preflight")
        print("=" * 60)
        checks = check_baseline_readiness(args.backend, args.model, agents)
        print("=" * 60)
        passed = sum(1 for c in checks if c)
        total = len(checks)
        if passed == total:
            print(f"All {total} checks passed. Ready to launch baselines.")
            return 0
        print(f"{passed}/{total} checks passed. Fix failures before launching.")
        return 1

    # Load profile if given — tells endpoint check which models to actually call
    profile_haiku_model: str | None = None
    profile_llm_step_model: str | None = None
    if args.profile:
        from craft_taskgen.config import PipelineProfile

        profile = PipelineProfile.from_toml(args.profile)
        profile_haiku_model = profile.haiku_model
        profile_llm_step_model = profile.llm_step_model or None
    elif args.check_endpoints:
        print(
            "  [warn] --check-endpoints without --profile: endpoint checks need "
            "a profile to know which models to test"
        )
        return 1

    print("Pre-flight checks")
    print("=" * 60)

    checks = [
        check_claude_cli(),
        check_docker(),
        check_env_file(),
        check_disk_space(args.min_disk_gb),
    ]
    # Docker memory is advisory (warn-only), not in pass/fail totals
    check_docker_memory(args.min_docker_gb)

    if candidate_files:
        checks.append(check_repos_exist(candidate_files))
    else:
        print("  [skip] repos/ check — no --candidates provided")

    if args.repos_csv:
        csv_ok = check_repos_csv(args.repos_csv)
        checks.append(csv_ok)
        if csv_ok and not args.skip_gh:
            # Estimate API calls: miner issues ~3 per repo (listing + per-PR stats).
            with open(args.repos_csv) as f:
                n_repos = sum(1 for _ in f) - 1  # minus header
            checks.append(check_gh_rate_limit(needed=3 * max(n_repos, 1)))

    if not args.skip_gh:
        checks.append(check_gh_cli())
    if not args.skip_harbor:
        checks.append(check_harbor())
    if not args.skip_harbor_lab:
        checks.append(check_harbor_lab())

    if args.check_endpoints:
        checks.append(check_claude_auth(model=profile_llm_step_model))
        # Test LiteLLM with summary model (Sonnet for fix-loop summaries)
        checks.append(check_litellm_endpoint(test_model="aws/anthropic/bedrock-claude-sonnet-4-6"))
        # Test Anthropic endpoint with the profile's haiku model (cheapest/fastest)
        checks.append(check_anthropic_endpoint(test_model=profile_haiku_model))

    print("=" * 60)
    passed = sum(1 for c in checks if c)
    total = len(checks)
    if passed == total:
        print(f"All {total} checks passed. Ready to launch.")
        return 0
    print(f"{passed}/{total} checks passed. Fix failures before launching a long run.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
