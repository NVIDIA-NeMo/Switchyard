# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "benchmark" / "run-baseline.sh"


def _write_fake_harbor(bin_dir: Path) -> Path:
    harbor = bin_dir / "harbor"
    harbor.write_text(
        """
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--version" ]]; then
    echo "harbor fake 1.0"
    exit 0
fi
if [[ "${1:-}" == "run" ]]; then
    shift
    jobs_dir=""
    job_name=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --jobs-dir) jobs_dir="$2"; shift 2 ;;
            --job-name) job_name="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    mkdir -p "${jobs_dir}/${job_name}"
    {
        printf 'OPENAI_API_KEY=%s\\n' "${OPENAI_API_KEY:-}"
        printf 'OPENAI_BASE_URL=%s\\n' "${OPENAI_BASE_URL:-}"
        printf 'ANTHROPIC_BASE_URL=%s\\n' "${ANTHROPIC_BASE_URL:-}"
        printf 'ANTHROPIC_AUTH_TOKEN=%s\\n' "${ANTHROPIC_AUTH_TOKEN:-}"
        printf 'ANTHROPIC_API_KEY=%s\\n' "${ANTHROPIC_API_KEY:-}"
        printf 'ANTHROPIC_MODEL=%s\\n' "${ANTHROPIC_MODEL:-}"
        printf 'ANTHROPIC_SMALL_FAST_MODEL=%s\\n' "${ANTHROPIC_SMALL_FAST_MODEL:-}"
        printf 'ANTHROPIC_CUSTOM_MODEL_OPTION=%s\\n' "${ANTHROPIC_CUSTOM_MODEL_OPTION:-}"
        printf 'CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=%s\\n' "${CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY:-}"
        printf 'CLOSED_BOOK_MODE=%s\\n' "${CLOSED_BOOK_MODE:-}"
        printf 'ALLOWED_HOSTS=%s\\n' "${ALLOWED_HOSTS:-}"
        printf 'CODEX_DISABLE_WEB_SEARCH=%s\\n' "${CODEX_DISABLE_WEB_SEARCH:-}"
        printf 'CODEX_MODEL_CATALOG_JSON=%s\\n' "${CODEX_MODEL_CATALOG_JSON:-}"
        printf 'OPENCODE_DISABLE_WEBFETCH=%s\\n' "${OPENCODE_DISABLE_WEBFETCH:-}"
    } > "${jobs_dir}/${job_name}/env.txt"
    printf '%s\\n' '{"stats":{"evals":{"fake":{"n_trials":1,"metrics":[{"mean":1.0}]}}}}' > "${jobs_dir}/${job_name}/result.json"
    exit 0
fi
echo "unexpected harbor args: $*" >&2
exit 2
""".lstrip()
    )
    harbor.chmod(0o755)
    return harbor


def _write_fake_docker(bin_dir: Path) -> Path:
    docker = bin_dir / "docker"
    docker.write_text(
        """
#!/usr/bin/env bash
set -euo pipefail
{
    printf '%q ' "$@"
    printf '\n'
} >> "${FAKE_DOCKER_LOG:?}"
if [[ "${1:-}" == "run" ]]; then
    echo "fake-container"
fi
""".lstrip()
    )
    docker.chmod(0o755)
    return docker


def _write_fake_harbor_patch(tmp_path: Path) -> Path:
    patch_file = tmp_path / "fake-harbor-agent-patches.diff"
    patch_file.write_text(
        """
--- a/harbor/agents/installed/base.py
+++ b/harbor/agents/installed/base.py
@@ -1 +1 @@
-unpatched
+patched
--- /dev/null
+++ b/harbor/switchyard_patch_id.txt
@@ -0,0 +1 @@
+switchyard-test-patch-v1
""".lstrip()
    )
    return patch_file


def _write_fake_harbor_python(tmp_path: Path, *, patched: bool = True) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    harbor_site = tmp_path / "fake-site-packages" / "harbor"
    base = harbor_site / "agents" / "installed" / "base.py"
    base.parent.mkdir(parents=True, exist_ok=True)
    base.write_text("patched\n" if patched else "unpatched\n")
    if patched:
        (harbor_site / "switchyard_patch_id.txt").write_text("switchyard-test-patch-v1\n")

    fake_python = tmp_path / "fake-harbor-python"
    fake_python.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
cat >/dev/null
printf '%s\\n' {shlex.quote(str(harbor_site))}
"""
    )
    fake_python.chmod(0o755)
    return fake_python


def _run_baseline(
    tmp_path: Path,
    *args: str,
    env: dict[str, str] | None = None,
    include_dataset: bool = True,
):
    fake_bin = tmp_path / "default-bin"
    fake_bin.mkdir(exist_ok=True)
    _write_fake_harbor(fake_bin)
    fake_patch = _write_fake_harbor_patch(tmp_path)
    fake_harbor_python = _write_fake_harbor_python(tmp_path)
    base_path = env.get("PATH", os.environ["PATH"]) if env else os.environ["PATH"]
    merged_env = {
        "PATH": f"{fake_bin}{os.pathsep}{base_path}",
        "HARBOR_BIN": str(fake_bin / "harbor"),
        "HARBOR_PYTHON": str(fake_harbor_python),
        "OPENROUTER_API_KEY": "or-test",  # pragma: allowlist secret
        "SWITCHYARD_HARBOR_PATCH_FILE": str(fake_patch),
    }
    if env:
        merged_env.update({key: value for key, value in env.items() if key != "PATH"})
    dataset_args: list[str] = []
    if include_dataset and "--harbor-path" not in args:
        dataset = tmp_path / "default-dataset"
        if not dataset.exists():
            _write_closed_book_dataset(dataset)
        dataset_args = ["--harbor-path", str(dataset)]
    return subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--output-dir",
            str(tmp_path / "out"),
            *dataset_args,
            *args,
        ],
        cwd=REPO,
        env=merged_env,
        text=True,
        capture_output=True,
        check=False,
    )


def _line_argv(stdout: str, prefix: str) -> list[str]:
    line = next(line for line in stdout.splitlines() if line.startswith(prefix))
    return shlex.split(line.removeprefix(prefix))


def _option_value(argv: list[str], option: str) -> str:
    return argv[argv.index(option) + 1]


def _option_values(argv: list[str], option: str) -> list[str]:
    return [argv[idx + 1] for idx, value in enumerate(argv[:-1]) if value == option]


def _write_route_profile(path: Path) -> Path:
    path.write_text(
        """
routes:
  tb-lite-random-routing:
    type: random_routing
    strong_probability: 0.5
    rng_seed: 444
    fallback_target_on_evict: strong
    strong:
      id: strong
      model: strong-model
    weak:
      id: weak
      model: weak-model
""".lstrip()
    )
    return path


def _write_closed_book_dataset(path: Path) -> Path:
    path.mkdir()
    (path / "switchyard_dataset_manifest.json").write_text(
        json.dumps(
            {
                "source_dataset": "openthoughts-tblite@2.0",
                "task_count": 1,
                "agent_versions": {"codex": "0.125.0"},
            }
        )
    )
    return path


def test_direct_mode_requires_model(tmp_path: Path) -> None:
    result = _run_baseline(tmp_path, "--dry-run")

    assert result.returncode != 0
    assert "--model is required when running direct upstream" in result.stderr


def test_direct_mode_defaults_to_openrouter_upstream_without_switchyard(tmp_path: Path) -> None:
    result = _run_baseline(
        tmp_path,
        "--model",
        "openai/gpt-5.2",
        "--agent",
        "codex",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    harbor = _line_argv(result.stdout, "HARBOR_CMD: ")
    assert _option_value(harbor, "--model") == "openai/gpt-5.2"
    agent_env = _option_values(harbor, "--ae")
    ca_env_prefixes = ("SSL_CERT_FILE=", "REQUESTS_CA_BUNDLE=", "CURL_CA_BUNDLE=", "GIT_SSL_CAINFO=")
    assert not any(
        value.startswith(ca_env_prefixes) for value in agent_env
    )
    assert "server_preset: direct" in result.stdout
    assert "server_mode:   direct" in result.stdout
    assert "upstream_url:  https://openrouter.ai/api/v1" in result.stdout
    assert "api_key_env:   OPENROUTER_API_KEY" in result.stdout
    assert "SERVER_CMD: <direct-upstream>" in result.stdout
    assert "switchyard serve" not in result.stdout
    assert "route_profile:" not in result.stdout


def test_direct_mode_uses_generic_upstream_key_when_set(tmp_path: Path) -> None:
    result = _run_baseline(
        tmp_path,
        "--model",
        "provider/model",
        "--agent",
        "codex",
        "--dry-run",
        env={
            "UPSTREAM_API_KEY": "upstream-test",  # pragma: allowlist secret
            "UPSTREAM_BASE_URL": "https://provider.example/v1",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "upstream_url:  https://provider.example/v1" in result.stdout
    assert "api_key_env:   UPSTREAM_API_KEY" in result.stdout


def test_direct_mode_requires_selected_api_key_env(tmp_path: Path) -> None:
    result = _run_baseline(
        tmp_path,
        "--model",
        "openai/gpt-5.2",
        "--dry-run",
        env={"OPENROUTER_API_KEY": ""},
    )

    assert result.returncode != 0
    assert "direct upstream requires $OPENROUTER_API_KEY to be set" in result.stderr


def test_direct_mode_rejects_server_only_options(tmp_path: Path) -> None:
    result = _run_baseline(
        tmp_path,
        "--model",
        "openai/gpt-5.2",
        "--server-extra",
        "--log-level=debug",
        "--dry-run",
    )

    assert result.returncode != 0
    assert "--server-extra requires --routing-profiles" in result.stderr

    result = _run_baseline(
        tmp_path,
        "--route-model",
        "tb-lite-random-routing",
        "--dry-run",
    )

    assert result.returncode != 0
    assert "--route-model requires --routing-profiles" in result.stderr


def test_dry_run_claude_code_opus_defaults_high_reasoning(tmp_path: Path) -> None:
    result = _run_baseline(
        tmp_path,
        "--routing-profiles",
        str(REPO / "benchmark" / "routing-profiles" / "tb-lite-single-opus-4-7.yaml"),
        "--route-model",
        "tb-lite-single-opus-4-7",
        "--agent",
        "claude-code",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    harbor = _line_argv(result.stdout, "HARBOR_CMD: ")
    assert _option_value(harbor, "--model") == "tb-lite-single-opus-4-7"
    assert "version=2.1.119" in _option_values(harbor, "--ak")
    assert "reasoning_effort=high" in _option_values(harbor, "--ak")
    assert "reasoning:     high" in result.stdout
    assert "harbor_server: http://switchyard:4000" in result.stdout


def test_dry_run_codex_gpt_defaults_high_reasoning(tmp_path: Path) -> None:
    result = _run_baseline(
        tmp_path,
        "--routing-profiles",
        str(REPO / "benchmark" / "routing-profiles" / "tb-lite-single-gpt-5-5.yaml"),
        "--route-model",
        "tb-lite-single-gpt-5-5",
        "--agent",
        "codex",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    harbor = _line_argv(result.stdout, "HARBOR_CMD: ")
    assert _option_value(harbor, "--model") == "tb-lite-single-gpt-5-5"
    assert "version=0.125.0" in _option_values(harbor, "--ak")
    assert "reasoning_effort=high" in _option_values(harbor, "--ak")
    catalog_env = next(
        value
        for value in _option_values(harbor, "--ae")
        if value.startswith("CODEX_MODEL_CATALOG_JSON=")
    )
    assert catalog_env.endswith("/codex_model_catalog.json")
    assert "codex_catalog:" in result.stdout


def test_dry_run_explicit_empty_reasoning_omits_kwarg(tmp_path: Path) -> None:
    profile = _write_route_profile(tmp_path / "routes.yaml")

    result = _run_baseline(
        tmp_path,
        "--routing-profiles",
        str(profile),
        "--route-model",
        "tb-lite-random-routing",
        "--agent",
        "claude-code",
        "--reasoning-effort",
        "",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    harbor = _line_argv(result.stdout, "HARBOR_CMD: ")
    assert not any(value.startswith("reasoning_effort=") for value in _option_values(harbor, "--ak"))
    assert "reasoning:     unset" in result.stdout


def test_dry_run_explicit_reasoning_overrides_default(tmp_path: Path) -> None:
    profile = _write_route_profile(tmp_path / "routes.yaml")

    result = _run_baseline(
        tmp_path,
        "--routing-profiles",
        str(profile),
        "--route-model",
        "tb-lite-random-routing",
        "--agent",
        "claude-code",
        "--reasoning-effort",
        "medium",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    harbor = _line_argv(result.stdout, "HARBOR_CMD: ")
    assert "reasoning_effort=medium" in _option_values(harbor, "--ak")


def test_harbor_path_is_required(tmp_path: Path) -> None:
    profile = _write_route_profile(tmp_path / "routes.yaml")

    result = _run_baseline(
        tmp_path,
        "--routing-profiles",
        str(profile),
        "--route-model",
        "tb-lite-random-routing",
        "--dry-run",
        include_dataset=False,
    )

    assert result.returncode != 0
    assert "--harbor-path is required" in result.stderr


def test_closed_book_dataset_rejects_unpatched_harbor(tmp_path: Path) -> None:
    dataset = _write_closed_book_dataset(tmp_path / "dataset")
    profile = _write_route_profile(tmp_path / "routes.yaml")
    fake_python = _write_fake_harbor_python(tmp_path / "unpatched", patched=False)

    result = _run_baseline(
        tmp_path,
        "--harbor-path",
        str(dataset),
        "--routing-profiles",
        str(profile),
        "--route-model",
        "tb-lite-random-routing",
        "--harbor-python",
        str(fake_python),
        "--dry-run",
        include_dataset=False,
    )

    assert result.returncode != 0
    assert "The current Harbor patch is not applied cleanly" in result.stderr
    assert "run-baseline.sh requires patched Harbor" in result.stderr
    assert "Verification used:" in result.stderr
    assert "patch -p1 <" in result.stderr


def test_dry_run_routing_profile_uses_switchyard_serve(tmp_path: Path) -> None:
    profile = _write_route_profile(tmp_path / "routes.yaml")

    result = _run_baseline(
        tmp_path,
        "--routing-profiles",
        str(profile),
        "--model",
        "tb-lite-random-routing",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    server = _line_argv(result.stdout, "SERVER_CMD: ")
    harbor = _line_argv(result.stdout, "HARBOR_CMD: ")
    # New form: uv run --no-sync switchyard --routing-profiles FILE -- serve ...
    assert server[:4] == ["uv", "run", "--no-sync", "switchyard"]
    assert _option_value(server, "--routing-profiles") == str(profile)
    assert "serve" in server
    assert _option_value(server, "--host") == "0.0.0.0"
    assert _option_value(server, "--port") == "4000"
    assert _option_value(harbor, "--model") == "openai/tb-lite-random-routing"
    assert "route_profile:" in result.stdout
    assert "route_model:   tb-lite-random-routing" in result.stdout


def test_dry_run_route_model_alias_uses_switchyard_serve(tmp_path: Path) -> None:
    profile = _write_route_profile(tmp_path / "routes.yaml")

    result = _run_baseline(
        tmp_path,
        "--routing-profiles",
        str(profile),
        "--route-model",
        "tb-lite-random-routing",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    server = _line_argv(result.stdout, "SERVER_CMD: ")
    harbor = _line_argv(result.stdout, "HARBOR_CMD: ")
    assert server[:4] == ["uv", "run", "--no-sync", "switchyard"]
    assert "serve" in server
    assert _option_value(harbor, "--model") == "openai/tb-lite-random-routing"
    assert "route_model:   tb-lite-random-routing" in result.stdout


def test_dry_run_routing_profile_honors_explicit_harbor_model(tmp_path: Path) -> None:
    profile = _write_route_profile(tmp_path / "routes.yaml")

    result = _run_baseline(
        tmp_path,
        "--routing-profiles",
        str(profile),
        "--route-model",
        "tb-lite-random-routing",
        "--harbor-model",
        "openai/custom-route-label",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    harbor = _line_argv(result.stdout, "HARBOR_CMD: ")
    assert _option_value(harbor, "--model") == "openai/custom-route-label"


def test_checked_in_routing_profiles_load(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")

    import switchyard.cli.route_bundle as route_bundle

    profiles = {
        "gpt-5-5-deepseek-gemini.yaml": "switchyard",
        "gpt-5-5-kimi-gemini.yaml": "switchyard",
        "tb-lite-llm-classifier-opus-deepseek-gemini.yaml": "switchyard",
        "tb-lite-llm-classifier-opus-kimi-gemini.yaml": "switchyard",
        "tb-lite-single-opus-4-7.yaml": "tb-lite-single-opus-4-7",
        "tb-lite-single-gpt-5-5.yaml": "tb-lite-single-gpt-5-5",
    }
    monkeypatch.setattr(route_bundle, "fetch_model_ids", lambda _base_url, _api_key: [])
    for filename, model in profiles.items():
        table = route_bundle.load_route_bundle_table(
            REPO / "benchmark" / "routing-profiles" / filename
        )
        assert model in table.registered_models()


def test_dry_run_harbor_extra(tmp_path: Path) -> None:
    profile = _write_route_profile(tmp_path / "routes.yaml")

    result = _run_baseline(
        tmp_path,
        "--routing-profiles",
        str(profile),
        "--route-model",
        "tb-lite-random-routing",
        "--harbor-extra",
        "--include-task-name=hello",
        "--harbor-extra",
        "--no-upload",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    harbor = _line_argv(result.stdout, "HARBOR_CMD: ")
    assert "--dataset" not in harbor
    assert "--include-task-name=hello" in harbor
    assert "--no-upload" in harbor


def test_dry_run_harbor_path_uses_local_dataset_and_closed_book_artifact(
    tmp_path: Path,
) -> None:
    dataset = _write_closed_book_dataset(tmp_path / "dataset")
    profile = _write_route_profile(tmp_path / "routes.yaml")

    result = _run_baseline(
        tmp_path,
        "--harbor-path",
        str(dataset),
        "--routing-profiles",
        str(profile),
        "--route-model",
        "tb-lite-random-routing",
        "--agent",
        "codex",
        "--dry-run",
        include_dataset=False,
    )

    assert result.returncode == 0, result.stderr
    harbor = _line_argv(result.stdout, "HARBOR_CMD: ")
    assert "--dataset" not in harbor
    assert _option_value(harbor, "--path") == str(dataset)
    assert _option_values(harbor, "--artifact") == [
        "/etc/proxy-ca/strip.jsonl",
        "/etc/proxy-ca/request_map.jsonl",
    ]
    assert "version=0.125.0" in _option_values(harbor, "--ak")
    assert "CODEX_DISABLE_WEB_SEARCH=1" in _option_values(harbor, "--ae")
    catalog_env = next(
        value
        for value in _option_values(harbor, "--ae")
        if value.startswith("CODEX_MODEL_CATALOG_JSON=")
    )
    assert catalog_env.endswith("/codex_model_catalog.json")
    verifier_env = _option_values(harbor, "--ve")
    assert "HTTP_PROXY=${SWITCHYARD_VERIFIER_HTTP_PROXY}" in verifier_env
    assert "HTTPS_PROXY=${SWITCHYARD_VERIFIER_HTTP_PROXY}" in verifier_env
    assert "NO_PROXY=localhost,127.0.0.1,proxy" in verifier_env
    assert "book_mode:     closed" in result.stdout
    assert "harbor_patch:  verified" in result.stdout
    assert "harbor_server: http://switchyard:4000" in result.stdout
    assert "docker_network:" in result.stdout
    assert "verifier_net: authenticated proxy egress" in result.stdout
    assert f"harbor_path:   {dataset}" in result.stdout


def test_dry_run_open_book_uses_proxy_topology_without_tool_disables(tmp_path: Path) -> None:
    dataset = _write_closed_book_dataset(tmp_path / "dataset")
    profile = _write_route_profile(tmp_path / "routes.yaml")

    result = _run_baseline(
        tmp_path,
        "--harbor-path",
        str(dataset),
        "--routing-profiles",
        str(profile),
        "--route-model",
        "tb-lite-random-routing",
        "--agent",
        "codex",
        "--book-mode",
        "open",
        "--dry-run",
        include_dataset=False,
    )

    assert result.returncode == 0, result.stderr
    harbor = _line_argv(result.stdout, "HARBOR_CMD: ")
    assert _option_value(harbor, "--path") == str(dataset)
    assert _option_values(harbor, "--artifact") == [
        "/etc/proxy-ca/strip.jsonl",
        "/etc/proxy-ca/request_map.jsonl",
    ]
    assert "CODEX_DISABLE_WEB_SEARCH=1" not in _option_values(harbor, "--ae")
    assert not _option_values(harbor, "--ve")
    assert "book_mode:     open" in result.stdout
    assert "harbor_server: http://switchyard:4000" in result.stdout
    assert "proxy_mode:    open" in result.stdout
    assert "verifier_net: authenticated proxy egress" not in result.stdout


def test_foreground_switchyard_mounts_run_local_routing_trace(tmp_path: Path) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    _write_fake_docker(tools)
    docker_log = tmp_path / "docker.log"
    profile = _write_route_profile(tmp_path / "routes.yaml")

    result = _run_baseline(
        tmp_path,
        "--routing-profiles",
        str(profile),
        "--route-model",
        "tb-lite-random-routing",
        "--foreground",
        "--skip-health-check",
        env={
            "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}",
            "FAKE_DOCKER_LOG": str(docker_log),
            "SWITCHYARD_CLOSED_BOOK_PREFLIGHT": "0",
            "SWITCHYARD_DOCKER_BUILD": "0",
        },
    )

    assert result.returncode == 0, result.stderr
    calls = [shlex.split(line) for line in docker_log.read_text().splitlines()]
    server_run = next(call for call in calls if call[0] == "run" and "--name" in call)
    run_dir = next((tmp_path / "out").iterdir())
    trace_dir = run_dir / "routing-traces"
    trace_jsonl = trace_dir / "routing_trace.jsonl"
    assert f"{trace_dir}:/var/lib/switchyard-traces" in _option_values(server_run, "-v")
    server_env = _option_values(server_run, "--env")
    assert (
        "SWITCHYARD_ROUTING_TRACE_JSONL=/var/lib/switchyard-traces/routing_trace.jsonl"
        in server_env
    )
    assert "SWITCHYARD_ROUTING_TRACE_CAPTURE_CONTENT=1" in server_env
    assert trace_jsonl.is_file()
    assert (trace_dir / "routing_trace_joined.jsonl").is_file()
    assert (trace_dir / "routing_trace_completeness.json").is_file()
    assert trace_dir.stat().st_mode & 0o777 == 0o700
    assert trace_jsonl.stat().st_mode & 0o777 == 0o600
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["routing_trace"]["status"] == "incomplete"
    assert manifest["routing_trace"]["capture_content"] is True
    assert manifest["routing_trace"]["raw_jsonl_status"] == "present"
    assert manifest["routing_trace"]["joined_jsonl_status"] == "present"
    assert manifest["routing_trace"]["report_json_status"] == "present"


def test_dry_run_claude_closed_book_merges_disallowed_tools(tmp_path: Path) -> None:
    dataset = _write_closed_book_dataset(tmp_path / "dataset")
    profile = _write_route_profile(tmp_path / "routes.yaml")

    result = _run_baseline(
        tmp_path,
        "--harbor-path",
        str(dataset),
        "--routing-profiles",
        str(profile),
        "--route-model",
        "tb-lite-random-routing",
        "--agent",
        "claude-code",
        "--harbor-extra",
        "--ak",
        "--harbor-extra",
        "disallowed_tools=Bash",
        "--dry-run",
        include_dataset=False,
    )

    assert result.returncode == 0, result.stderr
    harbor = _line_argv(result.stdout, "HARBOR_CMD: ")
    assert "version=2.1.119" in _option_values(harbor, "--ak")
    assert "disallowed_tools=Bash,WebFetch,WebSearch" in _option_values(harbor, "--ak")


def test_dry_run_opencode_closed_book_disables_webfetch(tmp_path: Path) -> None:
    dataset = _write_closed_book_dataset(tmp_path / "dataset")
    profile = _write_route_profile(tmp_path / "routes.yaml")

    result = _run_baseline(
        tmp_path,
        "--harbor-path",
        str(dataset),
        "--routing-profiles",
        str(profile),
        "--route-model",
        "tb-lite-random-routing",
        "--agent",
        "opencode",
        "--dry-run",
        include_dataset=False,
    )

    assert result.returncode == 0, result.stderr
    harbor = _line_argv(result.stdout, "HARBOR_CMD: ")
    assert _option_value(harbor, "--model") == "nvidia/tb-lite-random-routing"
    assert "version=1.14.31" in _option_values(harbor, "--ak")
    assert "OPENCODE_DISABLE_WEBFETCH=1" in _option_values(harbor, "--ae")


def test_task_list_file_expands_to_include_task_name(tmp_path: Path) -> None:
    task_list = tmp_path / "tasks.txt"
    task_list.write_text("# comment\nalpha\n\nbeta  # inline\n")
    profile = _write_route_profile(tmp_path / "routes.yaml")

    result = _run_baseline(
        tmp_path,
        "--routing-profiles",
        str(profile),
        "--route-model",
        "tb-lite-random-routing",
        "--task-list-file",
        str(task_list),
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    harbor = _line_argv(result.stdout, "HARBOR_CMD: ")
    assert _option_values(harbor, "--include-task-name") == ["alpha", "beta"]


def test_dry_run_uses_harbor_bin_override(tmp_path: Path) -> None:
    override_bin = tmp_path / "override-bin"
    override_bin.mkdir()
    expected_harbor = _write_fake_harbor(override_bin)
    profile = _write_route_profile(tmp_path / "routes.yaml")

    result = _run_baseline(
        tmp_path,
        "--routing-profiles",
        str(profile),
        "--route-model",
        "tb-lite-random-routing",
        "--dry-run",
        env={"HARBOR_BIN": str(expected_harbor)},
    )

    assert result.returncode == 0, result.stderr
    harbor = _line_argv(result.stdout, "HARBOR_CMD: ")
    assert harbor[0] == str(expected_harbor)
    assert f"harbor_cmd:    {expected_harbor} (override)" in result.stdout
