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
        "#!/usr/bin/env bash\n"
        "if [[ -n \"${FAKE_DOCKER_LOG:-}\" ]]; then\n"
        "    printf '%s\\n' \"$*\" >> \"${FAKE_DOCKER_LOG}\"\n"
        "fi\n"
        "exit 0\n"
    )
    docker.chmod(0o755)
    return docker


def _write_fake_curl(bin_dir: Path) -> Path:
    curl = bin_dir / "curl"
    curl.write_text(
        """
#!/usr/bin/env bash
set -euo pipefail
output=""
url=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -o) output="$2"; shift 2 ;;
        *) url="$1"; shift ;;
    esac
done
case "${url}" in
    */health) body='{"status":"ok"}' ;;
    */metrics) body='switchyard_total_requests 2' ;;
    */v1/stats) body='{"total_requests":2,"total_errors":0,"models":{"strong":{"calls":2}}}' ;;
    *) exit 22 ;;
esac
if [[ -n "${output}" ]]; then
    printf '%s\\n' "${body}" > "${output}"
else
    printf '%s\\n' "${body}"
fi
""".lstrip()
    )
    curl.chmod(0o755)
    return curl


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
    fake_server: bool = False,
):
    fake_bin = tmp_path / "default-bin"
    fake_bin.mkdir(exist_ok=True)
    _write_fake_harbor(fake_bin)
    if fake_server:
        _write_fake_docker(fake_bin)
        _write_fake_curl(fake_bin)
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


def _write_server_config(path: Path) -> Path:
    path.write_text(
        """
schema_version = 1

[llm_clients.test]
format = "openai_chat"
base_url = "https://example.test/v1"

[targets.strong]
id = "strong-model"
llm_client = "test"

[targets.weak]
id = "weak-model"
llm_client = "test"

[routes.random]
id = "tb-lite-random-routing"
type = "random"
targets = ["strong", "weak"]
weights = [1, 1]
seed = 444
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
                "agent_versions": {"codex": "0.144.5"},
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
    assert "switchyard-server" not in result.stdout
    assert "server_config:" not in result.stdout


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
    assert "--server-extra requires --server-config" in result.stderr

    result = _run_baseline(
        tmp_path,
        "--route-model",
        "tb-lite-random-routing",
        "--dry-run",
    )

    assert result.returncode != 0
    assert "--route-model requires --server-config" in result.stderr


def test_legacy_routing_profiles_are_rejected(tmp_path: Path) -> None:
    result = _run_baseline(
        tmp_path,
        "--routing-profiles",
        str(tmp_path / "routes.yaml"),
        "--model",
        "switchyard",
        "--dry-run",
    )

    assert result.returncode != 0
    assert "--routing-profiles is no longer supported" in result.stderr
    assert "use --server-config" in result.stderr


def test_dry_run_claude_code_opus_defaults_high_reasoning(tmp_path: Path) -> None:
    result = _run_baseline(
        tmp_path,
        "--server-config",
        str(REPO / "benchmark" / "server-configs" / "tb-lite-single-opus-4-7.toml"),
        "--route-model",
        "tb-lite-single-opus-4-7",
        "--agent",
        "claude-code",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    harbor = _line_argv(result.stdout, "HARBOR_CMD: ")
    assert _option_value(harbor, "--model") == "tb-lite-single-opus-4-7"
    assert "version=2.1.211" in _option_values(harbor, "--ak")
    assert "reasoning_effort=high" in _option_values(harbor, "--ak")
    assert "thinking=adaptive" in _option_values(harbor, "--ak")
    assert "reasoning:     high" in result.stdout
    assert "harbor_server: http://switchyard:4000" in result.stdout


def test_dry_run_codex_gpt_defaults_high_reasoning(tmp_path: Path) -> None:
    result = _run_baseline(
        tmp_path,
        "--server-config",
        str(REPO / "benchmark" / "server-configs" / "tb-lite-single-gpt-5-5.toml"),
        "--route-model",
        "tb-lite-single-gpt-5-5",
        "--agent",
        "codex",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    harbor = _line_argv(result.stdout, "HARBOR_CMD: ")
    assert _option_value(harbor, "--model") == "tb-lite-single-gpt-5-5"
    assert "version=0.144.5" in _option_values(harbor, "--ak")
    assert "reasoning_effort=high" in _option_values(harbor, "--ak")
    catalog_env = next(
        value
        for value in _option_values(harbor, "--ae")
        if value.startswith("CODEX_MODEL_CATALOG_JSON=")
    )
    assert catalog_env.endswith("/codex_model_catalog.json")
    assert "codex_catalog:" in result.stdout


def test_dry_run_explicit_empty_reasoning_omits_kwarg(tmp_path: Path) -> None:
    profile = _write_server_config(tmp_path / "routes.toml")

    result = _run_baseline(
        tmp_path,
        "--server-config",
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
    assert not any(value.startswith("thinking=") for value in _option_values(harbor, "--ak"))
    assert "reasoning:     unset" in result.stdout


def test_dry_run_explicit_reasoning_overrides_default(tmp_path: Path) -> None:
    profile = _write_server_config(tmp_path / "routes.toml")

    result = _run_baseline(
        tmp_path,
        "--server-config",
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
    assert "thinking=adaptive" in _option_values(harbor, "--ak")


def test_dry_run_explicit_claude_thinking_overrides_adaptive_default(tmp_path: Path) -> None:
    profile = _write_server_config(tmp_path / "routes.toml")

    result = _run_baseline(
        tmp_path,
        "--server-config",
        str(profile),
        "--route-model",
        "tb-lite-random-routing",
        "--agent",
        "claude-code",
        "--reasoning-effort",
        "high",
        "--harbor-extra",
        "--ak",
        "--harbor-extra",
        "thinking=disabled",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    harbor = _line_argv(result.stdout, "HARBOR_CMD: ")
    thinking = [value for value in _option_values(harbor, "--ak") if value.startswith("thinking=")]
    assert thinking == ["thinking=disabled"]


def test_harbor_path_is_required(tmp_path: Path) -> None:
    profile = _write_server_config(tmp_path / "routes.toml")

    result = _run_baseline(
        tmp_path,
        "--server-config",
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
    profile = _write_server_config(tmp_path / "routes.toml")
    fake_python = _write_fake_harbor_python(tmp_path / "unpatched", patched=False)

    result = _run_baseline(
        tmp_path,
        "--harbor-path",
        str(dataset),
        "--server-config",
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


def test_dry_run_server_config_uses_rust_switchyard_server(tmp_path: Path) -> None:
    profile = _write_server_config(tmp_path / "routes.toml")

    result = _run_baseline(
        tmp_path,
        "--server-config",
        str(profile),
        "--model",
        "tb-lite-random-routing",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    server = _line_argv(result.stdout, "SERVER_CMD: ")
    harbor = _line_argv(result.stdout, "HARBOR_CMD: ")
    assert server[0] == "switchyard-server"
    assert _option_value(server, "--config") == str(profile)
    assert _option_value(server, "--host") == "0.0.0.0"
    assert _option_value(server, "--port") == "4000"
    assert _option_value(harbor, "--model") == "openai/tb-lite-random-routing"
    assert "server_config:" in result.stdout
    assert "route_model:   tb-lite-random-routing" in result.stdout


def test_foreground_server_run_collects_rust_stats_endpoint(tmp_path: Path) -> None:
    profile = _write_server_config(tmp_path / "routes.toml")
    docker_log = tmp_path / "docker.log"

    result = _run_baseline(
        tmp_path,
        "--server-config",
        str(profile),
        "--route-model",
        "tb-lite-random-routing",
        "--foreground",
        fake_server=True,
        env={"FAKE_DOCKER_LOG": str(docker_log)},
    )

    assert result.returncode == 0, result.stderr
    run_dir = next((tmp_path / "out").glob("baseline-serve-serve-*"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    stats = json.loads((run_dir / "routing_stats_final.json").read_text())

    assert stats["total_requests"] == 2
    assert manifest["outcomes"]["routing_stats_json_status"] == "present"
    assert manifest["outcomes"]["server_metrics_prom_status"] == "present"
    assert "--env NVIDIA_API_KEY" in docker_log.read_text()


def test_dry_run_route_model_alias_uses_rust_switchyard_server(tmp_path: Path) -> None:
    profile = _write_server_config(tmp_path / "routes.toml")

    result = _run_baseline(
        tmp_path,
        "--server-config",
        str(profile),
        "--route-model",
        "tb-lite-random-routing",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    server = _line_argv(result.stdout, "SERVER_CMD: ")
    harbor = _line_argv(result.stdout, "HARBOR_CMD: ")
    assert server[0] == "switchyard-server"
    assert _option_value(server, "--config") == str(profile)
    assert _option_value(harbor, "--model") == "openai/tb-lite-random-routing"
    assert "route_model:   tb-lite-random-routing" in result.stdout


def test_dry_run_server_config_honors_explicit_harbor_model(tmp_path: Path) -> None:
    profile = _write_server_config(tmp_path / "routes.toml")

    result = _run_baseline(
        tmp_path,
        "--server-config",
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


def test_dry_run_harbor_extra(tmp_path: Path) -> None:
    profile = _write_server_config(tmp_path / "routes.toml")

    result = _run_baseline(
        tmp_path,
        "--server-config",
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
    profile = _write_server_config(tmp_path / "routes.toml")

    result = _run_baseline(
        tmp_path,
        "--harbor-path",
        str(dataset),
        "--server-config",
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
    assert _option_value(harbor, "--artifact") == "/etc/proxy-public/strip.jsonl"
    assert "version=0.144.5" in _option_values(harbor, "--ak")
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
    profile = _write_server_config(tmp_path / "routes.toml")

    result = _run_baseline(
        tmp_path,
        "--harbor-path",
        str(dataset),
        "--server-config",
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
    assert _option_value(harbor, "--artifact") == "/etc/proxy-public/strip.jsonl"
    assert "CODEX_DISABLE_WEB_SEARCH=1" not in _option_values(harbor, "--ae")
    assert not _option_values(harbor, "--ve")
    assert "book_mode:     open" in result.stdout
    assert "harbor_server: http://switchyard:4000" in result.stdout
    assert "proxy_mode:    open" in result.stdout
    assert "verifier_net: authenticated proxy egress" not in result.stdout


def test_dry_run_claude_closed_book_merges_disallowed_tools(tmp_path: Path) -> None:
    dataset = _write_closed_book_dataset(tmp_path / "dataset")
    profile = _write_server_config(tmp_path / "routes.toml")

    result = _run_baseline(
        tmp_path,
        "--harbor-path",
        str(dataset),
        "--server-config",
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
    assert "version=2.1.211" in _option_values(harbor, "--ak")
    assert "disallowed_tools=Bash,WebFetch,WebSearch" in _option_values(harbor, "--ak")


def test_dry_run_opencode_closed_book_disables_webfetch(tmp_path: Path) -> None:
    dataset = _write_closed_book_dataset(tmp_path / "dataset")
    profile = _write_server_config(tmp_path / "routes.toml")

    result = _run_baseline(
        tmp_path,
        "--harbor-path",
        str(dataset),
        "--server-config",
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
    assert "version=1.18.3" in _option_values(harbor, "--ak")
    assert "OPENCODE_DISABLE_WEBFETCH=1" in _option_values(harbor, "--ae")


def test_task_list_file_expands_to_include_task_name(tmp_path: Path) -> None:
    task_list = tmp_path / "tasks.txt"
    task_list.write_text("# comment\nalpha\n\nbeta  # inline\n")
    profile = _write_server_config(tmp_path / "routes.toml")

    result = _run_baseline(
        tmp_path,
        "--server-config",
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
    profile = _write_server_config(tmp_path / "routes.toml")

    result = _run_baseline(
        tmp_path,
        "--server-config",
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
