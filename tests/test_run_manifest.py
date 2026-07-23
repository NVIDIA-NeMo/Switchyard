# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "benchmark" / "run_manifest.py"


def _load_manifest_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("switchyard_benchmark_run_manifest", MANIFEST)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_write_manifest_schema_and_git_fields(tmp_path: Path) -> None:
    module = _load_manifest_module()
    out = tmp_path / "run_manifest.json"

    module.write_manifest(out, harbor={"dataset": "openthoughts-tblite@2.0"})

    manifest = json.loads(out.read_text())
    assert manifest["run"]["schema_version"] == module.SCHEMA_VERSION
    assert manifest["run"]["git_tree_kind"] in {"git-clean", "git-dirty", "not-git"}
    assert manifest["run"]["git_dirty"] in {True, False, None}
    assert manifest["harbor"]["dataset"] == "openthoughts-tblite@2.0"


def test_dataset_fingerprint_includes_task_list_digest(tmp_path: Path) -> None:
    module = _load_manifest_module()
    task_list = tmp_path / "tasks.txt"
    task_list.write_text("alpha\n")

    first = module.dataset_fingerprint("openthoughts-tblite@2.0", task_list)
    task_list.write_text("alpha\nbeta\n")
    second = module.dataset_fingerprint("openthoughts-tblite@2.0", task_list)

    assert first.startswith("sha256:")
    assert second.startswith("sha256:")
    assert first != second


def test_dataset_fingerprint_includes_local_path_digest(tmp_path: Path) -> None:
    module = _load_manifest_module()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "task.toml").write_text("[environment]\n")

    first = module.dataset_fingerprint(None, harbor_path=dataset)
    (dataset / "extra.txt").write_text("changed\n")
    second = module.dataset_fingerprint(None, harbor_path=dataset)

    assert first.startswith("sha256:")
    assert second.startswith("sha256:")
    assert first != second


def test_cli_write_applies_extra_fields(tmp_path: Path) -> None:
    module = _load_manifest_module()
    out = tmp_path / "run_manifest.json"
    run_dir = tmp_path / "run"
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    harbor_bin = tmp_path / "harbor"
    harbor_bin.write_text("#!/usr/bin/env bash\necho 'harbor test 9.9'\n")
    harbor_bin.chmod(0o755)

    rc = module._cli_main(
        [
            "write",
            "--output",
            str(out),
            "--server-preset",
            "serve",
            "--harbor-command-json",
            json.dumps([str(harbor_bin)]),
            "--harbor-path",
            str(dataset),
            "--agent",
            "terminus-2",
            "--harbor-model",
            "openai/gpt-5.2",
            "--n-concurrent",
            "1",
            "--max-retries",
            "0",
            "--agent-timeout-multiplier",
            "1.0",
            "--run-dir",
            str(run_dir),
            "--log-path",
            str(run_dir / "run.log"),
            "--harbor-result-json",
            str(run_dir / "harbor_result.json"),
            "--routing-stats-json",
            str(run_dir / "routing_stats_final.json"),
            "--extra",
            'server.note="external"',
        ]
    )

    assert rc == 0
    manifest = json.loads(out.read_text())
    assert manifest["run"]["harbor_command"] == [str(harbor_bin)]
    assert manifest["run"]["harbor_version"] == "harbor test 9.9"
    assert manifest["server"]["note"] == "external"
    assert "benchmark" not in manifest["harbor"]
    assert manifest["harbor"]["path"] == str(dataset)
    assert manifest["harbor"]["dataset_fingerprint"].startswith("sha256:")


def test_cli_write_records_routing_profile_digest(tmp_path: Path) -> None:
    module = _load_manifest_module()
    out = tmp_path / "run_manifest.json"
    run_dir = tmp_path / "run"
    profile = tmp_path / "routes.yaml"
    profile.write_text("routes:\n  tb-lite-random-routing:\n    type: noop\n")
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    rc = module._cli_main(
        [
            "write",
            "--output",
            str(out),
            "--server-preset",
            "serve",
            "--routing-profiles",
            str(profile),
            "--route-model",
            "tb-lite-random-routing",
            "--classifier-prompts-json",
            '{"tb-lite-random-routing":{"classifier_prompt_sha256":"abc"}}',
            "--harbor-path",
            str(dataset),
            "--agent",
            "terminus-2",
            "--harbor-model",
            "tb-lite-random-routing",
            "--n-concurrent",
            "1",
            "--max-retries",
            "0",
            "--agent-timeout-multiplier",
            "1.0",
            "--run-dir",
            str(run_dir),
            "--log-path",
            str(run_dir / "run.log"),
            "--harbor-result-json",
            str(run_dir / "harbor_result.json"),
            "--routing-stats-json",
            str(run_dir / "routing_stats_final.json"),
        ]
    )

    assert rc == 0
    manifest = json.loads(out.read_text())
    assert manifest["server"]["routing_profiles"] == str(profile)
    assert manifest["server"]["routing_profiles_digest"] == module.path_digest(profile)
    snapshot = run_dir / "routing_profiles" / profile.name
    assert manifest["server"]["routing_profiles_snapshot"] == str(snapshot)
    assert manifest["server"]["routing_profiles_snapshot_digest"] == module.path_digest(snapshot)
    assert snapshot.read_bytes() == profile.read_bytes()
    assert manifest["server"]["route_model"] == "tb-lite-random-routing"
    assert manifest["server"]["classifier_prompts"] == {
        "tb-lite-random-routing": {"classifier_prompt_sha256": "abc"}
    }


def test_cli_write_rejects_non_object_classifier_prompts(tmp_path: Path) -> None:
    module = _load_manifest_module()
    out = tmp_path / "run_manifest.json"
    run_dir = tmp_path / "run"

    rc = module._cli_main(
        [
            "write",
            "--output",
            str(out),
            "--server-preset",
            "serve",
            "--classifier-prompts-json",
            "[]",
            "--agent",
            "terminus-2",
            "--harbor-model",
            "tb-lite-random-routing",
            "--n-concurrent",
            "1",
            "--max-retries",
            "0",
            "--agent-timeout-multiplier",
            "1.0",
            "--run-dir",
            str(run_dir),
            "--log-path",
            str(run_dir / "run.log"),
            "--harbor-result-json",
            str(run_dir / "harbor_result.json"),
            "--routing-stats-json",
            str(run_dir / "routing_stats_final.json"),
        ]
    )

    assert rc == 2
    assert not out.exists()


def test_cli_write_records_direct_upstream_mode_without_routing_stats(tmp_path: Path) -> None:
    module = _load_manifest_module()
    out = tmp_path / "run_manifest.json"
    run_dir = tmp_path / "run"
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    rc = module._cli_main(
        [
            "write",
            "--output",
            str(out),
            "--server-preset",
            "direct",
            "--server-mode",
            "direct",
            "--server-config-json",
            '{"mode":"direct","upstream_api_key_env":"NVIDIA_API_KEY"}',
            "--harbor-base-url",
            "https://inference-api.nvidia.com/v1",
            "--upstream-base-url",
            "https://inference-api.nvidia.com/v1",
            "--upstream-api-key-env",
            "NVIDIA_API_KEY",
            "--harbor-path",
            str(dataset),
            "--agent",
            "codex",
            "--harbor-model",
            "openai/gpt-5.2",
            "--n-concurrent",
            "1",
            "--max-retries",
            "0",
            "--agent-timeout-multiplier",
            "1.0",
            "--run-dir",
            str(run_dir),
            "--log-path",
            str(run_dir / "run.log"),
            "--harbor-result-json",
            str(run_dir / "harbor_result.json"),
            "--routing-stats-json",
            str(run_dir / "routing_stats_final.json"),
            "--routing-stats-status",
            "not-requested",
        ]
    )

    assert rc == 0
    manifest = json.loads(out.read_text())
    assert manifest["server"]["preset"] == "direct"
    assert manifest["server"]["mode"] == "direct"
    assert manifest["server"]["upstream_base_url"] == "https://inference-api.nvidia.com/v1"
    assert manifest["server"]["upstream_api_key_env"] == "NVIDIA_API_KEY"
    assert manifest["server"]["routing_profiles"] is None
    assert manifest["outcomes"]["routing_stats_json_status"] == "not-requested"

    rc = module.finalize_manifest(out, harbor_rc=0, harbor_job_dir=run_dir / "jobs" / "job")

    assert rc == 0
    manifest = json.loads(out.read_text())
    assert manifest["outcomes"]["routing_stats_json_status"] == "not-requested"


def test_cli_write_records_closed_book_local_dataset_snapshot(tmp_path: Path) -> None:
    module = _load_manifest_module()
    out = tmp_path / "run_manifest.json"
    run_dir = tmp_path / "run"
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    source_manifest = dataset / "switchyard_dataset_manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "source_dataset": "openthoughts-tblite@2.0",
                "agent_versions": {"codex": "0.144.5"},
            }
        )
    )

    rc = module._cli_main(
        [
            "write",
            "--output",
            str(out),
            "--server-preset",
            "serve",
            "--harbor-path",
            str(dataset),
            "--agent",
            "codex",
            "--harbor-model",
            "openai/gpt-5.5",
            "--n-concurrent",
            "1",
            "--max-retries",
            "0",
            "--agent-timeout-multiplier",
            "1.0",
            "--closed-book-mode",
            "closed",
            "--closed-book-gateway-enforced",
            "1",
            "--closed-book-hosted-tools-disabled",
            "1",
            "--closed-book-proxy-strip-artifact",
            "/etc/proxy-public/strip.jsonl",
            "--agent-versions-json",
            '{"codex":"0.144.5"}',
            "--run-dir",
            str(run_dir),
            "--log-path",
            str(run_dir / "run.log"),
            "--harbor-result-json",
            str(run_dir / "harbor_result.json"),
            "--routing-stats-json",
            str(run_dir / "routing_stats_final.json"),
        ]
    )

    assert rc == 0
    manifest = json.loads(out.read_text())
    assert manifest["harbor"]["path"] == str(dataset)
    assert manifest["harbor"]["path_digest"] == module.path_digest(dataset)
    assert manifest["closed_book"]["mode"] == "closed"
    assert manifest["closed_book"]["gateway_enforced"] is True
    assert manifest["closed_book"]["hosted_tools_disabled"] is True
    assert manifest["closed_book"]["verifier_egress"] == "open-via-authenticated-proxy"
    assert manifest["closed_book"]["agent_versions"] == {"codex": "0.144.5"}
    snapshot = run_dir / "dataset" / "switchyard_dataset_manifest.json"
    assert manifest["closed_book"]["dataset_manifest_snapshot"] == str(snapshot)
    assert snapshot.read_text() == source_manifest.read_text()


def test_cli_write_records_open_book_proxy_mode(tmp_path: Path) -> None:
    module = _load_manifest_module()
    out = tmp_path / "run_manifest.json"
    run_dir = tmp_path / "run"
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "switchyard_dataset_manifest.json").write_text("{}\n")

    rc = module._cli_main(
        [
            "write",
            "--output",
            str(out),
            "--server-preset",
            "serve",
            "--harbor-path",
            str(dataset),
            "--agent",
            "codex",
            "--harbor-model",
            "tb-lite-single-gpt-5-5",
            "--n-concurrent",
            "1",
            "--max-retries",
            "0",
            "--agent-timeout-multiplier",
            "1.0",
            "--closed-book-mode",
            "open",
            "--closed-book-gateway-enforced",
            "1",
            "--closed-book-hosted-tools-disabled",
            "0",
            "--closed-book-proxy-strip-artifact",
            "/etc/proxy-public/strip.jsonl",
            "--run-dir",
            str(run_dir),
            "--log-path",
            str(run_dir / "run.log"),
            "--harbor-result-json",
            str(run_dir / "harbor_result.json"),
            "--routing-stats-json",
            str(run_dir / "routing_stats_final.json"),
        ]
    )

    assert rc == 0
    manifest = json.loads(out.read_text())
    assert manifest["closed_book"]["mode"] == "open"
    assert manifest["closed_book"]["gateway_enforced"] is True
    assert manifest["closed_book"]["hosted_tools_disabled"] is False
    assert manifest["closed_book"]["proxy_strip_log_status"] == "predicted"
    assert manifest["closed_book"]["verifier_egress"] == "open-via-proxy"


def test_finalize_copies_harbor_result_and_marks_stats(tmp_path: Path) -> None:
    module = _load_manifest_module()
    out = tmp_path / "run_manifest.json"
    run_dir = tmp_path / "run"
    job_dir = run_dir / "jobs" / "job"
    job_dir.mkdir(parents=True)
    (job_dir / "result.json").write_text('{"stats":{"evals":{}}}\n')
    stats = run_dir / "routing_stats_final.json"
    stats.parent.mkdir(parents=True, exist_ok=True)
    stats.write_text('{"total_requests":1}\n')

    module.write_manifest(
        out,
        outcomes={
            "harbor_result_json": str(run_dir / "harbor_result.json"),
            "harbor_result_json_status": "predicted",
            "routing_stats_json": str(stats),
            "routing_stats_json_status": "predicted",
            "harbor_rc": None,
        },
    )

    rc = module.finalize_manifest(out, harbor_rc=0, harbor_job_dir=job_dir, routing_stats=stats)

    assert rc == 0
    manifest = json.loads(out.read_text())
    assert manifest["outcomes"]["harbor_rc"] == 0
    assert manifest["outcomes"]["harbor_result_json_status"] == "present"
    assert manifest["outcomes"]["routing_stats_json_status"] == "present"
    assert (run_dir / "harbor_result.json").is_file()


def test_finalize_copies_proxy_strip_log_artifact(tmp_path: Path) -> None:
    module = _load_manifest_module()
    out = tmp_path / "run_manifest.json"
    run_dir = tmp_path / "run"
    job_dir = run_dir / "jobs" / "job" / "task" / "artifacts"
    job_dir.mkdir(parents=True)
    (job_dir / "strip.jsonl").write_text('{"removed":["web_search"]}\n')

    module.write_manifest(
        out,
        closed_book={
            "mode": "closed",
            "proxy_strip_log": str(run_dir / "proxy_strip_log.jsonl"),
            "proxy_strip_log_status": "predicted",
        },
        outcomes={
            "harbor_result_json": str(run_dir / "harbor_result.json"),
            "harbor_result_json_status": "predicted",
            "routing_stats_json": str(run_dir / "routing_stats_final.json"),
            "routing_stats_json_status": "predicted",
            "harbor_rc": None,
        },
    )

    rc = module.finalize_manifest(out, harbor_rc=0, harbor_job_dir=run_dir / "jobs" / "job")

    assert rc == 0
    manifest = json.loads(out.read_text())
    assert manifest["closed_book"]["proxy_strip_log_status"] == "present"
    assert (run_dir / "proxy_strip_log.jsonl").read_text() == '{"removed":["web_search"]}\n'


def _record(task: str, ts: str, model: str, tier: str, *, trial: str = "t", session: str = "s",
            **tokens: int) -> str:
    base = {"prompt_tokens": 0, "cached_tokens": 0, "cache_creation_tokens": 0,
            "completion_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0}
    base.update(tokens)
    return json.dumps({"task": task, "ts": ts, "trial_id": trial, "session_id": session,
                       "model": model, "tier": tier, **base})


def test_summarize_routing_log_groups_by_task_and_model(tmp_path: Path) -> None:
    module = _load_manifest_module()
    log = tmp_path / "routing_requests.jsonl"
    log.write_text(
        "\n".join([
            _record("task-a", "2026-01-01T00:00:01Z", "opus", "strong",
                    prompt_tokens=10, cached_tokens=6, completion_tokens=2,
                    reasoning_tokens=1, total_tokens=12),
            _record("task-a", "2026-01-01T00:00:02Z", "kimi", "weak",
                    prompt_tokens=4, completion_tokens=1, total_tokens=5),
            "not json",
            json.dumps(["not", "a", "dict"]),
        ]) + "\n"
    )

    summary = module.summarize_routing_log(log)

    assert summary["total_requests"] == 2
    task_a = summary["tasks"]["task-a"]
    assert task_a["requests"] == 2
    assert task_a["n_retries"] == 0
    assert task_a["retries"]["calls"] == 0
    assert task_a["final"]["calls"] == 2
    assert task_a["final"]["totals"] == {
        "prompt_tokens": 14, "cached_tokens": 6, "cache_creation_tokens": 0,
        "completion_tokens": 3, "reasoning_tokens": 1, "total_tokens": 17,
    }
    assert [(b["model"], b["tier"], b["calls"]) for b in task_a["final"]["models"]] == [
        ("kimi", "weak", 1), ("opus", "strong", 1),
    ]


def test_summarize_routing_log_nets_out_retried_attempt(tmp_path: Path) -> None:
    module = _load_manifest_module()
    log = tmp_path / "routing_requests.jsonl"
    # Same trial, two attempts: the earlier session was retried away.
    log.write_text(
        "\n".join([
            _record("task-a", "2026-01-01T00:00:01Z", "opus", "strong",
                    trial="trial-1", session="retry-sess",
                    prompt_tokens=100, completion_tokens=10, total_tokens=110),
            _record("task-a", "2026-01-01T00:05:01Z", "opus", "strong",
                    trial="trial-1", session="final-sess",
                    prompt_tokens=20, completion_tokens=3, total_tokens=23),
        ]) + "\n"
    )

    task_a = module.summarize_routing_log(log)["tasks"]["task-a"]

    assert task_a["n_retries"] == 1
    assert task_a["final"]["totals"]["total_tokens"] == 23
    assert task_a["retries"]["totals"]["total_tokens"] == 110
    assert task_a["final"]["calls"] == 1
    assert task_a["retries"]["calls"] == 1


def test_summarize_routing_log_keeps_all_concurrent_trials(tmp_path: Path) -> None:
    module = _load_manifest_module()
    log = tmp_path / "routing_requests.jsonl"
    # Two trials of the same task (k>1). Neither retried, so both are final even
    # though their calls interleave in time.
    log.write_text(
        "\n".join([
            _record("task-a", "2026-01-01T00:00:01Z", "opus", "strong",
                    trial="trial-1", session="s1", total_tokens=10),
            _record("task-a", "2026-01-01T00:00:02Z", "opus", "strong",
                    trial="trial-2", session="s2", total_tokens=20),
        ]) + "\n"
    )

    task_a = module.summarize_routing_log(log)["tasks"]["task-a"]

    assert task_a["n_retries"] == 0
    assert task_a["final"]["calls"] == 2
    assert task_a["final"]["totals"]["total_tokens"] == 30
    assert task_a["retries"]["calls"] == 0


def test_summarize_routing_log_untagged_records_all_final(tmp_path: Path) -> None:
    module = _load_manifest_module()
    log = tmp_path / "routing_requests.jsonl"
    # No trial_id -> no netting, everything counts as final.
    log.write_text(json.dumps({
        "task": "task-a", "ts": "2026-01-01T00:00:01Z", "session_id": "s1",
        "model": "opus", "tier": "strong", "total_tokens": 5,
    }) + "\n")

    task_a = module.summarize_routing_log(log)["tasks"]["task-a"]

    assert task_a["final"]["calls"] == 1
    assert task_a["retries"]["calls"] == 0
    assert task_a["n_retries"] == 0


def test_finalize_writes_routing_stats_by_task(tmp_path: Path) -> None:
    module = _load_manifest_module()
    out = tmp_path / "run_manifest.json"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    log = run_dir / "routing_requests.jsonl"
    log.write_text(json.dumps({
        "task": "task-a", "session_id": "s1", "model": "opus", "tier": "strong",
        "prompt_tokens": 10, "cached_tokens": 3, "cache_creation_tokens": 1,
        "completion_tokens": 2, "total_tokens": 12,
    }) + "\n")
    by_task = run_dir / "routing_stats_by_task.json"

    module.write_manifest(out, outcomes={"harbor_rc": None})
    rc = module.finalize_manifest(
        out, harbor_rc=0, routing_log=log, routing_stats_by_task=by_task,
    )

    assert rc == 0
    summary = json.loads(by_task.read_text())
    assert summary["tasks"]["task-a"]["final"]["models"][0]["calls"] == 1
    assert summary["tasks"]["task-a"]["final"]["models"][0]["cached_tokens"] == 3
    manifest = json.loads(out.read_text())
    assert manifest["outcomes"]["routing_stats_by_task_json_status"] == "present"


def test_finalize_marks_routing_stats_by_task_missing_without_log(tmp_path: Path) -> None:
    module = _load_manifest_module()
    out = tmp_path / "run_manifest.json"
    by_task = tmp_path / "routing_stats_by_task.json"

    module.write_manifest(out, outcomes={"harbor_rc": None})
    rc = module.finalize_manifest(
        out, harbor_rc=1,
        routing_log=tmp_path / "absent.jsonl", routing_stats_by_task=by_task,
    )

    assert rc == 0
    assert not by_task.exists()
    manifest = json.loads(out.read_text())
    assert manifest["outcomes"]["routing_stats_by_task_json_status"] == "missing"
