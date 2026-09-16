# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


artifacts = _load("vgr_holdout_artifacts", ROOT / "benchmark/vgr_holdout_artifacts.py")
launcher = _load("run_holdout_suite", ROOT / "benchmark/run_holdout_suite.py")
session_proxy = _load("holdout_session_proxy", ROOT / "benchmark/holdout_session_proxy.py")


def test_session_id_is_stable_across_turns_of_one_task() -> None:
    first = json.dumps(
        {
            "messages": [
                {"role": "system", "content": "Use tools."},
                {"role": "user", "content": "Create the calendar event."},
            ]
        }
    ).encode()
    later = json.dumps(
        {
            "messages": [
                {"role": "system", "content": "Use tools."},
                {"role": "user", "content": "Create the calendar event."},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
                {"role": "tool", "content": "created", "tool_call_id": "1"},
            ]
        }
    ).encode()

    assert session_proxy.session_id_for_payload(
        "automationbench", first
    ) == session_proxy.session_id_for_payload("automationbench", later)


def test_session_id_separates_tasks_and_suites() -> None:
    body = json.dumps({"messages": [{"role": "user", "content": "Task A"}]}).encode()
    other = json.dumps({"messages": [{"role": "user", "content": "Task B"}]}).encode()

    assert session_proxy.session_id_for_payload(
        "automationbench", body
    ) != session_proxy.session_id_for_payload("automationbench", other)
    assert session_proxy.session_id_for_payload(
        "automationbench", body
    ) != session_proxy.session_id_for_payload("appworld", body)


def test_session_id_rejects_requests_without_a_stable_conversation() -> None:
    with pytest.raises(ValueError, match="stable leading conversation"):
        session_proxy.session_id_for_payload("appworld", b'{"messages":[]}')


def test_dry_run_contains_all_full_holdout_cohorts(capsys: pytest.CaptureFixture[str]) -> None:
    assert launcher.main(["--dry-run"]) == 0
    preview = json.loads(capsys.readouterr().out)

    tb21 = preview["tb21"]
    assert tb21[:4] == ["uv", "run", "--no-sync", "harbor"]
    assert "bash" not in tb21
    assert "terminal-bench-2-1-closed-book" in " ".join(tb21)
    assert "--n-tasks" not in tb21

    automation = preview["automationbench_simple"]
    assert automation[automation.index("--domains") + 1] == "simple"
    assert automation[automation.index("--num-examples") + 1] == "-1"

    appworld = preview["appworld_easy"]
    assert appworld[appworld.index("--dataset-name") + 1] == "dev_easy"
    assert appworld[appworld.index("--agent-name") + 1] == "simplified_react_code_agent"
    assert preview["artifact_contract"]["counterfactual_gate"].startswith("blocked")


def test_appworld_cohort_and_model_match_the_frozen_holdout() -> None:
    model = launcher._appworld_model_module("http://127.0.0.1:4001/v1")

    assert len(launcher.APPWORLD_TASK_IDS) == 30
    assert '"model_id": "switchyard/vgr"' in model
    assert '"base_url": "http://127.0.0.1:4001/v1"' in model


def test_launcher_has_no_wsl_or_shell_dependency() -> None:
    source = (ROOT / "benchmark/run_holdout_suite.py").read_text()

    assert "wsl.exe" not in source
    assert "run-baseline.sh" not in source


def test_default_checkout_finds_harness_above_a_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    repository = workspace / "review" / "switchyard"
    harness = workspace / "AutomationBench"
    repository.mkdir(parents=True)
    harness.mkdir()
    monkeypatch.setattr(launcher, "REPO_ROOT", repository)

    assert launcher._default_checkout("AutomationBench") == harness


def test_server_command_persists_full_run_routing_records(tmp_path: Path) -> None:
    config = tmp_path / "routes.toml"
    config.write_text('schema_version = 1\n')

    command = launcher._server_command(
        config,
        4000,
        "holdout-network",
        "holdout-container",
        tmp_path,
    )

    mounts = [command[index + 1] for index, value in enumerate(command) if value == "--mount"]
    assert any("dst=/artifacts" in mount for mount in mounts)
    assert command[command.index("--routing-log-file") + 1] == (
        "/artifacts/routing_requests.jsonl"
    )


def _complete_artifact_fixture(run_dir: Path) -> None:
    (run_dir / "inputs").mkdir(parents=True)
    (run_dir / "inputs/server-config.toml").write_text('schema_version = 1\n')
    (run_dir / "run_manifest.json").write_text('{"schema_version": 2}\n')
    (run_dir / "routing_stats_final.json").write_text('{"total_requests": 2}\n')
    (run_dir / "server_metrics_final.prom").write_text("switchyard_requests_total 2\n")
    (run_dir / "tb21/jobs/job").mkdir(parents=True)
    (run_dir / "tb21/jobs/job/result.json").write_text('{"stats": {"passed": 1}}\n')
    (run_dir / "automationbench-simple.json").write_text(
        json.dumps(
            {
                "meta": {"total_tasks": 1},
                "summary": {"pass_rate": 1.0},
                "tasks": [{"name": "task-a", "passed": True}],
            }
        )
    )
    appworld = run_dir / "appworld-output/evaluations"
    appworld.mkdir(parents=True)
    (appworld / "dev_easy.json").write_text(
        '{"aggregate": {"task_goal_completion": 1.0}}\n'
    )


def test_artifact_summary_is_auditable_without_claiming_a_gate(tmp_path: Path) -> None:
    _complete_artifact_fixture(tmp_path)
    records = [
        {
            "task": "task-a",
            "session_id": "task-a",
            "vgr_predicted": "local",
            "vgr_effective": "local",
            "vgr_served": "local",
            "vgr_branch": "answer",
            "vgr_readiness_gate": "none",
            "vgr_short_circuit": "none",
            "model": "local-model",
            "tier": "local",
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
        },
        {
            "task": "task-a",
            "session_id": "task-a",
            "vgr_predicted": "cloud",
            "vgr_effective": "cloud",
            "vgr_served": "cloud",
            "vgr_branch": "coding",
            "vgr_readiness_gate": "none",
            "vgr_short_circuit": "none",
            "model": "cloud-model",
            "tier": "cloud",
            "prompt_tokens": 20,
            "completion_tokens": 4,
            "total_tokens": 24,
        },
    ]
    (tmp_path / "routing_requests.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\nnot-json\n"
    )

    result = artifacts.finalize_artifacts(tmp_path, require_complete=True)

    summary = json.loads((tmp_path / result["summary"]).read_text())
    assert summary["routing"]["records"] == 2
    assert summary["routing"]["invalid_records"] == 1
    assert summary["routing"]["effective"] == {"cloud": 1, "local": 1}
    assert summary["routing"]["tier_flips"] == 1
    assert summary["counterfactual_gate"]["status"] == "blocked"
    assert summary["benchmark_native"]["automationbench_simple"]["task_rows"] == 1
    task_rows = [
        json.loads(line) for line in (tmp_path / "per_task_routing.jsonl").read_text().splitlines()
    ]
    assert task_rows[0]["tokens"]["total_tokens"] == 36
    index = json.loads((tmp_path / "artifact_index.json").read_text())
    assert all(
        item["status"] == "present"
        for item in index["artifacts"].values()
        if item["required"]
    )


def test_frozen_labels_reproduce_per_benchmark_fpr_and_fnr(tmp_path: Path) -> None:
    _complete_artifact_fixture(tmp_path)
    (tmp_path / "routing_requests.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "task": "cloud-task",
                        "vgr_effective": "local",
                        "vgr_served": "local",
                    }
                ),
                json.dumps(
                    {
                        "task": "local-task",
                        "vgr_effective": "cloud",
                        "vgr_served": "cloud",
                    }
                ),
            ]
        )
        + "\n"
    )
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "decision_unit": "request",
                "labels": [
                    {
                        "benchmark": "automationbench_simple",
                        "task": "cloud-task",
                        "required_route": "cloud_required",
                    },
                    {
                        "benchmark": "automationbench_simple",
                        "task": "local-task",
                        "required_route": "local_required",
                    },
                ],
            }
        )
    )

    artifacts.finalize_artifacts(tmp_path, labels, require_complete=True)

    summary = json.loads((tmp_path / "machine_readable_summary.json").read_text())
    gate = summary["counterfactual_gate"]
    assert gate["status"] == "computed"
    benchmark = gate["benchmarks"]["automationbench_simple"]
    assert benchmark["fpr"]["numerator"] == 1
    assert benchmark["fpr"]["denominator"] == 1
    assert benchmark["fnr"]["numerator"] == 1
    assert benchmark["fnr"]["denominator"] == 1
    assert benchmark["fpr"]["wilson_95_confidence_interval"] is not None
    assert (tmp_path / "inputs/counterfactual_labels.json").is_file()


def test_counterfactual_labels_reject_duplicate_tasks(tmp_path: Path) -> None:
    _complete_artifact_fixture(tmp_path)
    (tmp_path / "routing_requests.jsonl").write_text("")
    labels = tmp_path / "labels.json"
    entry = {
        "benchmark": "tb21",
        "task": "task-1",
        "required_route": "local_required",
    }
    labels.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "decision_unit": "request",
                "labels": [entry, entry],
            }
        )
    )

    with pytest.raises(ValueError, match="duplicate counterfactual label"):
        artifacts.finalize_artifacts(tmp_path, labels)
