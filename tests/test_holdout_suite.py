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


def test_appworld_cohort_and_model_match_the_frozen_holdout() -> None:
    model = launcher._appworld_model_module("http://127.0.0.1:4001/v1")

    assert len(launcher.APPWORLD_TASK_IDS) == 30
    assert '"model_id": "switchyard/vgr"' in model
    assert '"base_url": "http://127.0.0.1:4001/v1"' in model


def test_launcher_has_no_wsl_or_shell_dependency() -> None:
    source = (ROOT / "benchmark/run_holdout_suite.py").read_text()

    assert "wsl.exe" not in source
    assert "run-baseline.sh" not in source
