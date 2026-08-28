# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from craft_taskgen import swebench_alignment


def _write_candidate_file(tmp_path):
    path = tmp_path / "teleport.json"
    path.write_text(
        json.dumps(
            {
                "repo": "teleport",
                "candidates": [
                    {
                        "sha": "abc123",
                        "merge_base_sha": "base123",
                        "source_task_id": "instance_1",
                        "source_metadata": {
                            "problem_statement": "Fix the login flow.",
                            "requirements": "- Preserve existing login behavior.",
                            "interface": "Keep the public login API unchanged.",
                        },
                    },
                    {
                        "sha": "def456",
                        "merge_base_sha": "base456",
                        "source_task_id": "instance_2",
                        "source_metadata": {
                            "problem_statement": "Fix the logout flow.",
                            "requirements": "- Preserve existing logout behavior.",
                            "interface": "Keep the public logout API unchanged.",
                        },
                    },
                ],
            }
        )
    )
    return path


def _write_state_file(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "created": "2026-01-01T00:00:00",
                "last_updated": "2026-01-01T00:00:00",
                "run_dir": str(tmp_path),
                "profile_data": {},
                "run_info": {},
                "tasks": {
                    "teleport-abc123": {
                        "task_id": "teleport-abc123",
                        "repo": "teleport",
                        "commit_sha": "abc123",
                        "description": "Fix login",
                        "base_sha": "base123",
                        "merge_base_sha": "base123",
                        "stage": "candidate",
                        "candidate_data": {
                            "source_task_id": "instance_1",
                            "source_metadata": {
                                "problem_statement": "Fix the login flow.",
                                "requirements": "- Preserve existing login behavior.",
                                "interface": "Keep the public login API unchanged.",
                            },
                        },
                    }
                },
            }
        )
    )
    return path


def test_swebench_alignment_happy_path(tmp_path, monkeypatch) -> None:
    state_path = _write_state_file(tmp_path)
    output_path = tmp_path / "out.jsonl"

    def fake_build_context(candidate, repos_dir):
        assert repos_dir == "repos"
        assert candidate.source_task_id == "instance_1"
        return {
            "instruction_md": candidate.problem_statement,
            "reference_test_bodies": [("tests/test_login.py", "def test_login(): pass\n")],
            "diff": "diff --git a/x b/x",
            "diff_truncated": False,
        }

    async def fake_judge(*, prompt, schema, model):
        assert "Fix the login flow." in prompt
        assert schema == swebench_alignment.ALIGNMENT_SCHEMA
        assert model == "test-model"
        return SimpleNamespace(
            result={"verdict": "ok", "reason": "looks aligned", "leakage_evidence": [], "v4_audit": {}},
            usage={"input_tokens": 10, "output_tokens": 5},
            model="openai/test-model",
            latency_s=1.25,
        )

    monkeypatch.setattr(swebench_alignment, "_build_context", fake_build_context)
    monkeypatch.setattr(swebench_alignment.llm_judge, "judge", fake_judge)
    monkeypatch.setattr(
        "sys.argv",
        [
            "craft-taskgen-swebench-align",
            "--state-json",
            str(state_path),
            "--output",
            str(output_path),
            "--model",
            "test-model",
            "--instance-id",
            "instance_1",
        ],
    )

    rc = swebench_alignment.main()

    assert rc == 0
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["status"] == "ok"
    assert rows[0]["verdict"] == "ok"
    assert rows[0]["reference_test_paths"] == ["tests/test_login.py"]
    state = json.loads(state_path.read_text())
    task = state["tasks"]["teleport-abc123"]
    assert task["alignment_verdict"] == "ok"
    assert task["alignment_reason"] == "looks aligned"


def test_swebench_alignment_promotes_ok_promising_tasks(tmp_path, monkeypatch) -> None:
    state_path = _write_state_file(tmp_path)
    output_path = tmp_path / "out.jsonl"

    def fake_build_context(candidate, repos_dir):
        return {
            "instruction_md": candidate.problem_statement,
            "reference_test_bodies": [("tests/test_login.py", "def test_login(): pass\n")],
            "diff": "diff --git a/x b/x",
            "diff_truncated": False,
        }

    async def fake_judge(*, prompt, schema, model):
        return SimpleNamespace(
            result={"verdict": "ok", "reason": "looks aligned", "leakage_evidence": [], "v4_audit": {}},
            usage={"input_tokens": 10, "output_tokens": 5},
            model="openai/test-model",
            latency_s=1.25,
        )

    state = json.loads(state_path.read_text())
    state["tasks"]["teleport-abc123"]["stage"] = "promising"
    state_path.write_text(json.dumps(state))

    monkeypatch.setattr(swebench_alignment, "_build_context", fake_build_context)
    monkeypatch.setattr(swebench_alignment.llm_judge, "judge", fake_judge)
    monkeypatch.setattr(
        "sys.argv",
        [
            "craft-taskgen-swebench-align",
            "--state-json",
            str(state_path),
            "--output",
            str(output_path),
            "--model",
            "test-model",
            "--promote-ok",
        ],
    )

    rc = swebench_alignment.main()

    assert rc == 0
    state = json.loads(state_path.read_text())
    task = state["tasks"]["teleport-abc123"]
    assert task["stage"] == "alignment_checked"
    assert task["task_dir"]
    assert (tmp_path / Path(task["task_dir"]).name / "instruction.md").read_text() == "Fix the login flow."
    assert task["instruction_words"] == 4


def test_swebench_alignment_include_interface(tmp_path, monkeypatch) -> None:
    state_path = _write_state_file(tmp_path)
    output_path = tmp_path / "out.jsonl"

    def fake_build_context(candidate, repos_dir):
        assert candidate.requirements == "- Preserve existing login behavior."
        assert candidate.interface == "Keep the public login API unchanged."
        return {
            "instruction_md": candidate.problem_statement,
            "reference_test_bodies": [("tests/test_login.py", "def test_login(): pass\n")],
            "diff": "diff --git a/x b/x",
            "diff_truncated": False,
        }

    async def fake_judge(*, prompt, schema, model):
        assert "Fix the login flow." in prompt
        assert "## Requirements" not in prompt
        assert "## Interface\nKeep the public login API unchanged." in prompt
        return SimpleNamespace(
            result={"verdict": "ok", "reason": "looks aligned", "leakage_evidence": [], "v4_audit": {}},
            usage={"input_tokens": 10, "output_tokens": 5},
            model="openai/test-model",
            latency_s=1.25,
        )

    state = json.loads(state_path.read_text())
    state["tasks"]["teleport-abc123"]["stage"] = "promising"
    state_path.write_text(json.dumps(state))

    monkeypatch.setattr(swebench_alignment, "_build_context", fake_build_context)
    monkeypatch.setattr(swebench_alignment.llm_judge, "judge", fake_judge)
    monkeypatch.setattr(
        "sys.argv",
        [
            "craft-taskgen-swebench-align",
            "--state-json",
            str(state_path),
            "--output",
            str(output_path),
            "--model",
            "test-model",
            "--promote-ok",
            "--include-interface",
        ],
    )

    rc = swebench_alignment.main()

    assert rc == 0
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert rows[0]["instruction_source"] == "problem_statement+interface"
    assert rows[0]["interface_included"] is True
    state = json.loads(state_path.read_text())
    task = state["tasks"]["teleport-abc123"]
    assert Path(task["task_dir"], "instruction.md").read_text() == (
        "Fix the login flow.\n\n## Interface\nKeep the public login API unchanged."
    )


def test_swebench_alignment_include_requirements_and_interface(tmp_path, monkeypatch) -> None:
    state_path = _write_state_file(tmp_path)
    output_path = tmp_path / "out.jsonl"

    def fake_build_context(candidate, repos_dir):
        assert candidate.requirements == "- Preserve existing login behavior."
        assert candidate.interface == "Keep the public login API unchanged."
        return {
            "instruction_md": candidate.problem_statement,
            "reference_test_bodies": [("tests/test_login.py", "def test_login(): pass\n")],
            "diff": "diff --git a/x b/x",
            "diff_truncated": False,
        }

    async def fake_judge(*, prompt, schema, model):
        assert "Fix the login flow." in prompt
        assert "## Requirements\n- Preserve existing login behavior." in prompt
        assert "## Interface\nKeep the public login API unchanged." in prompt
        assert prompt.index("## Requirements") < prompt.index("## Interface")
        return SimpleNamespace(
            result={"verdict": "ok", "reason": "looks aligned", "leakage_evidence": [], "v4_audit": {}},
            usage={"input_tokens": 10, "output_tokens": 5},
            model="openai/test-model",
            latency_s=1.25,
        )

    state = json.loads(state_path.read_text())
    state["tasks"]["teleport-abc123"]["stage"] = "promising"
    state_path.write_text(json.dumps(state))

    monkeypatch.setattr(swebench_alignment, "_build_context", fake_build_context)
    monkeypatch.setattr(swebench_alignment.llm_judge, "judge", fake_judge)
    monkeypatch.setattr(
        "sys.argv",
        [
            "craft-taskgen-swebench-align",
            "--state-json",
            str(state_path),
            "--output",
            str(output_path),
            "--model",
            "test-model",
            "--promote-ok",
            "--include-requirements",
            "--include-interface",
        ],
    )

    rc = swebench_alignment.main()

    assert rc == 0
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert rows[0]["instruction_source"] == "problem_statement+requirements+interface"
    assert rows[0]["requirements_included"] is True
    assert rows[0]["interface_included"] is True
    state = json.loads(state_path.read_text())
    task = state["tasks"]["teleport-abc123"]
    assert Path(task["task_dir"], "instruction.md").read_text() == (
        "Fix the login flow.\n\n"
        "## Requirements\n- Preserve existing login behavior.\n\n"
        "## Interface\nKeep the public login API unchanged."
    )


def test_swebench_alignment_does_not_revive_rejected_tasks(tmp_path, monkeypatch) -> None:
    state_path = _write_state_file(tmp_path)
    output_path = tmp_path / "out.jsonl"

    def fake_build_context(candidate, repos_dir):
        return {
            "instruction_md": candidate.problem_statement,
            "reference_test_bodies": [("tests/test_login.py", "def test_login(): pass\n")],
            "diff": "diff --git a/x b/x",
            "diff_truncated": False,
        }

    async def fake_judge(*, prompt, schema, model):
        return SimpleNamespace(
            result={"verdict": "ok", "reason": "looks aligned", "leakage_evidence": [], "v4_audit": {}},
            usage={"input_tokens": 10, "output_tokens": 5},
            model="openai/test-model",
            latency_s=1.25,
        )

    state = json.loads(state_path.read_text())
    state["tasks"]["teleport-abc123"]["stage"] = "rejected"
    state_path.write_text(json.dumps(state))

    monkeypatch.setattr(swebench_alignment, "_build_context", fake_build_context)
    monkeypatch.setattr(swebench_alignment.llm_judge, "judge", fake_judge)
    monkeypatch.setattr(
        "sys.argv",
        [
            "craft-taskgen-swebench-align",
            "--state-json",
            str(state_path),
            "--output",
            str(output_path),
            "--model",
            "test-model",
            "--promote-ok",
        ],
    )

    rc = swebench_alignment.main()

    assert rc == 0
    state = json.loads(state_path.read_text())
    task = state["tasks"]["teleport-abc123"]
    assert task["stage"] == "rejected"
    assert task.get("task_dir", "") == ""


def test_swebench_alignment_promote_existing_ok(tmp_path, monkeypatch) -> None:
    state_path = _write_state_file(tmp_path)
    state = json.loads(state_path.read_text())
    state["tasks"]["teleport-abc123"]["stage"] = "promising"
    state["tasks"]["teleport-abc123"]["alignment_verdict"] = "ok"
    state["tasks"]["teleport-abc123"]["alignment_reason"] = "looks aligned"
    state_path.write_text(json.dumps(state))

    monkeypatch.setattr(
        "sys.argv",
        [
            "craft-taskgen-swebench-align",
            "--state-json",
            str(state_path),
            "--promote-existing-ok",
        ],
    )

    rc = swebench_alignment.main()

    assert rc == 0
    state = json.loads(state_path.read_text())
    task = state["tasks"]["teleport-abc123"]
    assert task["stage"] == "alignment_checked"
    assert Path(task["task_dir"], "instruction.md").read_text() == "Fix the login flow."


def test_swebench_alignment_continues_on_error(tmp_path, monkeypatch) -> None:
    candidate_path = _write_candidate_file(tmp_path)
    output_path = tmp_path / "out.jsonl"

    def fake_build_context(candidate, repos_dir):
        if candidate.source_task_id == "instance_1":
            raise ValueError("missing git clone")
        return {
            "instruction_md": candidate.problem_statement,
            "reference_test_bodies": [("tests/test_logout.py", "def test_logout(): pass\n")],
            "diff": "diff --git a/y b/y",
            "diff_truncated": False,
        }

    async def fake_judge(*, prompt, schema, model):
        return SimpleNamespace(
            result={
                "verdict": "leaked",
                "reason": "too specific",
                "leakage_evidence": ["Fix logout"],
                "v4_audit": {},
            },
            usage={"input_tokens": 7, "output_tokens": 3},
            model="openai/test-model",
            latency_s=0.5,
        )

    monkeypatch.setattr(swebench_alignment, "_build_context", fake_build_context)
    monkeypatch.setattr(swebench_alignment.llm_judge, "judge", fake_judge)
    monkeypatch.setattr(
        "sys.argv",
        [
            "craft-taskgen-swebench-align",
            "--candidates",
            str(candidate_path),
            "--output",
            str(output_path),
            "--model",
            "test-model",
        ],
    )

    rc = swebench_alignment.main()

    assert rc == 0
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["status"] == "context_error"
    assert rows[0]["error"] == "missing git clone"
    assert rows[1]["status"] == "ok"
    assert rows[1]["verdict"] == "leaked"


def test_swebench_alignment_dry_run_writes_prompt(tmp_path, monkeypatch) -> None:
    state_path = _write_state_file(tmp_path)
    output_path = tmp_path / "out.jsonl"

    def fake_build_context(candidate, repos_dir):
        return {
            "instruction_md": candidate.problem_statement,
            "reference_test_bodies": [("tests/test_login.py", "def test_login(): pass\n")],
            "diff": "diff --git a/x b/x",
            "diff_truncated": False,
        }

    monkeypatch.setattr(swebench_alignment, "_build_context", fake_build_context)
    monkeypatch.setattr(
        "sys.argv",
        [
            "craft-taskgen-swebench-align",
            "--state-json",
            str(state_path),
            "--output",
            str(output_path),
            "--dry-run",
        ],
    )

    rc = swebench_alignment.main()

    assert rc == 0
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["status"] == "dry_run"
    assert "Fix the login flow." in rows[0]["prompt"]
