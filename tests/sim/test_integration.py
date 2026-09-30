# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor import through real native decisions and incremental report output."""

import hashlib
import json
import weakref
from dataclasses import asdict
from pathlib import Path

import pytest

from switchyard.runner import Runner
from switchyard.sim import Dataset, Outcome, Result, evaluate, load_harbor, score
from switchyard.sim.__main__ import _json_record, main
from tests.test_runner_bindings import JudgeStub, deployment
from tests.test_runner_bindings import judge as judge


def write_trial(
    root: Path, name: str, model: str, reward: float | None, *, prompt="easy task", codex=False
):
    trial = root / f"{name}__attempt"
    (trial / "agent").mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": name,
                "id": name,
                "task_checksum": f"checksum-{name}",
                "agent_info": {"model_info": {"name": model}},
                "verifier_result": {"rewards": {"reward": reward}},
                "agent_result": {"cost_usd": 1 if model == "weak/model" else 3},
            }
        )
    )
    steps = (
        [
            {"source": "system", "message": "agent instructions"},
            {"source": "user", "message": "workspace context"},
        ]
        if codex
        else []
    )
    steps += [
        {"source": "user", "message": prompt},
        {"source": "agent", "message": "FUTURE_ANSWER_MUST_NOT_LEAK"},
    ]
    (trial / "agent" / "trajectory.json").write_text(
        json.dumps({"schema_version": "ATIF-v1.7", "steps": steps})
    )


async def test_harbor_to_classifier_scoring_and_matched_metrics(judge: JudgeStub, tmp_path: Path):
    for name, prompt in (
        ("easy", "easy task"),
        ("hard", "TASK_REQUIRES_STRONG"),
        ("ungraded", "easy task"),
    ):
        write_trial(
            tmp_path / "weak",
            name,
            "weak/model",
            None if name == "ungraded" else float(name == "easy"),
            prompt=prompt,
            codex=True,
        )
        write_trial(tmp_path / "strong", name, "strong/model", 1, prompt=prompt)
    dataset = Dataset.from_runs(
        {target: load_harbor(tmp_path / target, target=target) for target in ("weak", "strong")},
        input_target="weak",
    )
    rows = []
    report = await evaluate(
        dataset,
        Runner.from_toml(deployment(judge.url)),
        route="auto",
        on_result=rows.append,
        price_call=lambda call: 0.01,
    )
    summary = report.to_dict()
    assert summary["counts"]["scored"] == 2
    assert summary["counts"]["unscored"] == 1
    assert summary["comparison"]["tasks"] == 2
    assert summary["comparison"]["routed_mean_reward"] == 1
    assert summary["comparison"]["targets"] == {"weak": 0.5, "strong": 1}
    assert summary["cost_comparison"]["tasks"] == 3
    assert summary["routing"]["calls"] == 3
    assert all(call["model"] == "judge/model" for call in judge.calls)
    assert "FUTURE_ANSWER_MUST_NOT_LEAK" not in json.dumps(judge.calls)
    assert len(judge.calls) == 3
    assert all(row.evidence["source"] == "llm-classifier" and row.decision_id for row in rows)
    assert all(row.fallbacks for row in rows)
    assert not report.complete


async def test_missing_task_prompt_keeps_system_constraints_in_native_routing(
    judge: JudgeStub, tmp_path: Path
):
    for target in ("weak", "strong"):
        write_trial(tmp_path / target, "task", f"{target}/model", float(target == "strong"))
    path = tmp_path / "weak" / "task__attempt" / "agent" / "trajectory.json"
    trajectory = json.loads(path.read_text())
    trajectory["steps"] = [
        {"source": "system", "message": "TASK_REQUIRES_STRONG"},
        {"source": "agent", "message": "FUTURE_ANSWER_MUST_NOT_LEAK"},
    ]
    path.write_text(json.dumps(trajectory))
    dataset = Dataset.from_runs(
        {
            target: load_harbor(tmp_path / target, target=target, task_inputs={"task": "easy task"})
            for target in ("weak", "strong")
        },
        input_target="weak",
    )
    rows = []
    config = deployment(judge.url).replace(
        'type = "llm_classifier"', 'type = "llm_classifier"\nrecent_turn_window = 0'
    )
    report = await evaluate(dataset, Runner.from_toml(config), route="auto", on_result=rows.append)
    assert report.complete
    assert rows[0].target == "strong"
    assert rows[0].outcome.reward == 1
    assert [call["model"] for call in judge.calls] == ["judge/model"]
    assert "TASK_REQUIRES_STRONG" in json.dumps(judge.calls)
    assert "easy task" in json.dumps(judge.calls)
    assert "FUTURE_ANSWER_MUST_NOT_LEAK" not in json.dumps(judge.calls)


@pytest.mark.parametrize("prompt_target", ["weak", "strong"])
async def test_native_completion_prompt_only_invalidates_its_selected_recording(
    judge: JudgeStub, tmp_path: Path, prompt_target: str
):
    for target in ("weak", "strong"):
        write_trial(tmp_path / target, "task", f"{target}/model", 1)
    dataset = Dataset.from_runs(
        {target: load_harbor(tmp_path / target, target=target) for target in ("weak", "strong")},
        input_target="weak",
    )
    prompt = "ADDED_COMPLETION_PROMPT"
    config = deployment(judge.url).replace(
        f"[targets.{prompt_target}]", f'[targets.{prompt_target}]\nsystem_prompt = "{prompt}"'
    )
    task = dataset.tasks[0]
    decision = await Runner.from_toml(config).decide(
        {"model": "auto", "messages": list(task.messages)}
    )
    rejected = prompt_target == "weak"
    assert decision.selected.target == "weak"
    assert (prompt in json.dumps(decision.outcome.request)) is rejected
    if rejected:
        with pytest.raises(ValueError, match="rewritten task request"):
            score(task, decision)
    else:
        assert score(task, decision).outcome == task.outcomes["weak"]

    rows = []
    report = await evaluate(
        dataset,
        Runner.from_toml(config),
        route="auto",
        on_result=rows.append,
        price_call=lambda _: 0.01,
    )
    (row,) = rows
    assert row.target == "weak" and row.model == "weak/model"
    assert (row.outcome is None) is rejected
    assert (row.error is not None) is rejected
    assert row.decision_id and row.evidence["source"] == "llm-classifier"
    assert row.fallbacks == ("strong",)
    assert row.algorithm == decision.outcome.metadata.algorithm
    if rejected:
        assert row.error == "recorded task outcomes cannot score a rewritten task request"
    assert row.routing_calls == 1 and row.routing_failed_calls == 0
    assert row.routing_cost_usd == 0.01
    assert row.routing_usage["input_tokens"] == 12
    assert row.routing_usage["output_tokens"] == 8
    summary = report.to_dict()
    assert summary["counts"]["scored"] == int(not rejected)
    assert summary["counts"]["errors"] == int(rejected)
    assert summary["comparison"]["tasks"] == int(not rejected)
    assert summary["cost_comparison"]["tasks"] == int(not rejected)
    assert summary["routing"]["cost_usd"]["total"] == 0.01
    assert [call["model"] for call in judge.calls] == ["judge/model", "judge/model"]
    assert prompt not in json.dumps(judge.calls)


@pytest.mark.parametrize("failed", [False, True], ids=["success", "failure"])
def test_cli_json_rows_preserve_exact_serialization(failed):
    selected = Outcome(
        trials=2,
        reward=-0.0,
        cost_usd=None,
        duration_seconds=5e-324,
        reward_trials=2,
        cost_trials=1,
        duration_trials=2,
    )
    ungraded = Outcome(
        trials=1,
        reward=None,
        cost_usd=1e308,
        duration_seconds=None,
        reward_trials=0,
        cost_trials=1,
        duration_trials=0,
    )
    result = Result(
        task_id="benchmark/\u03c0",
        baselines={"weak": selected, "strong": ungraded},
        target=None if failed else "weak",
        model=None if failed else "weak/model",
        outcome=None if failed else selected,
        decision_id=None if failed else "decision-1",
        algorithm=None if failed else "llm-classifier",
        evidence=None
        if failed
        else {"verdict": {"label": "\u590d\u6742", "values": [True, None, -0.0, 5e-324, 1e308]}},
        fallbacks=() if failed else ("strong",),
        routing_seconds=0.125,
        routing_calls=1,
        routing_failed_calls=int(failed),
        routing_usage={"input_tokens": None, "output_tokens": 0},
        error="routing failed: DecisionError" if failed else None,
        routing_error_kind="upstream_http" if failed else None,
        routing_error_status=503 if failed else None,
        routing_error_target="judge/model" if failed else None,
    )
    assert json.dumps(result, default=_json_record, allow_nan=False) == json.dumps(
        asdict(result), allow_nan=False
    )


@pytest.mark.parametrize("namespace", [None, "benchmark/v1"])
def test_cli_writes_flushed_rows_manifest_and_report_without_copying_config(
    judge: JudgeStub, tmp_path: Path, namespace: str | None
):
    write_trial(tmp_path / "recordings", "task", "weak/model", 1)
    config = tmp_path / "routes.toml"
    config.write_text(deployment(judge.url))
    output = tmp_path / "evaluation"
    args = [
        "--config",
        str(config),
        "--route",
        "fixed",
        "--run",
        f"weak={tmp_path / 'recordings'}",
        "--input-target",
        "weak",
        "--output",
        str(output),
    ]
    if namespace is not None:
        args.extend(["--dataset", namespace])
    assert main(args) == 0
    assert judge.calls == []
    report = json.loads((output / "report.json").read_text())
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    manifest = json.loads((output / "manifest.json").read_text())
    assert report["complete"] and len(rows) == 1
    assert rows[0]["task_id"] == (f"{namespace}/task" if namespace else "task")
    assert rows[0]["outcome"]["reward"] == 1
    assert manifest["dataset"] == namespace
    assert manifest["skip_invalid"] is False
    assert len(manifest["config_sha256"]) == 64
    assert "provider-secret" not in "".join(path.read_text() for path in output.iterdir())
    original = (output / "report.json").read_text()
    assert main(args) == 2
    assert (output / "report.json").read_text() == original


def test_cli_releases_excluded_trials_before_routing(
    judge: JudgeStub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    for target in ("weak", "strong"):
        write_trial(tmp_path / target, "shared", f"{target}/model", 1)
        write_trial(tmp_path / target, "bad", f"{target}/model", 0)
    write_trial(tmp_path / "weak", "weak-only", "weak/model", 0)
    write_trial(tmp_path / "weak", "bad-repeat", "weak/model", 0)
    invalid = tmp_path / "weak" / "bad-repeat__attempt" / "result.json"
    data = json.loads(invalid.read_text())
    data.update(task_name="bad", verifier_result={"rewards": {"reward": "invalid"}})
    invalid.write_text(json.dumps(data))
    config = tmp_path / "routes.toml"
    config.write_text(deployment(judge.url))
    run_refs = []
    trial_refs = []

    def capture(*args, **kwargs):
        run = load_harbor(*args, **kwargs)
        run_refs.append(weakref.ref(run))
        trial_refs.extend((trial.task_id, weakref.ref(trial)) for trial in run.trials)
        return run

    async def verify(dataset, *args, **kwargs):
        assert len(run_refs) == 2 and all(ref() is None for ref in run_refs)
        assert sum(name != "shared" for name, _ in trial_refs) == 3
        assert all(ref() is None for name, ref in trial_refs if name != "shared")
        (task,) = dataset.tasks
        assert task.task_id == "shared"
        assert {id(trial) for trials in task.trials.values() for trial in trials} == {
            id(ref()) for name, ref in trial_refs if name == "shared"
        }
        return await evaluate(dataset, *args, **kwargs)

    monkeypatch.setattr("switchyard.sim.__main__.load_harbor", capture)
    monkeypatch.setattr("switchyard.sim.__main__.evaluate", verify)
    output = tmp_path / "evaluation"
    assert (
        main(
            [
                "--config",
                str(config),
                "--route",
                "fixed",
                "--run",
                f"weak={tmp_path / 'weak'}",
                "--run",
                f"strong={tmp_path / 'strong'}",
                "--input-target",
                "weak",
                "--intersection",
                "--skip-invalid",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    manifest = json.loads((output / "manifest.json").read_text())
    report = json.loads((output / "report.json").read_text())
    assert manifest["skip_invalid"] is True
    coverage = manifest["coverage"]
    assert coverage.items() <= report["coverage"].items()
    assert coverage["tasks_seen"] == 3 and coverage["tasks_included"] == 1
    assert coverage["excluded_task_ids"] == ["bad", "weak-only"]
    assert coverage["cost_sources_by_target"] == {"weak": {"result": 3}, "strong": {"result": 2}}
    assert coverage["input_issues"][0]["task_id"] == "bad"
    assert coverage["input_issues"][0]["source"] == str(invalid)
    (row,) = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert row["task_id"] == "shared" and row["target"] == "weak"
    assert row["outcome"]["reward"] == 1 and report["complete"]
    assert judge.calls == []


@pytest.fixture
def cli_recording(judge: JudgeStub, tmp_path: Path):
    root = tmp_path / "recordings"
    write_trial(root, "task", "weak/model", 1)
    config = tmp_path / "routes.toml"
    config.write_text(deployment(judge.url))

    def invoke(run: Path, output: Path) -> int:
        return main(
            [
                "--config",
                str(config),
                "--route",
                "fixed",
                "--run",
                f"weak={run}",
                "--input-target",
                "weak",
                "--output",
                str(output),
            ]
        )

    return root, invoke


@pytest.mark.parametrize(
    "placement", ["jobs", "agent", "new-parent/agent", "evaluation", "output-alias", "input-alias"]
)
def test_cli_rejects_output_inside_recordings(cli_recording, placement, capsys, judge):
    root, invoke = cli_recording
    run = root
    output = root / placement
    if placement in ("output-alias", "input-alias"):
        alias = root.parent / "recording-alias"
        alias.symlink_to(root, target_is_directory=True)
        if placement == "output-alias":
            output = alias / "jobs"
        else:
            run = alias
            output = root / "jobs"
    before = sorted(root.rglob("*"))
    assert invoke(run, output) == 2
    assert "outside every recorded run directory" in capsys.readouterr().err
    assert sorted(root.rglob("*")) == before
    assert len(load_harbor(root, target="weak").trials) == 1
    assert judge.calls == []


@pytest.mark.parametrize("placement", ["sibling", "parent-traversal"])
def test_cli_resolves_outside_output_without_creating_input_directories(
    cli_recording, placement, judge
):
    root, invoke = cli_recording
    output = (
        root.parent / "recordings-evaluation"
        if placement == "sibling"
        else root / "agent" / ".." / ".." / "evaluation"
    )
    before = sorted(root.rglob("*"))
    assert invoke(root, output) == 0
    assert sorted(root.rglob("*")) == before
    assert len(load_harbor(root, target="weak").trials) == 1
    report = json.loads((output.resolve() / "report.json").read_text())
    assert report["complete"] and report["counts"]["scored"] == 1
    assert judge.calls == []


def test_cli_rejects_dangling_output_symlink(cli_recording, capsys, judge):
    root, invoke = cli_recording
    output = root.parent / "evaluation"
    destination = root.parent / "missing"
    output.symlink_to(destination, target_is_directory=True)
    assert invoke(root, output) == 2
    assert capsys.readouterr().err.startswith("switchyard.sim:")
    assert output.is_symlink() and not destination.exists()
    assert judge.calls == []


@pytest.mark.parametrize("location", ["run", "output-parent"])
def test_cli_reports_cyclic_paths_as_file_errors(cli_recording, location, capsys, judge):
    root, invoke = cli_recording
    loop = root.parent / "loop"
    loop.symlink_to(loop, target_is_directory=True)
    run = loop if location == "run" else root
    output = root.parent / "evaluation" if location == "run" else loop / "evaluation"
    assert invoke(run, output) == 2
    error = capsys.readouterr().err
    assert error.startswith("switchyard.sim:") and "Traceback" not in error
    assert not (root.parent / "evaluation").exists()
    assert len(load_harbor(root, target="weak").trials) == 1
    assert judge.calls == []


def test_cli_exits_incomplete_for_missing_reward(judge: JudgeStub, tmp_path: Path):
    write_trial(tmp_path / "recordings", "task", "weak/model", None)
    config = tmp_path / "routes.toml"
    config.write_text(deployment(judge.url))
    output = tmp_path / "evaluation"
    assert (
        main(
            [
                "--config",
                str(config),
                "--route",
                "fixed",
                "--run",
                f"weak={tmp_path / 'recordings'}",
                "--input-target",
                "weak",
                "--output",
                str(output),
            ]
        )
        == 1
    )
    assert json.loads((output / "report.json").read_text())["counts"]["unscored"] == 1


@pytest.mark.parametrize("status", [401, 503])
def test_cli_retains_safe_native_failure_details(judge: JudgeStub, tmp_path: Path, status: int):
    judge.status = status
    for target in ("weak", "strong"):
        write_trial(tmp_path / target, "task", f"{target}/model", 1)
    config = tmp_path / "routes.toml"
    config.write_text(deployment(judge.url))
    output = tmp_path / "evaluation"
    assert (
        main(
            [
                "--config",
                str(config),
                "--route",
                "auto",
                "--run",
                f"weak={tmp_path / 'weak'}",
                "--run",
                f"strong={tmp_path / 'strong'}",
                "--input-target",
                "weak",
                "--output",
                str(output),
            ]
        )
        == 1
    )
    row = json.loads((output / "results.jsonl").read_text())
    assert row["routing_error_kind"] == "upstream_http"
    assert row["routing_error_status"] == status
    assert row["routing_error_target"] == "judge/model"
    assert row["routing_calls"] == row["routing_failed_calls"] == 1
    assert row["routing_cost_usd"] is None
    assert row["error"] == "routing failed: DecisionError"
    assert "provider-secret" not in json.dumps(row)
    assert "private prompt" not in json.dumps(row)
    report = json.loads((output / "report.json").read_text())
    assert report["counts"]["errors"] == 1
    assert not report["complete"]


@pytest.mark.parametrize(
    "option",
    [
        "--concurrency=0",
        "--concurrency=-1",
        "--timeout=0",
        "--timeout=-1",
        "--timeout=nan",
        "--timeout=inf",
    ],
)
def test_cli_invalid_options_leave_output_path_available(
    judge: JudgeStub, tmp_path: Path, option: str
):
    write_trial(tmp_path / "recordings", "task", "weak/model", 1)
    config = tmp_path / "routes.toml"
    config.write_text(deployment(judge.url))
    output = tmp_path / "new-parent" / "evaluation"
    args = [
        "--config",
        str(config),
        "--route",
        "fixed",
        "--run",
        f"weak={tmp_path / 'recordings'}",
        "--input-target",
        "weak",
        "--output",
        str(output),
    ]
    assert main([*args, option]) == 2
    assert not output.parent.exists()
    assert judge.calls == []
    assert main(args) == 0
    assert (output / "report.json").exists()


@pytest.mark.parametrize(
    "problem", ["unknown-route", "missing-target", "model-mismatch", "response-route"]
)
def test_cli_route_preflight_leaves_output_path_available(
    judge: JudgeStub, tmp_path: Path, problem: str
):
    model = "different/model" if problem == "model-mismatch" else "weak/model"
    write_trial(tmp_path / "recordings", "task", model, 1)
    config = tmp_path / "routes.toml"
    config.write_text(deployment(judge.url, escalation=problem == "response-route"))
    output = tmp_path / "new-parent" / "evaluation"
    route = (
        "missing"
        if problem == "unknown-route"
        else ("auto" if problem in ("missing-target", "response-route") else "fixed")
    )
    args = [
        "--config",
        str(config),
        "--route",
        route,
        "--run",
        f"weak={tmp_path / 'recordings'}",
        "--input-target",
        "weak",
        "--output",
        str(output),
    ]
    assert main(args) == 2
    assert not output.parent.exists()
    assert judge.calls == []

    args[3] = "fixed"
    if problem == "model-mismatch":
        args.extend(["--model-alias", "different/model=weak/model"])
    assert main(args) == 0
    assert (output / "report.json").exists()


def test_manifest_hash_identifies_the_loaded_configuration_snapshot(
    judge: JudgeStub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    write_trial(tmp_path / "recordings", "task", "weak/model", 1)
    config = tmp_path / "routes.toml"
    original = deployment(judge.url).replace("\n", "\r\n").encode("utf-8")
    config.write_bytes(original)

    class ConcurrentEdit:
        @staticmethod
        def load(path):
            runner = Runner.load(path)
            config.write_bytes(original + b"\n# edited after loading\n")
            return runner

        @staticmethod
        def from_toml(source):
            runner = Runner.from_toml(source)
            config.write_bytes(original + b"\n# edited after loading\n")
            return runner

    monkeypatch.setattr("switchyard.sim.__main__.Runner", ConcurrentEdit)
    output = tmp_path / "evaluation"
    assert (
        main(
            [
                "--config",
                str(config),
                "--route",
                "fixed",
                "--run",
                f"weak={tmp_path / 'recordings'}",
                "--input-target",
                "weak",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert config.read_bytes() != original
    assert manifest["config_sha256"] == hashlib.sha256(original).hexdigest()
