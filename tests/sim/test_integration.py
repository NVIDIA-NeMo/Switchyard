# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor import through real native decisions and incremental report output."""

import hashlib
import json
from pathlib import Path

import pytest

from switchyard.runner import Runner
from switchyard.sim import Dataset, evaluate, load_harbor
from switchyard.sim.__main__ import main
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


def test_cli_writes_flushed_rows_manifest_and_report_without_copying_config(
    judge: JudgeStub, tmp_path: Path
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
    assert main(args) == 0
    assert judge.calls == []
    report = json.loads((output / "report.json").read_text())
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    manifest = json.loads((output / "manifest.json").read_text())
    assert report["complete"] and len(rows) == 1
    assert rows[0]["outcome"]["reward"] == 1
    assert len(manifest["config_sha256"]) == 64
    assert "provider-secret" not in "".join(path.read_text() for path in output.iterdir())
    original = (output / "report.json").read_text()
    assert main(args) == 2
    assert (output / "report.json").read_text() == original


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
