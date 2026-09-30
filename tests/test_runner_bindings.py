# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configured native decisions call only the providers needed by routing."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Thread
from typing import Any

import pytest

from switchyard.libsy import RoutingOutcome
from switchyard.runner import DecisionError, Runner


@dataclass
class JudgeStub:
    url: str = ""
    calls: list[dict[str, Any]] = field(default_factory=list)
    response_text: str | None = None
    status: int = 200
    started: Event = field(default_factory=Event)
    release: Event | None = None
    completed: Event = field(default_factory=Event)


@pytest.fixture
def judge(monkeypatch: pytest.MonkeyPatch) -> Iterator[JudgeStub]:
    monkeypatch.setenv("SWITCHYARD_RUNNER_TEST_KEY", "provider-secret")
    stub = JudgeStub()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            stub.calls.append(body)
            stub.started.set()
            if stub.release is not None:
                stub.release.wait(timeout=5)
            verdict = {
                "crux": "bounded task",
                "primary_rule": "SUP-1",
                "capability_boundary": "supported",
                "p_solve": 0.1 if "TASK_REQUIRES_STRONG" in json.dumps(body) else 0.9,
            }
            if stub.status == 200:
                payload = {
                    "id": "judge-response",
                    "model": body["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": stub.response_text or json.dumps(verdict),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 17,
                        "completion_tokens": 8,
                        "total_tokens": 25,
                        "prompt_tokens_details": {"cached_tokens": 5},
                        "completion_tokens_details": {"reasoning_tokens": 3},
                    },
                }
            else:
                payload = {"error": {"message": "echoed provider-secret and private prompt"}}
            encoded = json.dumps(payload).encode()
            self.send_response(stub.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            try:
                self.wfile.write(encoded)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                stub.completed.set()

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    stub.url = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        yield stub
    finally:
        if stub.release is not None:
            stub.release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def deployment(url: str, *, escalation: bool = False) -> str:
    classifier = (
        'mode = "escalation"\nescalation = { confirmations = 2 }'
        if escalation
        else 'base_threshold = 0.5\nclassify_trigger = "new_session"'
    )
    return f"""
schema_version = 1
[llm_clients.provider]
format = "openai_chat"
base_url = {json.dumps(url)}
api_key_env = "SWITCHYARD_RUNNER_TEST_KEY"
max_retries = 0
[targets.judge]
id = "judge/model"
llm_client = "provider"
[targets.strong]
id = "strong/model"
llm_client = "provider"
[targets.weak]
id = "weak/model"
llm_client = "provider"
[routes.classifier]
id = "auto"
type = "llm_classifier"
classifier_target = "judge"
strong_target = "strong"
weak_target = "weak"
{classifier}
[routes.fixed]
id = "fixed"
type = "passthrough"
target = "weak"
"""


def request(text: str = "easy task", *, model: str = "auto") -> dict[str, object]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": [{"type": "text", "text": text}]}],
    }


def session(identity: str) -> dict[str, str]:
    return {"x-switchyard-session-id": identity}


async def test_native_judge_only_decision_preserves_targets_and_usage(
    judge: JudgeStub, tmp_path: Path
) -> None:
    path = tmp_path / "routes.toml"
    path.write_text(deployment(judge.url))
    runner = Runner.load(path)

    targets = runner.validate_decision_route("auto")
    assert {(target.target, target.model) for target in targets} == {
        ("strong", "strong/model"),
        ("weak", "weak/model"),
    }
    assert judge.calls == []
    decision = await runner.decide(request(), headers=session("task-1"))

    assert (decision.selected.target, decision.selected.model) == ("weak", "weak/model")
    assert [(target.target, target.model) for target in decision.fallbacks] == [
        ("strong", "strong/model")
    ]
    assert isinstance(decision.outcome, RoutingOutcome)
    assert decision.outcome.selected_model_ids == ["weak/model", "strong/model"]
    assert decision.outcome.request["model"] == "weak/model"
    assert decision.outcome.response is None
    assert decision.outcome.metadata.evidence["source"] == "llm-classifier"
    assert [call["model"] for call in judge.calls] == ["judge/model"]
    (call,) = decision.calls
    assert call.model == "judge/model"
    assert call.is_success
    assert call.usage == {
        "input_tokens": 12,
        "cached_input_tokens": 5,
        "cache_creation_input_tokens": None,
        "output_tokens": 8,
        "total_tokens": 25,
        "reasoning_tokens": 3,
    }
    assert decision.duration_seconds >= call.duration_seconds >= 0


async def test_passthrough_decision_never_calls_answer_model(judge: JudgeStub) -> None:
    runner = Runner.from_toml(deployment(judge.url))
    decision = await runner.decide(request(model="fixed"))
    assert decision.selected.target == "weak"
    assert decision.calls == []
    assert decision.outcome.response is None
    assert judge.calls == []


async def test_affinity_and_observations_are_isolated_across_sessions_and_runners(
    judge: JudgeStub,
) -> None:
    runner = Runner.from_toml(deployment(judge.url))
    first = await runner.decide(request(), headers=session("shared"))
    retained = await runner.decide(request("TASK_REQUIRES_STRONG"), headers=session("shared"))
    weak, strong = await asyncio.gather(
        runner.decide(request(), headers=session("independent-weak")),
        runner.decide(request("TASK_REQUIRES_STRONG"), headers=session("independent-strong")),
    )
    fresh = await Runner.from_toml(deployment(judge.url)).decide(
        request("TASK_REQUIRES_STRONG"), headers=session("shared")
    )
    assert first.selected.target == retained.selected.target == weak.selected.target == "weak"
    assert strong.selected.target == fresh.selected.target == "strong"
    assert retained.calls == []
    assert retained.outcome.metadata.evidence["source"] == "retained"
    assert [len(decision.calls) for decision in (first, weak, strong, fresh)] == [1, 1, 1, 1]
    assert len(judge.calls) == 4


@pytest.mark.parametrize("advisor", [False, True])
async def test_response_based_route_requires_opt_in_before_calls(
    judge: JudgeStub, advisor: bool
) -> None:
    source = deployment(judge.url, escalation=True)
    if advisor:
        source = (
            source.split("[routes.classifier]")[0]
            + """
[routes.advisor]
id = "auto"
type = "advisor"
executor_target = "weak"
advisor_target = "judge"
"""
        )
    runner = Runner.from_toml(source)
    with pytest.raises(ValueError, match="allow_response=True"):
        runner.validate_decision_route("auto")
    with pytest.raises(ValueError, match="allow_response=True"):
        await runner.decide(request())
    assert runner.validate_decision_route("auto", allow_response=True)
    assert judge.calls == []


async def test_invalid_verdict_preserves_fail_open_evidence_and_call_cost(judge: JudgeStub) -> None:
    judge.response_text = "not valid routing JSON"
    runner = Runner.from_toml(deployment(judge.url))
    decision = await runner.decide(request(), headers=session("malformed"))
    assert decision.selected.target == "strong"
    assert decision.outcome.metadata.evidence == {
        "source": "fail_open",
        "reason_code": "parse_error",
    }
    assert len(decision.calls) == 1
    assert decision.calls[0].is_success
    assert decision.calls[0].usage["input_tokens"] == 12
    assert [call["model"] for call in judge.calls] == ["judge/model"]


async def test_provider_failure_keeps_observations_and_safe_diagnostics(judge: JudgeStub) -> None:
    judge.status = 503
    runner = Runner.from_toml(deployment(judge.url))
    with pytest.raises(DecisionError) as caught:
        await runner.decide(request())
    error = caught.value
    assert error.kind == "upstream_http"
    assert error.upstream_status == 503
    assert error.target == "judge/model"
    assert "provider-secret" not in str(error)
    assert "private prompt" not in str(error)
    (call,) = error.calls
    assert call.model == "judge/model"
    assert not call.is_success
    assert call.usage is None
    assert error.duration_seconds >= call.duration_seconds >= 0


async def test_bad_request_and_configuration_fail_before_calls(
    judge: JudgeStub, tmp_path: Path
) -> None:
    runner = Runner.from_toml(deployment(judge.url))
    with pytest.raises(ValueError, match="unknown route"):
        runner.validate_decision_route("missing")
    with pytest.raises(ValueError, match="non-empty model"):
        await runner.decide({"messages": []})
    with pytest.raises(ValueError):
        await runner.decide(request(), headers={"invalid\nheader": "value"})
    with pytest.raises(ValueError) as caught:
        Runner.from_toml('schema_version = "provider-secret"')
    assert "provider-secret" not in str(caught.value)
    assert "TOML" in str(caught.value)
    assert "byte" in str(caught.value)
    invalid_path = tmp_path / "invalid.toml"
    invalid_path.write_text('schema_version = "provider-secret"')
    with pytest.raises(ValueError) as caught:
        Runner.load(invalid_path)
    assert "provider-secret" not in str(caught.value)
    assert "byte" in str(caught.value)
    with pytest.raises(ValueError, match="NotFound"):
        Runner.load(tmp_path / "absent.toml")
    assert judge.calls == []


def test_configuration_diagnostics_name_missing_target_and_environment(
    judge: JudgeStub, monkeypatch: pytest.MonkeyPatch
) -> None:
    unknown = deployment(judge.url).replace('strong_target = "strong"', 'strong_target = "missing"')
    with pytest.raises(ValueError, match="unknown target missing"):
        Runner.from_toml(unknown)
    monkeypatch.delenv("SWITCHYARD_RUNNER_TEST_KEY")
    with pytest.raises(ValueError, match="api_key_env SWITCHYARD_RUNNER_TEST_KEY"):
        Runner.from_toml(deployment(judge.url))
    assert judge.calls == []


async def test_cancellation_does_not_complete_or_retain_a_decision(judge: JudgeStub) -> None:
    judge.release = Event()
    runner = Runner.from_toml(deployment(judge.url))
    future = asyncio.ensure_future(runner.decide(request(), headers=session("cancelled")))
    assert await asyncio.to_thread(judge.started.wait, 5)
    future.cancel()
    with pytest.raises(asyncio.CancelledError):
        await future
    judge.release.set()
    assert await asyncio.to_thread(judge.completed.wait, 5)
    # A fresh call using the same identity must still consult the judge.
    decision = await runner.decide(request("TASK_REQUIRES_STRONG"), headers=session("cancelled"))
    assert decision.selected.target == "strong"
    assert len(decision.calls) == 1
    assert [call["model"] for call in judge.calls] == ["judge/model", "judge/model"]
