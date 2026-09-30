# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configured native decisions call only the providers needed by routing."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Thread
from typing import Any

import pytest

from switchyard.libsy import RoutingOutcome
from switchyard.runner import DecisionError, Runner
from switchyard_rust.runner import _decide


@dataclass
class JudgeStub:
    url: str = ""
    calls: list[dict[str, Any]] = field(default_factory=list)
    request_headers: list[dict[str, str]] = field(default_factory=list)
    response_text: str | None = None
    status: int = 200
    started: Event = field(default_factory=Event)
    release: Event | None = None
    completed: Event = field(default_factory=Event)
    wait_for_disconnect: bool = False
    disconnected: Event = field(default_factory=Event)


@pytest.fixture
def judge(monkeypatch: pytest.MonkeyPatch) -> Iterator[JudgeStub]:
    monkeypatch.setenv("SWITCHYARD_RUNNER_TEST_KEY", "provider-secret")
    stub = JudgeStub()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            stub.calls.append(body)
            stub.request_headers.append(
                {name.lower(): value for name, value in self.headers.items()}
            )
            stub.started.set()
            if stub.wait_for_disconnect:
                self.connection.settimeout(5)
                try:
                    if self.rfile.read(1) == b"":
                        stub.disconnected.set()
                except ConnectionResetError:
                    stub.disconnected.set()
                finally:
                    stub.completed.set()
                return
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
                payload = {
                    "error": {
                        "message": f"echoed provider-secret and private prompt {self.headers.get('authorization', '')}"
                    }
                }
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


@pytest.mark.parametrize("forward_auth", [False, True])
async def test_caller_headers_are_request_scoped_and_respect_configured_auth(
    judge: JudgeStub, forward_auth: bool
) -> None:
    source = deployment(judge.url)
    if forward_auth:
        source = source.replace('api_key_env = "SWITCHYARD_RUNNER_TEST_KEY"', "forward_auth = true")
    runner = Runner.from_toml(source)
    await asyncio.gather(
        *(
            runner.decide(
                request(),
                headers={
                    **session(identity),
                    "authorization": f"Bearer caller-{identity}",
                    "x-request-id": identity,
                },
            )
            for identity in ("first", "second")
        )
    )
    assert {
        headers.get("x-request-id"): headers.get("authorization")
        for headers in judge.request_headers
    } == {
        identity: f"Bearer caller-{identity}" if forward_auth else "Bearer provider-secret"
        for identity in ("first", "second")
    }

    judge.status = 401
    with pytest.raises(DecisionError) as caught:
        await runner.decide(
            request(),
            headers={**session("failed"), "authorization": "Bearer caller-failed"},
        )
    assert caught.value.upstream_status == 401
    assert "caller-failed" not in str(caught.value)
    assert "provider-secret" not in str(caught.value)
    assert "private prompt" not in str(caught.value)


@pytest.mark.parametrize("kind", ["classifier", "random", "random-zero", "random-reversed"])
def test_ambiguous_completion_target_aliases_fail_before_calls(
    judge: JudgeStub, tmp_path: Path, kind: str
) -> None:
    source = deployment(judge.url).replace('id = "strong/model"', 'id = "weak/model"')
    if kind != "classifier":
        targets = '["strong", "weak"]' if kind == "random-reversed" else '["weak", "strong"]'
        weights = "[0, 1]" if kind == "random-zero" else "[1, 99]"
        source = (
            source.split("[routes.classifier]")[0]
            + f"""
[routes.random]
id = "auto"
type = "random"
targets = {targets}
weights = {weights}
seed = 1
"""
        )
    path = tmp_path / "routes.toml"
    path.write_text(source)
    for load in (lambda: Runner.from_toml(source), lambda: Runner.load(path)):
        with pytest.raises(ValueError, match="completion targets.*model"):
            load()
    assert judge.calls == []


async def test_same_model_aliases_in_separate_fixed_routes_keep_target_identity(
    judge: JudgeStub,
) -> None:
    source = deployment(judge.url).split("[routes.classifier]")[0]
    source = source.replace('id = "strong/model"', 'id = "weak/model"')
    for target in ("weak", "strong"):
        source += f"""
[routes.{target}]
id = "route-{target}"
type = "passthrough"
target = "{target}"
"""
    runner = Runner.from_toml(source)
    for target in ("weak", "strong"):
        selected = await runner.decide(request(model=f"route-{target}"))
        assert selected.selected.target == target
        assert selected.selected.model == "weak/model"
        assert [entry.target for entry in runner.validate_decision_route(f"route-{target}")] == [
            target
        ]
        assert selected.calls == []
    assert judge.calls == []


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


@pytest.mark.skipif(
    not os.supports_bytes_environ, reason="requires byte-valued environment variables"
)
def test_configuration_diagnostics_hide_non_unicode_api_keys(
    judge: JudgeStub, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(os.environb, b"SWITCHYARD_RUNNER_TEST_KEY", b"provider-secret\xff")
    source = deployment(judge.url)
    path = tmp_path / "routes.toml"
    path.write_text(source)
    for load in (lambda: Runner.from_toml(source), lambda: Runner.load(path)):
        with pytest.raises(ValueError, match="api_key_env SWITCHYARD_RUNNER_TEST_KEY") as caught:
            load()
        assert "provider-secret" not in str(caught.value)
        assert "not valid" in str(caught.value)
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


async def test_cancellation_before_first_step_does_not_start_native_work(judge: JudgeStub) -> None:
    runner = Runner.from_toml(deployment(judge.url))
    task = asyncio.create_task(runner.decide(request(), headers=session("never-started")))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    decision = await runner.decide(request(), headers=session("subsequent"))
    assert decision.selected.target == "weak"
    assert [call["model"] for call in judge.calls] == ["judge/model"]


@pytest.mark.parametrize("completion_error", [False, True])
async def test_repeated_cancellation_drains_bridge_before_propagating(
    completion_error: bool,
) -> None:
    bridge: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class PendingRunner:
        def _start_decision(self, *args: object, **kwargs: object) -> tuple[object, object]:
            started.set()
            return bridge, self

        def cancel(self) -> None:
            cancelled.set()

        def wait(self) -> None:
            assert bridge.done()

        def is_finished(self) -> bool:
            return False

    task = asyncio.create_task(_decide(PendingRunner(), request(), None, False))
    await started.wait()
    task.cancel("original cancellation")
    await cancelled.wait()
    assert not task.done()
    assert not bridge.cancelled()
    task.cancel("repeated cancellation")
    # Advance the event loop once so the repeated cancellation reaches the drain.
    await asyncio.sleep(0)
    assert not task.done()
    assert not bridge.cancelled()
    if completion_error:
        bridge.set_exception(RuntimeError("native failure while cancellation was draining"))
    else:
        bridge.set_result(None)
    with pytest.raises(asyncio.CancelledError) as caught:
        await task
    # Python 3.11 began propagating cancellation messages to task awaiters.
    if sys.version_info >= (3, 11):
        assert str(caught.value) == "original cancellation"
    assert bridge.done()
    assert not bridge.cancelled()


@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
async def test_decision_waits_for_completion_worker_to_exit(
    judge: JudgeStub, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    judge.release = Event()
    if outcome == "error":
        judge.status = 503
    runner = Runner.from_toml(deployment(judge.url))
    task = asyncio.create_task(runner.decide(request()))
    assert await asyncio.to_thread(judge.started.wait, 5)

    loop = asyncio.get_running_loop()
    if not hasattr(loop, "_write_to_self"):
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        pytest.skip("requires asyncio's socket wakeup implementation")
    write_to_self = loop._write_to_self
    entered = Event()
    release = Event()
    returned = Event()
    finished_before_worker_exit: list[bool] = []

    def gated_write() -> None:
        entered.set()
        write_to_self()
        assert release.wait(5)

    def observe_completion() -> None:
        if entered.wait(5):
            # The future's completion callback can run while its worker is still
            # inside call_soon_threadsafe. The public task must remain pending.
            finished_before_worker_exit.append(returned.wait(0.1))
        # Finishing this custom wakeup tail requires event-loop progress.
        loop.call_soon_threadsafe(release.set)

    observer = Thread(target=observe_completion)
    task.add_done_callback(lambda _: returned.set())
    monkeypatch.setattr(loop, "_write_to_self", gated_write)
    observer.start()
    try:
        if outcome == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            judge.release.set()
            if outcome == "error":
                with pytest.raises(DecisionError):
                    await task
            else:
                assert (await task).selected.target == "weak"
    finally:
        release.set()
        monkeypatch.setattr(loop, "_write_to_self", write_to_self)
        observer.join(timeout=5)
    assert finished_before_worker_exit == [False]


async def test_cancellation_first_arriving_during_bridge_join_is_drained() -> None:
    bridge: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    bridge.set_result(None)
    joining = Event()
    release = Event()

    class CompletedRunner:
        def _start_decision(self, *args: object, **kwargs: object) -> tuple[object, object]:
            return bridge, self

        def cancel(self) -> None:
            pytest.fail("routing was already complete when cancellation arrived")

        def wait(self) -> None:
            joining.set()
            assert release.wait(5)

        def is_finished(self) -> bool:
            return False

    task = asyncio.create_task(_decide(CompletedRunner(), request(), None, False))
    try:
        assert await asyncio.to_thread(joining.wait, 5)
        task.cancel("cancel during join")
        await asyncio.sleep(0)
        task.cancel("repeated cancellation")
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError) as caught:
        await task
    if sys.version_info >= (3, 11):
        assert str(caught.value) == "cancel during join"


def test_asyncio_shutdown_keeps_an_existing_bridge_join_alive() -> None:
    joining = Event()
    shutdown_started = Event()
    release = Event()
    returned = Event()
    joined = Event()
    finished_before_join: list[bool] = []

    async def main() -> None:
        bridge: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        bridge.set_result(None)

        class CompletedRunner:
            def _start_decision(self, *args: object, **kwargs: object) -> tuple[object, object]:
                return bridge, self

            def wait(self) -> None:
                joining.set()
                assert release.wait(5)
                joined.set()

            def is_finished(self) -> bool:
                return False

        task = asyncio.create_task(_decide(CompletedRunner(), request(), None, False))
        task.add_done_callback(lambda _: returned.set())
        assert await asyncio.to_thread(joining.wait, 5)
        asyncio.get_running_loop().call_soon(shutdown_started.set)

    def observe_shutdown() -> None:
        if shutdown_started.wait(5):
            finished_before_join.append(returned.wait(0.1))
        release.set()

    observer = Thread(target=observe_shutdown)
    observer.start()
    try:
        asyncio.run(main())
    finally:
        release.set()
        observer.join(timeout=5)
    assert joined.is_set()
    assert finished_before_join == [False]


@pytest.mark.parametrize("shutdown", [False, True], ids=["explicit-cancel", "asyncio-shutdown"])
async def test_cancellation_finishes_bridge_before_interpreter_shutdown(
    judge: JudgeStub, tmp_path: Path, shutdown: bool
) -> None:
    judge.wait_for_disconnect = True
    config = tmp_path / "routes.toml"
    config.write_text(deployment(judge.url))
    script = """
import asyncio
import sys
from switchyard.runner import Runner

async def main():
    runner = Runner.load(sys.argv[1])
    task = asyncio.ensure_future(runner.decide({
        "model": "auto",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "task"}]}],
    }))
    await asyncio.to_thread(sys.stdin.readline)
    if sys.argv[2] == "False":
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

asyncio.run(main())
"""
    # Repeat immediate process exit to exercise the completion thread scheduling
    # race, including cancellation performed by asyncio.run itself.
    for _ in range(5):
        judge.started.clear()
        judge.disconnected.clear()
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            script,
            str(config),
            str(shutdown),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            assert await asyncio.to_thread(judge.started.wait, 5)
            stdout, stderr = await asyncio.wait_for(process.communicate(b"\n"), timeout=10)
            assert process.returncode == 0, stderr.decode()
            assert not stdout
            assert not stderr, stderr.decode()
            assert await asyncio.to_thread(judge.disconnected.wait, 5)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
