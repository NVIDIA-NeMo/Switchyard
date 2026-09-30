# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded scheduling, cleanup, and recorded-outcome validity."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from switchyard.sim.dataset import Dataset
from switchyard.sim.evaluate import evaluate, score
from switchyard.sim.models import HarborRun, Trial


def dataset(count=4):
    trials = tuple(
        Trial(
            str(i),
            str(i),
            "fast",
            ({"role": "user", "content": [{"type": "text", "text": f"input {i}"}]},),
            reward=i / max(count, 1),
            cost_usd=2,
        )
        for i in range(count)
    )
    return Dataset.from_runs({"fast": HarborRun(trials)}, input_target="fast")


class FakeRunner:
    def __init__(self, delay=0.001):
        self.active = self.peak = self.calls = self.cancelled = 0
        self.delay = delay
        self.headers = []
        self.targets = [SimpleNamespace(target="fast", model="model-fast")]

    def validate_decision_route(self, route, *, allow_response):
        assert route == "auto" and allow_response is False
        return self.targets

    async def decide(self, request, *, headers, allow_response):
        assert allow_response is False
        self.headers.append(headers)
        self.calls += 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self.delay)
            return decision(request)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.active -= 1


def decision(request):
    return SimpleNamespace(
        selected=SimpleNamespace(target="fast", model="model-fast"),
        outcome=SimpleNamespace(request=request, response=None, metadata=None),
        calls=[],
        fallbacks=[],
        duration_seconds=0.01,
    )


async def test_bounded_concurrency_and_progressive_results():
    runner = FakeRunner()
    rows = []
    report = await evaluate(dataset(11), runner, route="auto", concurrency=3, on_result=rows.append)
    assert runner.peak == 3 and runner.active == 0
    assert len(rows) == 11
    assert report.to_dict()["counts"]["scored"] == 11
    assert all(row.routing_cost_usd == 0 for row in rows)
    assert len({headers["x-switchyard-session-id"] for headers in runner.headers}) == 11
    assert all(headers.get("x-switchyard-session-final") == "true" for headers in runner.headers)


async def test_sessions_are_isolated_between_evaluations():
    runner = FakeRunner()
    await evaluate(dataset(1), runner, route="auto")
    await evaluate(dataset(1), runner, route="auto")
    assert runner.headers[0] != runner.headers[1]


async def test_target_mapping_is_validated_before_any_calls():
    runner = FakeRunner()
    runner.targets.append(SimpleNamespace(target="unknown"))
    with pytest.raises(ValueError, match="do not match"):
        await evaluate(dataset(), runner, route="auto")
    assert runner.calls == 0


async def test_timeout_is_visible_and_does_not_claim_zero_cost():
    runner = FakeRunner(delay=1)
    rows = []
    report = await evaluate(dataset(2), runner, route="auto", timeout=0.01, on_result=rows.append)
    assert report.to_dict()["counts"]["errors"] == 2
    assert runner.active == 0 and runner.cancelled == 2
    assert all(row.routing_cost_usd is None and row.routing_calls is None for row in rows)


async def test_callback_failure_cancels_pending_work():
    class SlowRunner(FakeRunner):
        async def decide(self, request, **kwargs):
            self.delay = 0 if request["messages"][0]["content"][0]["text"] == "input 0" else 10
            return await super().decide(request, **kwargs)

    runner = SlowRunner()

    def fail(_):
        raise RuntimeError("sink failed")

    with pytest.raises(RuntimeError, match="sink failed"):
        await evaluate(dataset(10), runner, route="auto", concurrency=3, on_result=fail)
    assert runner.calls == 3 and runner.cancelled == 2 and runner.active == 0


async def test_caller_cancellation_drains_all_workers():
    runner = FakeRunner(delay=10)
    work = asyncio.create_task(evaluate(dataset(10), runner, route="auto", concurrency=2))
    while runner.active < 2:
        await asyncio.sleep(0)
    work.cancel()
    with pytest.raises(asyncio.CancelledError):
        await work
    assert runner.cancelled == 2 and runner.active == 0 and runner.calls == 2


@pytest.mark.parametrize("mutation", ["answer", "messages", "instructions", "tools", "target"])
def test_scorer_rejects_unsupported_counterfactuals(mutation):
    task = dataset(1).tasks[0]
    native = decision({"messages": list(task.messages)})
    if mutation == "answer":
        native.outcome.response = "new answer"
    elif mutation == "target":
        native.selected.target = "missing"
    else:
        native.outcome.request[mutation] = [{"text": "changed task"}]
    with pytest.raises(ValueError):
        score(task, native)


def test_routing_measurements_and_pricing_are_separate_from_task_cost():
    task = dataset(1).tasks[0]
    native = decision({"messages": list(task.messages)})
    native.calls = [
        SimpleNamespace(
            model="judge", is_success=True, usage={"input_tokens": 20, "output_tokens": 5}
        )
    ]
    row = score(task, native)
    assert row.routing_calls == 1 and row.routing_usage["input_tokens"] == 20
    assert row.routing_usage["cached_input_tokens"] is None
    assert row.routing_cost_usd is None and row.outcome.cost_usd == 2
    assert score(task, native, price_call=lambda call: 0.1).routing_cost_usd == 0.1


async def test_failure_is_an_error_without_exposing_provider_body():
    class BrokenRunner(FakeRunner):
        async def decide(self, *args, **kwargs):
            error = RuntimeError("provider response containing a secret")
            error.kind = "secret"
            error.upstream_status = 503
            error.target = "secret"
            raise error

    rows = []
    report = await evaluate(dataset(1), BrokenRunner(), route="auto", on_result=rows.append)
    assert report.to_dict()["counts"]["errors"] == 1
    assert "secret" not in rows[0].error
    assert rows[0].routing_error_kind is None
    assert rows[0].routing_error_status is None
    assert rows[0].routing_error_target is None


async def test_recorded_model_mismatch_fails_before_calls():
    data = dataset(1)
    recorded = replace(data.tasks[0].trials["fast"][0], model="different-model")
    data = Dataset.from_runs({"fast": HarborRun((recorded,))}, input_target="fast")
    runner = FakeRunner()
    with pytest.raises(ValueError, match="do not match configured model"):
        await evaluate(data, runner, route="auto")
    assert runner.calls == 0


async def test_conflicting_recorded_models_require_explicit_equivalent_aliases():
    data = dataset(1)
    first = replace(data.tasks[0].trials["fast"][0], model="model-fast")
    second = replace(first, trial_id="repeat", model="provider/model-fast")
    data = Dataset.from_runs({"fast": HarborRun((first, second))}, input_target="fast")
    runner = FakeRunner()
    with pytest.raises(ValueError, match="conflicting recorded models"):
        await evaluate(data, runner, route="auto")
    assert runner.calls == 0
    report = await evaluate(
        data, runner, route="auto", model_aliases={"provider/model-fast": "model-fast"}
    )
    validation = report.to_dict()["coverage"]["model_validation"]["fast"]
    assert validation == {
        "configured_model": "model-fast",
        "recorded_models": ["model-fast", "provider/model-fast"],
        "known_trials": 2,
        "unknown_trials": 0,
        "verified": True,
    }


async def test_unknown_models_are_visible_and_fixed_route_accepts_extra_baselines():
    fast = dataset(1).tasks[0].trials["fast"][0]
    strong = replace(fast, target="strong", model="other-model")
    data = Dataset.from_runs(
        {"fast": HarborRun((fast,)), "strong": HarborRun((strong,))}, input_target="fast"
    )
    report = await evaluate(data, FakeRunner(), route="auto")
    assert report.complete
    validation = report.to_dict()["coverage"]["model_validation"]
    assert validation["fast"]["unknown_trials"] == 1
    assert validation["fast"]["verified"] is False
    assert report.to_dict()["comparison"]["targets"].keys() == {"fast", "strong"}


@pytest.mark.parametrize("mode", ["raises", "nan", "negative", "failed-routing"])
async def test_pricing_errors_stop_and_drain_workers(mode):
    call = SimpleNamespace(model="judge", is_success=True, usage={"input_tokens": 1})

    class PaidRunner(FakeRunner):
        async def decide(self, request, **kwargs):
            self.delay = 0 if request["messages"][0]["content"][0]["text"] == "input 0" else 10
            native = await super().decide(request, **kwargs)
            if mode == "failed-routing":
                error = RuntimeError("provider failure")
                error.calls = [call]
                raise error
            native.calls = [call]
            return native

    def price(_):
        if mode in {"raises", "failed-routing"}:
            raise ValueError("pricing unavailable")
        return float("nan") if mode == "nan" else -1

    runner = PaidRunner()
    with pytest.raises(ValueError, match="pricing unavailable|price_call must return"):
        await evaluate(dataset(10), runner, route="auto", concurrency=3, price_call=price)
    assert runner.calls == 3 and runner.cancelled == 2 and runner.active == 0


async def test_unsupported_selection_retains_known_routing_cost():
    class RewritingRunner(FakeRunner):
        async def decide(self, request, **kwargs):
            native = await super().decide(request, **kwargs)
            native.outcome.request["instructions"] = [{"content": "changed task"}]
            native.calls = [SimpleNamespace(model="judge", is_success=True, usage={})]
            return native

    rows = []
    report = await evaluate(
        dataset(1),
        RewritingRunner(),
        route="auto",
        price_call=lambda _: 0.25,
        on_result=rows.append,
    )
    assert report.errors == 1
    assert rows[0].outcome is None
    assert rows[0].routing_cost_usd == 0.25


def test_public_scorer_checks_recorded_models_and_accepts_explicit_aliases():
    recorded = replace(dataset(1).tasks[0].trials["fast"][0], model="provider/model-fast")
    task = Dataset.from_runs({"fast": HarborRun((recorded,))}, input_target="fast").tasks[0]
    native = decision({"messages": list(task.messages)})
    with pytest.raises(ValueError, match="do not match configured model"):
        score(task, native)
    row = score(task, native, model_aliases={"provider/model-fast": "model-fast"})
    assert row.error is None and row.outcome == task.outcomes["fast"]
    native.selected.model = "unrelated-model"
    with pytest.raises(ValueError, match="do not match configured model"):
        score(task, native, model_aliases={"provider/model-fast": "model-fast"})


def test_usage_crosses_native_boundary_once_per_call():
    class CountingCall:
        model = "judge"
        is_success = True
        reads = 0

        @property
        def usage(self):
            self.reads += 1
            return {"input_tokens": 3, "output_tokens": 1}

    task = dataset(1).tasks[0]
    native = decision({"messages": list(task.messages)})
    calls = [CountingCall(), CountingCall()]
    native.calls = calls
    row = score(task, native)
    assert row.routing_usage["input_tokens"] == 6
    assert row.routing_usage["output_tokens"] == 2
    assert row.routing_usage["cached_input_tokens"] is None
    assert [call.reads for call in calls] == [1, 1]
