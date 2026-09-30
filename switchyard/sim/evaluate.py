# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route and score completed tasks through the public Switchyard Python API."""

from __future__ import annotations

import asyncio
import copy
import math
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

from .dataset import Dataset
from .models import Result, Task, Trial
from .report import Report

if TYPE_CHECKING:
    from switchyard.runner import Decision, RoutingCall, Runner

PriceCall = Callable[["RoutingCall"], float | None]
_TOKEN_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_creation_input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
)


def score(
    task: Task,
    decision: Decision,
    *,
    price_call: PriceCall | None = None,
    model_aliases: Mapping[str, str] | None = None,
) -> Result:
    """Score a native decision against recorded outcomes, without making calls.

    Saved outcomes support target selection only. They cannot score an answer
    produced during routing or changes to the original task conversation.
    ``price_call`` may estimate a routing call's USD cost using caller-owned
    pricing; return ``None`` when usage or rates are unavailable.
    Known recorded model IDs must match the decision, optionally using explicit
    ``model_aliases``. Missing recorded IDs cannot be verified; ``evaluate``
    reports their coverage separately.
    """
    _validate_selection(task, decision, _aliases(model_aliases))
    return _result(task, decision, price_call=price_call)


def _validate_selection(task: Task, decision: Decision, aliases: Mapping[str, str]) -> None:
    target = decision.selected.target
    if target not in task.outcomes:
        raise ValueError(f"selected target {target!r} has no recorded outcome")
    _validate_models(task.trials.get(target, ()), target, decision.selected.model, aliases)
    outcome = decision.outcome
    if outcome.response is not None:
        raise ValueError("recorded task outcomes cannot score a response produced during routing")
    request = outcome.request
    if (
        request.get("messages") != list(task.messages)
        or request.get("instructions")
        or request.get("tools")
    ):
        raise ValueError("recorded task outcomes cannot score a rewritten task request")


def _aliases(model_aliases: Mapping[str, str] | None) -> dict[str, str]:
    aliases = dict(model_aliases or {})
    if any(
        not isinstance(name, str) or not name.strip() for pair in aliases.items() for name in pair
    ):
        raise ValueError("model_aliases must map non-empty recorded IDs to configured IDs")
    return aliases


def _validate_models(
    trials: Iterable[Trial], target: str, model: str, aliases: Mapping[str, str]
) -> dict[str, object]:
    recorded_models: set[str] = set()
    known = unknown = 0
    for trial in trials:
        if trial.model is None:
            unknown += 1
        elif not isinstance(trial.model, str) or not trial.model.strip():
            raise ValueError(f"invalid recorded model for target {target!r}")
        else:
            known += 1
            recorded_models.add(trial.model)
    resolved_models = {aliases.get(recorded, recorded) for recorded in recorded_models}
    if len(resolved_models) > 1:
        raise ValueError(
            f"conflicting recorded models for target {target!r}: {sorted(resolved_models)!r}"
        )
    if resolved_models and resolved_models != {model}:
        raise ValueError(
            f"recorded models {sorted(recorded_models)!r} do not match configured model "
            f"{model!r} for target {target!r}; provide model_aliases only for equivalent IDs"
        )
    return {
        "configured_model": model,
        "recorded_models": sorted(recorded_models),
        "known_trials": known,
        "unknown_trials": unknown,
        "verified": known > 0 and unknown == 0,
    }


def _result(
    task: Task,
    decision: Decision,
    *,
    price_call: PriceCall | None,
    error: str | None = None,
) -> Result:
    metadata = decision.outcome.metadata
    calls = decision.calls
    return Result(
        task_id=task.task_id,
        baselines=task.outcomes,
        target=decision.selected.target,
        model=decision.selected.model,
        outcome=task.outcomes.get(decision.selected.target) if error is None else None,
        decision_id=metadata.outcome_id if metadata else None,
        algorithm=metadata.algorithm if metadata else None,
        evidence=metadata.evidence if metadata else None,
        fallbacks=tuple(target.target for target in decision.fallbacks),
        routing_seconds=decision.duration_seconds,
        routing_calls=len(calls),
        routing_failed_calls=sum(not call.is_success for call in calls),
        routing_cost_usd=_cost(calls, price_call),
        routing_usage=_usage(calls),
        error=error,
    )


def _cost(calls: Sequence[RoutingCall], price_call: PriceCall | None) -> float | None:
    if not calls:
        return 0.0
    if price_call is None:
        return None
    values = [price_call(call) for call in calls]
    for value in values:
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError("price_call must return a finite non-negative cost or None")
    return (
        math.fsum(value for value in values if value is not None)
        if all(value is not None for value in values)
        else None
    )


def _usage(calls: Sequence[RoutingCall]) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    usages = [call.usage for call in calls]
    for field in _TOKEN_FIELDS:
        values = [usage.get(field) if usage is not None else None for usage in usages]
        result[field] = (
            sum(value for value in values if value is not None)
            if all(value is not None for value in values)
            else None
        )
    return result


async def evaluate(
    dataset: Dataset,
    runner: Runner,
    *,
    route: str,
    concurrency: int = 8,
    timeout: float = 60.0,
    on_result: Callable[[Result], None] | None = None,
    price_call: PriceCall | None = None,
    model_aliases: Mapping[str, str] | None = None,
) -> Report:
    """Evaluate a validated cohort with bounded work and progressive results.

    Use a fresh Runner for each independent experiment. The runner owns native
    algorithm state and retries. Each task gets a distinct session ID. Use
    ``concurrency=1`` for reproducible task ordering with a seeded random router.
    Known recorded model IDs must match configured targets. ``model_aliases``
    maps recorded IDs to configured IDs explicitly; provider prefixes are never
    guessed. Unknown model IDs remain visible in the report's input coverage.

    ``timeout`` covers the complete routing decision. Errors become result rows;
    callback errors or cancellation stop and drain pending work. Keep callbacks
    short. Cancelling local work cannot revoke provider requests already sent.
    """
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    if not dataset.tasks or len({task.task_id for task in dataset.tasks}) != len(dataset.tasks):
        raise ValueError("dataset must contain non-empty, unique tasks")
    targets = runner.validate_decision_route(route, allow_response=False)
    configured = {target.target for target in targets}
    if not configured or not configured.issubset(dataset.targets):
        raise ValueError(
            f"configured targets {sorted(configured)!r} do not match available recorded targets {sorted(dataset.targets)!r}"
        )
    aliases = _aliases(model_aliases)
    model_validation = {
        target.target: _validate_models(
            (trial for task in dataset.tasks for trial in task.trials[target.target]),
            target.target,
            target.model,
            aliases,
        )
        for target in targets
    }
    run_id = uuid.uuid4().hex
    report = Report(
        len(dataset.tasks),
        dataset.targets,
        run_id=run_id,
        concurrency=concurrency,
        coverage={**dataset.coverage, "model_validation": model_validation},
    )

    async def route_task(index: int, task: Task) -> Result:
        started = time.monotonic()
        try:
            decision = await asyncio.wait_for(
                runner.decide(
                    {"model": route, "messages": copy.deepcopy(list(task.messages))},
                    headers={"x-switchyard-session-id": f"sim-{run_id}-{index}"},
                    allow_response=False,
                ),
                timeout=timeout,
            )
        except Exception as error:
            # Native decision errors expose completed observations without provider
            # bodies. Other errors (including timeouts) cannot prove call counts.
            calls = getattr(error, "calls", None)
            return Result(
                task.task_id,
                task.outcomes,
                routing_seconds=time.monotonic() - started,
                routing_calls=len(calls) if calls is not None else None,
                routing_failed_calls=sum(not call.is_success for call in calls)
                if calls is not None
                else None,
                routing_cost_usd=_cost(calls, price_call) if calls is not None else None,
                routing_usage=_usage(calls) if calls is not None else {},
                error=f"routing failed: {type(error).__name__}",
            )
        score_error = None
        try:
            _validate_selection(task, decision, aliases)
        except ValueError as error:
            score_error = str(error)
        return _result(task, decision, price_call=price_call, error=score_error)

    pending: set[asyncio.Task[Result]] = set()
    done: set[asyncio.Task[Result]] = set()
    remaining = iter(enumerate(dataset.tasks))

    def fill() -> None:
        while len(pending) < concurrency:
            item = next(remaining, None)
            if item is None:
                break
            pending.add(asyncio.create_task(route_task(*item)))

    try:
        fill()
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            # Consume every completed task before refilling. All callbacks run on
            # this coordinator, so sinks need no locks.
            for future in done:
                result = future.result()
                report.add(result)
                if on_result is not None:
                    on_result(result)
            fill()
    finally:
        for future in pending:
            future.cancel()
        if pending or done:
            await asyncio.gather(*pending, *done, return_exceptions=True)
    return report
