# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ATIF documents and their task-routing projection, independent of the producer."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .models import Trial

_SUPPORTED_VERSIONS = frozenset(f"ATIF-v1.{minor}" for minor in range(8))


class Trajectory:
    """An owned ATIF document, including fields unused by task simulation.

    Validates the version and steps container on construction. Input projection
    validates the fields it consumes; this is not a full ATIF schema validator.
    Converter functions can return this type without inheriting a base class.
    """

    def __init__(self, data: Mapping[str, Any]) -> None:
        _steps(data)
        self._data = deepcopy(dict(data))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Trajectory:
        """Copy a JSON-compatible ATIF document without dropping extension fields."""
        return cls(data)

    def to_dict(self) -> dict[str, Any]:
        """Return an independent copy, including the complete recorded history."""
        return deepcopy(self._data)

    def initial_messages(self, *, task_input: str | None = None) -> tuple[dict[str, object], ...]:
        """Extract text input before the first agent step, without later evidence.

        ``task_input`` supplies original task text when the recording lacks it
        or contains copied continuation context. Copied context may summarize
        earlier execution, so it is never used as task-routing input.
        """
        return _initial_messages(self._data, task_input)

    def to_trial(
        self,
        *,
        task_id: str,
        trial_id: str,
        target: str,
        reward: float | None = None,
        cost_usd: float | None = None,
        cost_source: str | None = None,
        duration_seconds: float | None = None,
        task_checksum: str | None = None,
        model: str | None = None,
        source: str | None = None,
        usage: Mapping[str, int | None] | None = None,
        error: str | None = None,
        task_input: str | None = None,
    ) -> Trial:
        """Project into task evidence without retaining the full trajectory.

        Identity, outcomes, and accounting are explicit: ATIF alone does not
        define verifier rewards or a cross-run task identity. Model identity is
        also explicit because per-step models can override the ATIF agent default.
        """
        return Trial(
            task_id=task_id,
            trial_id=trial_id,
            target=target,
            messages=self.initial_messages(task_input=task_input),
            reward=reward,
            cost_usd=cost_usd,
            cost_source=cost_source,
            duration_seconds=duration_seconds,
            task_checksum=task_checksum,
            model=model,
            source=source,
            usage=dict(usage) if usage is not None else {},
            error=error,
        )


def _steps(data: Mapping[str, Any]) -> list[Any]:
    if not isinstance(data, Mapping):
        raise ValueError("ATIF trajectory must be an object")
    version = data.get("schema_version")
    if not isinstance(version, str) or version not in _SUPPORTED_VERSIONS:
        raise ValueError("unsupported or missing ATIF schema_version")
    steps = data.get("steps")
    if not isinstance(steps, list):
        raise ValueError("ATIF steps must be an array")
    return steps


def _initial_messages(
    data: Mapping[str, Any], task_input: str | None
) -> tuple[dict[str, object], ...]:
    """Share projection with file importers without copying unused history."""
    messages: list[dict[str, object]] = []
    has_user_input = False
    for step in _steps(data):
        if not isinstance(step, dict):
            raise ValueError("ATIF step must be an object")
        role = step.get("source")
        if role == "agent":
            break
        if role not in ("system", "user"):
            raise ValueError("unsupported ATIF input source")
        copied = step.get("is_copied_context")
        if copied is not None and not isinstance(copied, bool):
            raise ValueError("ATIF is_copied_context must be a boolean or null")
        if copied:
            if task_input is None:
                raise ValueError("copied ATIF context requires explicit task_input")
            return _task_messages(task_input)
        content = step.get("message")
        blocks: list[dict[str, str]] = []
        if isinstance(content, str):
            blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    raise ValueError("ATIF message content must be an object")
                if block.get("type") != "text" or not isinstance(block.get("text"), str):
                    raise ValueError("only text ATIF input is supported")
                blocks.append({"type": "text", "text": block["text"]})
        else:
            raise ValueError("ATIF input message must contain text")
        if blocks:
            messages.append({"role": role, "content": blocks})
            has_user_input |= role == "user" and any(block["text"].strip() for block in blocks)
    return tuple(messages) if has_user_input else _task_messages(task_input)


def _task_messages(text: str | None) -> tuple[dict[str, object], ...]:
    if text is None:
        raise ValueError("missing initial task input; provide task_inputs explicitly")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("task input must be a non-empty string")
    return ({"role": "user", "content": [{"type": "text", "text": text}]},)
