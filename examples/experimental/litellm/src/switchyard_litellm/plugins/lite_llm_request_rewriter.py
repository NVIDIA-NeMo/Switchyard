# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply Switchyard request deltas at LiteLLM's selected-deployment boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import CallTypes

_PATCH_FIELDS = frozenset(
    {
        "messages",
        "tools",
        "tool_choice",
        "temperature",
        "top_p",
        "top_k",
        "max_completion_tokens",
        "response_format",
        "reasoning_effort",
        "stream",
        "parallel_tool_calls",
        "prompt_cache_key",
        "prompt_cache_retention",
        "safety_identifier",
        "service_tier",
        "store",
        "stream_options",
        "top_logprobs",
        "user",
        "stop",
    }
)


class LiteLLMRequestRewriter(CustomLogger):
    """Act as the callback half of a dual-role LiteLLM routing plugin."""

    def __init__(self) -> None:
        super().__init__()

    async def async_pre_call_deployment_hook(
        self,
        kwargs: dict[str, Any],
        call_type: CallTypes | None,
    ) -> dict[str, Any] | None:
        """Apply this plugin's private request patch after deployment selection."""
        del call_type
        located = self._locate_patch(kwargs)
        if located is None:
            return None
        metadata_key, metadata, signals, switchyard_signal, patch = located

        raw_set_values = patch.get("set", {})
        if not isinstance(raw_set_values, Mapping):
            raise ValueError("Switchyard LiteLLM request patch set must be a mapping")
        set_values = cast(Mapping[str, object], raw_set_values)
        unknown_set_fields = sorted(set(set_values) - _PATCH_FIELDS)
        if unknown_set_fields:
            raise ValueError(
                "Switchyard LiteLLM request patch cannot set fields: "
                + ", ".join(unknown_set_fields)
            )
        raw_remove_values = patch.get("remove", [])
        if (
            isinstance(raw_remove_values, (str, bytes, bytearray))
            or not isinstance(raw_remove_values, Sequence)
            or not all(isinstance(item, str) for item in raw_remove_values)
        ):
            raise ValueError("Switchyard LiteLLM request patch remove must be a string sequence")
        unknown_remove_fields = sorted(set(raw_remove_values) - _PATCH_FIELDS)
        if unknown_remove_fields:
            raise ValueError(
                "Switchyard LiteLLM request patch cannot remove fields: "
                + ", ".join(unknown_remove_fields)
            )

        rewritten = dict(kwargs)
        for field in cast(Sequence[str], raw_remove_values):
            rewritten.pop(field, None)
        rewritten.update(set_values)

        sanitized_signal = dict(switchyard_signal)
        sanitized_signal.pop("request_patch", None)
        sanitized_signals = dict(signals)
        sanitized_signals["switchyard"] = sanitized_signal
        sanitized_metadata = dict(metadata)
        sanitized_metadata["routing_plugin_signals"] = sanitized_signals
        rewritten[metadata_key] = sanitized_metadata
        return rewritten

    @staticmethod
    def _locate_patch(
        kwargs: Mapping[str, object],
    ) -> tuple[
        str,
        Mapping[str, object],
        Mapping[str, object],
        Mapping[str, object],
        Mapping[str, object],
    ] | None:
        for metadata_key in ("litellm_metadata", "metadata"):
            raw_metadata = kwargs.get(metadata_key)
            if not isinstance(raw_metadata, Mapping):
                continue
            metadata = cast(Mapping[str, object], raw_metadata)
            raw_signals = metadata.get("routing_plugin_signals")
            if not isinstance(raw_signals, Mapping):
                continue
            signals = cast(Mapping[str, object], raw_signals)
            raw_switchyard_signal = signals.get("switchyard")
            if not isinstance(raw_switchyard_signal, Mapping):
                continue
            switchyard_signal = cast(Mapping[str, object], raw_switchyard_signal)
            raw_patch = switchyard_signal.get("request_patch")
            if not isinstance(raw_patch, Mapping):
                continue
            return (
                metadata_key,
                metadata,
                signals,
                switchyard_signal,
                cast(Mapping[str, object], raw_patch),
            )
        return None


__all__ = ["LiteLLMRequestRewriter"]
