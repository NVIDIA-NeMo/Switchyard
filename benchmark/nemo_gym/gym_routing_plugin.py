# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize Gym responses and record the serving model before LiteLLM applies its alias."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from litellm.integrations.custom_logger import CustomLogger


def response_payload(response: Any) -> dict[str, Any]:
    """Keep missing detail counts unknown while satisfying Gym's object-shaped fields."""
    payload = response.model_dump()
    for item in payload.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "reasoning":
            continue
        if item.get("summary") is None:
            item["summary"] = []
        for content in item.get("content") or []:
            if isinstance(content, dict) and content.get("type") == "output_text":
                content["type"] = "reasoning_text"
    usage = payload.get("usage")
    if isinstance(usage, dict):
        for key, leaf in (
            ("input_tokens_details", "cached_tokens"),
            ("output_tokens_details", "reasoning_tokens"),
        ):
            if usage.get(key) is None:
                usage[key] = {leaf: None}
            elif isinstance(usage[key], dict):
                usage[key].setdefault(leaf, None)
    hidden = getattr(response, "_hidden_params", None)
    if hidden is not None:
        payload["_hidden_params"] = hidden
    return payload


class GymRoutingPlugin(CustomLogger):
    """Provide response compatibility and final-model attribution, not routing or accounting."""

    def __init__(self, results: Path) -> None:
        super().__init__()
        self.results = results

    async def async_post_call_success_hook(
        self, data: dict[str, Any], user_api_key_dict: Any, response: Any
    ) -> dict[str, Any]:
        """Save only response identity and the serving deployment; never prompts or credentials."""
        payload = response_payload(response)
        route = data.get("model")
        metadata = data.get("litellm_metadata") or {}
        record = {"response_id": payload.get("id"), "model": metadata.get("deployment")}
        if (
            route not in ("fixed", "routed")
            or not all(isinstance(value, str) and value.strip() for value in record.values())
            or record["model"] in ("fixed", "routed")
        ):
            raise ValueError("Missing response ID or serving deployment for Gym model attribution")
        directory = self.results / route
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "models.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        return payload


PLUGIN = (
    GymRoutingPlugin(Path(os.environ["NEMO_GYM_LITELLM_RESULTS"]))
    if os.environ.get("NEMO_GYM_LITELLM_RESULTS")
    else None
)
