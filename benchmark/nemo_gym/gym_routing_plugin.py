# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record LiteLLM request evidence and preserve Gym-compatible usage detail shapes."""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.router import RoutingContext
from switchyard_litellm import RandomRoutingPlugin
from switchyard_litellm.configuration import load_routing_plugin


def response_payload(response: Any) -> dict[str, Any]:
    """Keep missing detail counts unknown while satisfying Gym's object-shaped fields."""
    payload = response.model_dump() if hasattr(response, "model_dump") else deepcopy(dict(response))
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
    """Join routing and callback instances using one runner-supplied gateway identity."""

    def __init__(
        self, routing_path: Path, profile_path: Path, results: Path, instance_id: str
    ) -> None:
        super().__init__()
        if not instance_id.strip():
            raise ValueError("The runner must supply a gateway instance ID")
        self.routing = load_routing_plugin(routing_path)
        if not isinstance(self.routing, RandomRoutingPlugin):
            raise ValueError("This example requires Switchyard Random routing")
        self.results = results
        profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
        models = {
            route: [
                entry["litellm_params"]["model"]
                for entry in profile["model_list"]
                if entry["model_name"] == route
            ]
            for route in ("fixed", "routed")
        }
        if len(models["fixed"]) != 1 or len(set(models["routed"])) != 2:
            raise ValueError("Define one fixed target and two distinct routed targets")
        if models["fixed"][0] not in models["routed"]:
            raise ValueError("The fixed target must be one of the routed targets")
        self.runtime = {
            "mode": "litellm_libsy",
            "instance_id": instance_id,
            "litellm_version": version("litellm"),
            "switchyard_version": version("nemo-switchyard"),
            "fastapi_version": version("fastapi"),
            "starlette_version": version("starlette"),
            "routing_plugin": "switchyard_litellm.RandomRoutingPlugin",
            "models": models,
            "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
            "routing_sha256": hashlib.sha256(routing_path.read_bytes()).hexdigest(),
            "callback_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "provider_base_sha256": hashlib.sha256(
                os.environ.get("NVIDIA_BASE_URL", "").encode()
            ).hexdigest(),
        }
        results.mkdir(parents=True, exist_ok=True)
        (results / "litellm-runtime.json").write_text(
            json.dumps(self.runtime, indent=2) + "\n", encoding="utf-8"
        )

    async def run(self, context: RoutingContext) -> RoutingContext:
        """Measure the actual libsy decision without including provider inference."""
        started = perf_counter()
        result = await self.routing.run(context)
        result.signals["switchyard"]["routing_ms"] = (perf_counter() - started) * 1000
        return result

    async def async_pre_call_deployment_hook(
        self, kwargs: dict[str, Any], call_type: Any
    ) -> dict[str, Any] | None:
        """Keep the configured plugin's deployment callback behavior intact."""
        return await self.routing.async_pre_call_deployment_hook(kwargs, call_type)

    def _record(self, data: dict[str, Any], event: dict[str, Any]) -> None:
        """Write only allowlisted evidence, never request content or credentials."""
        route, request_id = data.get("model"), data.get("litellm_call_id")
        if route not in ("fixed", "routed"):
            raise ValueError("This example accepts only the fixed and routed model groups")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("LiteLLM request ID is missing")
        directory = self.results / route
        directory.mkdir(parents=True, exist_ok=True)
        record = {
            "instance_id": self.runtime["instance_id"],
            "route": route,
            "request_id": request_id,
            **event,
        }
        with (directory / "litellm-calls.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    async def async_pre_call_hook(
        self, user_api_key_dict: Any, cache: Any, data: dict[str, Any], call_type: Any
    ) -> dict[str, Any]:
        """Record a request before routing so interrupted evidence can be detected."""
        self._record(data, {"event": "start"})
        if data.get("stream"):
            raise ValueError("This example requires non-streaming requests")
        return data

    async def async_post_call_success_hook(
        self, data: dict[str, Any], user_api_key_dict: Any, response: Any
    ) -> dict[str, Any]:
        """Record the final response before LiteLLM replaces its model with the public alias."""
        payload = response_payload(response)
        metadata = data.get("litellm_metadata") or {}
        signal = (metadata.get("routing_plugin_signals") or {}).get("switchyard") or {}
        self._record(
            data,
            {
                "event": "finish",
                "status_code": 200,
                "error_type": None,
                "response_id": payload.get("id"),
                "response_status": payload.get("status"),
                "selected_model": signal.get("selected_model_id"),
                "deployment_model": metadata.get("deployment"),
                "tokens_total": (payload.get("usage") or {}).get("total_tokens"),
                "routing_ms": signal.get("routing_ms"),
            },
        )
        return payload

    async def async_post_call_failure_hook(
        self,
        request_data: dict[str, Any],
        original_exception: Exception,
        user_api_key_dict: Any,
        traceback_str: str | None = None,
    ) -> None:
        """Record failed gateway requests without inventing unreported token usage."""
        metadata = request_data.get("litellm_metadata") or {}
        signal = (metadata.get("routing_plugin_signals") or {}).get("switchyard") or {}
        self._record(
            request_data,
            {
                "event": "finish",
                "status_code": getattr(original_exception, "status_code", None),
                "error_type": type(original_exception).__name__,
                "response_id": None,
                "response_status": None,
                "selected_model": signal.get("selected_model_id"),
                "deployment_model": metadata.get("deployment"),
                "tokens_total": None,
                "routing_ms": signal.get("routing_ms"),
            },
        )


PLUGIN = (
    GymRoutingPlugin(
        Path(os.environ["SWITCHYARD_LITELLM_CONFIG"]),
        Path(os.environ["NEMO_GYM_LITELLM_PROFILE"]),
        Path(os.environ["NEMO_GYM_LITELLM_RESULTS"]),
        os.environ["NEMO_GYM_LITELLM_INSTANCE_ID"],
    )
    if os.environ.get("NEMO_GYM_LITELLM_RESULTS")
    else None
)
