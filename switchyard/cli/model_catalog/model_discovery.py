# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible model discovery for YAML route bundles."""

import httpx


class ModelDiscoveryError(RuntimeError):
    """Raised when an upstream model catalog cannot be fetched."""


def fetch_model_ids(
    base_url: str,
    api_key: str,
    timeout_s: float = 10.0,
) -> list[str]:
    """Return sorted model IDs from an OpenAI-compatible ``GET /models``."""

    try:
        with httpx.Client(timeout=timeout_s) as client:
            response = client.get(
                f"{base_url.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            body = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise ModelDiscoveryError(str(exc)) from exc

    raw_items = body.get("data", []) if isinstance(body, dict) else body
    if not isinstance(raw_items, list):
        raise ModelDiscoveryError("GET /models response did not contain a model list")

    model_ids: list[str] = []
    for item in raw_items:
        if isinstance(item, str):
            model_ids.append(item)
            continue
        if not isinstance(item, dict):
            continue
        for key in ("id", "model", "name"):
            value = item.get(key)
            if isinstance(value, str) and value:
                model_ids.append(value)
                break
    return sorted(set(model_ids))


__all__ = ["ModelDiscoveryError", "fetch_model_ids"]
