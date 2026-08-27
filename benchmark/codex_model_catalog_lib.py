# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for generating benchmark Codex model-catalog JSON.

Codex only forwards reasoning controls for models it recognizes as
reasoning-capable. Switchyard often exposes synthetic route ids, so the
benchmark harness generates a tiny catalog that maps those route ids onto
Codex's bundled GPT-5-style metadata.
"""

from __future__ import annotations

import copy
import json
import logging
import subprocess
from collections.abc import Sequence
from typing import Any, TypeAlias

logger = logging.getLogger(__name__)

_CODEX_CATALOG_TEMPLATE_MODEL = "gpt-5.5"
_CODEX_CATALOG_LOAD_TIMEOUT_S = 2.0

CodexModelCatalogEntry: TypeAlias = tuple[str, str, str]


def _fallback_codex_model_template() -> dict[str, Any]:
    """Return the minimum model metadata shape accepted by Codex."""
    return {
        "slug": "switchyard",
        "display_name": "Switchyard",
        "description": "Switchyard-routed model.",
        "default_reasoning_level": "xhigh",
        "supported_reasoning_levels": [
            {"effort": "low", "description": "Fast responses with lighter reasoning"},
            {"effort": "medium", "description": "Balances speed and reasoning depth"},
            {"effort": "high", "description": "Greater reasoning depth"},
            {"effort": "xhigh", "description": "Extra high reasoning depth"},
        ],
        "shell_type": "shell_command",
        "visibility": "list",
        "supported_in_api": True,
        "priority": 0,
        "additional_speed_tiers": [],
        "availability_nux": None,
        "upgrade": None,
        "base_instructions": "You are Codex, a coding agent.",
        "supports_reasoning_summaries": True,
        "default_reasoning_summary": "none",
        "support_verbosity": True,
        "default_verbosity": "low",
        "apply_patch_tool_type": "freeform",
        "web_search_tool_type": "text",
        "truncation_policy": {"mode": "tokens", "limit": 10000},
        "supports_parallel_tool_calls": True,
        "supports_image_detail_original": False,
        "context_window": 128000,
        "max_context_window": 128000,
        "effective_context_window_percent": 95,
        "experimental_supported_tools": [],
        "input_modalities": ["text"],
        "supports_search_tool": False,
    }


def _copy_string_keyed_dict(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _load_codex_model_template(codex_bin: str) -> dict[str, Any]:
    """Copy Codex's bundled catalog shape so custom models stay compatible."""
    try:
        raw = subprocess.check_output(
            [codex_bin, "debug", "models", "--bundled"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=_CODEX_CATALOG_LOAD_TIMEOUT_S,
        )
        catalog = json.loads(raw)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        logger.debug("failed to load bundled Codex model catalog", exc_info=True)
        return _fallback_codex_model_template()

    catalog_dict = _copy_string_keyed_dict(catalog)
    models = catalog_dict.get("models") if catalog_dict is not None else None
    if not isinstance(models, list):
        return _fallback_codex_model_template()

    for raw_model in models:
        model = _copy_string_keyed_dict(raw_model)
        if model is not None and model.get("slug") == _CODEX_CATALOG_TEMPLATE_MODEL:
            return model
    for raw_model in models:
        model = _copy_string_keyed_dict(raw_model)
        if model is not None:
            return model
    return _fallback_codex_model_template()


def _build_codex_model_catalog(
    codex_bin: str,
    entries: Sequence[CodexModelCatalogEntry],
) -> dict[str, list[dict[str, Any]]]:
    """Build Codex catalog JSON for Switchyard route ids."""
    template = _load_codex_model_template(codex_bin)
    models: list[dict[str, Any]] = []
    for priority, (model_id, display_name, description) in enumerate(entries):
        model = copy.deepcopy(template)
        model["slug"] = model_id
        model["display_name"] = display_name
        model["description"] = description
        model["priority"] = priority
        model["visibility"] = "list"
        model["supported_in_api"] = True
        model["availability_nux"] = None
        model["upgrade"] = None
        models.append(model)
    return {"models": models}


def _codex_model_display_name(model_id: str) -> str:
    """Return a compact display name for a model id."""
    return model_id.rsplit("/", maxsplit=1)[-1]
