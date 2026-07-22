# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from switchyard.cli.route_bundle import build_route_bundle_table
from switchyard.lib.processors.llm_classifier import (
    DEFAULT_MAX_REQUEST_CHARS,
    LLMClassifierRequestProcessor,
)
from switchyard.lib.processors.llm_classifier.presets import profile_default_prompt

RESOLVER = Path(__file__).resolve().parents[1] / "benchmark" / "resolve_classifier_prompts.py"


def _load_resolver_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "switchyard_benchmark_resolve_classifier_prompts",
        RESOLVER,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(tmp_path: Path, bundle: dict) -> Path:
    path = tmp_path / "routes.yaml"
    path.write_text(yaml.safe_dump(bundle))
    return path


def _bundle(prompt: str | None = None) -> dict:
    classifier: dict[str, object] = {
        "model": "c",
        "api_key": "k",
        "base_url": "https://example.invalid/v1",
        "recent_turn_window": 2,
        "max_request_chars": 1024,
    }
    if prompt is not None:
        classifier["prompt"] = prompt
    return {
        "routes": {
            "r/llm": {
                "type": "deterministic",
                "fallback_target_on_evict": "strong",
                "profile": "coding_agent",
                "classifier": classifier,
                "strong": {
                    "model": "s",
                    "api_key": "k",
                    "base_url": "https://example.invalid/v1",
                },
                "weak": {
                    "model": "w",
                    "api_key": "k",
                    "base_url": "https://example.invalid/v1",
                },
            },
        },
    }


def test_explicit_prompt_override_is_recorded(tmp_path: Path) -> None:
    module = _load_resolver_module()
    out = module.resolve_classifier_prompts(_write(tmp_path, _bundle("MY PROMPT")))

    assert out["r/llm"]["source"] == "override"
    assert out["r/llm"]["classifier_prompt"] == "MY PROMPT"
    assert out["r/llm"]["classifier_prompt_sha256"] == hashlib.sha256(
        b"MY PROMPT"
    ).hexdigest()
    assert out["r/llm"]["profile"] == "coding_agent"
    assert out["r/llm"]["max_request_chars"] == 1024
    assert out["r/llm"]["recent_turn_window"] == 2


def test_profile_default_is_resolved(tmp_path: Path) -> None:
    module = _load_resolver_module()
    out = module.resolve_classifier_prompts(_write(tmp_path, _bundle()))

    assert out["r/llm"]["source"] == "profile_default"
    assert out["r/llm"]["classifier_prompt"] == profile_default_prompt("coding_agent")


def test_default_profile_is_general_when_absent(tmp_path: Path) -> None:
    module = _load_resolver_module()
    bundle = _bundle()
    del bundle["routes"]["r/llm"]["profile"]

    out = module.resolve_classifier_prompts(_write(tmp_path, bundle))

    assert out["r/llm"]["profile"] == "general"


def test_non_deterministic_routes_are_skipped(tmp_path: Path) -> None:
    module = _load_resolver_module()
    bundle = {
        "routes": {
            "r/model": {"type": "model", "target": "m"},
            "r/rand": {"type": "random_routing"},
        },
    }

    assert module.resolve_classifier_prompts(_write(tmp_path, bundle)) == {}


def test_whitespace_only_prompt_uses_profile_default(tmp_path: Path) -> None:
    module = _load_resolver_module()
    out = module.resolve_classifier_prompts(_write(tmp_path, _bundle("   ")))

    assert out["r/llm"]["source"] == "profile_default"


def test_context_defaults_are_recorded(tmp_path: Path) -> None:
    module = _load_resolver_module()
    bundle = _bundle()
    classifier = bundle["routes"]["r/llm"]["classifier"]
    del classifier["recent_turn_window"]
    del classifier["max_request_chars"]

    out = module.resolve_classifier_prompts(_write(tmp_path, bundle))

    assert out["r/llm"]["recent_turn_window"] == 4
    assert out["r/llm"]["max_request_chars"] == DEFAULT_MAX_REQUEST_CHARS


def test_subminimum_max_request_chars_uses_default(tmp_path: Path) -> None:
    module = _load_resolver_module()
    bundle = _bundle()
    classifier = bundle["routes"]["r/llm"]["classifier"]
    classifier["max_request_chars"] = 255

    out = module.resolve_classifier_prompts(_write(tmp_path, bundle))

    assert out["r/llm"]["max_request_chars"] == DEFAULT_MAX_REQUEST_CHARS


def test_resolver_matches_server_effective_prompt_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_resolver_module()
    bundle = _bundle("CUSTOM SERVER PROMPT")
    monkeypatch.setattr("switchyard.cli.route_bundle.fetch_model_ids", lambda *_: [])

    table = build_route_bundle_table(bundle)
    switchyard = table.lookup_switchyard("r/llm")
    classifier = next(
        c for c in switchyard.iter_components()
        if isinstance(c, LLMClassifierRequestProcessor)
    )

    resolved = module.resolve_classifier_prompts(_write(tmp_path, bundle))["r/llm"]
    assert resolved["classifier_prompt"] == classifier._config.system_prompt
    assert resolved["max_request_chars"] == classifier._config.max_request_chars
    assert resolved["recent_turn_window"] == classifier._config.recent_turn_window


def test_prompt_env_vars_match_route_bundle_expansion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_resolver_module()
    monkeypatch.setenv("CUSTOM_CLASSIFIER_PROMPT", "expanded prompt")

    out = module.resolve_classifier_prompts(
        _write(tmp_path, _bundle("${CUSTOM_CLASSIFIER_PROMPT}")),
    )

    assert out["r/llm"]["classifier_prompt"] == "expanded prompt"
