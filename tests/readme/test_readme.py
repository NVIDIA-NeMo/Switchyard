# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Executable coverage for ``README.md``.

Companion to ``tests/getting_started/``. Three guards:

* the "Use as a Python library" snippet executes (via ``--markdown-docs`` +
  the passthrough→noop fixture in ``conftest.py``);
* README route examples validate against the route-bundle schema;
* canonical routing pages still build stage_router and LLM-routing profile examples;
* every CLI subcommand / flag the README names still exists.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import yaml as pyyaml
from markdown_it import MarkdownIt

from switchyard.cli import route_bundle as rb
from switchyard.cli.switchyard_cli import _build_parser
from switchyard_rust.profiles import parse_profile_config_str

REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATH = REPO_ROOT / "README.md"
ROUTING_DOC_PATHS = (
    REPO_ROOT / "docs" / "routing_algorithms" / "stage_router_routing.md",
    REPO_ROOT / "docs" / "routing_algorithms" / "llm_classifier_routing.md",
    REPO_ROOT / "docs" / "routing_algorithms" / "overview.md",
)


@pytest.fixture(scope="module")
def readme_text() -> str:
    return README_PATH.read_text()


def _code_blocks(text: str, lang: str) -> list[str]:
    # markdown-it-py handles indented fences + trailing whitespace correctly,
    # which a naive triple-backtick regex does not.
    md = MarkdownIt()
    return [
        token.content
        for token in md.parse(text)
        if token.type == "fence" and token.info.strip() == lang
    ]


def test_python_snippet_tripwire(readme_text: str) -> None:
    # Guards the shape the conftest's passthrough-profile→noop fixture depends on, plus
    # the dict-access fix (call() returns a dict, not an object with `.body`).
    assert (
        "from switchyard import ChatRequest, PassthroughProfileConfig, ProfileSwitchyard"
        in readme_text
    ), (
        "README Python snippet's imports moved — update conftest.py."
    )
    assert "ProfileSwitchyard(PassthroughProfileConfig(" in readme_text, (
        "README snippet no longer builds a passthrough profile — update the "
        "markdown-docs fixture in conftest.py."
    )
    assert 'response["choices"][0]["message"]["content"]' in readme_text, (
        "README snippet's response access changed. `Switchyard.call()` returns "
        "a JSON-compatible dict — `response.body` would raise AttributeError."
    )


def test_all_yaml_route_blocks_in_readme_validate_against_the_schema(
    readme_text: str,
) -> None:
    # Schema/key validation, not a full chain build: building README's routes
    # is NOT hermetic — `passthrough` with `discover: true` does a live catalog
    # fetch and `latency_service` polls. The
    # schema layer (route type + per-type key allowlist) is what we can check
    # offline, and it catches the likeliest drift: a renamed `type:` or a key
    # that no longer exists on that type.
    blocks = _code_blocks(readme_text, "yaml")
    assert blocks, "README unexpectedly has no yaml blocks"

    validated_routes = 0
    for idx, block in enumerate(blocks):
        payload = pyyaml.safe_load(block)
        if not isinstance(payload, dict) or "routes" not in payload:
            continue
        for model_id, route_raw in payload["routes"].items():
            route = rb._normalize_route(model_id, route_raw)
            try:
                route_type = rb._route_type(model_id, route)
                rb._validate_route_keys(model_id, route, route_type)
            except rb.RouteBundleConfigError as exc:
                raise AssertionError(
                    f"YAML block {idx} route {model_id!r} in README.md failed "
                    f"schema validation: {exc}\n\nBlock:\n{block}"
                ) from exc
            validated_routes += 1

    assert validated_routes, "no README yaml block parsed as a route bundle"


def test_stage_router_and_llm_routing_yaml_blocks_in_canonical_docs_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-readme")

    built_profiles = 0
    for path in ROUTING_DOC_PATHS:
        for idx, block in enumerate(_code_blocks(path.read_text(), "yaml")):
            payload = pyyaml.safe_load(block)
            if not isinstance(payload, dict) or not {
                "endpoints",
                "targets",
                "profiles",
            }.issubset(payload):
                continue
            relevant_profile_ids = [
                profile_id
                for profile_id, profile in payload["profiles"].items()
                if profile.get("type") in {"stage_router", "llm-routing"}
            ]
            if not relevant_profile_ids:
                continue
            try:
                plan = parse_profile_config_str(block).resolve()
                for profile_id in relevant_profile_ids:
                    plan.build_profile(profile_id)
            except Exception as exc:
                raise AssertionError(
                    f"YAML block {idx} profiles {relevant_profile_ids!r} in {path} "
                    f"failed profile-config build: {exc}\n\nBlock:\n{block}"
                ) from exc
            built_profiles += len(relevant_profile_ids)

    assert built_profiles >= 2


def _subparsers(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return action.choices  # type: ignore[return-value]


def test_cli_parser_exposes_every_subcommand_the_readme_names() -> None:
    parser = _build_parser()
    subs = _subparsers(parser)
    for cmd in ("serve", "launch", "configure"):
        assert cmd in subs, f"top-level `switchyard {cmd}` is documented but missing"
        assert subs[cmd].format_help().strip()

    launch_subs = _subparsers(subs["launch"])
    for target in ("claude", "codex", "openclaw"):
        assert target in launch_subs, (
            f"`switchyard launch {target}` is documented but missing"
        )


def test_cli_parser_advertises_documented_flags() -> None:
    parser = _build_parser()
    subs = _subparsers(parser)
    # --routing-profiles is a global switchyard flag now, not on serve
    assert "--routing-profiles" in parser.format_help()
    assert "--port" in subs["serve"].format_help()

    claude_help = _subparsers(subs["launch"])["claude"].format_help()
    for flag in ("--base-url", "--model"):
        assert flag in claude_help, (
            f"`launch claude {flag}` documented but missing from --help"
        )
