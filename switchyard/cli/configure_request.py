# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The typed inputs for ``switchyard configure``.

This is the one place that lists every field ``cmd_configure`` reads, and the
default for each. Two callers build it: the ``configure`` CLI command (via
``from_namespace``) and first-run launch setup (via the constructor). Because
both build the same type, a field the command needs but a caller forgets is
caught by mypy or at construction time — not as an ``AttributeError`` deep
inside ``cmd_configure`` after the user has already started a launch.
"""

import argparse
from dataclasses import dataclass, fields
from typing import Literal

# Which defaults `configure` writes. Matches the CLI's --target choices.
ConfigureTarget = Literal["all", "provider", "claude", "codex", "openclaw"]


@dataclass(frozen=True, kw_only=True)
class ConfigureRequest:
    """Everything ``cmd_configure`` reads. Each default lives here, once."""

    # One of these picks a mode (matches the CLI's mutually-exclusive group).
    show: bool = False
    reset: bool = False
    list_models: bool = False
    skill_distillation: str | None = None
    disable_skill_distillation: bool = False

    # Which provider and scope to configure.
    json: bool = False
    target: ConfigureTarget = "all"
    provider: str | None = None
    base_url: str | None = None
    api_key: str | None = None

    # Per-launcher endpoint overrides. None means "use the provider default".
    claude_model: str | None = None
    claude_base_url: str | None = None
    claude_api_key: str | None = None
    codex_model: str | None = None
    codex_base_url: str | None = None
    codex_api_key: str | None = None
    openclaw_model: str | None = None
    openclaw_base_url: str | None = None
    openclaw_api_key: str | None = None

    # Cross-cutting behavior. routing_profiles is the global --routing-profiles
    # flag, merged in here so configure sees it alongside the rest.
    routing_profiles: str | None = None
    no_model_discovery: bool = False
    no_tui: bool = False

    # Extra inputs for the read-only --show / --list-models modes.
    query: str | None = None
    limit: int = 50
    check: bool = False

    # Set only by first-run launch setup; the CLI has no flags for these.
    reuse_existing_provider: bool = False
    prompt_default_api_key: str | None = None
    prompt_default_api_key_source: str | None = None

    @classmethod
    def from_namespace(cls, ns: argparse.Namespace) -> "ConfigureRequest":
        """Build a request from parsed CLI args.

        This is the single spot where the loosely-typed argparse namespace
        becomes a typed request. argparse fills every configure flag plus the
        global routing_profiles; the launch-setup-only fields fall back to the
        defaults above. Unknown namespace attributes (argparse's own ``func``,
        etc.) are dropped so they can't leak in.
        """
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in vars(ns).items() if key in known})


__all__ = ["ConfigureRequest", "ConfigureTarget"]
