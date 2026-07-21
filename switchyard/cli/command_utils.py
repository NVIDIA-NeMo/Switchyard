# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for CLI command modules."""

from __future__ import annotations

import getpass
import logging
import sys

from switchyard.cli.model_catalog.model_discovery import (
    ModelDiscoveryError,
    fetch_model_ids,
)
from switchyard.cli.tui.launch_config_wizard import LaunchConfigWizard
from switchyard.cli.tui.tui_session import TuiSession

logger = logging.getLogger(__name__)

ROUTING_MODE_CHOICES: list[str] = ["single", "random", "deterministic"]


def quiet_dependency_loggers() -> None:
    """Keep third-party INFO logs from interrupting interactive CLI prompts."""

    for noisy in ("httpx", "httpcore", "openai", "anthropic", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def parse_probability(value: str | float | None, *, default: float = 0.5) -> float:
    """Parse and validate a strong-model probability value."""

    if value is None:
        return default
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise SystemExit("strong probability must be between 0.0 and 1.0.")
    return parsed


def strip_forwarded_args(args: list[str] | None) -> list[str]:
    """Strip argparse's leading ``--`` sentinel from passthrough args."""

    forwarded = args or []
    if forwarded and forwarded[0] == "--":
        return forwarded[1:]
    return forwarded


def is_interactive_terminal() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def build_launch_config_wizard(no_tui: bool) -> LaunchConfigWizard:
    return LaunchConfigWizard(
        session=TuiSession.from_current_terminal(
            force_plain=no_tui,
            input_fn=input,
            secret_fn=getpass.getpass,
            output_fn=print,
        ),
    )


def discover_models(base_url: str, api_key: str, *, disabled: bool) -> list[str]:
    if disabled:
        return []
    try:
        return fetch_model_ids(base_url, api_key)
    except ModelDiscoveryError as exc:
        logger.warning("Could not fetch model catalog from %s: %s", base_url, exc)
        return []


__all__ = [
    "ROUTING_MODE_CHOICES",
    "build_launch_config_wizard",
    "discover_models",
    "is_interactive_terminal",
    "parse_probability",
    "quiet_dependency_loggers",
    "strip_forwarded_args",
]
