# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight telemetry headers for outbound LLM SDK calls.

Switchyard sends its package version as a single HTTP header on upstream LLM
requests.  This lets downstream request logs attribute traffic to a Switchyard
version without a side channel or additional reporting infrastructure.

Set ``SWITCHYARD_TELEMETRY_OPT_OUT=1`` to suppress the header.  The legacy
``NEMO_SWITCHYARD_TELEMETRY_OPT_OUT`` name is also honored for compatibility
with pre-rename environments.
"""

from __future__ import annotations

import importlib.metadata
import logging
import os
from functools import lru_cache

log = logging.getLogger(__name__)

HEADER_NAME = "X-Switchyard-Version"
OPT_OUT_ENVVAR = "SWITCHYARD_TELEMETRY_OPT_OUT"
LEGACY_OPT_OUT_ENVVAR = "NEMO_SWITCHYARD_TELEMETRY_OPT_OUT"

_FALSEY_VALUES = {"", "0", "false", "no"}


def _is_truthy_opt_out_value(value: str | None) -> bool:
    """Return whether *value* should opt out of telemetry headers."""
    if value is None:
        return False
    return value.strip().lower() not in _FALSEY_VALUES


def _is_opted_out() -> bool:
    """Return whether telemetry headers are disabled by environment."""
    return any(
        _is_truthy_opt_out_value(os.environ.get(name))
        for name in (OPT_OUT_ENVVAR, LEGACY_OPT_OUT_ENVVAR)
    )


@lru_cache(maxsize=1)
def _get_version() -> str:
    """Read the installed ``switchyard`` package version once."""
    try:
        return importlib.metadata.version("nemo-switchyard")
    except Exception:
        log.debug("telemetry: could not read switchyard package version", exc_info=True)
        return "unknown"


def get_telemetry_headers() -> dict[str, str]:
    """Return headers to attach to outbound LLM SDK clients.

    Returns an empty dict when telemetry is opted out, so callers can pass or
    merge the result unconditionally.
    """
    if _is_opted_out():
        return {}
    return {HEADER_NAME: _get_version()}
