# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for version lookups after the package rename."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from unittest.mock import patch


def _mock_version(name: str) -> str:
    """Simulate an environment that only has ``nemo-switchyard`` installed."""
    if name == "nemo-switchyard":
        return "1.0.0"
    raise PackageNotFoundError(name)


def test_intake_payload_builder_version_uses_switchyard_package() -> None:
    from switchyard.lib.processors import intake_payload_builder

    intake_payload_builder._switchyard_version.cache_clear()
    with patch(
        "switchyard.lib.processors.intake_payload_builder.version",
        side_effect=_mock_version,
    ):
        result = intake_payload_builder._switchyard_version()

    assert result == "1.0.0"


def test_intake_payload_builder_version_fallback_is_unknown() -> None:
    from switchyard.lib.processors import intake_payload_builder

    intake_payload_builder._switchyard_version.cache_clear()
    with patch(
        "switchyard.lib.processors.intake_payload_builder.version",
        side_effect=PackageNotFoundError("nemo-switchyard"),
    ):
        result = intake_payload_builder._switchyard_version()

    assert result == "unknown"
