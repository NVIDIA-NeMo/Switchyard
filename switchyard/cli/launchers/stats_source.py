# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stats snapshot interface shared by launcher presentation code."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol


class StatsSource(Protocol):
    """Provides the current launcher-session statistics."""

    def snapshot_sync(self) -> Mapping[str, object]:
        """Return the latest statistics snapshot."""
        ...


__all__ = ["StatsSource"]
