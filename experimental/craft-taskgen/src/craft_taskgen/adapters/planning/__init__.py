# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Planning track Harbor task converter.

Reads planning TaskCandidate JSON files (one file per task, produced upstream
by the bootstrap step) and writes Harbor task directories with a binary
F2P + P2P reward gate.
"""

from __future__ import annotations

from craft_taskgen.adapters.planning.converter import run_convert

__all__ = ["run_convert"]
