# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the global output-token cap."""

from __future__ import annotations

from craft_taskgen.baselines import OUTPUT_TOKEN_CAP


def test_output_token_cap_is_int():
    assert isinstance(OUTPUT_TOKEN_CAP, int)


def test_output_token_cap_in_sane_range():
    # Low end: a cap below ~1k would truncate every trial.
    # High end: a cap above ~200k is effectively uncapped and defeats the purpose.
    assert 1024 <= OUTPUT_TOKEN_CAP <= 200_000
