# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import version

import switchyard


def test_dunder_version_matches_installed_metadata() -> None:
    assert switchyard.__version__ == version("nemo-switchyard")
