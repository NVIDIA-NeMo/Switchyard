#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
mkdir -p /logs/verifier
python3 /tests/test_runner.py
python3 /tests/score.py
exit 0
