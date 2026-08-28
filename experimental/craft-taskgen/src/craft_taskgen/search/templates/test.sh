#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Harbor verifier entry point — delegates to standalone Python scorer.
set -euo pipefail

mkdir -p /logs/verifier

python3 /tests/test_runner.py

# Always exit 0 — reward.txt carries the score.
exit 0
