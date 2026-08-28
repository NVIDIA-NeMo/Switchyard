# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor task adapters: convert synthesized task candidates into Harbor task directories.

Each adapter lives in its own subpackage (e.g. `search_native/`) and exposes a
`run_convert(...)` function. The `cli.py` dispatcher routes `--adapter <name>` to
the right adapter.
"""
