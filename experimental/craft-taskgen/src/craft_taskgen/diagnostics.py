# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic file helpers for cross-step context preservation."""

from __future__ import annotations

import os


def _next_diagnostic_path(task_dir: str, name: str) -> str:
    """Return next numbered diagnostic file path: {task_dir}/diagnostics/{NNN}_{name}.md."""
    diag_dir = os.path.join(task_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    existing = [f for f in os.listdir(diag_dir) if f.endswith(".md")]
    n = len(existing) + 1
    return os.path.join(diag_dir, f"{n:03d}_{name}.md")


def _write_diagnostic(path: str, content: str) -> None:
    """Write diagnostic content to a file."""
    with open(path, "w") as f:
        f.write(content)
