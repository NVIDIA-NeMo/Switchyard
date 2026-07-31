# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small helpers shared by the remaining CLI commands."""

import logging


def quiet_dependency_loggers() -> None:
    """Keep third-party INFO logs out of launcher terminal UIs."""

    for noisy in ("httpx", "httpcore", "openai", "anthropic", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def strip_forwarded_args(args: list[str] | None) -> list[str]:
    """Strip argparse's leading ``--`` sentinel from forwarded arguments."""

    forwarded = args or []
    if forwarded and forwarded[0] == "--":
        return forwarded[1:]
    return forwarded


__all__ = ["quiet_dependency_loggers", "strip_forwarded_args"]
