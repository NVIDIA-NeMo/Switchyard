# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rust components-v2 profile server bindings."""

from switchyard_rust.core import _load_native

DEFAULT_MAX_ATOF_IDENTITIES = 10_000
DEFAULT_MAX_ATOF_HISTORY_PER_IDENTITY = 256
DEFAULT_MAX_ATOF_DEDUPE_ENTRIES = 100_000
DEFAULT_MAX_ATOF_RETAINED_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_ATOF_EVENT_BYTES = 256 * 1024
DEFAULT_MAX_ATOF_BATCH_BYTES = 4 * 1024 * 1024


def run_profile_server(
    config_path: str,
    host: str = "127.0.0.1",
    port: int = 4000,
    backlog: int = 65_535,
    dry_run: bool = False,
    atof_bearer_token: str | None = None,
    atof_max_identities: int = DEFAULT_MAX_ATOF_IDENTITIES,
    atof_max_history_per_identity: int = DEFAULT_MAX_ATOF_HISTORY_PER_IDENTITY,
    atof_max_dedupe_entries: int = DEFAULT_MAX_ATOF_DEDUPE_ENTRIES,
    atof_max_retained_bytes: int = DEFAULT_MAX_ATOF_RETAINED_BYTES,
    atof_max_event_bytes: int = DEFAULT_MAX_ATOF_EVENT_BYTES,
    atof_max_batch_bytes: int = DEFAULT_MAX_ATOF_BATCH_BYTES,
) -> None:
    """Run the bounded Rust profile server and optional Relay ATOF receiver."""
    _load_native().run_profile_server(
        config_path,
        host,
        port,
        backlog,
        dry_run,
        atof_bearer_token,
        atof_max_identities,
        atof_max_history_per_identity,
        atof_max_dedupe_entries,
        atof_max_retained_bytes,
        atof_max_event_bytes,
        atof_max_batch_bytes,
    )


__all__ = [
    "DEFAULT_MAX_ATOF_BATCH_BYTES",
    "DEFAULT_MAX_ATOF_DEDUPE_ENTRIES",
    "DEFAULT_MAX_ATOF_EVENT_BYTES",
    "DEFAULT_MAX_ATOF_HISTORY_PER_IDENTITY",
    "DEFAULT_MAX_ATOF_IDENTITIES",
    "DEFAULT_MAX_ATOF_RETAINED_BYTES",
    "run_profile_server",
]
