# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configured Rust routing with requirements supplied by a trusted Python host."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, final

from switchyard_rust._native import load_native

if TYPE_CHECKING:
    from switchyard_rust.libsy import RoutingOutcome

    @final
    class Runner:
        """Reusable deployment routes retaining algorithm and task state."""

        @staticmethod
        def from_toml(source: str) -> "Runner":
            """Load TOML and native clients; invalid configuration raises ServerConfigError."""
            ...

        async def decide(
            self,
            request: Mapping[str, object],
            *,
            headers: Mapping[str, str] | None = None,
            required_labels: Sequence[str] | None = None,
            task_id: str | None = None,
        ) -> RoutingOutcome:
            """Route normalized input; the host serves the selected completion and fallbacks.

            The trusted host supplies requirements and an authorized task ID. ``None`` leaves
            requirements unspecified; ``[]`` explicitly supplies an empty requirement set.
            Neither clears retained task requirements. Native clients serve routing-time calls.
            Candidate-specific rewrites are in ``outcome.candidate_requests``.
            Invalid requests raise ValueError; execution failures raise LibsyError.
            """
            ...


def __getattr__(name: str) -> object:
    if name == "Runner":
        native: Any = load_native()
        return native.runner.Runner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Runner"]
