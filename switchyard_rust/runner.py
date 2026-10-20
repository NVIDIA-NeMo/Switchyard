# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TOML-configured routing with native routing-time clients and host-owned completions."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, final

from switchyard_rust._native import load_native

if TYPE_CHECKING:
    from switchyard_rust.libsy import RoutingOutcome

    @final
    class Runner:
        """Persistent native routes loaded from a version-1 deployment."""

        @staticmethod
        def from_toml(source: str) -> "Runner":
            """Load TOML and native clients; invalid configuration raises ServerConfigError."""
            ...

        async def decide(
            self,
            request: Mapping[str, object],
            *,
            headers: Mapping[str, str] | None = None,
        ) -> RoutingOutcome:
            """Route normalized input; native clients handle routing-time calls.

            The host serves completions unless the outcome already includes an answer.
            """
            ...


def __getattr__(name: str) -> object:
    if name == "Runner":
        native: Any = load_native()
        return native.runner.Runner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Runner"]
