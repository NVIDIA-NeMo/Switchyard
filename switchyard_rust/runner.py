# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native configured routing decisions without a local server."""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from typing import TYPE_CHECKING, Any, final

from switchyard_rust._native import load_native

_EXPORTS = frozenset({"Decision", "DecisionError", "DecisionTarget", "RoutingCall", "Runner"})

if TYPE_CHECKING:
    from switchyard_rust.libsy import RoutingOutcome

    class DecisionError(RuntimeError):
        """Routing failure with safe diagnostics and completed-call observations."""

        kind: str
        upstream_status: int | None
        target: str | None
        calls: list[RoutingCall]
        duration_seconds: float

    @final
    class DecisionTarget:
        @property
        def target(self) -> str: ...

        @property
        def model(self) -> str: ...

    @final
    class RoutingCall:
        """A logical provider call; duration includes backend retries.

        Usage is the native normalized protocol mapping: ``input_tokens`` excludes
        cache reads/writes. Output and reasoning counts retain the native codec's
        semantics; reasoning can be a subset of output and must not be added blindly.
        Missing usage or token fields are unknown, not zero. Failed attempts and
        streamed routing calls may not report usage.
        """

        @property
        def model(self) -> str: ...

        @property
        def is_success(self) -> bool: ...

        @property
        def duration_seconds(self) -> float: ...

        @property
        def usage(self) -> dict[str, int | None] | None: ...

    @final
    class Decision:
        @property
        def selected(self) -> DecisionTarget: ...

        @property
        def fallbacks(self) -> list[DecisionTarget]: ...

        @property
        def outcome(self) -> RoutingOutcome: ...

        @property
        def calls(self) -> list[RoutingCall]: ...

        @property
        def duration_seconds(self) -> float: ...

    @final
    class Runner:
        """Native route configuration, clients, and shared algorithm state.

        Give independent tasks distinct session headers. Process turns sharing
        a session in order. Use a separate runner for independent experiments.
        Cancelling a decision cancels local routing; a provider may still finish
        or bill an already submitted request.
        """

        @staticmethod
        def load(path: str | PathLike[str]) -> Runner: ...

        @staticmethod
        def from_toml(source: str) -> Runner: ...

        def validate_decision_route(
            self, model: str, *, allow_response: bool = False
        ) -> list[DecisionTarget]:
            """Validate without calls and return the configured completion targets."""
            ...

        async def decide(
            self,
            request: Mapping[str, object],
            *,
            headers: Mapping[str, str] | None = None,
            allow_response: bool = False,
        ) -> Decision:
            """Route normalized IR; response-based algorithms need explicit opt-in."""
            ...


def __getattr__(name: str) -> object:
    if name in _EXPORTS:
        native: Any = load_native()
        return getattr(native.runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted(_EXPORTS)
