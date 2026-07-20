# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal bindings for Rust-owned libsy algorithms."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

from switchyard_rust.core import _load_native

_EXPORTS = frozenset({"Algorithm", "LibsyError", "LlmTarget", "noop", "random"})


class LlmClient(Protocol):
    """Structural interface for a Python-hosted model client."""

    async def call(
        self,
        request: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Call the configured target and return an aggregate neutral response."""
        ...


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import final

    from switchyard_rust.core import SwitchyardRuntimeError

    class LibsyError(SwitchyardRuntimeError): ...

    @final
    class LlmTarget:
        def __init__(self, name: str, client: LlmClient) -> None: ...

        @property
        def name(self) -> str: ...

    @final
    class Algorithm:
        async def run(
            self,
            request: Mapping[str, object],
        ) -> tuple[list[dict[str, object]], dict[str, object]]: ...

    def noop() -> Algorithm: ...

    def random(targets: Sequence[LlmTarget]) -> Algorithm: ...


def __getattr__(name: str) -> object:
    if name in _EXPORTS:
        native: Any = _load_native()
        return getattr(native.libsy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [*sorted(_EXPORTS), "LlmClient"]
