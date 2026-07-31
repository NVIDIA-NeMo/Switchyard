# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Rust Switchyard server host."""

from __future__ import annotations

from os import PathLike
from typing import TYPE_CHECKING, Any, final

from switchyard_rust.core import _load_native

if TYPE_CHECKING:

    @final
    class Server:
        """Running loopback instance of the native Switchyard server."""

        def __init__(self, config: str | PathLike[str], *, port: int = 0) -> None: ...

        @property
        def port(self) -> int: ...

        @property
        def base_url(self) -> str: ...

        def close(self, *, timeout_secs: float = 2.0) -> None: ...

        def __enter__(self) -> Server: ...

        def __exit__(
            self,
            exception_type: type[BaseException] | None,
            exception: BaseException | None,
            traceback: object | None,
        ) -> bool: ...


def __getattr__(name: str) -> object:
    if name == "Server":
        native: Any = _load_native()
        return native.server.Server
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Server"]
