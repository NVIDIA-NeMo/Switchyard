# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, access-ordered LRU cache keyed by a session key."""

from __future__ import annotations

from collections import OrderedDict
from typing import Generic, TypeVar, cast

_V = TypeVar("_V")
_MISSING = object()


class SessionCache(Generic[_V]):
    """Retain the most recently accessed session values up to a fixed capacity."""

    def __init__(self, max_sessions: int) -> None:
        if max_sessions < 0:
            raise ValueError("max_sessions must be non-negative")
        self._max_sessions = max_sessions
        self._values: OrderedDict[str, _V] = OrderedDict()

    @property
    def max_sessions(self) -> int:
        """Return the configured cache capacity."""
        return self._max_sessions

    def get(self, key: str) -> _V | None:
        """Return a cached value and mark it as most recently used."""
        value = self._values.get(key, _MISSING)
        if value is _MISSING:
            return None
        self._values.move_to_end(key)
        return cast(_V, value)

    def put(self, key: str, value: _V) -> None:
        """Insert a value and evict the least recently used entry if needed."""
        if self._max_sessions == 0:
            return
        self._values[key] = value
        self._values.move_to_end(key)
        if len(self._values) > self._max_sessions:
            self._values.popitem(last=False)

    def values(self) -> list[_V]:
        """Return the retained values in least-to-most-recent order."""
        return list(self._values.values())

    def __len__(self) -> int:
        return len(self._values)


__all__ = ["SessionCache"]
