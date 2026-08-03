# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RouteTable — route inbound requests to per-model runtimes.

Stores a static mapping of model name → callable runtime. The launcher
populates the table at startup after building all configured runtimes. The
three HTTP endpoints read ``body["model"]``, call
:meth:`lookup_switchyard`, and dispatch to the returned chain before making any
backend calls.
"""

import logging
from collections.abc import Iterator, Mapping
from typing import Any, ClassVar, Protocol, TypeAlias

from switchyard.lib.model_listing import model_entry
from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.roles import TranslatedResponse
from switchyard_rust.core import ChatRequest

log = logging.getLogger(__name__)


#: Type accepted by app factories and launcher runtimes — either a single
#: :class:`Switchyard` chain or a :class:`RouteTable` that dispatches
#: by inbound model id.
class ChainRuntime(Protocol):
    """Runtime accepted by the table and FastAPI dispatcher."""

    async def call(
        self,
        request: ChatRequest,
        *,
        ctx: ProxyContext | None = None,
    ) -> TranslatedResponse:
        """Execute one request through the runtime."""
        ...

    def iter_components(self) -> list[Any]:
        """Return lifecycle components in startup order."""
        ...


SwitchyardApp: TypeAlias = "ChainRuntime | RouteTable"


class RouteTable:
    """Table that maps model names to pre-built runtimes.

    Register one chain per model the proxy should handle explicitly. Unknown
    models raise ``KeyError`` so endpoint handlers can return ``model_not_found``
    instead of silently forwarding a request to the wrong backend.
    """

    #: Same key as :class:`Switchyard` so app factories store this under the
    #: attribute the V2 endpoint handlers already read.
    state_key: ClassVar[str] = "switchyard"

    def __init__(self) -> None:
        self._by_model: dict[str, ChainRuntime] = {}
        self._metadata_by_model: dict[str, dict[str, Any]] = {}
        self._model_listing_warnings: list[str] = []
        self._default_model: str | None = None
        # Last model id that `lookup_switchyard` successfully resolved.
        # Updated on every request ingress (each endpoint calls
        # `lookup_switchyard` once per request before any backend work).
        # Read by the launcher's live stats footer so the displayed model
        # tracks what the user actually picked via /model, with no delay
        # for streaming responses.
        self._last_looked_up: str | None = None

    def register(
        self,
        model: str,
        switchyard: ChainRuntime,
        metadata: Mapping[str, Any] | None = None,
        default: bool = False,
    ) -> None:
        """Register *switchyard* as the exact-match chain for *model*."""
        self._by_model[model] = switchyard
        self._metadata_by_model[model] = dict(metadata or {})
        if default:
            self._default_model = model
        log.debug("RouteTable: registered chain for model=%r", model)

    def registered_models(self) -> list[str]:
        """Return registered model ids in registration order."""
        return list(self._by_model)

    def set_default_model(self, model: str) -> None:
        """Mark *model* as the default entry advertised by ``/v1/models``."""
        if model not in self._by_model:
            raise KeyError(model)
        self._default_model = model

    def default_model(self) -> str | None:
        """Return the advertised default model id, falling back to first entry."""
        if self._default_model in self._by_model:
            return self._default_model
        return next(iter(self._by_model), None)

    def items(self) -> Iterator[tuple[str, ChainRuntime, dict[str, Any]]]:
        """Iterate ``(model_id, chain, metadata)`` triples in registration order.

        Used when a caller needs to merge one table into another — e.g. a
        caller that composes multiple route tables.
        """
        for model in self._by_model:
            yield model, self._by_model[model], dict(self._metadata_by_model.get(model, {}))

    def lookup_switchyard(self, model: str) -> ChainRuntime:
        """Return the chain for *model*.

        Records *model* as the last successfully resolved id (see
        :attr:`last_looked_up`).

        Raises:
            KeyError: *model* is unregistered.
        """
        chain = self._by_model.get(model)
        if chain is not None:
            log.debug("RouteTable: model=%r → registered chain", model)
            self._last_looked_up = model
            return chain
        raise KeyError(model)

    @property
    def last_looked_up(self) -> str | None:
        """Model id from the most recent successful :meth:`lookup_switchyard`.

        ``None`` until the first request arrives. Set at request ingress, so
        the value reflects the model the user is *currently* sending traffic
        for — useful for live launcher TUIs that want to display which
        table entry the client most recently picked.
        """
        return self._last_looked_up

    def registered_model_entries(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible model entries with optional metadata."""
        entries: list[dict[str, Any]] = []
        for model in self._by_model:
            metadata = dict(self._metadata_by_model.get(model, {}))
            entries.append(model_entry(model, metadata=metadata))
        return entries

    def add_model_listing_warning(self, warning: str) -> None:
        """Record non-fatal model catalog discovery warnings for ``/v1/models``."""
        if warning not in self._model_listing_warnings:
            self._model_listing_warnings.append(warning)

    def model_listing_warnings(self) -> list[str]:
        """Return non-fatal model catalog warnings in discovery order."""
        return list(self._model_listing_warnings)

    def iter_components(self) -> list[Any]:
        """Return all chain components across registered chains, deduplicated.

        Components shared by object identity are returned once so endpoint and
        shutdown hooks are not double-registered.
        """
        seen: set[int] = set()
        result: list[Any] = []
        for switchyard in self._by_model.values():
            for component in switchyard.iter_components():
                if id(component) in seen:
                    continue
                seen.add(id(component))
                result.append(component)
        return result
