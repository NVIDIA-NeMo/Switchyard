# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route-selection response headers for spend/tokenomics attribution.

Maps the :data:`CTX_ROUTE_SELECTION` record a routing backend stored on
``ctx`` (see :mod:`switchyard.lib.proxy_context`) to the ``x-switchyard-*``
response headers a front proxy such as LiteLLM copies into its parent
spend-log row. Shared by the success serializer (``dispatch``) and the
error path (``upstream_error``) — a failure that happens *after* a billed
upstream success must still expose the selection, or the provider spend-log
row's correlation id becomes unjoinable.
"""

from collections.abc import Mapping

from switchyard.lib.proxy_context import CTX_ROUTE_SELECTION, ProxyContext

#: Response headers exposing the route selection behind an upstream call,
#: carrying the same correlation id the provider row received via the
#: outbound ``x-litellm-spend-logs-metadata`` header.
ROUTER_MODEL_HEADER = "x-switchyard-router-model"
SELECTED_MODEL_HEADER = "x-switchyard-selected-model"
SELECTED_PROVIDER_HEADER = "x-switchyard-selected-provider"
ROUTER_CORRELATION_ID_HEADER = "x-switchyard-router-correlation-id"

_ROUTE_SELECTION_RESPONSE_HEADERS = (
    (ROUTER_MODEL_HEADER, "router_model"),
    (SELECTED_MODEL_HEADER, "router_selected_model"),
    (SELECTED_PROVIDER_HEADER, "router_selected_provider"),
    (ROUTER_CORRELATION_ID_HEADER, "router_correlation_id"),
)


def _is_header_value_safe(value: str) -> bool:
    """Whether *value* can be emitted as an HTTP/1.1 response-header value.

    ``router_model`` echoes the client-supplied model string, so it must be
    re-validated as header material: Starlette encodes response-header values
    as latin-1 (a non-encodable value would fail response construction after
    the upstream call already succeeded and was billed), and CTL characters —
    CR/LF above all — would be a response-splitting vector on permissive
    ASGI stacks.
    """
    try:
        value.encode("latin-1")
    except UnicodeEncodeError:
        return False
    return not any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value)


def route_selection_headers(ctx: ProxyContext) -> dict[str, str]:
    """Response headers for the route selection recorded on *ctx*, if any.

    Empty when no routing backend recorded a selection (passthrough chains,
    failures before any upstream success). A recorded field that is absent or
    not emittable as a header value is skipped — headers never carry
    placeholder or unsafe values.
    """
    selection = ctx.metadata.get(CTX_ROUTE_SELECTION)
    if not isinstance(selection, Mapping):
        return {}
    headers: dict[str, str] = {}
    for header_name, selection_key in _ROUTE_SELECTION_RESPONSE_HEADERS:
        value = selection.get(selection_key)
        if isinstance(value, str) and value and _is_header_value_safe(value):
            headers[header_name] = value
    return headers
