# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the FastAPI app factory wiring."""

from __future__ import annotations

from typing import Protocol

from fastapi import FastAPI
from fastapi.testclient import TestClient

from switchyard.lib.endpoints.base import Endpoint
from switchyard.lib.proxy_context import CTX_SWITCHYARD_FALLBACK
from switchyard.server.switchyard_app import build_switchyard_app


class _RequestWithBody(Protocol):
    body: dict[str, object]


class _RecordingSwitchyard:
    def __init__(self, *, fallback: str | None = None) -> None:
        self.requests: list[_RequestWithBody] = []
        self.fallback = fallback

    async def call(
        self,
        request: _RequestWithBody,
        *,
        ctx: object | None = None,
    ) -> dict[str, object]:
        self.requests.append(request)
        if self.fallback is not None and hasattr(ctx, "metadata"):
            ctx.metadata[CTX_SWITCHYARD_FALLBACK] = self.fallback
        return {
            "id": "resp-test",
            "object": "response",
            "model": request.body["model"],
            "output": [],
        }


class _MarkerEndpoint(Endpoint):
    def register(self, app: FastAPI) -> None:
        @app.get("/marker")
        async def marker() -> dict[str, str]:
            return {"status": "ok"}


class _EndpointContributor:
    def get_endpoint(self) -> Endpoint:
        return _MarkerEndpoint()


class _SwitchyardWithComponents(_RecordingSwitchyard):
    def iter_components(self) -> list[_EndpointContributor]:
        return [_EndpointContributor()]


def test_app_exposes_switchyard_under_endpoint_state_key() -> None:
    switchyard = _RecordingSwitchyard()

    app = build_switchyard_app(switchyard)  # type: ignore[arg-type]

    assert app.state.switchyard is switchyard
    assert app.state.switchyard is switchyard


def test_responses_endpoint_uses_app_factory_switchyard() -> None:
    switchyard = _RecordingSwitchyard()
    app = build_switchyard_app(switchyard)  # type: ignore[arg-type]

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "ping"},
        )

    assert response.status_code == 200
    assert response.json()["model"] == "test-model"
    assert len(switchyard.requests) == 1


def test_endpoint_surfaces_switchyard_fallback_header() -> None:
    switchyard = _RecordingSwitchyard(fallback="classifier_error")
    app = build_switchyard_app(switchyard)  # type: ignore[arg-type]

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "ping"},
        )

    assert response.status_code == 200
    assert response.headers["x-switchyard-fallback"] == "classifier_error"
    assert response.json()["model"] == "test-model"


def test_app_registers_component_contributed_endpoints() -> None:
    app = build_switchyard_app(_SwitchyardWithComponents())  # type: ignore[arg-type]

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/marker")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
