# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the OpenTelemetry SDK bootstrap (switchyard.lib.observability)."""

from __future__ import annotations

import importlib


def _fresh():
    obs = importlib.import_module("switchyard.lib.observability")
    obs.reset_for_test()
    return obs


def test_init_is_noop_when_disabled(monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    obs = _fresh()
    assert obs.init_observability() is False
    assert obs.is_enabled() is False
    assert obs.get_tracer() is None
    assert obs.get_meter() is None


def test_init_is_noop_without_endpoint(monkeypatch):
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    obs = _fresh()
    assert obs.init_observability() is False
    assert obs.is_enabled() is False


def test_init_idempotent_when_enabled(monkeypatch):
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    obs = _fresh()
    assert obs.init_observability() is True
    first_id = obs.service_instance_id()
    assert obs.init_observability() is True  # second call no-ops, stays enabled
    assert obs.service_instance_id() == first_id  # id stable across idempotent calls
    assert obs.is_enabled() is True
    assert obs.get_tracer() is not None
    assert obs.get_meter() is not None
    assert first_id
