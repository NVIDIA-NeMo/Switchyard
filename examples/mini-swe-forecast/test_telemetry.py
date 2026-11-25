# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline checks for causal boundaries, trace parenting and score semantics."""

import json
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from scores import score_rows
from telemetry import ForecastObserver, ir_metadata


def observed_state():
    return {
        "state_id": "state-1",
        "model_call_index": 1,
        "ir_snapshot": {
            "artifact_state": {"coverage": "observed", "observed_targets": ["example.py"]},
            "execution_state": {
                "has_new_result": True,
                "pending_tool_calls": [],
                "result_updates": [
                    {
                        "tool_call_id": "tool-1",
                        "operations": [
                            {"category": "Verification", "subtype": "test"},
                            {"category": "Modification", "subtype": "edit"},
                        ],
                        "ir_update": {
                            "features": {"test": 1},
                            "result": "PRIVATE TOOL OUTPUT",
                            "outcome": {"exit_code": 0, "raw": "PRIVATE"},
                        },
                    }
                ],
            },
        },
    }


class FakeBackend:
    def __init__(self, path):
        self.checkpoint = path
        self.manifest = {"variant": "B"}
        (path / "manifest.json").write_text(json.dumps(self.manifest))

    def features(self, config, task_id, messages, source_key):
        history = [observed_state()] if messages else []
        usage = [{"output_tokens": 42, "cache_read_input_tokens": 12}] if history else []
        return {"observed_calls": len(history)}, history, usage

    def predict(self, features, view="call_level"):
        assert features == [{"observed_calls": 0}]
        if view == "task_start":
            predictions = {
                "task_start.peak_context_tokens": {
                    "class_id": 0,
                    "probabilities": [1.0, 0.0],
                    "lower_inclusive": 0,
                    "upper_exclusive": 100,
                }
            }
        else:
            predictions = {
                "next_call.output_tokens": {
                    "class_id": 1,
                    "probabilities": [0.2, 0.8],
                    "lower_inclusive": 100,
                    "upper_exclusive": None,
                },
                "remaining_task.model_calls": {"count_estimate": 3.5},
            }
        return [{"predictions": predictions}]


@pytest.fixture
def observer(tmp_path, monkeypatch):
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instance = ForecastObserver(FakeBackend(tmp_path), {}, "test-task", tmp_path, provider)
    yield instance, exporter, tmp_path
    instance.close()


def test_forecast_precedes_call_and_links_actuals(observer):
    instance, exporter, path = observer
    agent = SimpleNamespace(messages=[])
    with instance.call(agent, 1) as headers:
        assert [s.name for s in exporter.get_finished_spans()] == ["lightgbm.forecast"]
        trace_id, parent_id = headers["traceparent"].split("-")[1:3]
        agent.messages.append({"role": "assistant"})
    forecast, parent = exporter.get_finished_spans()
    assert f"{parent.context.trace_id:032x}" == trace_id
    assert f"{parent.context.span_id:016x}" == parent_id
    assert forecast.parent.span_id == parent.context.span_id
    assert forecast.attributes["ir.source.phase"] == "Initial"
    assert parent.attributes["ir.observed.phase"] == "Mixed"
    assert parent.attributes["actual.output_tokens"] == 42
    assert forecast.attributes["forecast.next_call.output_tokens.upper_unbounded"]
    assert all(
        not k.startswith(("gen_ai.usage", "langfuse.observation.cost"))
        for span in (forecast, parent)
        for k in span.attributes
    )
    assert "PRIVATE" not in json.dumps(dict(parent.attributes))
    record = json.loads((path / "forecasts.jsonl").read_text())
    assert record["source_ir"]["after_model_call"] == 0
    assert json.loads((path / "actuals.jsonl").read_text())["call_index"] == 1
    rows = score_rows(record)
    assert rows == score_rows(record)
    assert len({r["id"] for r in rows}) == len(rows)
    assert all(r["observationId"] == f"{forecast.context.span_id:016x}" for r in rows)
    values = {r["name"]: r["value"] for r in rows}
    assert "prediction.next_call.output_tokens.upper_exclusive" not in values
    assert values["prediction.next_call.output_tokens.upper_unbounded"] == 1
    assert values["prediction.remaining_task.model_calls.count_estimate"] == 3.5


def test_failed_call_does_not_invent_usage_or_export_error_body(observer):
    instance, exporter, path = observer
    with pytest.raises(RuntimeError):
        with instance.call(SimpleNamespace(messages=[]), 1):
            raise RuntimeError("PRIVATE provider response")
    assert not (path / "actuals.jsonl").exists()
    parent = exporter.get_finished_spans()[-1]
    assert parent.status.description == "RuntimeError"
    assert not parent.events


def test_mismatched_boundary_is_rejected(observer):
    instance, _, path = observer
    with pytest.raises(ValueError, match="boundary"):
        with instance.call(SimpleNamespace(messages=[]), 2):
            pytest.fail("A mismatched call must not reach the acting model")
    assert not (path / "forecasts.jsonl").exists()


def test_ir_preserves_mixed_categories_without_raw_outputs():
    metadata = ir_metadata([observed_state()])
    assert metadata["phase"] == "Mixed"
    assert metadata["categories"] == ["Modification", "Verification"]
    assert metadata["results"][0]["outcome"] == {"exit_code": 0}
    assert "PRIVATE" not in json.dumps(metadata)
