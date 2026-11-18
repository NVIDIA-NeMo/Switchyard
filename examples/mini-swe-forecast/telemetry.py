# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export causal B forecasts and IR metadata, linked to the acting model request."""

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.propagate import inject
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode
from scores import ScoreClient


def ir_metadata(history):
    """Preserve mixed categories and per-result labels; do not invent a phase."""
    if not history:
        return {"after_model_call": 0, "phase": "Initial", "categories": [], "results": []}
    state = history[-1]
    snapshot = state["ir_snapshot"]
    execution = snapshot["execution_state"]
    results = []
    for arrival in execution["result_updates"]:
        ops = arrival["operations"]
        results.append(
            {
                "tool_call_id": arrival["tool_call_id"],
                "categories": sorted({o["category"] for o in ops if o.get("category")}),
                "subtypes": sorted({o["subtype"] for o in ops if o.get("subtype")}),
                "ir_update": {
                    k: v
                    for k, v in (arrival["ir_update"] or {}).items()
                    if k in ("features", "window")
                },
                "outcome": {
                    k: v
                    for k, v in (arrival["ir_update"] or {}).get("outcome", {}).items()
                    if k
                    in (
                        "reported_status",
                        "execution",
                        "exit_code",
                        "confirmed_operation_indices",
                        "patched_files",
                        "error_signature",
                        "check_signature",
                        "test_counts",
                    )
                },
            }
        )
    categories = sorted({c for r in results for c in r["categories"]})
    artifact = snapshot["artifact_state"]
    return {
        "model_call_state_id": state["state_id"],
        "after_model_call": state["model_call_index"],
        "phase": categories[0] if len(categories) == 1 else ("Mixed" if categories else "Unknown"),
        "categories": categories,
        "results": results,
        "has_new_result": execution["has_new_result"],
        "pending_tool_call_count": len(execution["pending_tool_calls"]),
        "artifact_coverage": artifact.get("coverage"),
        "filesystem_reconstructed": artifact.get("filesystem_reconstructed"),
        "observed_targets": artifact.get("observed_targets", []),
    }


def prediction_attributes(predictions):
    attrs = {}
    for target, result in predictions.items():
        prefix = "forecast." + target
        for key, value in result.items():
            if value is not None:
                attrs[prefix + "." + key] = value
        if "upper_exclusive" in result:
            attrs[prefix + ".upper_unbounded"] = result["upper_exclusive"] is None
    return attrs


class ForecastObserver:
    def __init__(self, backend, config, task_id, output, provider=None):
        self.bundle = backend
        checkpoint = backend.checkpoint
        if self.bundle.manifest["variant"] != "B":
            raise ValueError("Live requirement annotations are not configured; use B")
        self.config, self.task_id, self.output = config, task_id, Path(output)
        self.run_id = self.output.name
        self.checkpoint_id = Path(checkpoint).parent.name + "/B"
        self.checkpoint_sha = hashlib.sha256(
            (Path(checkpoint) / "manifest.json").read_bytes()
        ).hexdigest()
        if provider is None:
            if not os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"):
                raise ValueError("Configure OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
            provider = TracerProvider(
                resource=Resource.create({"service.name": "mini-swe-lightgbm"})
            )
            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(), schedule_delay_millis=1000)
            )
        self.provider = provider
        self.tracer = provider.get_tracer("mini-swe-lightgbm-demo")
        self.scores = ScoreClient() if os.environ.get("LANGFUSE_BASE_URL") else None
        self.score_pool = ThreadPoolExecutor(max_workers=1) if self.scores else None
        self.score_jobs = []

    def write(self, filename, record):
        with (self.output / filename).open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    @contextmanager
    def call(self, agent, step):
        common = {
            "demo.run_id": self.run_id,
            "demo.task_id": self.task_id,
            "demo.call_index": step,
            "session.id": self.run_id,
            "langfuse.session.id": self.run_id,
            "langfuse.trace.name": self.task_id + f" / call {step}",
            "openinference.span.kind": "CHAIN",
        }
        with self.tracer.start_as_current_span(
            "agent.call", attributes=common, record_exception=False, set_status_on_exception=False
        ) as parent:
            span_context = parent.get_span_context()
            trace_id = format(span_context.trace_id, "032x")
            started = time.perf_counter()
            try:
                with self.tracer.start_as_current_span(
                    "lightgbm.forecast",
                    attributes={
                        **common,
                        "openinference.span.kind": "CHAIN",
                        "forecast.variant": "B",
                        "forecast.checkpoint": self.checkpoint_id,
                        "forecast.checkpoint_sha256": self.checkpoint_sha,
                        "forecast.boundary": "before_model_call",
                        "forecast.remaining_calls_includes_upcoming": True,
                    },
                    record_exception=False,
                    set_status_on_exception=False,
                ) as span:
                    feature, history, _ = self.bundle.features(
                        self.config, self.task_id, agent.messages, self.run_id
                    )
                    if len(history) != step - 1:
                        raise ValueError("Forecast boundary does not match observed calls")
                    predictions = self.bundle.predict([feature])[0]["predictions"]
                    if step == 1:
                        predictions.update(
                            self.bundle.predict([feature], "task_start")[0]["predictions"]
                        )
                    metadata = ir_metadata(history)
                    payload = {"predictions": predictions, "source_ir": metadata}
                    span.set_attributes(prediction_attributes(predictions))
                    span.set_attribute("ir.source.phase", metadata["phase"])
                    span.set_attribute("ir.source.categories", metadata["categories"])
                    span.set_attribute("ir.source.metadata_json", json.dumps(metadata))
                    span.set_attribute("output.mime_type", "application/json")
                    span.set_attribute("output.value", json.dumps(payload))
                    record = {
                        "run_id": self.run_id,
                        "call_index": step,
                        "trace_id": trace_id,
                        "forecast_span_id": format(span.get_span_context().span_id, "016x"),
                        "checkpoint": self.checkpoint_id,
                        **payload,
                        "forecast_seconds": time.perf_counter() - started,
                    }
                    self.write("forecasts.jsonl", record)
                    self.write("features.jsonl", feature)
                    if self.scores:
                        self.score_jobs.append(
                            (trace_id, self.score_pool.submit(self.scores.send, record))
                        )
                headers = {}
                inject(headers)
                yield headers
            except Exception as error:
                parent.set_status(Status(StatusCode.ERROR, type(error).__name__))
                raise
            finally:
                # Reconstruct only observed responses; termination may have no response.
                _, history, usage = self.bundle.features(
                    self.config, self.task_id, agent.messages, self.run_id
                )
                if len(history) == step:
                    actual = usage[-1]
                    result_ir = ir_metadata(history)
                    parent.set_attribute("ir.observed.phase", result_ir["phase"])
                    parent.set_attribute("ir.observed.metadata_json", json.dumps(result_ir))
                    parent.set_attribute("actual.usage_json", json.dumps(actual))
                    # Custom attributes avoid billing the same usage twice in Langfuse.
                    for key, value in actual.items():
                        if value is not None:
                            parent.set_attribute("actual." + key, value)
                    self.write(
                        "actuals.jsonl",
                        {
                            "call_index": step,
                            "trace_id": trace_id,
                            "usage": actual,
                            "observed_ir": result_ir,
                        },
                    )

    def close(self):
        self.provider.force_flush(timeout_millis=10000)
        self.provider.shutdown()
        if self.score_pool:
            self.score_pool.shutdown(wait=True)
            for trace_id, job in self.score_jobs:
                try:
                    receipt = job.result()
                except Exception as error:
                    receipt = {"trace_id": trace_id, "error_type": type(error).__name__}
                self.write("score-export.jsonl", receipt)
