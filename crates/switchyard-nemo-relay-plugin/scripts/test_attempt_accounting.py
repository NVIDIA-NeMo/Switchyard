# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise a packaged native plugin with Relay 0.10 and a synthetic provider."""

import argparse
import asyncio
import json
import math
import os
import platform
import sys
import tempfile
import threading
import time
import tomllib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version
from pathlib import Path

import nemo_relay as relay
from nemo_relay import codecs, model_pricing, plugin, subscribers

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from attempt_costs import summarize  # noqa: E402


class Provider(BaseHTTPRequestHandler):
    scenario = "reported"
    calls = 0
    cancel_started = threading.Event()

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path in {"/decisions", "/systemone"}:
            if self.scenario == "decision_cancel":
                Provider.cancel_started.set()
                time.sleep(0.2)
            if request["model"].endswith("error"):
                self.reply(503, {"error": "synthetic private failure"})
                return
            time.sleep(0.02)
            answer = {"type": "choice", "name": "route", "choice": "capable_first", "confidence": 1.0,
                      "probabilities": [{"value": "capable_first", "probability": 1.0}, {"value": "efficient_first", "probability": 0.0}]}
            body = {"model": request["model"], "answers": [answer], "usage": {
                "input_tokens": 6, "output_tokens": 2, "total_tokens": 8,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            }}
            if self.path == "/systemone":
                body = {"answers": {"route": {"type": "choice", "choice": "capable_first", "confidence": 1.0, "probabilities": {"capable_first": 1.0, "efficient_first": 0.0}}}}
                if request["model"].endswith("reported_zero"):
                    body["usage"] = {"provider_cost": 0.0}
            self.reply(200, body)
            return
        first = Provider.calls % 2 == 0
        Provider.calls += 1
        if self.scenario.startswith("decision_"):
            first = False
        usage = {"prompt_tokens": 11 if first else 7, "completion_tokens": 3 if first else 2}
        if self.scenario in {"reported", "missing"} or self.scenario.startswith("decision_"):
            usage["cost"] = 0.02 if first else 0.03
        if self.scenario == "zero":
            usage = {"cost": 0.0}
        if self.scenario == "missing" and first:
            usage = {}
        body = {"model": request["model"], "usage": usage}
        if first:
            body["error"] = {"message": "synthetic failure"}
        else:
            body.update(id="r1", choices=[{
                "index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop",
            }])
        self.reply(503 if first else 200, body)

    def reply(self, status, body):
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Retry-After", "0")
        # A body charge must override this valid header fallback.
        if self.scenario == "reported":
            self.send_header("x-litellm-response-cost", "1.0")
        self.end_headers()
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *_args):
        pass


async def run(manifest, output, include_decisions=False):
    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    events = []
    subscribers.register("attempt-accounting-test", lambda e: events.append(e.to_dict()))
    activation = None
    previous_key = os.environ.get("SWITCHYARD_REQ4_FIXTURE_KEY")
    try:
        deployment = output / "switchyard.toml"
        deployment.write_text(f'''schema_version = 1
[llm_clients.local]
format = "openai_chat"
base_url = "http://127.0.0.1:{server.server_port}/v1"
max_retries = 1
failure_cooldown_ms = 0
[targets.default]
id = "test/model"
llm_client = "local"
[targets.unpriced]
id = "test/unpriced"
llm_client = "local"
[routes.default]
id = "switchyard/default"
type = "passthrough"
target = "default"
[routes.unpriced]
id = "switchyard/unpriced"
type = "passthrough"
target = "unpriced"
''')
        if include_decisions:
            os.environ["SWITCHYARD_REQ4_FIXTURE_KEY"] = "synthetic-decision-key"
            with deployment.open("a") as stream:
                for client, wire, endpoint in [("judge", "openai_decisions", "decisions"), ("system", "system_one", "systemone")]:
                    stream.write(f'''\n[decision_clients.{client}]
format = "{wire}"
endpoint = "http://127.0.0.1:{server.server_port}/{endpoint}"
api_key_env = "SWITCHYARD_REQ4_FIXTURE_KEY"
timeout_ms = 2000
''')
                for name in ["free", "paid", "unpriced", "missing", "reported_zero", "error"]:
                    client = "system" if name in {"missing", "reported_zero"} else "judge"
                    stream.write(f'''\n[decision_targets.judge_{name}]
id = "test/decision-{name}"
decision_client = "{client}"
[routes.decision_{name}]
id = "switchyard/decision_{name}"
type = "llm_classifier"
classifier_target = "judge_{name}"
strong_target = "default"
weak_target = "unpriced"
fail_open = false
[routes.decision_{name}.decision]
cutoff = 0.5
candidates = {{ capable_first = "default", efficient_first = "unpriced" }}
criteria = {{ capable_first = "Capable", efficient_first = "Efficient" }}
instructions = {{ task = "Choose" }}
evidence = {{ description = "Synthetic decision context" }}
''')
        config = output / "plugins.toml"
        config.write_text(f'''[[plugins.dynamic]]
manifest = {json.dumps(str(manifest))}
[plugins.dynamic.config]
switchyard_config_path = {json.dumps(str(deployment))}
[plugins.policy.overrides."nvidia.switchyard"]
attestation = "integrity_only"
''')
        prices = model_pricing.ComponentSpec(model_pricing.PricingConfig(sources=[
            model_pricing.InlineSource(model_pricing.PricingCatalog(entries=[
                model_pricing.ModelPricing(
                    provider="anthropic", model_id="test/model", pricing_as_of="2026-11-04",
                    pricing_source="synthetic", rates=model_pricing.TokenPricingRates(
                        input_per_million=1000, output_per_million=2000,
                    ),
                ),
                *[model_pricing.ModelPricing(
                    provider="anthropic", model_id=f"test/decision-{name}", pricing_as_of="2026-11-04",
                    pricing_source="synthetic", rates=model_pricing.TokenPricingRates(
                        input_per_million=0 if name == "free" else 1000,
                        output_per_million=0 if name == "free" else 2000,
                    ),
                ) for name in ["free", "paid", "missing"]],
            ])),
        ]))
        activation = await plugin.initialize(plugin.PluginConfig(components=[prices]), config)
        (output / "environment.json").write_text(json.dumps({
            "relay_version": version("nemo-relay"), "platform": platform.platform(),
            "bundle_sha256": tomllib.loads(manifest.read_text())["integrity"]["sha256"],
        }, indent=2))

        async def forbidden(_request):
            raise AssertionError("native plugin did not intercept the request")

        for scenario, expected_subtotal, expected_total in [
            ("reported", 0.05, 0.05), ("missing", 0.03, None),
            ("estimated", 0.028, 0.028), ("unpriced", 0.0, None), ("zero", 0.0, 0.0),
        ]:
            Provider.scenario = scenario
            Provider.calls = 0
            route = "unpriced" if scenario == "unpriced" else "default"
            with relay.scope.scope(scenario, relay.ScopeType.Agent):
                await relay.llm.execute(
                    "openai.chat_completions",
                    relay.LLMRequest({}, {"model": f"switchyard/{route}", "messages": [{"role": "user", "content": "synthetic test"}]}),
                    forbidden, codec=codecs.OpenAIChatCodec(), response_codec=codecs.OpenAIChatCodec(),
                )
            await subscribers.flush_async()
            workflow = next(e["uuid"] for e in events if e.get("name") == scenario and e.get("scope_category") == "start")
            report = summarize(events, workflow)
            assert Provider.calls == 2
            assert len(report["attempts"]) == 2
            assert report["attempts"][0]["outcome"] == "error"
            assert report["coverage_complete"]
            assert math.isclose(report["known_cost_subtotal_usd"], expected_subtotal)
            if expected_total is None:
                assert report["total_cost_usd"] is None
            else:
                assert math.isclose(report["total_cost_usd"], expected_total)
            # ATOF replay, missing records, and mixed workflows must not inflate totals.
            assert summarize(events + events, workflow) == report
            dropped = report["attempts"][0]["uuid"]
            assert summarize([e for e in events if e["uuid"] != dropped], workflow)["total_cost_usd"] is None
            if scenario == "reported":
                assert all(a["cost"]["source"] == "provider_reported" for a in report["attempts"])
            if scenario == "estimated":
                assert all(a["cost"]["source"] == "model_pricing" for a in report["attempts"])
            if scenario == "zero":
                assert all(a["usage_status"] == "missing" for a in report["attempts"])
            (output / f"{scenario}.json").write_text(json.dumps(report, indent=2))
            print(f"{scenario}: subtotal={report['known_cost_subtotal_usd']}, total={report['total_cost_usd']}")
        if include_decisions:
            Provider.scenario = "decision_workflow"
            for name, subtotal, total in [("free", 0.03, 0.03), ("paid", 0.04, 0.04),
                                          ("unpriced", 0.03, None), ("missing", 0.03, None), ("reported_zero", 0.03, 0.03), ("error", 0.0, None)]:
                await check_decisions(events, output, name, name, 1, subtotal, total, forbidden)
            await check_decisions(events, output, "repeated", "paid", 2, 0.08, 0.08, forbidden)
            await asyncio.gather(
                check_decisions(events, output, "concurrent-a", "paid", 1, 0.04, 0.04, forbidden),
                check_decisions(events, output, "concurrent-b", "paid", 1, 0.04, 0.04, forbidden),
            )
            await check_cancelled_decision(events, output, forbidden)
        (output / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    finally:
        (output / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
        if activation is not None:
            await activation.close()
        subscribers.deregister("attempt-accounting-test")
        server.shutdown()
        server.server_close()
        thread.join()
        if previous_key is None:
            os.environ.pop("SWITCHYARD_REQ4_FIXTURE_KEY", None)
        else:
            os.environ["SWITCHYARD_REQ4_FIXTURE_KEY"] = previous_key


async def check_decisions(events, output, label, model, count, subtotal, total, forbidden):
    with relay.use_scope_stack(relay.create_scope_stack()):
        await check_decisions_in_scope(events, output, label, model, count, subtotal, total, forbidden)


async def check_decisions_in_scope(events, output, label, model, count, subtotal, total, forbidden):
    headers = {} if label == "concurrent-a" else {"x-switchyard-task-id": f"task-{label}", "x-switchyard-turn-id": "turn-1", "x-switchyard-request-id": f"request-{label}"}
    with relay.scope.scope(f"decision-{label}", relay.ScopeType.Agent):
        for _ in range(count):
            try:
                await relay.llm.execute("openai.chat_completions", relay.LLMRequest(headers, {
                    "model": f"switchyard/decision_{model}", "messages": [{"role": "user", "content": "synthetic decision request"}],
                }), forbidden, codec=codecs.OpenAIChatCodec(), response_codec=codecs.OpenAIChatCodec())
            except RuntimeError:
                if model != "error":
                    raise
            else:
                assert model != "error", "expected a decision error"
    await subscribers.flush_async()
    workflow = next(e["uuid"] for e in events if e.get("name") == f"decision-{label}" and e.get("scope_category") == "start")
    report = summarize(events, workflow)
    assert len(report["decisions"]) == count, report
    assert report["coverage_complete"], report
    assert math.isclose(report["known_cost_subtotal_usd"], subtotal), report
    if total is None:
        assert report["total_cost_usd"] is None, report
    else:
        assert math.isclose(report["total_cost_usd"], total), report
    ends = {e["uuid"]: e for e in events if e.get("kind") == "scope" and e.get("scope_category") == "end"}
    for decision in report["decisions"]:
        event = ends[decision["uuid"]]
        assert event["name"] == "libsy.decision_call"
        assert event["category"] == "llm"
        assert decision["context_status"] == "bound"
        parent = ends[event["parent_uuid"]]
        assert parent["name"] == "openai.chat_completions" and parent["parent_uuid"] == workflow
        if headers:
            assert event["metadata"]["task_id"] == f"task-{label}"
        else:
            assert "task_id" not in event["metadata"]
        assert event["data"]["model"] == f"test/decision-{model}"
        dropped = [e for e in events if e["uuid"] != decision["uuid"]]
        assert summarize(dropped, workflow)["total_cost_usd"] is None
    assert summarize(events + events, workflow) == report
    (output / f"decision-{label}.json").write_text(json.dumps(report, indent=2))
    print(f"decision-{label}: subtotal={report['known_cost_subtotal_usd']}, total={report['total_cost_usd']}")


async def check_cancelled_decision(events, output, forbidden):
    Provider.scenario = "decision_cancel"
    Provider.cancel_started.clear()
    with relay.use_scope_stack(relay.create_scope_stack()):
        with relay.scope.scope("decision-cancelled", relay.ScopeType.Agent):
            await subscribers.flush_async()
            workflow = next(e["uuid"] for e in events if e.get("name") == "decision-cancelled")
            async def invoke():
                await relay.llm.execute("openai.chat_completions", relay.LLMRequest({}, {
                    "model": "switchyard/decision_paid", "messages": [{"role": "user", "content": "cancelled request"}],
                }), forbidden, codec=codecs.OpenAIChatCodec(), response_codec=codecs.OpenAIChatCodec())

            task = asyncio.create_task(invoke())
            try:
                assert await asyncio.to_thread(Provider.cancel_started.wait, 1.0), "decision did not start"
            finally:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            for _ in range(100):
                await subscribers.flush_async()
                if any(e.get("name") == "libsy.decision_call" and e.get("scope_category") == "end" and e.get("metadata", {}).get("outcome") == "cancelled" for e in events):
                    break
                await asyncio.sleep(0.01)
        await subscribers.flush_async()
        report = summarize(events, workflow)
    assert len(report["decisions"]) == 1, report
    decision = report["decisions"][0]
    assert decision["outcome"] == "cancelled" and decision["usage_status"] == "missing", report
    assert decision["context_status"] == "bound", report
    scopes = {e["uuid"]: e for e in events if e.get("kind") == "scope"}
    parent = scopes[scopes[decision["uuid"]]["parent_uuid"]]
    assert parent["name"] == "openai.chat_completions" and parent["parent_uuid"] == workflow
    assert report["total_cost_usd"] is None
    (output / "decision-cancelled.json").write_text(json.dumps(report, indent=2))
    print("decision-cancelled: usage=missing, total=None, native workflow preserved")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-decisions", action="store_true")
    args = parser.parse_args()
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        asyncio.run(run(args.manifest.resolve(), args.output.resolve(), args.include_decisions))
    else:
        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(run(args.manifest.resolve(), Path(directory), args.include_decisions))


if __name__ == "__main__":
    main()
