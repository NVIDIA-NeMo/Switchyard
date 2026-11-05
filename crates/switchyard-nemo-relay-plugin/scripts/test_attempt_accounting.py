# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise a packaged native plugin with Relay 0.10 and a synthetic provider."""

import argparse
import asyncio
import json
import math
import platform
import sys
import tempfile
import threading
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

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        first = Provider.calls % 2 == 0
        Provider.calls += 1
        usage = {"prompt_tokens": 11 if first else 7, "completion_tokens": 3 if first else 2}
        if self.scenario in {"reported", "missing"}:
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
        raw = json.dumps(body).encode()
        self.send_response(503 if first else 200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Retry-After", "0")
        # A body charge must override this valid header fallback.
        if self.scenario == "reported":
            self.send_header("x-litellm-response-cost", "1.0")
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *_args):
        pass


async def run(manifest, output):
    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    events = []
    subscribers.register("attempt-accounting-test", lambda e: events.append(e.to_dict()))
    activation = None
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
            ends = {e["uuid"]: e for e in events if e.get("scope_category") == "end"}
            for record in report["attempts"]:
                attempt = ends[record["uuid"]]
                assert attempt["name"] == "libsy.upstream_attempt"
                assert attempt["metadata"]["switchyard.accounting.schema_version"] == 1
                parent = ends[attempt["parent_uuid"]]
                assert parent["name"] == "libsy.client_call"
                assert parent["metadata"]["switchyard.accounting.unit"] == "summary"
                assert parent["metadata"]["switchyard.accounting.schema_version"] == 1
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
        (output / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    finally:
        if activation is not None:
            await activation.close()
        subscribers.deregister("attempt-accounting-test")
        server.shutdown()
        server.server_close()
        thread.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        asyncio.run(run(args.manifest.resolve(), args.output.resolve()))
    else:
        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(run(args.manifest.resolve(), Path(directory)))


if __name__ == "__main__":
    main()
