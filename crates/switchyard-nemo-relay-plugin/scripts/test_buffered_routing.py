# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check buffered routing attribution with a packaged plugin and Relay 0.10.

Run with an isolated Relay configuration and a loopback-only network policy.
The synthetic judge streams even though the classifier requests a verdict.
"""

import argparse
import asyncio
import json
import math
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import nemo_relay as relay
from nemo_relay import codecs, model_pricing, plugin, subscribers


class Provider(BaseHTTPRequestHandler):
    calls = Counter()
    lock = threading.Lock()

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        model = request["model"]
        with self.lock:
            self.calls[model] += 1
        judge = model == "test/judge"
        content = '{"target":"efficient"}' if judge else "answer"
        usage = {"prompt_tokens": 11 if judge else 7, "completion_tokens": 3 if judge else 2}
        if judge:
            assert request["stream"] is True
        streaming = request.get("stream", False)
        if streaming:
            frames = [
                {"id": "judge", "model": model, "choices": [
                    {"index": 0, "delta": {"role": "assistant", "content": content}},
                ]},
                {"id": "judge", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                {"id": "judge", "choices": [], "usage": usage},
            ]
            body = ("".join("data: " + json.dumps(frame) + "\n\n" for frame in frames)
                    + "data: [DONE]\n\n").encode()
        else:
            body = json.dumps({"id": "answer", "model": model, "usage": usage, "choices": [
                {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"},
            ]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream" if streaming else "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


def check_workflow(events, label):
    ends = {e["uuid"]: e for e in events if e.get("kind") == "scope" and e.get("scope_category") == "end"}
    workflow = next(e for e in ends.values() if e.get("name") == label)
    outer, = [e for e in ends.values() if e.get("name") == "openai.chat_completions"
              and e.get("parent_uuid") == workflow["uuid"]]
    calls = [e for e in ends.values() if e.get("name") == "libsy.client_call"
             and e.get("parent_uuid") == outer["uuid"]]
    assert len(calls) == 2, (label, calls)
    assert {e["metadata"]["selected_model"] for e in calls} == {"test/judge", "test/answer"}
    total = 0.0
    for call in calls:
        # Exactly one start/end pair: replay must not create another observation.
        pair = [e for e in events if e.get("uuid") == call["uuid"]]
        assert Counter(e["scope_category"] for e in pair) == {"start": 1, "end": 1}
        assert all(e["parent_uuid"] == outer["uuid"] for e in pair)
        assert call["metadata"]["outcome"] == "ok"
        judge = call["metadata"]["selected_model"] == "test/judge"
        usage = codecs.AnthropicMessagesCodec().decode_response(call["data"]).usage
        assert usage["prompt_tokens"] == (11 if judge else 7), usage
        assert usage["completion_tokens"] == (3 if judge else 2), usage
        assert usage["cost"]["currency"] == "USD", usage
        total += usage["cost"]["total"]
    assert math.isclose(total, 0.028), total
    return {"workflow_uuid": workflow["uuid"], "calls": len(calls), "logical_call_cost_usd": total}


async def run(manifest, output):
    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    events = []
    subscribers.register("buffered-routing-test", lambda event: events.append(event.to_dict()))
    activation = None
    try:
        deployment = output / "switchyard.toml"
        deployment.write_text(f'''schema_version = 1
[llm_clients.local]
format = "openai_chat"
base_url = "http://127.0.0.1:{server.server_port}/v1"
max_retries = 0
[targets.judge]
id = "test/judge"
llm_client = "local"
extra_body = {{stream = true}}
[targets.answer]
id = "test/answer"
llm_client = "local"
[routes.test]
id = "switchyard/test"
type = "llm_classifier"
mode = "custom"
models = {{judge = ["judge"], efficient = ["answer"], any = ["answer"]}}
default_target = "efficient"
prompt = "Select efficient."
response_schema = '{{"type":"object","properties":{{"target":{{"type":"string","enum":["efficient"]}}}},"required":["target"],"additionalProperties":false}}'
policy = {{type = "target_selector", selector = "/target"}}
''')
        config = output / "plugins.toml"
        config.write_text(f'''[[plugins.dynamic]]
manifest = {json.dumps(str(manifest))}
[plugins.dynamic.config]
switchyard_config_path = {json.dumps(str(deployment))}
[plugins.policy.overrides."nvidia.switchyard"]
attestation = "integrity_only"
''')
        pricing = model_pricing.ComponentSpec(model_pricing.PricingConfig(sources=[
            model_pricing.InlineSource(model_pricing.PricingCatalog(entries=[
                model_pricing.ModelPricing(
                    provider="anthropic", model_id=model, pricing_as_of="2026-11-05",
                    pricing_source="synthetic", rates=model_pricing.TokenPricingRates(
                        input_per_million=1000, output_per_million=2000,
                    ),
                ) for model in ["test/judge", "test/answer"]
            ])),
        ]))
        activation = await plugin.initialize(plugin.PluginConfig(components=[pricing]), config)

        async def forbidden(_request):
            raise AssertionError("plugin did not intercept the request")

        async def forbidden_stream(_request):
            raise AssertionError("plugin did not intercept the stream")
            yield  # Make this an async generator, as required by Relay.

        async def execute(label, streaming):
            chunks = []
            request = relay.LLMRequest({}, {"model": "switchyard/test", "stream": streaming,
                                           "messages": [{"role": "user", "content": label}]})
            with relay.use_scope_stack(relay.create_scope_stack()):
                with relay.scope.scope(label, relay.ScopeType.Agent):
                    if streaming:
                        stream = await relay.llm.stream_execute(
                            "openai.chat_completions", request, forbidden_stream, chunks.append,
                            lambda: {"model": "test/answer", "choices": []},
                            codec=codecs.OpenAIChatCodec(), response_codec=codecs.OpenAIChatCodec(),
                        )
                        async for _ in stream:
                            pass
                        assert chunks
                    else:
                        await relay.llm.execute("openai.chat_completions", request, forbidden,
                                                codec=codecs.OpenAIChatCodec(), response_codec=codecs.OpenAIChatCodec())

        await execute("buffered", False)
        await execute("streamed", True)
        await asyncio.gather(execute("concurrent-a", True), execute("concurrent-b", True))
        await subscribers.flush_async()
        results = {label: check_workflow(events, label)
                   for label in ["buffered", "streamed", "concurrent-a", "concurrent-b"]}
        assert Provider.calls == {"test/judge": 4, "test/answer": 4}, Provider.calls
        all_calls = [e for e in events if e.get("name") == "libsy.client_call" and e.get("scope_category") == "end"]
        assert len(all_calls) == 8, "unexpected detached or duplicated model call"
        (output / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
        (output / "results.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(results, indent=2))
    finally:
        if activation is not None:
            await activation.close()
        subscribers.deregister("buffered-routing-test")
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(args.manifest.resolve(), args.output.resolve()))
