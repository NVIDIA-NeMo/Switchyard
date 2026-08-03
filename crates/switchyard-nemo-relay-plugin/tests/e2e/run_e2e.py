# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process E2E for the Switchyard plugin against a real NeMo Relay host."""

from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, cast

HERE = Path(__file__).resolve().parent
CRATE_ROOT = HERE.parents[1]
TARGET_AUTHORIZATION = "Bearer target-e2e"
TARGET_AUTHORIZATION_ENV = "SWITCHYARD_E2E_TARGET_AUTHORIZATION"


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def capture(process: subprocess.Popen[str], sink: list[str]) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        sink.append(line.rstrip())


def http_json(base: str, path: str) -> dict[str, Any]:
    with urllib.request.urlopen(f"{base}{path}", timeout=5) as response:
        return cast(dict[str, Any], json.loads(response.read()))


def request(relay_url: str, path: str, body: dict[str, Any]) -> tuple[int, bytes]:
    request = urllib.request.Request(
        f"{relay_url}{path}",
        data=json.dumps(body).encode(),
        headers={
            "content-type": "application/json",
            "authorization": "Bearer e2e",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return response.status, response.read()


def request_until_stream_error(
    relay_url: str, path: str, body: dict[str, Any]
) -> tuple[int, bytes, bool]:
    request = urllib.request.Request(
        f"{relay_url}{path}",
        data=json.dumps(body).encode(),
        headers={
            "content-type": "application/json",
            "authorization": "Bearer e2e",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        try:
            return response.status, response.read(), False
        except http.client.IncompleteRead as error:
            return response.status, error.partial, True


def stream_events(raw: bytes) -> list[dict[str, object]]:
    return [
        json.loads(line[6:]) for line in raw.decode().splitlines() if line.startswith("data: {")
    ]


def response_text(protocol: str, response: dict[str, Any]) -> str:
    if protocol == "openai_chat":
        return str(response["choices"][0]["message"]["content"])
    if protocol == "openai_responses":
        return str(response["output"][0]["content"][0]["text"])
    return str(response["content"][0]["text"])


def stream_text(protocol: str, events: list[dict[str, Any]]) -> str:
    if protocol == "openai_chat":
        return "".join(
            str(event["choices"][0]["delta"].get("content", ""))
            for event in events
            if event.get("choices")
        )
    if protocol == "openai_responses":
        return "".join(
            str(event.get("delta", ""))
            for event in events
            if event.get("type") == "response.output_text.delta"
        )
    return "".join(
        str(event.get("delta", {}).get("text", ""))
        for event in events
        if event.get("type") == "content_block_delta"
    )


CASES: tuple[tuple[str, str, dict[str, Any]], ...] = (
    (
        "openai_chat",
        "/v1/chat/completions",
        {
            "model": "caller/chat",
            "messages": [{"role": "user", "content": "hello"}],
            "caller_extension": {"preserve": True},
        },
    ),
    (
        "openai_responses",
        "/v1/responses",
        {
            "model": "caller/responses",
            "input": "hello",
            "caller_extension": {"preserve": True},
        },
    ),
    (
        "anthropic_messages",
        "/v1/messages",
        {
            "model": "caller/anthropic",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hello"}],
            "caller_extension": {"preserve": True},
        },
    ),
)


def plugin_config(
    manifest: Path,
    provider_url: str,
    atof_directory: Path,
    algorithm: str,
    targets: str,
    defaults: str,
    *,
    max_retries: int = 1,
) -> str:
    return f"""\
version = 1

[[plugins.dynamic]]
manifest = {json.dumps(str(manifest))}

[plugins.dynamic.config]
version = 2
priority = 0
max_retries = {max_retries}

{algorithm}

[plugins.dynamic.config.default_targets]
{defaults}

{targets.format(provider_url=provider_url)}

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 3

[components.config.atof]
enabled = true

[[components.config.atof.sinks]]
type = "file"
mode = "overwrite"
output_directory = {json.dumps(str(atof_directory))}
filename = "events.jsonl"
"""


class RelayScenario:
    def __init__(
        self,
        relay_bin: Path,
        root: Path,
        provider_url: str,
        name: str,
        config: str,
    ) -> None:
        self.relay_bin = relay_bin
        self.root = root / name
        self.provider_url = provider_url
        self.config = config
        self.port = free_port()
        self.process: subprocess.Popen[str] | None = None
        self.log: list[str] = []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def atof_path(self) -> Path:
        return self.root / "atof" / "events.jsonl"

    def __enter__(self) -> RelayScenario:
        (self.root / ".nemo-relay").mkdir(parents=True)
        (self.root / ".nemo-relay" / "plugins.toml").write_text(self.config, encoding="utf-8")
        subprocess.run(
            [str(self.relay_bin), "plugins", "enable", "nvidia.switchyard"],
            cwd=self.root,
            env={**os.environ, TARGET_AUTHORIZATION_ENV: TARGET_AUTHORIZATION},
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.process = subprocess.Popen(
            [
                str(self.relay_bin),
                "--bind",
                f"127.0.0.1:{self.port}",
                "--openai-base-url",
                f"{self.provider_url}/v1",
                "--log-level",
                "warn",
            ],
            cwd=self.root,
            env={**os.environ, TARGET_AUTHORIZATION_ENV: TARGET_AUTHORIZATION},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        threading.Thread(target=capture, args=(self.process, self.log), daemon=True).start()
        deadline = time.time() + 20
        while True:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"Relay exited early ({self.process.returncode}):\n" + "\n".join(self.log[-40:])
                )
            try:
                with urllib.request.urlopen(f"{self.url}/healthz", timeout=1) as response:
                    if response.status == 200:
                        return self
            except (OSError, urllib.error.URLError):
                pass
            if time.time() > deadline:
                raise TimeoutError("Relay did not become healthy")
            time.sleep(0.1)

    def __exit__(self, *_: object) -> None:
        assert self.process is not None
        self.process.send_signal(signal.SIGINT)
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        (self.root / "relay.log").write_text("\n".join(self.log) + "\n", encoding="utf-8")
        if self.process.returncode:
            raise RuntimeError(
                f"Relay exited with {self.process.returncode}:\n" + "\n".join(self.log[-40:])
            )

    def marks(self, name: str, expected: int = 1, timeout: float = 5) -> list[dict[str, object]]:
        deadline = time.time() + timeout
        while True:
            try:
                events = [
                    json.loads(line)
                    for line in self.atof_path.read_text(encoding="utf-8").splitlines()
                    if line
                ]
            except FileNotFoundError:
                events = []
            matches = [event for event in events if event.get("name") == name]
            if len(matches) >= expected or time.time() > deadline:
                return matches
            time.sleep(0.05)


def run_same_protocol(
    relay_bin: Path, root: Path, manifest: Path, provider_url: str
) -> dict[str, object]:
    config = plugin_config(
        manifest,
        provider_url,
        root / "same" / "atof",
        '[plugins.dynamic.config.algorithm]\nkind = "random"\nseed = 7',
        """\
[plugins.dynamic.config.targets.chat]
model = "fake/header-target"
protocol = "openai_chat"
base_url = "{provider_url}/v1"
weight = 1

[plugins.dynamic.config.targets.chat.header_env]
authorization = "SWITCHYARD_E2E_TARGET_AUTHORIZATION"

[plugins.dynamic.config.targets.chat.headers]
x-switchyard-target = "same"
""",
        'openai_chat = "chat"',
    )
    with RelayScenario(relay_bin, root, provider_url, "same", config) as relay:
        template = dict(CASES[0][2])
        template["stream"] = False
        status, raw = request(relay.url, CASES[0][1], template)
        buffered = json.loads(raw)
        assert status == 200
        assert buffered["provider_extension"] == {"preserved": True}
        assert buffered["system_fingerprint"] == "fp_dynamic_plugin"

        template["stream"] = True
        status, raw = request(relay.url, CASES[0][1], template)
        events = stream_events(raw)
        assert status == 200
        assert len(events) == 2
        assert all(event["provider_extension"] == {"preserved": True} for event in events)
        assert all(event["system_fingerprint"] == "fp_dynamic_plugin" for event in events)
    assert len(relay.marks("switchyard.routing.decision", 2)) == 2

    replayed = {"openai_chat": len(events)}
    for protocol, path, template in CASES[1:]:
        config = plugin_config(
            manifest,
            provider_url,
            root / f"same-{protocol}" / "atof",
            '[plugins.dynamic.config.algorithm]\nkind = "random"\nseed = 7',
            f"""\
[plugins.dynamic.config.targets.target]
model = "fake/preserve-{protocol}"
protocol = "{protocol}"
base_url = "{{provider_url}}/v1"
weight = 1
""",
            f'{protocol} = "target"',
        )
        with RelayScenario(relay_bin, root, provider_url, f"same-{protocol}", config) as scenario:
            body = dict(template)
            body["stream"] = False
            status, raw = request(scenario.url, path, body)
            response = json.loads(raw)
            assert status == 200
            assert response["provider_extension"] == {"preserved": True}

            body["stream"] = True
            status, raw = request(scenario.url, path, body)
            events = stream_events(raw)
            assert status == 200
            assert events
            assert all(event["provider_extension"] == {"preserved": True} for event in events)
        assert len(scenario.marks("switchyard.routing.decision", 2)) == 2
        replayed[protocol] = len(events)

    return {"buffered_unknown_fields": True, "stream_events_replayed": replayed}


def run_random(relay_bin: Path, root: Path, manifest: Path, provider_url: str) -> dict[str, object]:
    config = plugin_config(
        manifest,
        provider_url,
        root / "random" / "atof",
        '[plugins.dynamic.config.algorithm]\nkind = "random"\nseed = 42',
        """\
[plugins.dynamic.config.targets.chat]
model = "fake/chat"
protocol = "openai_chat"
base_url = "{provider_url}/v1"
weight = 1

[plugins.dynamic.config.targets.responses]
model = "fake/responses"
protocol = "openai_responses"
base_url = "{provider_url}/v1"
weight = 1

[plugins.dynamic.config.targets.anthropic]
model = "fake/anthropic"
protocol = "anthropic_messages"
base_url = "{provider_url}/v1"
weight = 1
""",
        ('openai_chat = "chat"\nopenai_responses = "responses"\nanthropic_messages = "anthropic"'),
    )
    with RelayScenario(relay_bin, root, provider_url, "random", config) as relay:
        models: set[str] = set()
        for protocol, path, template in CASES:
            for _ in range(4):
                body = dict(template)
                body["stream"] = False
                status, raw = request(relay.url, path, body)
                response = json.loads(raw)
                assert status == 200
                assert response_text(protocol, response)
                models.add(response["model"])

        streams = {}
        for protocol, path, template in CASES:
            body = dict(template)
            body["stream"] = True
            status, raw = request(relay.url, path, body)
            events = stream_events(raw)
            assert status == 200
            streams[protocol] = stream_text(protocol, events)
            assert streams[protocol]

        def concurrent_call(index: int) -> str:
            body = dict(CASES[0][2])
            body["stream"] = False
            body["messages"] = [{"role": "user", "content": f"concurrent {index}"}]
            status, raw = request(relay.url, CASES[0][1], body)
            assert status == 200
            return response_text("openai_chat", json.loads(raw))

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            concurrent_results = list(executor.map(concurrent_call, range(12)))
        assert all(concurrent_results)

        assert models == {"fake/chat", "fake/responses", "fake/anthropic"}
    decisions = relay.marks("switchyard.routing.decision", 27)
    requested = relay.marks("switchyard.routing.requested", 27)
    assert len(decisions) == 27
    assert len(requested) == 27
    assert all(event["parent_uuid"] for event in decisions)
    assert {
        event["data"]["selected_target"]  # type: ignore[index]
        for event in decisions
    } == {"chat", "responses", "anthropic"}
    return {
        "models": sorted(models),
        "stream_text": streams,
        "concurrent_calls": len(concurrent_results),
        "routing_decisions": len(decisions),
    }


def classifier_config(
    manifest: Path,
    provider_url: str,
    atof: Path,
    classifier_model: str,
) -> str:
    return plugin_config(
        manifest,
        provider_url,
        atof,
        """\
[plugins.dynamic.config.algorithm]
kind = "llm_classifier"
classifier_target = "classifier"
weak_target = "weak"
strong_target = "strong"
base_threshold = 0.5
min_confidence = 0.5
session_affinity = false
message_hash_fallback = false
""",
        f"""\
[plugins.dynamic.config.targets.classifier]
model = "{classifier_model}"
protocol = "openai_chat"
base_url = "{{provider_url}}/v1"

[plugins.dynamic.config.targets.weak]
model = "fake/weak"
protocol = "openai_responses"
base_url = "{{provider_url}}/v1"

[plugins.dynamic.config.targets.strong]
model = "fake/strong"
protocol = "anthropic_messages"
base_url = "{{provider_url}}/v1"

[plugins.dynamic.config.targets.fallback]
model = "fake/fallback"
protocol = "openai_chat"
base_url = "{{provider_url}}/v1"
""",
        ('openai_chat = "fallback"\nopenai_responses = "weak"\nanthropic_messages = "strong"'),
    )


def run_classifier(
    relay_bin: Path, root: Path, manifest: Path, provider_url: str
) -> dict[str, object]:
    weak_config = classifier_config(
        manifest,
        provider_url,
        root / "classifier-weak" / "atof",
        "fake/classifier",
    )
    with RelayScenario(relay_bin, root, provider_url, "classifier-weak", weak_config) as weak_relay:
        for protocol, path, template in CASES:
            body = dict(template)
            body["stream"] = False
            status, raw = request(weak_relay.url, path, body)
            response = json.loads(raw)
            assert status == 200
            assert response["model"] == "fake/weak"
            assert response_text(protocol, response) == "responses from fake/weak"

            body["stream"] = True
            status, raw = request(weak_relay.url, path, body)
            events = stream_events(raw)
            assert status == 200
            assert stream_text(protocol, events) == "responses from fake/weak"

        def concurrent_classifier_call(index: int) -> str:
            protocol, path, template = CASES[index % len(CASES)]
            body = dict(template)
            body["stream"] = False
            status, raw = request(weak_relay.url, path, body)
            response = json.loads(raw)
            assert status == 200
            assert response["model"] == "fake/weak"
            return response_text(protocol, response)

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            concurrent_results = list(executor.map(concurrent_classifier_call, range(6)))
        assert all(result == "responses from fake/weak" for result in concurrent_results)

    weak_decisions = weak_relay.marks("switchyard.routing.decision", 12)
    assert len(weak_decisions) == 12
    assert all(
        event["data"]["algorithm"] == "llm_task_classifier"  # type: ignore[index]
        and event["data"]["selected_target"] == "weak"  # type: ignore[index]
        and event["data"]["routing_tier"] == "weak"  # type: ignore[index]
        for event in weak_decisions
    )

    strong_config = classifier_config(
        manifest,
        provider_url,
        root / "classifier-strong" / "atof",
        "fake/classifier-strong",
    )
    with RelayScenario(
        relay_bin, root, provider_url, "classifier-strong", strong_config
    ) as strong_relay:
        body = dict(CASES[0][2])
        body["stream"] = False
        status, raw = request(strong_relay.url, CASES[0][1], body)
        response = json.loads(raw)
        assert status == 200
        assert response["model"] == "fake/strong"
        assert response_text("openai_chat", response) == "anthropic from fake/strong"

        body["stream"] = True
        status, raw = request(strong_relay.url, CASES[0][1], body)
        events = stream_events(raw)
        assert status == 200
        assert stream_text("openai_chat", events) == "anthropic from fake/strong"

    strong_decisions = strong_relay.marks("switchyard.routing.decision", 2)
    assert len(strong_decisions) == 2
    assert all(
        event["data"]["algorithm"] == "llm_task_classifier"  # type: ignore[index]
        and event["data"]["selected_target"] == "strong"  # type: ignore[index]
        and event["data"]["routing_tier"] == "strong"  # type: ignore[index]
        for event in strong_decisions
    )
    return {
        "weak_decisions": len(weak_decisions),
        "strong_decisions": len(strong_decisions),
        "protocols": [protocol for protocol, _, _ in CASES],
        "concurrent_calls": len(concurrent_results),
    }


def stage_config(manifest: Path, provider_url: str, atof: Path) -> str:
    return plugin_config(
        manifest,
        provider_url,
        atof,
        """\
[plugins.dynamic.config.algorithm]
kind = "stage_router"
capable_target = "capable"
efficient_target = "efficient"
picker = "efficient_first"
confidence_threshold = 0.5
recent_turn_window = 3
capable_system_prompt = "diagnose before editing"
efficient_system_prompt = "follow the settled plan"

[plugins.dynamic.config.algorithm.handoff_notes]
escalation_note = "continue the failed diagnosis"
only_on_wrong_signal_escalation = true

[plugins.dynamic.config.algorithm.classifier]
target = "classifier"
base_threshold = 0.5
min_confidence = 0.5
recent_turn_window = 3
""",
        """\
[plugins.dynamic.config.targets.classifier]
model = "fake/stage-classifier"
protocol = "openai_chat"
base_url = "{provider_url}/v1"

[plugins.dynamic.config.targets.efficient]
model = "fake/stage-efficient"
protocol = "openai_chat"
base_url = "{provider_url}/v1"

[plugins.dynamic.config.targets.capable]
model = "fake/stage-capable"
protocol = "openai_chat"
base_url = "{provider_url}/v1"
""",
        'openai_chat = "efficient"',
    )


def run_stage_router(
    relay_bin: Path, root: Path, manifest: Path, provider_url: str
) -> dict[str, object]:
    config = stage_config(manifest, provider_url, root / "stage-router" / "atof")
    with RelayScenario(relay_bin, root, provider_url, "stage-router", config) as relay:
        ordinary = dict(CASES[0][2])
        ordinary["stream"] = False
        status, raw = request(relay.url, CASES[0][1], ordinary)
        response = json.loads(raw)
        assert status == 200
        assert response["model"] == "fake/stage-efficient"

        failed_turn = {
            "model": "caller/chat",
            "messages": [
                {"role": "user", "content": "fix the build"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_stage",
                            "type": "function",
                            "function": {
                                "name": "Bash",
                                "arguments": json.dumps({"command": "cargo test"}),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_stage",
                    "content": "fatal runtime error: out of memory",
                },
            ],
            "caller_extension": {"preserve": True},
            "stream": True,
        }
        status, raw = request(relay.url, CASES[0][1], failed_turn)
        events = stream_events(raw)
        assert status == 200
        assert stream_text("openai_chat", events) == "chat from fake/stage-capable"

    decisions = relay.marks("switchyard.routing.decision", 2)
    assert [event["data"]["selected_target"] for event in decisions] == [  # type: ignore[index]
        "efficient",
        "capable",
    ]
    assert [event["data"]["routing_tier"] for event in decisions] == [  # type: ignore[index]
        "weak",
        "strong",
    ]
    requests = http_json(provider_url, "/requests")
    classifier_requests = cast(list[dict[str, Any]], requests["fake/stage-classifier"])
    capable_requests = cast(list[dict[str, Any]], requests["fake/stage-capable"])
    assert len(classifier_requests) == 1, "only the ambiguous turn should reach the classifier"
    assert len(capable_requests) == 1
    capable = capable_requests[0]
    assert capable["caller_extension"] == {"preserve": True}
    messages = cast(list[dict[str, Any]], capable["messages"])
    assert messages[0] == {"role": "system", "content": "diagnose before editing"}
    assert any(
        "continue the failed diagnosis" in str(message.get("content", ""))
        for message in messages
    )
    return {
        "efficient_decisions": 1,
        "capable_decisions": 1,
        "classifier_fallback_calls": len(classifier_requests),
        "mutated_request_reached_provider": True,
        "unknown_fields_preserved": True,
    }


def single_target_config(
    manifest: Path,
    provider_url: str,
    atof: Path,
    selected_model: str,
    fallback_model: str,
    *,
    max_retries: int = 1,
) -> str:
    return plugin_config(
        manifest,
        provider_url,
        atof,
        '[plugins.dynamic.config.algorithm]\nkind = "random"\nseed = 1',
        f"""\
[plugins.dynamic.config.targets.selected]
model = "{selected_model}"
protocol = "openai_chat"
base_url = "{{provider_url}}/v1"

[plugins.dynamic.config.targets.fallback]
model = "{fallback_model}"
protocol = "openai_chat"
base_url = "{{provider_url}}/v1"
""",
        'openai_chat = "fallback"',
        max_retries=max_retries,
    )


def reselection_config(
    manifest: Path,
    provider_url: str,
    atof: Path,
    failing_model: str,
    succeeding_model: str,
    fallback_model: str,
) -> str:
    return plugin_config(
        manifest,
        provider_url,
        atof,
        '[plugins.dynamic.config.algorithm]\nkind = "random"\nseed = 6',
        f"""\
[plugins.dynamic.config.targets.a_fail]
model = "{failing_model}"
protocol = "openai_chat"
base_url = "{{provider_url}}/v1"
weight = 1

[plugins.dynamic.config.targets.b_success]
model = "{succeeding_model}"
protocol = "openai_chat"
base_url = "{{provider_url}}/v1"
weight = 1

[plugins.dynamic.config.targets.z_fallback]
model = "{fallback_model}"
protocol = "openai_chat"
base_url = "{{provider_url}}/v1"
weight = 0
""",
        'openai_chat = "z_fallback"',
        max_retries=1,
    )


def run_retry_and_fallback(
    relay_bin: Path, root: Path, manifest: Path, provider_url: str
) -> dict[str, object]:
    retry_config = single_target_config(
        manifest,
        provider_url,
        root / "retry" / "atof",
        "fake/retry-once",
        "fake/retry-fallback",
    )
    with RelayScenario(relay_bin, root, provider_url, "retry", retry_config) as relay:
        body = dict(CASES[0][2])
        body["stream"] = False
        status, raw = request(relay.url, CASES[0][1], body)
        assert status == 200
        assert json.loads(raw)["model"] == "fake/retry-once"
    retry_marks = relay.marks("switchyard.routing.retry")
    assert len(retry_marks) == 1
    assert retry_marks[0]["data"] == {
        "attempt": 1,
        "retryable": True,
        "failure_kind": "http",
        "http_status": 503,
    }
    assert not relay.marks("switchyard.routing.fallback", expected=0)

    fallback_config = single_target_config(
        manifest,
        provider_url,
        root / "fallback" / "atof",
        "fake/always-fail",
        "fake/trusted-fallback",
    )
    with RelayScenario(relay_bin, root, provider_url, "fallback", fallback_config) as relay:
        body = dict(CASES[0][2])
        body["stream"] = False
        status, raw = request(relay.url, CASES[0][1], body)
        assert status == 200
        assert json.loads(raw)["model"] == "fake/trusted-fallback"
    error_marks = relay.marks("switchyard.routing.error")
    assert len(error_marks) == 1
    assert error_marks[0]["data"] == {
        "attempt": 1,
        "retryable": False,
        "failure_kind": "http",
        "http_status": 400,
    }
    assert len(relay.marks("switchyard.routing.fallback")) == 1

    reselection = reselection_config(
        manifest,
        provider_url,
        root / "retry-reselection" / "atof",
        "fake/reselect-fail",
        "fake/reselect-success",
        "fake/reselect-fallback",
    )
    with RelayScenario(relay_bin, root, provider_url, "retry-reselection", reselection) as relay:
        body = dict(CASES[0][2])
        body["stream"] = False
        status, raw = request(relay.url, CASES[0][1], body)
        assert status == 200
        assert json.loads(raw)["model"] == "fake/reselect-success"
    decisions = relay.marks("switchyard.routing.decision", 2)
    assert [event["data"]["selected_target"] for event in decisions] == [  # type: ignore[index]
        "a_fail",
        "b_success",
    ]
    assert len(relay.marks("switchyard.routing.retry")) == 1
    assert not relay.marks("switchyard.routing.fallback", expected=0)

    calls = http_json(provider_url, "/calls")
    assert calls["fake/retry-once"] == 2
    assert calls.get("fake/retry-fallback", 0) == 0
    assert calls["fake/always-fail"] == 1
    assert calls["fake/trusted-fallback"] == 1
    assert calls["fake/reselect-fail"] == 1
    assert calls["fake/reselect-success"] == 1
    assert calls.get("fake/reselect-fallback", 0) == 0
    return {
        "retry_attempts": calls["fake/retry-once"],
        "fallback_calls": calls["fake/trusted-fallback"],
        "retry_reselected": ["a_fail", "b_success"],
    }


def run_stream_reliability(
    relay_bin: Path, root: Path, manifest: Path, provider_url: str
) -> dict[str, object]:
    retry_config = single_target_config(
        manifest,
        provider_url,
        root / "stream-retry" / "atof",
        "fake/retry-stream-once",
        "fake/retry-stream-fallback",
        max_retries=1,
    )
    with RelayScenario(relay_bin, root, provider_url, "stream-retry", retry_config) as relay:
        body = dict(CASES[0][2])
        body["stream"] = True
        status, raw = request(relay.url, CASES[0][1], body)
        events = stream_events(raw)
        assert status == 200
        assert len(events) == 2
        assert raw.decode().count("data: [DONE]") == 1
        assert stream_text("openai_chat", events) == "chat from fake/retry-stream-once"
    retry_marks = relay.marks("switchyard.routing.retry")
    assert len(retry_marks) == 1
    assert retry_marks[0]["data"] == {
        "attempt": 1,
        "retryable": True,
        "failure_kind": "http",
        "http_status": 503,
    }
    assert not relay.marks("switchyard.routing.fallback", expected=0)
    assert len(relay.marks("switchyard.routing.requested", expected=2)) == 2
    assert len(relay.marks("switchyard.routing.decision", expected=2)) == 2

    fallback_config = single_target_config(
        manifest,
        provider_url,
        root / "stream-fallback" / "atof",
        "fake/always-fail-stream",
        "fake/trusted-stream-fallback",
        max_retries=1,
    )
    with RelayScenario(relay_bin, root, provider_url, "stream-fallback", fallback_config) as relay:
        body = dict(CASES[0][2])
        body["stream"] = True
        status, raw = request(relay.url, CASES[0][1], body)
        events = stream_events(raw)
        assert status == 200
        assert len(events) == 2
        assert raw.decode().count("data: [DONE]") == 1
        assert stream_text("openai_chat", events) == "chat from fake/trusted-stream-fallback"
    error_marks = relay.marks("switchyard.routing.error")
    assert len(error_marks) == 1
    assert error_marks[0]["data"] == {
        "attempt": 1,
        "retryable": False,
        "failure_kind": "http",
        "http_status": 400,
    }
    assert len(relay.marks("switchyard.routing.fallback")) == 1
    assert len(relay.marks("switchyard.routing.requested")) == 1
    assert len(relay.marks("switchyard.routing.decision")) == 1

    late_config = single_target_config(
        manifest,
        provider_url,
        root / "stream-late-failure" / "atof",
        "fake/late-stream-failure",
        "fake/late-stream-fallback",
        max_retries=1,
    )
    with RelayScenario(relay_bin, root, provider_url, "stream-late-failure", late_config) as relay:
        body = dict(CASES[0][2])
        body["stream"] = True
        status, raw, saw_stream_error = request_until_stream_error(relay.url, CASES[0][1], body)
        late_events = stream_events(raw)
        assert status == 200
        assert saw_stream_error
        assert len(late_events) == 1
        assert stream_text("openai_chat", late_events) == "committed before failure"
        assert "data: [DONE]" not in raw.decode()
    late_error_marks = relay.marks("switchyard.routing.error")
    assert len(late_error_marks) == 1
    assert late_error_marks[0]["data"] == {
        "attempt": 1,
        "retryable": True,
        "failure_kind": "non_http",
        "non_http_kind": "transport",
    }
    assert not relay.marks("switchyard.routing.retry", expected=0)
    assert not relay.marks("switchyard.routing.fallback", expected=0)
    assert len(relay.marks("switchyard.routing.requested")) == 1
    assert len(relay.marks("switchyard.routing.decision")) == 1

    empty_config = single_target_config(
        manifest,
        provider_url,
        root / "stream-empty" / "atof",
        "fake/empty-stream",
        "fake/empty-stream-fallback",
        max_retries=1,
    )
    with RelayScenario(relay_bin, root, provider_url, "stream-empty", empty_config) as relay:
        body = dict(CASES[0][2])
        body["stream"] = True
        status, raw = request(relay.url, CASES[0][1], body)
        events = stream_events(raw)
        assert status == 200
        assert stream_text("openai_chat", events) == "chat from fake/empty-stream-fallback"
    empty_error_marks = relay.marks("switchyard.routing.error")
    assert len(empty_error_marks) == 1
    assert empty_error_marks[0]["data"] == {
        "attempt": 1,
        "retryable": False,
        "failure_kind": "non_http",
        "non_http_kind": "invalid_response",
    }
    assert len(relay.marks("switchyard.routing.fallback")) == 1
    assert not relay.marks("switchyard.routing.retry", expected=0)

    stream_reselection = reselection_config(
        manifest,
        provider_url,
        root / "stream-reselection" / "atof",
        "fake/reselect-stream-fail",
        "fake/reselect-stream-success",
        "fake/reselect-stream-fallback",
    )
    with RelayScenario(
        relay_bin, root, provider_url, "stream-reselection", stream_reselection
    ) as relay:
        body = dict(CASES[0][2])
        body["stream"] = True
        status, raw = request(relay.url, CASES[0][1], body)
        events = stream_events(raw)
        assert status == 200
        assert stream_text("openai_chat", events) == "chat from fake/reselect-stream-success"
    decisions = relay.marks("switchyard.routing.decision", 2)
    assert [event["data"]["selected_target"] for event in decisions] == [  # type: ignore[index]
        "a_fail",
        "b_success",
    ]
    assert len(relay.marks("switchyard.routing.retry")) == 1
    assert not relay.marks("switchyard.routing.fallback", expected=0)

    calls = http_json(provider_url, "/calls")
    assert calls["fake/retry-stream-once"] == 2
    assert calls.get("fake/retry-stream-fallback", 0) == 0
    assert calls["fake/always-fail-stream"] == 1
    assert calls["fake/trusted-stream-fallback"] == 1
    assert calls["fake/late-stream-failure"] == 1
    assert calls.get("fake/late-stream-fallback", 0) == 0
    assert calls["fake/empty-stream"] == 1
    assert calls["fake/empty-stream-fallback"] == 1
    assert calls["fake/reselect-stream-fail"] == 1
    assert calls["fake/reselect-stream-success"] == 1
    assert calls.get("fake/reselect-stream-fallback", 0) == 0
    return {
        "retry_attempts": calls["fake/retry-stream-once"],
        "fallback_calls": calls["fake/trusted-stream-fallback"],
        "late_events_before_failure": len(late_events),
        "late_error_marks": len(late_error_marks),
        "empty_stream_fallback_calls": calls["fake/empty-stream-fallback"],
        "retry_reselected": ["a_fail", "b_success"],
    }


def run_unmanaged_passthrough(
    relay_bin: Path, root: Path, manifest: Path, provider_url: str
) -> dict[str, object]:
    config = plugin_config(
        manifest,
        provider_url,
        root / "unmanaged-passthrough" / "atof",
        '[plugins.dynamic.config.algorithm]\nkind = "random"\nseed = 5',
        """\
[plugins.dynamic.config.targets.responses]
model = "fake/unused-responses-target"
protocol = "openai_responses"
base_url = "{provider_url}/v1"
weight = 1
""",
        'openai_responses = "responses"',
    )
    with RelayScenario(relay_bin, root, provider_url, "unmanaged-passthrough", config) as relay:
        body = dict(CASES[0][2])
        body["model"] = "fake/unmanaged-passthrough"
        body["stream"] = False
        status, raw = request(relay.url, CASES[0][1], body)
        response = json.loads(raw)
        assert status == 200
        assert response["model"] == "fake/unmanaged-passthrough"

        body["stream"] = True
        status, raw = request(relay.url, CASES[0][1], body)
        events = stream_events(raw)
        assert status == 200
        assert len(events) == 2
        assert raw.decode().count("data: [DONE]") == 1
        assert stream_text("openai_chat", events) == "chat from fake/unmanaged-passthrough"

    assert not relay.marks("switchyard.routing.requested", expected=0)
    assert not relay.marks("switchyard.routing.decision", expected=0)
    assert not relay.marks("switchyard.routing.retry", expected=0)
    assert not relay.marks("switchyard.routing.error", expected=0)
    assert not relay.marks("switchyard.routing.fallback", expected=0)
    calls = http_json(provider_url, "/calls")
    assert calls["fake/unmanaged-passthrough"] == 2
    assert calls.get("fake/unused-responses-target", 0) == 0
    return {
        "buffered": True,
        "streaming": True,
        "provider_calls": calls["fake/unmanaged-passthrough"],
        "switchyard_marks": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--relay-bin",
        type=Path,
        default=os.environ.get("NEMO_RELAY_BIN"),
        required="NEMO_RELAY_BIN" not in os.environ,
    )
    parser.add_argument(
        "--plugin-library",
        type=Path,
        default=os.environ.get("SWITCHYARD_PLUGIN_LIBRARY"),
        required="SWITCHYARD_PLUGIN_LIBRARY" not in os.environ,
    )
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    relay_bin = args.relay_bin.resolve()
    plugin_library = args.plugin_library.resolve()
    if not relay_bin.is_file():
        parser.error(f"Relay binary does not exist: {relay_bin}")
    if not plugin_library.is_file():
        parser.error(f"plugin library does not exist: {plugin_library}")

    temporary = None
    if args.keep_temp:
        root = Path(tempfile.mkdtemp(prefix="switchyard-relay-plugin-e2e-"))
    else:
        temporary = tempfile.TemporaryDirectory(prefix="switchyard-relay-plugin-e2e-")
        root = Path(temporary.name)
    bundle = root / "bundle"
    subprocess.run(
        [
            sys.executable,
            str(CRATE_ROOT / "scripts" / "package_bundle.py"),
            "--library",
            str(plugin_library),
            "--output",
            str(bundle),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(relay_bin),
            "plugins",
            "validate",
            str(bundle / "relay-plugin.toml"),
        ],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )

    provider_port = free_port()
    provider_url = f"http://127.0.0.1:{provider_port}"
    provider_log: list[str] = []
    provider = subprocess.Popen(
        [
            sys.executable,
            "-u",
            str(HERE / "fake_provider.py"),
            "--port",
            str(provider_port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    threading.Thread(target=capture, args=(provider, provider_log), daemon=True).start()
    try:
        deadline = time.time() + 10
        while True:
            try:
                if http_json(provider_url, "/healthz")["ok"]:
                    break
            except (OSError, urllib.error.URLError):
                pass
            if provider.poll() is not None:
                raise RuntimeError("fake provider exited early:\n" + "\n".join(provider_log[-40:]))
            if time.time() > deadline:
                raise TimeoutError("fake provider did not become healthy")
            time.sleep(0.05)

        summary = {
            "same_protocol_preservation": run_same_protocol(
                relay_bin, root, bundle / "relay-plugin.toml", provider_url
            ),
            "random": run_random(relay_bin, root, bundle / "relay-plugin.toml", provider_url),
            "llm_classifier": run_classifier(
                relay_bin, root, bundle / "relay-plugin.toml", provider_url
            ),
            "stage_router": run_stage_router(
                relay_bin, root, bundle / "relay-plugin.toml", provider_url
            ),
            "reliability": run_retry_and_fallback(
                relay_bin, root, bundle / "relay-plugin.toml", provider_url
            ),
            "stream_reliability": run_stream_reliability(
                relay_bin, root, bundle / "relay-plugin.toml", provider_url
            ),
            "unmanaged_passthrough": run_unmanaged_passthrough(
                relay_bin, root, bundle / "relay-plugin.toml", provider_url
            ),
        }
        recorded = [
            path
            for path in root.rglob("*")
            if path.suffix in {".jsonl", ".log", ".toml"}
            and TARGET_AUTHORIZATION in path.read_text(encoding="utf-8", errors="replace")
        ]
        assert not recorded, f"target credential was recorded in {recorded}"
        summary["target_headers"] = {
            "source_credentials_replaced": True,
            "credential_recorded": False,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        provider.terminate()
        try:
            provider.wait(timeout=5)
        except subprocess.TimeoutExpired:
            provider.kill()
            provider.wait()
        if args.keep_temp:
            print(f"preserved E2E directory: {root}")
        elif temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    main()
