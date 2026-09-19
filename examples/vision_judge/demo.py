# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run four live image-routing checks and save an image-free, credential-free trace."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HUB = "https://inference-api.nvidia.com"
SAMPLE_SHA256 = "04a9ef879f493b109f51527e72b73350b67927308819e8584a0be1ab28b019bd"
MODELS = {
    "cosmos": "nvidia/nvidia/cosmos3-nano-reasoner",
    "astra": "openai/openai/gpt-6-astra",
}
TASKS = {
    "description": "Describe the main objects and scene in this image in two sentences.",
    "pointing": (
        "Point to the largest car in the image. Choose one answer and briefly explain: "
        "A: (370,722), B: (386,718), C: (335,721), D: (397,714)."
    ),
}


def image_hashes(value: Any) -> list[str]:
    """Collect image digests from either provider's wire format without retaining pixels."""
    if isinstance(value, list):
        return [digest for item in value for digest in image_hashes(item)]
    if not isinstance(value, dict):
        return []
    if value.get("type") in {"image_url", "input_image"}:
        url = value["image_url"]
        if isinstance(url, dict):
            url = url["url"]
        return [hashlib.sha256(base64.b64decode(url.split(",", 1)[1])).hexdigest()]
    return [digest for item in value.values() for digest in image_hashes(item)]


def response_text(body: dict[str, Any]) -> str:
    """Read final assistant text from Chat Completions or Responses."""
    if "choices" in body:
        return body["choices"][0]["message"].get("content") or ""
    return "".join(
        part.get("text", "")
        for item in body.get("output", [])
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    )


def recorder(calls: list[dict[str, Any]]) -> ThreadingHTTPServer:
    """Forward only the two inference paths and retain safe request/response evidence."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            if self.path not in {"/v1/chat/completions", "/v1/responses"}:
                self.send_error(404)
                return
            payload = self.rfile.read(int(self.headers["Content-Length"]))
            request_body = json.loads(payload)
            call = {
                "phase": "judge" if "response_format" in request_body else "answer",
                "model": request_body["model"],
                "path": self.path,
                "image_sha256": image_hashes(request_body),
            }
            started = time.monotonic()
            request = urllib.request.Request(
                HUB + self.path,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": self.headers["Authorization"],
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    status, body = response.status, response.read()
            except urllib.error.HTTPError as error:
                status, body = error.code, error.read()
            except (OSError, urllib.error.URLError):
                status, body = 502, b'{"error":{"message":"upstream connection failed"}}'
            call.update(status=status, seconds=round(time.monotonic() - started, 3))
            if status == 200:
                parsed = json.loads(body)
                call.update(text=response_text(parsed), usage=parsed.get("usage", {}))
            calls.append(call)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ThreadingHTTPServer(("127.0.0.1", 0), Handler)


def check_case(case: dict[str, Any], image_digest: str, judge_limit: int) -> None:
    """Reject fallback, missing output, or changed/dropped answer images."""
    judge, answer = case["calls"]
    verdict = json.loads(judge["text"])
    expected_judge = [image_digest] if judge_limit else []
    checks = (
        set(verdict) == {"target", "reason"},
        isinstance(verdict.get("reason"), str),
        judge["phase"] == "judge",
        judge["model"] == MODELS["cosmos"],
        judge["status"] == answer["status"] == case["status"] == 200,
        judge["image_sha256"] == expected_judge,
        answer["phase"] == "answer",
        answer["image_sha256"] == [image_digest],
        MODELS[verdict["target"]] == answer["model"] == case["selected_model"],
        bool(answer["text"]),
        answer["text"] == case["answer"],
    )
    if not all(checks):
        raise RuntimeError(f"Routing verification failed for {case['route']}/{case['task']}")
    case["verified"] = True


def main() -> None:
    """Launch Switchyard, exercise both judge modes, and verify actual upstream payloads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="VANTAGE pointing JPEG")
    parser.add_argument("--server", type=Path, default=Path("target/debug/switchyard-server"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("NVIDIA_API_KEY"):
        parser.error("Set NVIDIA_API_KEY before running live inference.")
    image = args.image.read_bytes()
    digest = hashlib.sha256(image).hexdigest()
    if digest != SAMPLE_SHA256:
        parser.error("Use the pinned VANTAGE pointing image documented in README.md.")
    image_url = "data:image/jpeg;base64," + base64.b64encode(image).decode()
    calls: list[dict[str, Any]] = []
    config_text = Path(__file__).with_name("routes.toml").read_text()
    report: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "image_sha256": digest,
        "cases": [],
    }
    proxy = recorder(calls)
    worker = threading.Thread(target=proxy.serve_forever, daemon=True)
    worker.start()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}/v1"
    process = None
    try:
        with tempfile.TemporaryDirectory(prefix="switchyard-vision-") as directory:
            config = Path(directory) / "routes.toml"
            config.write_text(config_text.replace(HUB, f"http://127.0.0.1:{proxy.server_port}"))
            with (Path(directory) / "server.log").open("w") as log:
                process = subprocess.Popen(
                    [
                        str(args.server.resolve()),
                        "--config",
                        str(config),
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(port),
                    ],
                    stdout=log,
                    stderr=log,
                )
                for _ in range(100):
                    if process.poll() is not None:
                        raise RuntimeError(
                            "Switchyard exited during startup; validate routes.toml."
                        )
                    try:
                        with urllib.request.urlopen(base_url + "/models", timeout=1):
                            break
                    except urllib.error.URLError:
                        time.sleep(0.1)
                else:
                    raise RuntimeError("Switchyard did not become ready.")
                for mode, limit in [("text", 0), ("image", 1)]:
                    for task, prompt in TASKS.items():
                        case: dict[str, Any] = {"route": f"vision/{mode}-judge", "task": task}
                        report["cases"].append(case)
                        calls.clear()
                        request = urllib.request.Request(
                            base_url + "/chat/completions",
                            data=json.dumps(
                                {
                                    "model": case["route"],
                                    "max_tokens": 2048,
                                    "stream": False,
                                    "messages": [
                                        {
                                            "role": "user",
                                            "content": [
                                                {"type": "text", "text": prompt},
                                                {
                                                    "type": "image_url",
                                                    "image_url": {"url": image_url},
                                                },
                                            ],
                                        }
                                    ],
                                }
                            ).encode(),
                            headers={"Content-Type": "application/json"},
                        )
                        print(f"Running {case['route']}/{task} ...", flush=True)
                        try:
                            with urllib.request.urlopen(request, timeout=360) as response:
                                case.update(
                                    status=response.status,
                                    selected_model=response.headers.get(
                                        "x-model-router-selected-model"
                                    ),
                                    answer=response_text(json.load(response)),
                                )
                        finally:
                            case["calls"] = list(calls)
                        check_case(case, digest, limit)
                        print(f"Verified: {case['selected_model']}", flush=True)
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        proxy.shutdown()
        proxy.server_close()
        worker.join()
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
