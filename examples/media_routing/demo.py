# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11"
# dependencies = ["Pillow>=10"]
# ///
"""Verify live image/video routing, including the actual prepared judge and answer payloads."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
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

from PIL import Image

HUB = "https://inference-api.nvidia.com"
REVISION = "ad5297f645ba90830478a4c6a72a3a7ab077a2f7"
VIDEO_URL = (
    f"https://huggingface.co/datasets/nvidia/PhysicalAI-VANTAGE-Bench/resolve/{REVISION}/"
    "data/vqa/videos/drivesim___Collision___Real___Collision_4.mp4"
)
IMAGE_SHA = "04a9ef879f493b109f51527e72b73350b67927308819e8584a0be1ab28b019bd"
VIDEO_SHA = "38ce99e16f3b99edf5ded024521824c6cdfbd504e13d74ceee1a31b3d5cc345e"
MODELS = {
    "judge": "nvidia/nvidia/cosmos3-nano-reasoner",
    "cosmos": "nvidia/nvidia/cosmos3-nano-reasoner",
    "astra": "openai/openai/gpt-6-astra",
    "qwen": "nvidia/qwen/qwen3.5-35b-a3b",
    "gemini": "gcp/google/gemini-2.5-flash",
}


def media_evidence(value: Any) -> list[dict[str, Any]]:
    """Record only media types, digests, dimensions and frame sample labels."""
    if isinstance(value, list):
        return [entry for item in value for entry in media_evidence(item)]
    if not isinstance(value, dict):
        return []
    kind = value.get("type")
    if kind in {"image_url", "input_image", "video_url", "file"}:
        if kind == "file":
            payload = value["file"]
            url = payload.get("file_data") or payload["file_id"]
        else:
            url = value["video_url" if kind == "video_url" else "image_url"]
            if isinstance(url, dict):
                url = url["url"]
        entry: dict[str, Any] = {"type": kind, "inline": url.startswith("data:")}
        if entry["inline"]:
            data = base64.b64decode(url.split(",", 1)[1])
            entry.update(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
            if kind in {"image_url", "input_image"}:
                with Image.open(io.BytesIO(data)) as image:
                    entry.update(width=image.width, height=image.height)
        else:
            entry["url_sha256"] = hashlib.sha256(url.encode()).hexdigest()
        return [entry]
    if kind in {"text", "input_text"} and value.get("text", "").startswith("[video sample "):
        return [{"sample": value["text"]}]
    return [entry for item in value.values() for entry in media_evidence(item)]


def response_text(body: dict[str, Any]) -> str:
    if "choices" in body:
        return body["choices"][0]["message"].get("content") or ""
    return "".join(
        part.get("text", "")
        for item in body.get("output", [])
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    )


def recorder(calls: list[dict[str, Any]]) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            if self.path not in {"/v1/chat/completions", "/v1/responses"}:
                self.send_error(404)
                return
            payload = self.rfile.read(int(self.headers["Content-Length"]))
            parsed = json.loads(payload)
            call = {
                "phase": "judge" if "response_format" in parsed else "answer",
                "model": parsed["model"],
                "path": self.path,
                "media": media_evidence(parsed),
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


def verify(case: dict[str, Any], expected: str, source_kind: str) -> None:
    routed = case["route"].endswith("-judge")
    if len(case["calls"]) != (2 if routed else 1):
        raise RuntimeError("Unexpected call count or fallback")
    answer = case["calls"][-1]
    checks = [
        answer["phase"] == "answer",
        answer["status"] == case["status"] == 200,
        bool(case["answer"]),
    ]
    if routed:
        judge = case["calls"][0]
        verdict = json.loads(judge["text"])
        case["intended_target"] = expected
        case["judge_target_agreement"] = verdict["target"] == expected
        expected = verdict["target"]
        judge_images = [item for item in judge["media"] if item.get("type") == "image_url"]
        checks.extend(
            [
                judge["phase"] == "judge",
                judge["status"] == 200,
                len(judge_images) == (0 if case["route"] == "media/text-judge" else 1),
                all(max(item["width"], item["height"]) <= 384 for item in judge_images),
                not any(item.get("type") in {"video_url", "file"} for item in judge["media"]),
            ]
        )
    checks.append(MODELS[expected] == answer["model"] == case["selected_model"])
    answer_images = [
        item for item in answer["media"] if item.get("type") in {"image_url", "input_image"}
    ]
    if source_kind == "image":
        checks.append([item["sha256"] for item in answer_images] == [IMAGE_SHA])
    elif expected == "astra":
        samples = [
            float(item["sample"].split()[2][:-2]) for item in answer["media"] if "sample" in item
        ]
        checks.extend(
            [
                len(answer_images) == 6,
                len(samples) == 6,
                samples == sorted(samples),
                samples[-1] > 6.9,
            ]
        )
    else:
        native = [item for item in answer["media"] if item.get("type") in {"video_url", "file"}]
        checks.append(
            len(native) == 1
            and native[0]["type"] == ("file" if expected == "gemini" else "video_url")
        )
        if native:
            checks.append(
                native[0].get("sha256") == VIDEO_SHA
                if source_kind == "video_inline"
                else native[0].get("url_sha256") == hashlib.sha256(VIDEO_URL.encode()).hexdigest()
            )
    case["verified"] = all(checks)
    if not case["verified"]:
        raise RuntimeError(f"Verification failed: {case['name']}; inspect saved report")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--server", type=Path, default=Path("target/debug/switchyard-server"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("NVIDIA_API_KEY"):
        parser.error("Set NVIDIA_API_KEY to run live inference")
    image, video = args.image.read_bytes(), args.video.read_bytes()
    if (
        hashlib.sha256(image).hexdigest() != IMAGE_SHA
        or hashlib.sha256(video).hexdigest() != VIDEO_SHA
    ):
        parser.error("Use the pinned VANTAGE samples documented in README.md")
    sources = {
        "image": {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image).decode()},
        },
        "video_inline": {
            "type": "video_url",
            "video_url": {"url": "data:video/mp4;base64," + base64.b64encode(video).decode()},
        },
        "video_url": {"type": "video_url", "video_url": {"url": VIDEO_URL}},
    }
    cases = [
        (
            "image-text-judge",
            "text-judge",
            "image",
            "astra",
            "Point to the largest car. Give approximate image coordinates.",
        ),
        (
            "image-vision-judge",
            "vision-judge",
            "image",
            "astra",
            "Point to the largest car. Give approximate image coordinates.",
        ),
        (
            "video-text-judge-native",
            "text-judge",
            "video_inline",
            "qwen",
            "Describe the broad scene in this video in two sentences.",
        ),
        (
            "video-vision-judge-frames",
            "vision-judge",
            "video_inline",
            "astra",
            "Describe the temporal order of the main events in this video.",
        ),
        (
            "video-url-counting",
            "vision-judge",
            "video_url",
            "gemini",
            "Count the visible cars near the end of this video. Explain briefly.",
        ),
        (
            "video-url-description",
            "vision-judge",
            "video_url",
            "qwen",
            "Describe the broad scene in this video in two sentences.",
        ),
    ]
    cases.extend(
        [
            (
                "gemini-url-file",
                "gemini",
                "video_url",
                "gemini",
                "Describe the main events in this video briefly.",
            ),
            (
                "gemini-inline-file",
                "gemini",
                "video_inline",
                "gemini",
                "Describe the main events in this video briefly.",
            ),
            (
                "cosmos-inline-native",
                "cosmos-native",
                "video_inline",
                "cosmos",
                "Describe the main events in this video briefly.",
            ),
        ]
    )
    config_text = Path(__file__).with_name("routes.toml").read_text()
    report: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "image_sha256": IMAGE_SHA,
        "video_sha256": VIDEO_SHA,
        "cases": [],
    }
    calls: list[dict[str, Any]] = []
    proxy = recorder(calls)
    worker = threading.Thread(target=proxy.serve_forever, daemon=True)
    worker.start()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}/v1"
    process = None
    try:
        with tempfile.TemporaryDirectory(prefix="switchyard-media-demo-") as directory:
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
                        raise RuntimeError("Switchyard exited during startup; validate routes.toml")
                    try:
                        with urllib.request.urlopen(base_url + "/models", timeout=1):
                            break
                    except (urllib.error.URLError, TimeoutError):
                        time.sleep(0.1)
                else:
                    raise RuntimeError("Switchyard did not become ready")
                for name, mode, source_kind, expected, prompt in cases:
                    case: dict[str, Any] = {"name": name, "route": f"media/{mode}"}
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
                                            sources[source_kind],
                                        ],
                                    }
                                ],
                            }
                        ).encode(),
                        headers={"Content-Type": "application/json"},
                    )
                    print(f"Running {name} ...", flush=True)
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
                    verify(case, expected, source_kind)
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
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
