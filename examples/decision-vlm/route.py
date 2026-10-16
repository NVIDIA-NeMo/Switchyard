#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ask Switchyard to choose a VQA target using images or sampled video frames."""

import argparse
import base64
import io
import json
import time
import urllib.request
from pathlib import Path

import cv2
from PIL import Image


def image_part(image: Image.Image, max_edge: int) -> dict:
    image = image.convert("RGB")
    image.thumbnail((max_edge, max_edge))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": url}}


def video_parts(path: Path, frames: int, max_edge: int) -> list[dict]:
    capture = cv2.VideoCapture(str(path))
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or count <= 0:
            raise ValueError(f"Cannot read video: {path}")
        parts = []
        for i in range(min(frames, count)):
            index = round(i * (count - 1) / max(min(frames, count) - 1, 1))
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Cannot decode frame {index}: {path}")
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            parts.append(image_part(image, max_edge))
        return parts
    finally:
        capture.release()


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True, help="VQA question and answer options")
    parser.add_argument("--category", help="Known question category, without the answer")
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--video", type=Path)
    parser.add_argument("--frames", type=positive_int, default=4)
    parser.add_argument("--max-edge", type=positive_int, default=384)
    parser.add_argument("--url", default="http://127.0.0.1:18232/v1/decision")
    parser.add_argument("--route", default="switchyard/decision-vqa")
    args = parser.parse_args()

    images = []
    for path in args.image:
        with Image.open(path) as image:
            images.append(image_part(image, args.max_edge))
    question = args.question
    if args.category:
        question += f"\nQuestion category: {args.category}"
    if args.video:
        frames = video_parts(args.video, args.frames, args.max_edge)
        question += f"\nThe last {len(frames)} images are video frames in time order."
        images.extend(frames)
    body = {
        "input_format": "openai_chat",
        "request": {
            "model": args.route,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": question}, *images],
                }
            ],
        },
    }
    started = time.monotonic()
    request = urllib.request.Request(
        args.url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=190) as response:
        decision = json.load(response)
    print(
        json.dumps(
            {
                "question": args.question,
                "category": args.category,
                "images": len(images),
                "seconds": round(time.monotonic() - started, 3),
                "decision": decision,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
