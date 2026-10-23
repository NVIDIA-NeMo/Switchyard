# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from route import image_part, video_parts


def decode_part(part: dict) -> Image.Image:
    url = part["image_url"]["url"]
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))


class MediaTest(unittest.TestCase):
    def test_image_is_bounded_and_keeps_aspect_ratio(self):
        part = image_part(Image.new("RGBA", (800, 400), "red"), 384)
        with decode_part(part) as image:
            self.assertEqual(image.size, (384, 192))
            self.assertEqual(image.mode, "RGB")

    def test_video_samples_in_time_order_without_duplicate_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 2, (32, 32))
            self.assertTrue(writer.isOpened())
            for value in (30, 120, 220):
                writer.write(np.full((32, 32, 3), value, dtype=np.uint8))
            writer.release()
            parts = video_parts(path, 8, 32)
            self.assertEqual(len(parts), 3)
            values = []
            for part in parts:
                with decode_part(part) as image:
                    values.append(image.getpixel((16, 16))[0])
            self.assertLess(values[0], values[1])
            self.assertLess(values[1], values[2])
            endpoints = video_parts(path, 2, 32)
            with decode_part(endpoints[0]) as first, decode_part(endpoints[1]) as last:
                self.assertLess(first.getpixel((16, 16))[0], 50)
                self.assertGreater(last.getpixel((16, 16))[0], 200)


if __name__ == "__main__":
    unittest.main()
