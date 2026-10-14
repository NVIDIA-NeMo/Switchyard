# Generated media fixtures

These synthetic fixtures contain no dataset content. They are distributed under
the repository's Apache-2.0 license.

`colors.mp4`: two seconds of a 64×32 test pattern, four frames per second, H.264.

```bash
ffmpeg -f lavfi -i testsrc=size=64x32:rate=4:duration=2 -c:v libx264 -pix_fmt yuv420p colors.mp4
```

`oriented.jpg`: an 80×40 red image with EXIF orientation 6 (rotate 90° clockwise),
created with Pillow. The resize test expects a portrait output after orientation.

`vp8.webm` and `vp9.webm`: synthetic 256×256 color clips copied from
`ai-dynamo/frontend-crates` commit `4448b929bb2187645e83d1195dc9a85819dd5145`,
`multimodal/tests/fixtures/video_decode/`, Apache-2.0. Their generator is
`multimodal/tests/fixtures/video_decode/generate.py` in that revision.
