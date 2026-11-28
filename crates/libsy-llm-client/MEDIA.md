# Media settings

Configure `[targets.<name>.media]` to prepare images and videos for an endpoint.
The LLM client prepares a separate outgoing body for each judge, answer, or
fallback call. The original request stays available to the other targets.
Omit the media table to keep existing behavior.

## Configure judge and answer inputs

This example gives the judge one small video frame and the answer up to eight.
Still images sent to the judge are resized to fit 384 pixels. The judge rejects
requests with more than one image or video; choose the input before routing.

```toml
[targets.judge.media]
image_max_edge = 384
max_images = 1
video = "frames"
video_max_frames = 1
frame_max_edge = 384

[targets.answer.media]
video = "frames"
video_max_frames = 8
frame_max_edge = 768
```

For a text-only target, set `max_images = 0` and `video = "omit"`.
The route's `judge_max_images` setting affects existing image blocks only;
video processing belongs in the target's media settings.

| Setting | Default | Behavior |
|---|---|---|
| `image_max_edge` | unset | Resize still images to fit this edge (1–4096 pixels); preserve aspect ratio and never upscale. |
| `max_images` | unset | Maximum images and requested video frames per call (0–128). Unset allows all; zero omits them; a positive limit rejects excess input. |
| `video` | `passthrough` | Video handling mode; see below. |
| `video_max_frames` | 4 | Maximum samples per video (1–64). |
| `frame_max_edge` | 640 | Maximum frame edge (1–4096 pixels); preserve aspect ratio and never upscale. |
| `max_input_bytes` | 33554432 | Maximum downloaded or decoded inline bytes per locally processed source (1 byte–256 MiB). |
| `max_output_bytes` | 33554432 | Maximum combined prepared image bytes before base64 encoding (1 byte–256 MiB). |
| `timeout_ms` | 30000 | Maximum wait for preparation (1–300000 ms), including downloads and decoder queueing. An active decode may finish after the timeout. |

Image limits cover all selected messages and nested tool results. A positive
`max_images` reserves `video_max_frames` images per video before downloading it,
even if the clip later yields fewer frames. Excess input returns HTTP 400,
including on a fail-open judge. The limit never selects or reorders images.

## Video modes

| Mode | Outgoing content |
|---|---|
| `passthrough` | Keep the video block produced by translation. |
| `frames` | Replace each video with timestamped JPEG images. |
| `video_url` | Chat Completions `video_url` block. Requires an `openai_chat` target. |
| `file` | Gemini-on-Inference-Hub video `file` block. Requires an `openai_chat` target. |
| `omit` | Replace videos with text markers without fetching them. |

Native modes forward the existing URL or inline data without fetching or
transcoding it. They do not count toward `max_images`. The endpoint must support
the selected format. Responses `input_video` and Anthropic video source blocks
are intermediate formats; use `frames` for endpoints that accept images only.

Frame sampling uses uniformly spaced indices on the video track, including its
endpoints. One sample uses the first frame; a short clip returns at most its
available frames. Timestamps are presentation times, and longer audio does not
extend sampling. Uniform sampling can miss brief events.

## Sources and limits

- Use public HTTP(S) URLs or base64 data URIs. Local paths, private addresses,
  and provider-managed file IDs cannot be fetched. Downloads validate DNS and
  redirects and receive no inference credentials or environment proxy settings.
- Still images support PNG, JPEG, and WebP. Resizing applies EXIF orientation,
  preserves transparency, and retains provider fields such as image detail.
- Local processing accepts at most 64 sources per call and applies the configured
  byte limits. A timeout stops waiting; active decoding retains its worker slot
  until it finishes and can delay later requests.
- The server separately limits the JSON body to 32 MiB, including base64 overhead.
  Raise this with `--max-request-body-bytes`; see the
  [CLI reference](../../docs/cli_reference.md).

## Runtime requirements

Media processing uses the published `dynamo-multimodal` 0.2 release with the
`fetch` and `video-decode` features.

Building from source requires CMake and a C/C++ toolchain.
Video decoding requires 64-bit Linux. H.264, HEVC, VP9, and AV1 use NVDEC when
supported by the GPU and matching NVIDIA driver libraries. VP8 and VP9 also have
a CPU path through system libvpx, which is included in the server image.
Unsupported formats or missing decoders return an error. FFmpeg is not required.
