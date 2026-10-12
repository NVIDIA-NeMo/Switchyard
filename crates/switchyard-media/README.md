# switchyard-media

Prepares media in an outgoing provider JSON body. `libsy-llm-client` invokes this
crate after translation and endpoint overrides, before retries. Core `libsy` has
no dependency on this crate. The caller's original routing request is unchanged.

Supported content: Chat `image_url` / `video_url`, Responses `input_image` /
`input_video`, Anthropic image/video source blocks, and Gemini-on-Hub video
`file` blocks. Video source blocks in Responses/Anthropic are intermediate
representations: configure `video = "frames"` for endpoints that only accept images.
Only message content, nested tool results and Responses function outputs are visited.

```toml
[targets.judge.media]
image_max_edge = 384
video = "frames"
video_max_frames = 1
frame_max_edge = 384
max_images = 1

[targets.answer.media]
video = "frames"
video_max_frames = 8
frame_max_edge = 768
```

An absent media table preserves existing behavior. Fields have these defaults:

| Field | Default | Meaning |
|---|---|---|
| `image_max_edge` | unset | Downscale still images to fit this edge; JPEG output (PNG for alpha), no upscale |
| `max_images` | unset | Reject calls exceeding N images and extracted frames; zero omits them; unset is unlimited |
| `video` | `passthrough` | `frames`, `video_url`, `file`, `omit`, or unchanged |
| `video_max_frames` | 4 | Uniform frames per video; one frame uses midpoint |
| `frame_max_edge` | 640 | Extracted JPEG frame maximum edge; no upscale |
| `max_input_bytes` | 33554432 | Limit per source fetched/decoded for local processing |
| `max_output_bytes` | 33554432 | Combined prepared image bytes before base64 encoding |
| `timeout_ms` | 30000 | Total preparation deadline, including queueing and downloads |

`video_url` and `file` require a Chat endpoint. `file` emits the video format used
by Gemini on Inference Hub; it is not a generic file upload API. Native modes
forward existing URLs/inline data without downloading, transcoding, or resizing.
Set `video = "omit"` and `max_images = 0` for a text-only judge.

Positive `max_images` limits are checked before downloads or decoding, including
images in nested tool results and every frame requested by `video_max_frames`.
Excess input returns HTTP 400 through the server, including on a fail-open judge.
The limit never selects or reorders images. Choose images before routing and use
`video_max_frames` to choose the number of frames per video.

`judge_max_images` uses the same unlimited/zero/reject contract on the judge's
selected input images before target preparation. A target's `media.max_images`
also counts frames that its video settings will extract. Native videos are not
counted as images; use `video = "omit"` to remove them. Other targets still prepare
their own copy of the original request.

Video frame extraction requires `ffmpeg` and `ffprobe` on PATH, with MP4/MOV or
Matroska/WebM demuxers and the input codec enabled. These are runtime dependencies;
image-only and native-video calls need neither executable. A missing tool or failed
decode returns an error instead of silently dropping media. Samples include text
labels with requested seek times, not exact decoded frame timestamps. Uniform
sampling can miss brief events; benchmark temporal accuracy must be evaluated separately.

Locally processed media can be base64 data URIs or public HTTPS URLs. Downloads
use no inference credentials, no environment proxy, validated DNS/redirect targets,
and bounded bodies. Local paths, private network URLs and provider-managed file IDs
are not fetched. Use inline data for local files. At most 64 sources are processed
per call; two decode jobs run at once per client. PNG/JPEG/WebP still images are
supported. All other provider fields, including image detail/cache hints, are retained.

## Borrowed code

Adapted from [ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo) revision
`c3deae7507717409d4d1ff0d6f7e575180bd03d7`, Apache-2.0:

- `lib/llm/src/preprocessor/media/loader.rs`: blocked IP ranges and DNS/redirect
  validation approach, reduced to an unauthenticated public-media fetcher.
- `lib/llm/src/preprocessor/media/decoders/image/backends/image_reader.rs`:
  bounded `ImageReader` setup. Tensor conversion is replaced with image resizing.
- `lib/llm/src/preprocessor/media/decoders/video.rs`: `get_target_times` sampling
  calculation. FFmpeg subprocesses replace Dynamo's linked video/tensor stack.

Original NVIDIA copyright and Apache-2.0 headers are retained. No SGLang or
private GitLab code is copied; Dynamo supplied the small reusable pieces needed.

## Embedded Rust use

Use the same settings without TOML through the LLM client's model configuration:

```rust
use switchyard_llm_client::{Backend, MediaConfig, ModelConfig, VideoMode};

fn judge_endpoint(backend: Backend) -> ModelConfig {
    ModelConfig::new("judge-model", backend, None).with_media(MediaConfig {
        image_max_edge: Some(384),
        video: VideoMode::Frames,
        video_max_frames: 1,
        frame_max_edge: 384,
        max_images: Some(1),
        ..MediaConfig::default()
    })
}
```

`MediaProcessor` can also prepare an owned provider JSON body directly. Discard the
body if preparation fails; it may be partially modified. The LLM client follows
this rule and never modifies the routing driver's original request.
