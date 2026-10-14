# Media preparation

`libsy-llm-client` prepares media in an outgoing provider JSON body after
translation and endpoint overrides, before retries. Processing lives in its
private `media` module; core `libsy` has no media-processing dependencies.
The caller's original routing request is unchanged.
Each target with media settings gets a reusable Dynamo fetcher at client setup,
with its own input size limit. All targets share the client's two decoding slots.

Media fetching, EXIF-aware JPEG decoding, RGB resizing, video decoding and frame
JPEG encoding use `dynamo-multimodal`, pinned to
[frontend-crates PR #378 at `3954d9b`](https://github.com/ai-dynamo/frontend-crates/tree/3954d9bcd481b4077bd4711528166eef9c4f977a/multimodal).
RGB downscaling uses Lanczos. Other still-image formats retain the bounded `image`
decoder, including transparency and higher bit depths. PNG output for transparency,
and provider formatting remain in the client.
Dynamo's optional Rayon pool is disabled; decoding uses the client's bounded workers.
Building from source requires CMake and a C/C++ toolchain for libjpeg-turbo.
This pin uses the unreleased `video-decode` feature. Registry publication requires
a Dynamo release containing that feature; registry version `0.1.1` does not provide it.

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
| `video_max_frames` | 4 | Maximum uniform samples per video; one sample uses the first frame |
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

Video decoding uses Dynamo exclusively. On 64-bit Linux, NVDEC supports H.264,
HEVC, VP9 and AV1 when the GPU and NVIDIA driver support the codec. VP8 and VP9
also decode on the CPU through system libvpx. The server image includes libvpx;
GPU decoding requires matching CUDA/NVDEC driver libraries in the runtime.
Unsupported codecs, formats, platforms or missing decoders return an error.

Switchyard supplies uniformly spaced frame indices and a Lanczos resize target to
Dynamo's `decode_frames`. Dynamo resizes each selected frame during decoding and
skips keyframe groups that contain no selected frames when the stream supports it.
Returned timestamps are real presentation times.
One sample selects the first frame; requests above the source frame count return
all available frames. Sampling uses the video timeline, so longer audio does not
extend it. Uniform sampling can miss brief events.

Dynamo's selected RGB frames are capped at 128 MiB after resizing; the full-size
frames are not accumulated. Each source frame's RGB scratch buffer is also
limited to 128 MiB. JPEG outputs also obey `max_output_bytes`.
The preparation deadline covers the request's wait;
a running native blocking decoder completes in the background after cancellation,
holding its shared worker permit until it finishes.

Locally processed media can be base64 data URIs or public HTTP(S) URLs. Dynamo's
`MediaFetcher` validates DNS and redirects and limits downloaded bytes. It receives
no inference credentials or environment proxy. Each target reuses its fetcher across requests. Inline limits count decoded bytes; Dynamo also bounds
the encoded payload before decoding. The preparation deadline covers fetching.
Local paths, private network URLs and provider-managed file IDs
are not fetched. Use inline data for local files. At most 64 sources are processed
per call; two decode jobs run at once per client. PNG/JPEG/WebP still images are
supported. All other provider fields, including image detail/cache hints, are retained.

## Borrowed code

Adapted from [ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo) revision
`c3deae7507717409d4d1ff0d6f7e575180bd03d7`, Apache-2.0:

- `lib/llm/src/preprocessor/media/decoders/image/backends/image_reader.rs`:
  bounded `ImageReader` setup. Tensor conversion is replaced with image resizing.

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

The private processor modifies only the outgoing provider body. The client
discards that body if preparation fails and preserves the original routing request.
