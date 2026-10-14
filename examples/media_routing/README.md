# Image and video routing

The [configuration](routes.toml) gives the Cosmos judge a 384px image or one
384px video frame. It routes the original request to Astra, Qwen, or Gemini.
Astra receives six frames, Qwen receives native video, and Gemini receives a
video `file` block. Answer still images stay unchanged.

## Run

Install libvpx for CPU VP8/VP9 decoding, or use a compatible NVIDIA GPU with
matching CUDA/NVDEC driver libraries for H.264, HEVC and AV1. Set `NVIDIA_API_KEY`
for Inference Hub. From the
repository root, start the server:

```bash
cargo run -p switchyard-server --locked -- \
  --config examples/media_routing/routes.toml --port 8080
```

In another terminal, replace `/path/to/clip.mp4` with a local H.264 video and send
the request. This calls the live judge and answer endpoints:

```bash
video=$(base64 -w0 /path/to/clip.mp4)
curl --fail-with-body http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' --data-binary @- <<EOF
{"model":"media/vision-judge","messages":[{"role":"user","content":[
  {"type":"text","text":"Describe what happens in the video."},
  {"type":"video_url","video_url":{"url":"data:video/mp4;base64,$video"}}
]}]}
EOF
```

For a still image, use an `image_url` block containing a public HTTPS URL or a
base64 data URI. To make the judge text-only, set its media `max_images = 0`
and `video = "omit"`.

See the [media reference](../../crates/libsy-llm-client/MEDIA.md) for formats,
limits, and embedded Rust use. Preparation runs in the LLM client, independently
for each target. Frame sampling can miss brief events; this example demonstrates
request handling and does not measure routing accuracy.
