# switchyard-server

`switchyard-server` exposes libsy algorithms through OpenAI Chat Completions, OpenAI Responses,
and Anthropic Messages endpoints. The current CLI builds uniform-random routes over one shared
upstream backend.

```bash
cargo run -p switchyard-server -- \
  --route switchyard/general=model/a,model/b \
  --route switchyard/coding=model/c,model/d \
  --base-url https://example.com/v1 \
  --api-key "$API_KEY"
```

Select a route by sending its `switchyard/*` ID as the request's `model`.
