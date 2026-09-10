# Verification-Gated Routing

Verification-Gated Routing (VGR) generates a candidate on a local tier, gathers
bounded evidence about that exact candidate, and serves it only when the policy
licenses a local commit. All other outcomes use the cloud tier.

Tool-calling turns continue on the local tier without being mistaken for a
terminal answer. Missing context, unavailable verifiers, malformed verdicts,
and expired decision deadlines fail closed to cloud.

## Configure a route

Declare local, cloud, and optional verifier targets using the normal deployment
schema, then reference their target names:

```toml
[routes.assistant]
id = "assistant"
type = "vgr"
local_target = "local"
cloud_target = "cloud"
judge_target = "local"
cloud_judge_target = "cloud"
mode = "shadow"
deadline_seconds = 30
task_typing = true
breaker_threshold = 5
breaker_cooldown_seconds = 30
```

`judge_target` defaults to `local_target`. `cloud_judge_target` is optional.
The local and cloud targets must resolve to distinct model IDs.

## Serving modes

- `off` skips candidate generation and serves cloud. This is the default.
- `shadow` computes a decision but serves cloud.
- `evaluate` serves the policy decision for isolated evaluation.
- `active` serves the policy decision in production and requires
  `active_approval = "prospective-validation-and-canary-approved"`.

VGR buffers a local candidate before releasing it. A streamed candidate that is
licensed is replayed as a stream; a rejected candidate is not sent to the
client.

## Verify route identity

Every completion response includes `x-switchyard-route-type: vgr` and
`x-model-router-selected-model`, which identifies the tier that served the
response. `/v1/models` also includes `"route_type": "vgr"` for the route.

```bash
curl -s http://127.0.0.1:8000/v1/models
curl -i http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"assistant","messages":[{"role":"user","content":"Hello"}]}'
```
