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
confirmed_recovery_min_clean_tail = 1
```

`judge_target` defaults to `local_target`. `cloud_judge_target` is optional.
The local and cloud targets must resolve to distinct model IDs. When the local
backend reports its live context capacity, VGR republishes it through
`/v1/models`. `confirmed_recovery_min_clean_tail`, when set, lets an agentic
run that recovered from tool errors commit locally once that many trailing tool
results are clean and the cloud judge confirms the evidence.

## Serving modes

- `off` skips candidate generation and serves cloud. This is the default.
- `shadow` computes a decision but serves cloud.
- `evaluate` serves the policy decision for isolated evaluation.
- `active` serves the policy decision in production and requires
  `active_approval = "prospective-validation-and-canary-approved"`.

VGR buffers a local candidate before releasing it. A streamed candidate that is
licensed is replayed as a stream; a rejected candidate is not sent to the
client.

`x-model-router-selected-model` on each response names the tier that served it.
