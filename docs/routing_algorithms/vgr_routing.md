# Verification-Gated Routing

Verification-Gated Routing (VGR) generates a candidate on a local tier. For a
final answer, it gathers bounded evidence about that exact candidate and serves
it only when the policy allows it. All other final-answer outcomes use the cloud
tier. Missing context, unavailable verifiers, malformed verdicts, and expired
decision deadlines fail closed to cloud for final answers.

Tool-calling turns continue on the local tier unless the turn judge provides
evidence to escalate. An unavailable or inconclusive turn judge leaves the tool
turn on the local tier. A local endpoint failure triggers cloud fallback.

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
The local and cloud targets must resolve to distinct model IDs. Set the route's
`context_window` to advertise a configured limit through `/v1/models`; this works
with any supported local service and takes precedence over discovery.

Context discovery is disabled by default. For llama.cpp, set
`context_discovery = "llama_cpp"` on the local target's `[llm_clients.<name>]`
entry and leave the route's `context_window` unset. Switchyard then reads the
service's `/props` response and advertises a positive context capacity when
available. Missing or invalid properties leave the capacity unknown; routing
continues normally. Discovery does not configure the service or load a model.

See [the local-server example](../../examples/vgr-local-servers.toml)
for an active llama.cpp configuration and commented Ollama/vLLM alternatives.
For Ollama and vLLM, set `context_window` to match the running service; automatic
discovery for those servers is not implemented.

The server caches the model catalog, including unavailable discovery results,
for one minute. Discovery probes run concurrently with a two-second total budget
per refresh. Completed discoveries are retained when that budget expires.

`confirmed_recovery_min_clean_tail`, when set, lets an agentic
run that recovered from tool errors commit locally once that many trailing tool
results are clean and the cloud judge confirms the evidence.

## Agentic handoff

These settings are off by default:

```toml
agentic_handoff = true
local_turn_budget_seconds = 600
```

`agentic_handoff` tells the cloud tier when it takes over tool-using work that
the local tier previously served. The notice can include the local tier's
unchecked final message. It is sent at most once per user turn, and only after
new local work has been served since the previous handoff. The handoff preserves
the conversation's existing messages and content. `local_turn_budget_seconds`
escalates a user turn once the local tier has spent that much wall-clock time on it.

The notice covers the current request and asks the cloud tier to preserve earlier
work, inspect its effects, and repeat or undo steps only when needed.

Escalations release when a new user turn is decided locally. Set
`latch_escalation = true` to retain the cloud tier across later user turns instead.
All configured VGR durations must be positive.

## Hermes session headers

Use a Hermes build that supports both provider options below. The session header
must stay the same throughout a conversation; each conversation needs its own ID.

```yaml
model:
  default: assistant
  provider: custom
  base_url: http://localhost:4000/v1
providers:
  switchyard:
    api: http://localhost:4000/v1
    session_affinity_header: x-switchyard-session-id
    initiator_header: true
```

`x-initiator: user` releases the turn hold for a person's new request or steer.
`x-initiator: agent` keeps the hold through tool calls, recovery nudges, and
automatic continuations. Retries retain the original call's tag. Without a
recognized tag, Switchyard uses the message shape to identify a user turn.
`x-session-id` is also accepted as a session-header alias.

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
