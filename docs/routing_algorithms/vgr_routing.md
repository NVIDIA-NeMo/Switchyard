# Verification-Gated Routing

Verification-Gated Routing (VGR) generates a candidate on an efficient tier. For a
final answer, it gathers bounded evidence about that exact candidate and serves
it only when the policy allows it. All other final-answer outcomes use the capable
tier. Missing context, unavailable verifiers, malformed verdicts, and expired
decision deadlines fail closed to the capable tier for final answers.

Tool-calling turns continue on the efficient tier unless the turn judge provides
evidence to escalate. An unavailable or inconclusive turn judge leaves the tool
turn on the efficient tier. An efficient endpoint failure triggers capable fallback.

## Configure a route

Declare efficient, capable, and optional verifier targets using the normal deployment
schema, then reference their target names:

```toml
[routes.assistant]
id = "assistant"
type = "vgr"
efficient_target = "efficient"
capable_target = "capable"
judge_target = "efficient"
capable_judge_target = "capable"
mode = "shadow"
deadline_seconds = 30
task_typing = true
breaker_threshold = 5
breaker_cooldown_seconds = 30
confirmed_recovery_min_clean_tail = 1
```

`judge_target` defaults to `efficient_target`. `capable_judge_target` is optional.
The efficient and capable targets must resolve to distinct model IDs. Set the route's
`context_window` to advertise a configured limit through `/v1/models`; this works
with any supported local service and takes precedence over discovery.

Context discovery is disabled by default. For llama.cpp, set
`context_discovery = "llama_cpp"` on the efficient target's `[llm_clients.<name>]`
entry and leave the route's `context_window` unset. Switchyard then reads the
service's `/props` response and advertises a positive context capacity when
available. Missing or invalid properties leave the capacity unknown; routing
continues normally. Discovery does not configure the service or load a model.

See [the local-server example](../../examples/vgr-local-servers.toml)
for an active llama.cpp configuration and commented Ollama/vLLM alternatives.
For Ollama and vLLM, set `context_window` to match the running service; automatic
discovery for those servers is not implemented.

The judge reads its verdicts from the first token's logprobs, so its scored calls
turn thinking off: they send `reasoning_effort = "none"` and
`chat_template_kwargs = {"enable_thinking": false}` on `openai_chat` targets.
Candidates and the judge's deliberation keep the target's own setting. If a Chat
backend explicitly rejects this template switch with HTTP 400 or 422, the SDK
retries once without it, within the same deadline. Other request fields and the
configured transient retry budget are preserved. The target's `omit_body_fields`
can omit the switch from the first call. Custom hosts must handle provider
compatibility themselves.

The server caches the model catalog, including unavailable discovery results,
for one minute. Discovery probes run concurrently with a two-second total budget
per refresh. Completed discoveries are retained when that budget expires.

`confirmed_recovery_min_clean_tail`, when set, lets an agentic
run that recovered from tool errors commit on the efficient tier once that many trailing tool
results are clean and the capable judge confirms the evidence.

## Agentic handoff

These settings are off by default:

```toml
agentic_handoff = true
efficient_turn_budget_seconds = 600
```

`agentic_handoff` tells the capable tier when it takes over tool-using work that
the efficient tier previously served. The notice can include the efficient tier's
unchecked final message. It is sent at most once per user turn, and only after
new efficient work has been served since the previous handoff. The handoff preserves
the conversation's existing messages and content. `efficient_turn_budget_seconds`
escalates a user turn once the efficient tier has spent that much wall-clock time on it.

Optional experimental compaction condenses history from the efficient tier into a cached digest
when handing off. Enable it on the VGR route:

```toml
[routes.assistant]
agentic_handoff = true
compact_handoff = true
```

`compact_handoff` defaults to `false` and requires `agentic_handoff = true`.
Omitting it or setting it to `false` preserves the full history. Later requests
reuse a digest only when their original history prefix matches; a new user turn
clears it. An `x-initiator: agent` message keeps the digest, while
`x-initiator: user` clears it.

The digest is lossy: user updates and tool output are clipped. Non-text tool
results become `[non-text content]`; non-text user updates, including images,
are omitted without a marker. Cache savings and task-success effects have not
been measured in a matched evaluation.

The notice covers the current request and asks the capable tier to preserve earlier
work, inspect its effects, and repeat or undo steps only when needed.

Escalations release when a new user turn is decided on the efficient tier. Set
`latch_escalation = true` to retain the capable tier across later user turns instead.
Shadow mode always releases the hold at the next user turn.
All configured VGR durations must be positive.

## Execution limits

`bounded_efficient` defaults to `true`. Set `bounded_efficient = false` on the
VGR route to disable its decision deadline and `efficient_turn_budget_seconds`.
Candidate generation, task typing, and verification then wait for completion.
Client and upstream HTTP timeouts still apply. Configure those timeouts in the
harness and Switchyard client for long-running decisions. Configured durations
must remain positive even when these VGR limits are disabled.

## Hermes auxiliary routing

The [local-server example](../../examples/vgr-local-servers.toml) exposes
`switchyard/only-efficient` through the existing passthrough router. Hermes can
use it for titles and approval checks, independently of the conversation's VGR
route. Configure Hermes against the same Switchyard server:

```yaml
auxiliary:
  title_generation:
    provider: custom
    model: switchyard/only-efficient
    base_url: http://localhost:4000/v1
  approval:
    provider: custom
    model: switchyard/only-efficient
    base_url: http://localhost:4000/v1
    extra_body:
      reasoning_effort: none
      chat_template_kwargs:
        enable_thinking: false
```

Use the approval thinking controls only with endpoints that support them.
Adjust auxiliary timeouts for endpoint queueing and allow concurrent requests
on the efficient server when titles must run alongside the conversation.

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

## Served-answer reporting

VGR opts into served-answer reporting through `Algorithm::tracks_answers`.
The SDK summarizes answers only for opted-in algorithms with a session ID;
other routers keep their responses and streams unchanged. Custom algorithms and
classifiers default to no tracking. Stateful `FallThrough` routes opt in when a
classifier requests reports.

Custom libsy hosts can check `tracks_answers()` and report
`AnswerSummary::of(&response)` through `Algorithm::observe_answer` for aggregate
responses. For streams, report `final_reply = false` when serving begins and
report the final summary before forwarding the terminal chunk. Use the request's
session ID; requests without one retain the existing behavior.

Without a recognized `x-initiator` header, reported incomplete answers and
immediate recovery nudges retain the turn hold and budget. A reply to a complete
final answer releases them after a two-second quiet period. Explicit user/agent
headers take precedence. Hosts that omit reporting retain message-shape detection.

## Verification and capable-tier outages

Policy version `1.2` allows the default verification branch to accept a clean
efficient attempt when the capable judge approves both the evidence and the
answer. A negative or unavailable evidence verdict skips the answer check.

Typed factual answers are verified on the answer itself, including
later user turns without tool work. A readout of at least 0.9 can serve the
efficient answer; lower scores require capable-tier verification. Scored verdicts
require yes and no together to hold at least half the first token's probability.

Coding without tool history can serve the efficient attempt with a clean readout of at least
0.6, or a readout of at least 0.2 plus capable-judge affirmation. Other final-answer
branches can use a separate scored verdict after efficient deliberation.

In `active` mode, an unavailable capable tier can fall back to an unverified efficient
answer. That call receives the original request without the capable handoff note.
Fallback is disabled by the kill switch, an open efficient circuit breaker, or an
image request the efficient tier cannot serve.

## Serving modes

- `off` skips candidate generation and serves the capable tier. This is the default.
- `shadow` computes a decision but serves the capable tier. It continues evaluating
  subsequent steps until VGR escalates, then holds the capable tier for the rest
  of the user turn. A new user turn resumes evaluation, including when
  `latch_escalation = true`.
- `evaluate` serves the policy decision for isolated evaluation.
- `active` serves the policy decision in production and requires
  `active_approval = "prospective-validation-and-canary-approved"`.

VGR buffers an efficient candidate before releasing it. A streamed candidate that is
licensed is replayed as a stream; a rejected candidate is not sent to the
client.

`x-model-router-selected-model` on each response names the tier that served it.
