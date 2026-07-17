# Sub-Agent Routing Infrastructure

**Status:** Proposed

## Why this matters

Frontier models are becoming orchestrators, not only answer generators. Anthropic describes
[Claude Fable 5](https://www.anthropic.com/claude/fable) as capable of planning across stages,
delegating to sub-agents, and checking their work over long-running tasks. Its
[system card](https://www-cdn.anthropic.com/2f9323abbcc4abe219577539efe19a623c9ca2bd/Claude%20Fable%205%20%26%20Claude%20Mythos%205%20System%20Card.pdf)
also evaluates blocking sub-agents, fixed-agent teams, and asynchronous sub-agents. In those
evaluations, the multi-agent variants outperformed the single-agent variants, and asynchronous
workers improved latency at the cost of using more tokens.

This changes the role of the proxy. Once the harness declares that a request belongs to an isolated
worker, the proxy should preserve that boundary instead of treating it as an ordinary turn and
allowing the normal pipeline to reinterpret it. The orchestrator and worker may intentionally use
different models, even when the worker model is not cheaper.

The routing opportunity exists because a sub-agent task crosses a real inference boundary:

- it receives a bounded delegation prompt and its own context;
- it issues separate model requests with its own agent identity;
- it can run concurrently with the parent and other workers; and
- it returns a result or summary to the orchestrator rather than extending the parent's reasoning
  inline.

The tasks are isolated at the model-request boundary, even when they are not logically independent.
A workflow may still contain dependencies, such as research before implementation or review after
implementation. The harness owns that task graph; Switchyard only sees separately attributable
inference requests. That is enough to route each worker without understanding or reimplementing the
orchestration plan.

### Benefits for Switchyard

Switchyard is the proxy between the harness and the model provider. That position creates two
primary responsibilities when the request is marked as sub-agent work:

1. **Preserve input-cache continuity.** Prompt/KV caches are normally scoped to a model and often
   to an endpoint or replica. If Switchyard ignores the sub-agent signal and lets every worker turn
   pass through an ordinary per-request router, one sub-agent loop may hop between models or
   targets. The growing worker prefix then has to be processed again instead of reusing the cache
   created by earlier turns. A configured `subagent_target` keeps worker requests on an intentional,
   stable target at the Switchyard layer.
2. **Honor the user's model expectation.** A user or harness may deliberately assign one model to
   orchestration and another to delegated execution, review, or exploration. Sending the marked
   worker request through the normal router can override that division of responsibility. The
   profile-level `subagent_target` makes the expected worker model an explicit deployment contract.

This infrastructure does not implement a new prompt cache. It preserves the conditions under which
the upstream cache can work: stable model/target selection and a cache-compatible wire format. If a
target itself spreads requests across replicas that do not share KV state, provider-side or
load-balancer affinity is still required.

The current repository already has the pieces needed to enforce those responsibilities:

- **Existing cache semantics.** Switchyard already preserves Anthropic `cache_control` on native
  Anthropic paths, injects cache breakpoints when an OpenAI-shaped client is routed to an Anthropic
  target in supported profiles, and records cached and cache-creation tokens per model/tier.
- **Stable target bypass.** For a profile that normally runs a classifier, random choice, or
  multi-stage router, a recognized sub-agent request can take the existing passthrough path directly
  instead of recomputing the worker's model on every turn.
- **Protocol-independent behavior.** Codex enters through OpenAI Responses and Claude Code through
  Anthropic Messages, but both endpoints already create provider-neutral `ChatRequest` values and
  retain normalized request headers for the profile.
- **Reuse of production behavior.** Existing target resolution, backend-format selection,
  translation, streaming, error handling, lifecycle, and usage accounting remain on the request
  path. Sub-agent routing selects an existing branch rather than creating a parallel serving stack.
- **Capacity isolation.** A `subagent_target` can point at a dedicated model or separately scaled
  endpoint, keeping bursts of parallel worker traffic from consuming the orchestrator's capacity.
- **Measurable rollout.** Existing profile routing metadata and shared statistics can attribute
  target selection, cached tokens, cache-creation tokens, latency, and errors to the sub-agent
  branch without labeling metrics with high-cardinality agent IDs.
- **Safe opt-in.** Because the behavior belongs to each profile and is absent by default, operators
  can enable it route by route. Profiles without `subagent_target` retain their current behavior.

Lower inference cost or routing latency may follow from choosing a different worker target, but
neither is required. The core value is preserving cache locality and honoring the model boundary
expressed by the user, harness, and profile configuration.

## Objective

Let a Switchyard profile send sub-agent work to a dedicated model while leaving its normal
routing behavior unchanged.

The design should:

- recognize sub-agent requests from Codex and Claude Code;
- preserve a stable, cache-compatible target for a sub-agent loop;
- honor the profile's explicit worker-model choice;
- reuse Switchyard targets, profiles, backends, translation, streaming, and statistics;
- work with any routing profile rather than being implemented separately by every router; and
- be inactive when the selected profile does not configure a sub-agent target.

## Task-type handling

Treat every recognized delegated user task equally. All of them use the profile's single
`subagent_target`; Switchyard does not classify prompts or select targets by task type. This keeps
the worker target stable for input-cache continuity and honors the user's orchestrator-versus-worker
model choice.

Codex exposes partial subtypes such as `review` and `collab_spawn`, while Claude Code does not expose
a portable semantic task type. Switchyard may retain an explicit subtype as metadata, but it does
not affect routing. Codex `review` and `collab_spawn` therefore route identically.

Codex `compact` and `memory_consolidation` maintain harness context rather than execute delegated
user work, so they remain on normal routing. Differentiated worker routing can be reconsidered only
if a harness supplies a stable explicit signal and production evidence shows that users need a
different model. The initial contract remains one binary sub-agent signal and one target.

## Capability contract

When Codex or Claude Code identifies a request as sub-agent work, and the selected Switchyard
route appoints a sub-agent model, Switchyard sends the request to that model. Requests without a
recognized signal or without an appointment continue through normal routing.

One possible configuration surface is an optional common profile field:

```yaml
profiles:
  coding:
    type: stage_router
    capable: anthropic/claude-opus-4.7
    efficient: moonshotai/kimi-k2.6
    subagent_target: moonshotai/kimi-k2.6
```

In this example, `subagent_target` refers to an existing entry in `targets`. The final field name
and schema may evolve with the profile system; the contract is that the appointment is explicit,
optional, and resolved through existing Switchyard targets.

At runtime, the selected profile behaves as follows:

```text
request
  -> select route from request model
  -> normalize harness agent metadata
  -> route appoints a sub-agent model and request is recognized sub-agent work?
       yes -> send to the appointed sub-agent model
       no  -> existing route behavior
  -> existing response translation and delivery
```

This is shared routing infrastructure. Individual routers do not need to parse Codex or Claude
Code headers. The implementation may use a direct target override, a common wrapper, a
router-native policy, affinity-aware selection, or another shared mechanism. The observable
behavior above is the design constraint.

## Relationship to agent-aware routing

The agent-aware routing work on the base branch provides two useful foundations:

- normalized agent context separates harness-specific header parsing from routing policy; and
- optional sub-agent affinity can keep requests from one child agent on a stable selection.

This proposal builds on those primitives but defines a different layer: an operator-facing
capability to appoint a model for sub-agent work across Codex and Claude Code. The random-router
affinity example is one possible implementation path, not the product contract.

Affinity is not required when the appointment resolves to one fixed target. It becomes useful when
the appointed path is itself a router or model pool whose concrete selection could otherwise vary
across turns in the same worker loop.

The remaining follow-up work is to expose the appointment through the profile system, normalize
Claude Code's child-agent signal alongside Codex, and preserve production streaming, translation,
and statistics behavior on the appointed path. Header normalization should remain shared request
metadata infrastructure so it is useful even when no affinity policy is enabled.

## Configuration behavior

The appointment is opt-in per route and should be available to every routing profile without
adding router-specific implementations. It resolves through the existing target system. An
unknown appointed target is a configuration error, while an absent appointment produces the same
runtime route as today.

The request body's `model` still selects the top-level route first. A request that names a target
directly continues to use that target's normal passthrough route; it does not inherit an unrelated
route's sub-agent appointment.

## Request detection

Detection is a small, deterministic function over normalized request metadata. Shared ingress
code may derive that metadata from case-insensitive, trimmed headers, but routing policies should
not parse harness-specific headers independently.

| Client | Signal | Sub-agent request |
| --- | --- | --- |
| Codex | `x-openai-subagent` | `collab_spawn` or `review` |
| Claude Code | `x-claude-code-agent-id` | Any non-empty value |
| Explicit Switchyard client | `x-switchyard-is-subagent` | `true` |

Codex values such as `compact` and `memory_consolidation` are not treated as sub-agent work in
the initial implementation. Unknown or malformed values fall through to normal profile routing.
New client values should be added deliberately with captured request fixtures and tests rather
than inferred from header presence alone.

Claude Code documents `x-claude-code-agent-id` as present only for requests from an in-process
sub-agent or teammate. The initial policy treats both as isolated worker traffic.
`x-claude-code-parent-agent-id` is retained for lineage and cost attribution but is not required to
make the binary routing decision.

Codex may also provide session, thread, turn, and parent identifiers. These are useful for
observability and optional affinity, but the recognized `x-openai-subagent` subtype remains the
binary routing signal. Agent and parent-agent identifiers must not become high-cardinality metric
labels.

These client signals should be verified against the upstream implementations when detection is
implemented:

- [Codex request headers](https://github.com/openai/codex/blob/main/codex-rs/codex-api/src/requests/headers.rs)
- [Claude Code gateway request headers](https://code.claude.com/docs/en/llm-gateway)

## Runtime behavior

| Route appointment | Sub-agent signal | Result |
| --- | --- | --- |
| None | Absent or present | Existing route behavior |
| Configured | Absent or unrecognized | Existing route behavior |
| Configured | Recognized | Appointed sub-agent model |

The decision runs after inbound request translation has produced the normal `ChatRequest`, so it
does not depend on the client wire format. It must work for the endpoints used by both harnesses:

- Codex through OpenAI Responses;
- Claude Code through Anthropic Messages; and
- OpenAI Chat Completions when used by another compatible client.

The appointed path must not buffer or rewrite the request or response. Streaming and non-streaming
requests continue through the existing target backend and translation engine.

If the configured sub-agent target fails, Switchyard returns the normal target error. It must not
silently send the request back through the main router, because that would make cost, capability,
and failure behavior unpredictable.

## Fast activation benchmark

[OpenThoughts-TBLite](https://huggingface.co/datasets/open-thoughts/OpenThoughts-TBLite)
provides 100 difficulty-calibrated, TB2-style tasks intended for faster iteration. It is a good
end-to-end smoke test for this infrastructure because it runs through Harbor with the real Codex or
Claude Code harness, but the benchmark alone does not guarantee that either harness will delegate.

For the activation run, use a Harbor prompt template that encourages native delegation without
prescribing a decomposition:

```jinja2
Consider using subagents for independent parts of this task when doing so would improve speed or
quality. Choose the decomposition and number of subagents based on the work, then coordinate their
results before completing the task.

{{ instruction }}
```

Run one task first and repeat the same command with `claude-code`:

```bash
harbor run \
  --dataset openthoughts-tblite \
  --n-tasks 1 \
  --n-concurrent 1 \
  --agent codex \
  --model <parent-model> \
  --ak prompt_template_path=<absolute-path-to-template>
```

The initial signal is not the benchmark score. It is whether the harness delegates and, when it
does, whether the parent request follows normal profile routing while marked child requests use
the appointed sub-agent model. Run the same task once without the prompt template as a negative
control. After this path is stable, expand to a fixed TBLite subset for repeatable comparisons and
retain full TB2 as the higher-difficulty validation.
