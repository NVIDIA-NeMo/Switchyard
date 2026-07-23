# LLM Classifier Routing

LLM classifier routing asks a classifier model to evaluate each request, then
sends the request to a `weak` or `strong` backend. Use it when routing should
depend on request content, tool use, context needs, or risk level instead of a
fixed traffic split.

The classifier runs before the selected backend. Low-confidence and abstained
results use the configured default tier. Classifier errors do the same when
`classifier_fail_open` is enabled, which is the default. The built-in two-tier
policies default to `strong`.

## Choose a policy

Set `profile` for the traffic you expect:

| `profile` | Use for | Default tier mapping |
|---|---|---|
| `general` | Mixed chat or API traffic | `simple` uses `weak`; all higher tiers use `strong`. |
| `coding_agent` | Claude Code, Codex, Cursor-style agents | `simple` and `medium` use `weak`; `complex` and `reasoning` use `strong`. Tool-planning turns can escalate. |
| `openclaw` | OpenClaw personal-assistant traffic | `simple` and `medium` use `weak`; `complex` and `reasoning` use `strong`. Tool orchestration and high-risk external actions can escalate. |

For coding-agent traffic, start with `profile: coding_agent`.

## Configure a classifier route

Define the strong, weak, and classifier models in a `deterministic` route:

```yaml
defaults:
  api_key: ${OPENROUTER_API_KEY}
  base_url: https://openrouter.ai/api/v1
  format: openai

routes:
  smart:
    type: deterministic
    profile: coding_agent
    classifier:
      model: nvidia/nemotron-3-nano-30b-a3b
      min_confidence: 0.6
      fail_open: true
      recent_turn_window: 4
    strong:
      model: openai/gpt-4o
    weak:
      model: openai/gpt-4o-mini
    fallback_target_on_evict: strong
```

Start the server with:

```bash
switchyard --routing-profiles routes.yaml -- serve --port 4000
```

The route ID (`smart`) is the model ID clients select for classifier-based
routing.

Try the profile with representative requests:

```bash
# Coding task: expected to use the strong tier.
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer dummy" -H "Content-Type: application/json" \
  -d '{"model":"smart","messages":[{"role":"user","content":"Plan and implement a multi-file API change."}],"max_tokens":200}'

# Simple question: expected to use the weak tier.
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer dummy" -H "Content-Type: application/json" \
  -d '{"model":"smart","messages":[{"role":"user","content":"What is 2+2? Reply with just the number."}],"max_tokens":50}'
```

Treat these as smoke checks, not fixed test vectors: the classifier model and
prompt determine the verdict.

## Useful options

| Configuration path | Use it when |
|---|---|
| `classifier.min_confidence` | Low-confidence results should use the default tier instead of the classifier policy. |
| `classifier.fail_open` | Classifier errors should use the default tier rather than fail the client request. |
| `classifier.recent_turn_window` | The classifier needs more or less recent conversation and tool context. |
| `alignment_min_confidence` | A classifier recommendation should only raise the policy tier above this confidence. |
| `default_tier` | Abstain, low-confidence, and fail-open decisions should use a tier other than the default `strong`. |
| `tier_mapping` | The four classifier policy tiers need a custom mapping to `weak` or `strong`. |

For a self-hosted strong, weak, or classifier target, configure it like any
other OpenAI-compatible endpoint. See
[Self-hosted targets](overview.md#self-hosted-targets).

## Session affinity

LLM classifier routing supports optional session affinity through
`DeterministicRoutingConfig`. Set `session_affinity: true` to share one affinity
store between the classifier and tier selector. After any configured
`affinity_warmup_turns`, the first confident verdict pins the tier. Later turns
reuse that tier before classification, so they skip the classifier call;
abstain, low-confidence, missing-signal, and fail-open decisions do not pin.

Configure these fields on the `type: deterministic` entry in the `routes:`
bundle. See [Session Affinity](sticky_routing.md) for YAML and
[How session affinity composes](overview.md#how-session-affinity-composes) for
the interaction with routing decisions.

If the per-request classifier cost is too high, use
[Stage-Router Routing](stage_router_routing.md), which can route many turns from tool and
agent-progress signals without an extra classifier call.
