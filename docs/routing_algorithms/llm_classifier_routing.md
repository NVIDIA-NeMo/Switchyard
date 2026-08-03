# LLM Classifier Routing

LLM classifier routing asks a judge model whether the efficient model is likely
to complete a request, then routes the request to a weak or strong target.

## Configure a classifier route

This example uses the packaged classifier prompt as intended: it estimates
whether the weak target can complete the task, and keeps the first routing
decision for later requests in the same conversation.

```toml
schema_version = 1

[llm_clients.openrouter]
format = "openai_chat"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"

[targets.classifier]
id = "openai/gpt-4o-mini"
llm_client = "openrouter"

[targets.strong]
id = "openai/gpt-4o"
llm_client = "openrouter"

[targets.weak]
id = "z-ai/glm-5.2"
llm_client = "openrouter"

[routes.smart]
id = "smart"
type = "llm_classifier"
classifier_target = "classifier"
strong_target = "strong"
weak_target = "weak"
base_threshold = 0.5
session_affinity = true
message_hash_fallback = true
```

`message_hash_fallback` is best-effort: independent sessions with the same
first user message share an affinity key. Prefer an explicit
`x-switchyard-session-id` when repeated opening prompts are possible.

The target table names are local references. Their `id` values are the model
identifiers sent to the upstream provider. The route's `id`, `smart`, is the
model name clients send to Switchyard.

## How the decision works

The classifier target returns a structured verdict containing:

- `p_solve`: the estimated probability that the weak model completes the task.
- `confidence`: confidence in the assessment.
- `capability_boundary`: `supported`, `uncertain`, `unsupported`, or `unmatched`.

For a usable verdict, Switchyard routes to `weak_target` when `p_solve` is
greater than or equal to the applicable threshold. Otherwise it routes to
`strong_target`:

- `supported` uses `base_threshold`.
- `uncertain`, `unsupported`, and `unmatched` use
  `capability_elevated_floor` when configured, or `base_threshold` otherwise.

An abstention, an invalid or unparseable verdict, a judge failure, or confidence
below `min_confidence` routes to `strong_target`. Raising either threshold sends
more traffic to the strong model.

## Tuning options

| Key | Default | Meaning |
|---|---|---|
| `base_threshold` | required | Lowest `p_solve` that routes a supported task to `weak_target`. Must be between `0` and `1`. |
| `min_confidence` | `0.0` | Lowest judge confidence that permits weak routing. Must be between `0` and `1`. |
| `capability_elevated_floor` | unset | Higher threshold for uncertain, unsupported, and unmatched tasks. It must be between `0` and `1` and greater than `base_threshold`. |
| `recent_turn_window` | unset | When unset, the judge sees only the newest user message. When set to `N`, it sees client system/developer instructions, the opening user task, and the last `N` conversation messages after that task. `0` keeps only the instructions and opening task. |
| `session_affinity` | `false` | Retains the first selected target for a session and reuses it on later requests. |
| `message_hash_fallback` | `false` | When session metadata is absent, keys affinity from the first user-message text. Requires `session_affinity = true`. |
| `prompt` | packaged capability prompt | Replaces the classifier's system prompt. The packaged verdict schema and routing policy remain active. |
| `max_output_tokens` | `4096` | Maximum completion tokens available to the classifier verdict. Must be at least `1`. |

### Override the classifier prompt

Set `prompt` on the route when the packaged capability rubric does not describe
your weak model. A `{{RESPONSE_SCHEMA}}` placeholder is optional. When present,
Switchyard replaces it with the packaged capability-verdict schema.

```toml
[routes.smart]
id = "smart"
type = "llm_classifier"
classifier_target = "classifier"
strong_target = "strong"
weak_target = "weak"
base_threshold = 0.5
prompt = """
Estimate whether the weak target can complete the request.
Return JSON matching this schema:
{{RESPONSE_SCHEMA}}
"""
```

The override changes the instructions only. The judge must still return the
packaged fields such as `p_solve`, `confidence`, and `capability_boundary`.

### Classifier calibration

The packaged prompt is calibrated for one specific weak model, assumes it
receives the initial task, and describes a sticky routing decision. It is not a
model-neutral weak-target evaluator.

Without affinity, the runtime judges every request; by default, it sends the
newest user message rather than the initial task. The example enables affinity
and its message-hash fallback so the first request is judged and later requests
with the same opening task reuse that decision. If you use a different weak
model or per-request routing, tune the thresholds before treating classifier
decisions as production policy.

## Session affinity

With `session_affinity = true`, the first selected target is retained for the
request's session identity. This includes `strong_target` when it was selected
as the fallback for an unavailable or unusable judge verdict. There is no
warmup period. Later requests with the same identity reuse the target before
classification, so the judge call is skipped.

Affinity is process-local. Clients can send `x-switchyard-session-id`, or enable
`message_hash_fallback` to key requests without session metadata from the first
user-message text.

## Run the route

After [building the Rust server](../getting_started.md#build-the-server), export
the provider credential, validate the configuration, and start the release
binary:

```bash
export OPENROUTER_API_KEY="your-openrouter-key"  # pragma: allowlist secret
./target/release/switchyard-server --config routes.toml --dry-run
./target/release/switchyard-server --config routes.toml \
  --host 127.0.0.1 --port 4000
```

Send a request using the route ID:

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"smart","messages":[{"role":"user","content":"Explain why the sky appears blue."}]}'
```

Treat the selected target as model-dependent output, not a fixed test result.
