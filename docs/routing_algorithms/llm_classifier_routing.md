# LLM Classifier Routing

LLM classifier routing supports capability classification, trajectory escalation,
and custom schema-driven routing across two or more targets.

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
mode = "capability"
classifier_target = "classifier"
strong_target = "strong"
weak_target = "weak"
base_threshold = 0.5
threshold_step = 0.1
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
- `capability_boundary`: `supported`, `uncertain`, `unsupported`, or `unmatched`.
- `primary_rule`: the capability-card rule that determines the boundary.
- `crux`: the hardest material requirement for whole-task success.

For a usable verdict, Switchyard routes to `weak_target` when `p_solve` is
greater than or equal to the applicable threshold. Otherwise it routes to
`strong_target`:

- `supported` uses `base_threshold`.
- `uncertain` and `unmatched` use `base_threshold + threshold_step`.
- `unsupported` uses `base_threshold + 2 * threshold_step`.

An invalid, inconsistent, or unparseable verdict, or a judge failure, routes to
`strong_target`. Raising either knob sends more traffic to the strong model.

## Judge model compatibility

The judge must return complete, schema-valid JSON in normal assistant `content`.
Switchyard does not parse provider-specific reasoning fields such as
`reasoning_content`. If `content` is empty or unparseable, the route falls back
to `strong_target` even when the judge request returned HTTP 200. With session
affinity, that fallback can be reused without another judge call.

When a vLLM-compatible provider supports `enable_thinking`, configure it on the
judge target through `extra_body`:

```toml
[targets.classifier]
extra_body = { chat_template_kwargs = { enable_thinking = false } }
```

`enable_thinking` is a provider-specific vLLM option, not a general requirement
for reasoning models. Other model/provider pairs may work with reasoning enabled
or use a different control. Verify the judge response shape before deployment.
If reasoning remains enabled, set `max_output_tokens` high enough for both the
reasoning and final JSON. A truncated verdict has the same fail-open result. See
the [target-level `extra_body` reference](../../crates/switchyard-server/CONFIGURATION.md#add-an-llm-client-and-target)
for the server merge behavior.

## Tuning options

| Key | Default | Meaning |
|---|---|---|
| `base_threshold` | required | Lowest `p_solve` that routes a supported task to `weak_target`. Must be between `0` and `1`. |
| `threshold_step` | `0.0` | Amount added for each boundary step. Must be finite and non-negative, and `base_threshold + 2 * threshold_step` must not exceed `1`. |
| `recent_turn_window` | unset | When unset, the judge sees the opening user task and the latest user message when they differ. When set to `N`, it sees the opening user task and the last `N` conversation messages after that task. `0` keeps only the opening task. Client system and developer instructions are not shown to the judge. |
| `session_affinity` | `false` | Retains the first selected target for a session and reuses it on later requests. |
| `message_hash_fallback` | `false` | When session metadata is absent, keys affinity from the first user-message text. Requires `session_affinity = true`. |
| `prompt` | packaged capability prompt | Replaces the classifier's system prompt. The packaged verdict schema and routing policy remain active. |
| `max_output_tokens` | `4096` | Maximum completion tokens available to the classifier verdict. Must be at least `1`. |

### Override the classifier prompt

Set `prompt` on the route when the packaged capability rubric does not describe
your weak model. Switchyard sends the response schema separately through the
provider's structured-output request; do not copy it into the prompt.

```toml
[routes.smart]
id = "smart"
type = "llm_classifier"
mode = "capability"
classifier_target = "classifier"
strong_target = "strong"
weak_target = "weak"
base_threshold = 0.5
prompt = """
Estimate whether the weak target can complete the request.
Return exactly one JSON object matching the response schema supplied with the request.
"""
```

The override changes the instructions only. The judge must still return the
packaged `crux`, `primary_rule`, `capability_boundary`, and `p_solve` fields.

## Custom multi-target routing

Custom mode accepts an inner JSON Schema and a policy that reads the validated
verdict. This example routes across four configured targets:

```toml
[routes.smart]
id = "smart"
type = "llm_classifier"
mode = "custom"
classifier_target = "classifier"
targets = ["fast", "balanced", "reasoning", "premium"]
default_target = "premium"
prompt = """
Choose the best configured target for this request.
Return JSON matching the response schema supplied with the request.
"""
response_schema = '''
{
  "type": "object",
  "properties": {
    "decision": {
      "type": "object",
      "properties": {
        "target": {
          "type": "string",
          "enum": ["fast", "balanced", "reasoning", "premium"]
        }
      },
      "required": ["target"],
      "additionalProperties": false
    }
  },
  "required": ["decision"],
  "additionalProperties": false
}
'''

[routes.smart.policy]
type = "target_selector"
selector = "/decision/target"
```

The names in `targets` reference existing target tables. Switchyard passes the
schema to the provider in a strict structured-output wrapper and validates the
returned JSON again. `jsonptr` resolves the selector against that verdict. A
missing, non-string, or unknown target falls back to `default_target`.

This separation applies to every classifier mode. Prompts containing the legacy
`{{RESPONSE_SCHEMA}}` placeholder are rejected during configuration validation.

### Forecast and policy assumptions

The packaged prompt forecasts whole-task success for a generic efficient agent.
It produces a probability and capability boundary but does not choose a route.
The deterministic policy applies `base_threshold` and `threshold_step` after
generation.

Without affinity, the runtime judges every request. By default, it sends the
opening task and the latest user follow-up when they differ. Set
`recent_turn_window` when intervening conversation context affects the forecast.
If a client sends only a follow-up fragment without the opening task, enable
affinity or include the task history. Threshold tuning changes routing policy;
it cannot recover missing task context.

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

After [installing the Rust server](../getting_started.md#install-the-server), export
the provider credential, validate the configuration, and start the release
binary:

```bash
export OPENROUTER_API_KEY="your-openrouter-key"  # pragma: allowlist secret
switchyard-server --config routes.toml --dry-run
switchyard-server --config routes.toml \
  --host 127.0.0.1 --port 4000
```

Send a request using the route ID:

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"smart","messages":[{"role":"user","content":"Explain why the sky appears blue."}]}'
```

Treat the selected target as model-dependent output, not a fixed test result.
