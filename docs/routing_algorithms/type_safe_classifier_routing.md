# TypeSafe Classifier Routing

TypeSafe classifier routing decides a route's target with
[TypeSafe](https://docs.typesafe.ai/introduction)'s Jev "System One Model"
instead of a chat-completion judge. Jev is non-generative: it returns a typed,
probabilistic `Choice` directly from a sampling layer rather than generating
and parsing text, so this classifier never calls one of the deployment's own
`[llm_clients]` targets for its judgment step. It supports two or more targets,
the same way `llm_classifier` custom mode does.

Use it when you want a classifier call that is not itself a chat completion —
for example to avoid a judge model's own latency and reasoning-token cost. Use
`llm_classifier` instead when the classifier should be one of your existing
chat-completion targets, or when you need capability/escalation mode.

## Configure a TypeSafe client

Every `type_safe_classifier` route shares one deployment-wide TypeSafe client,
configured once in `[type_safe_client]`:

```toml
[type_safe_client]
api_key_env = "TYPESAFE_API_KEY"
# base_url = "https://api.typesafe.ai"   # optional, defaults to the production endpoint
# model = "jev-latest"                    # optional, defaults to "jev-latest"
```

`api_key_env` names an environment variable read at startup. As with every
other credential in a deployment, the key itself never appears in the TOML
file or in logs. A route using `type = "type_safe_classifier"` fails to build
when the deployment has no `[type_safe_client]` table.

## Configure a route

```toml
schema_version = 1

[type_safe_client]
api_key_env = "TYPESAFE_API_KEY"

[llm_clients.openrouter]
format = "openai_chat"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"

[targets.strong]
id = "openai/gpt-4o"
llm_client = "openrouter"

[targets.weak]
id = "openai/gpt-4o-mini"
llm_client = "openrouter"

[routes.smart]
id = "smart"
type = "type_safe_classifier"
default_target = "efficient"
base_threshold = 0.6
question = "Which model tier does this conversation need?"

[routes.smart.options]
capable = "Needs multi-step reasoning, ambiguous instructions, or high-stakes correctness."
efficient = "A short, well-specified request."

[routes.smart.models]
capable = ["strong"]
efficient = ["weak"]
any = ["strong", "weak"]
```

`options` is the set of labels Jev is asked to choose between, each paired
with the natural-language criteria for when it applies. Every `options` key
must also be a key of `models`, and every group in `models` (`any` included)
must contain at least one target. Unlike `llm_classifier`, there is no
`models.judge` group — the judgment step is TypeSafe itself, not one of your
runtime models — and `judge` may not be used as an `options` label or as
`default_target`.

## How the decision works

Each request's opening task (and latest user follow-up, when it differs) is
flattened into plain text and sent to TypeSafe as `state`, alongside `options`
as the `Choice` criteria. TypeSafe returns a label and a confidence in
`[0, 1]`.

- A confidence at or above `base_threshold` routes to the first model in the
  matching `options` label's `models` group.
- A confidence below `base_threshold`, an unresolved or unconfigured label, an
  out-of-range or non-numeric confidence, or a failed TypeSafe request all
  fall back to `default_target` instead. This classifier never errors the
  request: every provider failure mode fails open.

## Tuning options

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `options` | Yes | — | Labeled criteria offered to TypeSafe, keyed by name. Each key must also be a key of `models`. |
| `default_target` | Yes | — | Group used when TypeSafe fails, returns an unconfigured label, or answers below `base_threshold`. Any group except `judge`. |
| `base_threshold` | Yes | — | Lowest confidence that is trusted, in `[0, 1]`. |
| `question` | No | generic tier-selection prompt | Instruction sent as TypeSafe's `instructions` field. |
| `classify_trigger` | No | `every_request` | When the classifier re-decides. Same semantics as `llm_classifier`: `every_request`, `user_turn`, or `new_session`. |
| `message_hash_fallback` | No | `false` | Keys affinity on the first user message when session metadata is absent. Requires `classify_trigger = "new_session"`. |
| `recent_turn_window` | No | unset | When unset, TypeSafe sees the opening task and latest user follow-up, when present. When set, it also sees trailing turns. |

## Run the route

After [installing the Rust server](../getting_started.md#install-the-server), export
both the upstream provider credential and the TypeSafe API key, validate the
configuration, and start the release binary:

```bash
export OPENROUTER_API_KEY="your-openrouter-key"  # pragma: allowlist secret
export TYPESAFE_API_KEY="your-typesafe-key"      # pragma: allowlist secret
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
