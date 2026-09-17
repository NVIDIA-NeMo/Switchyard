# TypeSafe Classifier Routing

TypeSafe classifier routing decides a route's target with
[TypeSafe](https://docs.typesafe.ai/introduction)'s Jev "System One Model"
instead of a chat-completion judge. Jev is non-generative: it returns a typed,
probabilistic `Choice` directly from a sampling layer rather than generating
and parsing text, so this classifier never calls one of the deployment's own
`[llm_clients]` targets for its judgment step. It compares two or more target
models configured by the user.

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
routing_description = "Best for complex, multi-step work and high-stakes correctness."

[targets.weak]
id = "openai/gpt-4o-mini"
llm_client = "openrouter"
routing_description = "Best for short, simple, well-specified requests."

[routes.smart]
id = "smart"
type = "type_safe_classifier"
default_target = "weak"
base_threshold = 0.6
candidates = ["strong", "weak"]
question = "Which configured model is the best fit for this conversation?"
```

`candidates` names the configured targets that Jev compares. Each candidate
must be unique, and `default_target` must name one of them. The optional
`routing_description` on a target tells Jev when that model is a good fit. If
it is omitted, Switchyard uses the target name and model ID. A route can
override a target's description with `candidate_descriptions`:

```toml
[routes.smart.candidate_descriptions]
strong = "Prefer for difficult coding and long-context analysis."
```

The names `any` and `judge` are reserved and cannot be candidates.

## How the decision works

Each request's opening task (and latest user follow-up, when it differs) is
flattened into plain text and sent to TypeSafe as `state`, alongside the
candidate descriptions as `Choice` criteria. Switchyard sends up to three
fixed candidate orders in one request, averages their probability
distributions, and chooses the candidate with the highest average probability.
It computes confidence from that averaged distribution using TypeSafe's Choice
confidence formula. The full probabilities and decision latency are recorded
in routing evidence.

- A confidence at or above `base_threshold` routes to the selected target.
- A confidence below `base_threshold`, an invalid response, or a failed
  TypeSafe request falls back to `default_target`. This classifier never
  errors the request: every provider failure mode fails open.

## Tuning options

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `candidates` | Yes | — | Two or more unique target names offered to TypeSafe. `any` and `judge` are reserved. |
| `candidate_descriptions` | No | target description or generated text | Route-specific descriptions keyed by candidate target name. |
| `default_target` | Yes | — | Candidate used when TypeSafe fails or answers below `base_threshold`. |
| `base_threshold` | Yes | — | Lowest confidence that is trusted, in `[0, 1]`. |
| `question` | No | generic model-selection prompt | Instruction sent as TypeSafe's `instructions` field. |
| `classify_trigger` | No | `every_request` | When the classifier re-decides. Same semantics as `llm_classifier`: `every_request`, `user_turn`, or `new_session`. |
| `message_hash_fallback` | No | `false` | Keys affinity on the first user message when session metadata is absent. Requires `classify_trigger = "new_session"`. |
| `recent_turn_window` | No | unset | When unset, TypeSafe sees the opening task and latest user follow-up, when present. When set, it also sees trailing turns. |

`targets.<name>.routing_description` is optional. It supplies reusable facts
about a target for every TypeSafe route that includes it. A route-level
`candidate_descriptions` entry takes precedence.

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
