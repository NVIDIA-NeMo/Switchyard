# Coding agents with your existing login

Run Codex or Claude Code through Switchyard using the login already saved by your
CLI. Switchyard forwards that credential to the same provider, including classifier
calls. You do not need to put a key in the server config.

Both recipes use [composite routing](../routing_algorithms/composite_routing.md).
A classifier chooses the default tier for each user turn. The stage router uses
tool results to adjust that choice during the turn. Keep every target, including
the classifier, with the same provider when using
[`forward_auth`](../reference/toml_schema.md#llm_clientsname).

From a checkout of current `main`, install the server, then choose one recipe.
Each uses local port `4123`.

```bash
cargo install --locked --path crates/switchyard-server
```

## Codex with OpenAI

Sign in with `codex login`. This example uses Terra to classify and routes between
Sol and Luna. Use model IDs available to your ChatGPT account.

Save as `codex-routing.toml`:

```toml
schema_version = 1

[llm_clients.chatgpt]
format = "openai_responses"
base_url = "https://chatgpt.com/backend-api/codex"
forward_auth = true

[targets.capable]
id = "gpt-5.6-sol"
llm_client = "chatgpt"

[targets.efficient]
id = "gpt-5.6-luna"
llm_client = "chatgpt"

[targets.judge]
id = "gpt-5.6-terra"
llm_client = "chatgpt"
extra_body = { store = false, stream = true }
omit_body_fields = ["max_output_tokens"]

[routes.switchyard]
id = "switchyard"
type = "composite"

[routes.switchyard.classifier]
target = "judge"
base_threshold = 0.5
classify_trigger = "user_turn"

[routes.switchyard.stage]
capable_target = "capable"
efficient_target = "efficient"
confidence_threshold = 0.5
```

The ChatGPT backend requires `store = false` and `stream = true`, and rejects
`max_output_tokens`. The judge settings above adapt its generated request.
`omit_body_fields` requires a build newer than v0.3.0. Codex already supplies the
right fields for the answer requests.

Start the server:

```bash
switchyard-server --config codex-routing.toml --dry-run
switchyard-server --config codex-routing.toml --host 127.0.0.1 --port 4123
```

Create `~/.codex/switchyard.config.toml` (or place it beside your `config.toml` if
you use a custom Codex config directory):

```toml
model = "switchyard"
model_provider = "switchyard"

[model_providers.switchyard]
name = "Switchyard"
base_url = "http://127.0.0.1:4123/v1"
wire_api = "responses"
requires_openai_auth = true
```

In another terminal, launch Codex:

```bash
codex --profile switchyard
```

`requires_openai_auth` makes Codex send its saved OpenAI login. Switchyard forwards
it and the account headers to the ChatGPT backend. See the
[Codex configuration reference](https://developers.openai.com/codex/config-reference/)
for profile and provider settings.

## Claude Code with Anthropic

Sign in to Claude Code with your Claude account first. This example uses Haiku to
classify and routes between Opus and Sonnet. Use model IDs available to your account.

Save as `claude-routing.toml`:

```toml
schema_version = 1

[llm_clients.anthropic]
format = "anthropic_messages"
base_url = "https://api.anthropic.com"
forward_auth = true

[targets.capable]
id = "claude-opus-5-5"
llm_client = "anthropic"

[targets.efficient]
id = "claude-sonnet-5-5"
llm_client = "anthropic"

[targets.judge]
id = "claude-haiku-4-5-20251001"
llm_client = "anthropic"

[routes.switchyard]
id = "switchyard"
type = "composite"

[routes.switchyard.classifier]
target = "judge"
base_threshold = 0.5
classify_trigger = "user_turn"

[routes.switchyard.stage]
capable_target = "capable"
efficient_target = "efficient"
confidence_threshold = 0.5
```

Start the server:

```bash
switchyard-server --config claude-routing.toml --dry-run
switchyard-server --config claude-routing.toml --host 127.0.0.1 --port 4123
```

In another terminal, launch Claude Code:

```bash
env -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN \
  ANTHROPIC_BASE_URL=http://127.0.0.1:4123 \
  ANTHROPIC_DEFAULT_OPUS_MODEL=switchyard \
  ANTHROPIC_DEFAULT_SONNET_MODEL=switchyard \
  ANTHROPIC_DEFAULT_HAIKU_MODEL=switchyard \
  claude --model switchyard
```

Do not append `/v1` to Claude Code's base URL. The CLI adds it. The model overrides
keep its Opus, Sonnet, and Haiku aliases pointed at the configured route.

Leave gateway credentials and `apiKeyHelper` unset when using your saved login.
A placeholder `ANTHROPIC_AUTH_TOKEN` would replace the real credential. Switchyard
forwards Anthropic's credential and the `oauth-*` beta marker required for OAuth.
See [Anthropic's gateway authentication docs](https://code.claude.com/docs/en/llm-gateway#subscriptions-and-gateways).

## Check the routing

Send a short prompt, then inspect the server's counters:

```bash
curl -s http://127.0.0.1:4123/v1/stats \
  | jq '{answers: .models, classifier: .classifier.models}'
```

The answer appears under the selected model. The classifier has separate counters.
Both CLIs send session headers that Switchyard uses to retain the classifier's
choice between user turns. Stop the local server with `Ctrl-C`.
