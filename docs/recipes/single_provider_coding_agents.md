# Interactive coding agents with single-provider auth

Route Codex or Claude Code between models from one provider using your saved CLI
login. Every target, including the classifier, must use that provider.

These [composite routing](../routing_algorithms/composite_routing.md) recipes use a
classifier to choose the starting tier for each user turn. Stage adjusts routing
during tool calls. [`forward_auth`](../reference/toml_schema.md#llm_clientsname)
passes your login through to the models.

Choose one recipe. Each uses local port `4123`.

## Codex with OpenAI

Sign in with `codex login`. Terra classifies, and Stage routes between Sol and Luna.

This recipe needs Codex classifier support added after v0.3.0. Install from a
checkout of current `main`:

```bash
cargo install --locked --path crates/switchyard-server
```

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

## Claude Code with Anthropic

Sign in to Claude Code with your Claude account. Haiku classifies, and Stage routes
between Opus and Sonnet.

Install the latest release:

```bash
cargo install --locked switchyard-server
```

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

Use your saved Claude login, without an `apiKeyHelper` or gateway token.

## Check the routing

Send a prompt, then check which models handled the answer and classification:

```bash
curl -s http://127.0.0.1:4123/v1/stats \
  | jq '{answers: .models, classifier: .classifier.models}'
```
