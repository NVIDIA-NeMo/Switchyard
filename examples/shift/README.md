# Shift-1 with direct selection

[Shift](https://shift.ac) runs Shift-1, a router for Codex and Claude Code. In
this example, Switchyard asks Shift-1 which model to use for each request and
then serves that model. Codex requests go to GPT-6.1-Sol or GPT-6-Astra. Claude
Code requests go to Claude Sonnet 5.5 or Claude Opus 5.5.

Shift answers decisions at `https://api.shift.ac/v1/decisions` in the OpenAI
Decisions format, and serves the models with the same key. Switchyard uses the
key in `SHIFT_API_KEY` for decisions. Codex and Claude Code send the same key,
and Switchyard forwards it with each model call. See
[direct selection](../../docs/routing_algorithms/llm_classifier_routing.md#direct-selection)
for how the route works.

## Run it

You need a Shift API key with your OpenAI key, Anthropic key, or both
connected in Shift.

```bash
export SHIFT_API_KEY=relay_...
cargo run -p switchyard-server --release -- --config examples/shift/switchyard.toml --port 4123
```

Point Codex at the `codex` route. Add this to `~/.codex/config.toml`:

```toml
[profiles.shift]
model = "codex"
model_provider = "switchyard"

[model_providers.switchyard]
name = "Switchyard"
base_url = "http://127.0.0.1:4123/v1"
wire_api = "responses"
env_key = "SHIFT_API_KEY"
```

Then run `codex --profile shift`.

Point Claude Code at the `claude_code` route:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:4123 ANTHROPIC_AUTH_TOKEN="$SHIFT_API_KEY" claude --model claude_code
```

`curl -s http://127.0.0.1:4123/v1/stats` shows which models served the
requests.

## How it routes

Each route lists Shift's model names as its groups. Shift reads the
conversation and answers with one of them. Switchyard serves the target in that
group.

By default, a decision includes the opening task and the last 10 messages.
Shift-1 reads more of the session than that, so the example sets
`recent_turn_window = 128`.

Each route offers one provider's models. Shift rejects a route that mixes
providers. When a decision fails, the route serves `default_target`. Set
`fail_open = false` to stop the request instead.
