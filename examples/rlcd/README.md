# RLCD decision routing

Serve an RLCD (calibrated decision) model behind its own OpenAI-compatible
endpoint and let it choose the best target per request.

## What you need

- An endpoint that serves a decision model. The most-liked open Jev
  counterpart is
  [`AlexWortega/openjev`](https://huggingface.co/AlexWortega/openjev), a
  Qwen3.5 cross-encoder that scores a task against every candidate option and
  returns one probability per option. Any OpenAI-compatible server that
  answers the decision prompt with one JSON object works.
- Two or more completion targets (the candidates the decision model chooses
  between).

## Run

Start the decision endpoint, then point `routes.toml` at it and the completion
targets through `[llm_clients]`:

```toml
schema_version = 1

[llm_clients.rlcd]
format = "openai_chat"
base_url = "http://localhost:8080/v1"
api_key_env = "RLCD_API_KEY"

[llm_clients.openrouter]
format = "openai_chat"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"

[targets.decision]
id = "Qwen-2.5-1B-RLCD"
llm_client = "rlcd"

[targets.strong]
id = "openai/gpt-5"
llm_client = "openrouter"

[targets.weak]
id = "openai/gpt-4o-mini"
llm_client = "openrouter"

[routes.decide]
id = "switchyard/rlcd"
type = "rlcd"
classifier_target = "decision"
targets = ["weak", "strong"]
default_target = "weak"
```

Send requests to the route:

```bash
# from the repository root: run the Rust server against this example config
cargo run -p switchyard-server -- examples/rlcd/routes.toml

curl http://localhost:8000/v1/chat/completions \
  -h 'content-type: application/json' \
  -d '{
    "model": "switchyard/rlcd",
    "messages": [{"role": "user", "content": "Refactor this module and add tests."}]
  }'
```

Switchyard sends the task plus a numbered option list to the decision model,
routes to the option with the highest probability, and falls back to
`default_target` when the verdict cannot be used. A failed decision call stops
the request, exactly like a failed classifier judge.

See
[RLCD Decision Routing](../../docs/routing_algorithms/rlcd_routing.md) for the
full behavior, including the exact verdict schema the decision endpoint must
return.