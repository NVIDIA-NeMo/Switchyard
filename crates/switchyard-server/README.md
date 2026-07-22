# switchyard-server

`switchyard-server` exposes libsy algorithms through OpenAI Chat Completions, OpenAI Responses,
and Anthropic Messages endpoints. A TOML file explicitly defines the LLM clients, target backends,
and algorithm routes served by the process.

```toml
# routes.toml
schema_version = 1

[targets."model/a"]
backend = { type = "openai_chat", base_url = "https://example.com/v1", api_key_env = "API_KEY" }

[targets."model/b"]
backend = { type = "openai_chat", base_url = "https://example.com/v1", api_key_env = "API_KEY" }

[routes."switchyard/general"]
type = "random"
targets = ["model/a", "model/b"]

[routes."switchyard/classified"]
type = "llm_classifier"
classifier_target = "model/a"
strong_target = "model/a"
weak_target = "model/b"
threshold = 0.5
```

```bash
export API_KEY="..."
cargo run -p switchyard-server -- --config routes.toml
```

Each target name is also the exact model ID sent upstream. Select an algorithm route by sending its
`switchyard/*` route ID as the inbound request's `model`.

Targets use a shared `translating_http` client by default. Define `llm_clients` and set a target's
`llm_client` only when it needs a separate named client. Supported backend types: `openai_chat`,
`openai_responses`, and `anthropic_messages`. Supported algorithms: `noop`, `random`, and
`llm_classifier`. An `api_key_env` value names an environment variable; the TOML never contains the
secret itself. If the field is omitted, the backend is called without authentication.

See [CONFIGURATION.md](CONFIGURATION.md) to add a target, backend type, or algorithm.
