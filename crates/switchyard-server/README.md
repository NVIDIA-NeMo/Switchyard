# switchyard-server

`switchyard-server` exposes libsy algorithms through OpenAI Chat Completions, OpenAI Responses,
and Anthropic Messages endpoints. A TOML file explicitly defines the LLM clients, targets, and
algorithm routes served by the process.

```toml
# routes.toml
schema_version = 1

[llm_clients.example]
format = "openai_chat"
base_url = "https://example.com/v1"
api_key_env = "API_KEY"

[targets.model_a]
id = "model/a"
llm_client = "example"

[targets.model_b]
id = "model/b"
llm_client = "example"

[routes.general]
id = "switchyard/general"
type = "random"
targets = ["model_a", "model_b"]

[routes.classified]
id = "switchyard/classified"
type = "llm_classifier"
classifier_target = "model_a"
strong_target = "model_a"
weak_target = "model_b"
threshold = 0.5
```

```bash
export API_KEY="..."
cargo run -p switchyard-server -- --config routes.toml
```

Target and route table names are local references. A target's `id` is the exact model ID sent
upstream, and a route's `id` is the model clients send to select that algorithm.

Each target references an entry under `llm_clients`. The client `type` defaults to `translating`;
supported formats are `openai_chat`, `openai_responses`, and `anthropic_messages`. Supported
algorithms are `noop`, `random`, and `llm_classifier`. An `api_key_env` value names an environment
variable; the TOML never contains the secret itself. If omitted, the client sends no authentication.

See [CONFIGURATION.md](CONFIGURATION.md) to add an LLM client, target, or algorithm.
