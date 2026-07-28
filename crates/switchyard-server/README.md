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
weights = [1, 3]
seed = 42

[routes.classified]
id = "switchyard/classified"
type = "llm_classifier"
classifier_target = "model_a"
strong_target = "model_a"
weak_target = "model_b"
threshold = 0.5

[routes.passthrough]
id = "switchyard/passthrough"
type = "passthrough"
target = "model_a"
```

```bash
export API_KEY="..."
cargo run -p switchyard-server -- --config routes.toml
```

The server logs exactly one structured terminal event per LLM request: successful responses at
`INFO`, 4xx responses at `WARN`, and 5xx responses at `ERROR`. Set
`RUST_LOG=switchyard_server=debug,libsy=debug` to include routing decisions and nested failure
details. A streaming failure is logged separately because it can occur after the response starts.

Target and route table names are local references. A target's `id` is the exact model ID sent
upstream, and a route's `id` is the model clients send to select that algorithm.

Each target references an entry under `llm_clients`. All configured clients use
`TranslatingLlmClient`; supported formats are `openai_chat`, `openai_responses`, and
`anthropic_messages`. Supported algorithms are `noop`, `random`, `passthrough`, and
`llm_classifier`. An `api_key_env` value names an environment variable; the TOML
never contains the secret itself. If omitted, the client sends no authentication.

Random-route `weights` are relative, follow target order, and do not need to sum to one. Omit them
for equal weighting. The optional `seed` reproduces the selection sequence for the same call order.

See [CONFIGURATION.md](CONFIGURATION.md) to add an LLM client, target, or algorithm.
