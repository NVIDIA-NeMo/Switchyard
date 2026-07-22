# switchyard-server

`switchyard-server` exposes libsy algorithms through OpenAI Chat Completions, OpenAI Responses,
and Anthropic Messages endpoints. A YAML file explicitly defines the LLM clients, target backends,
and algorithm routes served by the process.

```yaml
# routes.yaml
schema_version: 1

llm_clients:
  primary:
    type: translating_http

targets:
  model/a:
    llm_client: primary
    backend:
      type: openai_chat
      base_url: https://example.com/v1
      api_key_env: API_KEY

  model/b:
    llm_client: primary
    backend:
      type: openai_chat
      base_url: https://example.com/v1
      api_key_env: API_KEY

routes:
  switchyard/general:
    type: random
    targets:
      - model/a
      - model/b

  switchyard/classified:
    type: llm_classifier
    classifier_target: model/a
    strong_target: model/a
    weak_target: model/b
    threshold: 0.5
```

```bash
export API_KEY="..."
cargo run -p switchyard-server -- --config routes.yaml
```

Each target name is also the exact model ID sent upstream. Select an algorithm route by sending its
`switchyard/*` route ID as the inbound request's `model`.

Supported client type: `translating_http`. Supported backend types: `openai_chat`,
`openai_responses`, and `anthropic_messages`. Supported algorithms: `noop`, `random`, and
`llm_classifier`. An `api_key_env` value names an environment variable; the YAML never contains the
secret itself. If the field is omitted, the backend is called without authentication.

To add another built-in algorithm:

1. Implement and export it from `libsy`.
2. Add its typed fields to `AlgorithmConfig` in `src/config.rs`.
3. Add one construction arm to `build_algorithm` and cover it with a config test.
