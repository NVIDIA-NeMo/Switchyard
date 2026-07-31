# Random Routing

Random routing selects one configured target for each request without inspecting
the prompt. It supports two or more targets.

Use it for A/B tests, benchmark baselines, and controlled traffic splits. Use a
content-aware routing algorithm instead when the request itself should determine
the target.

## Configure a random route

This example sends approximately 30% of requests to a strong model and 70% to a
weak model:

```toml
schema_version = 1

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

[routes.ab_test]
id = "ab-test"
type = "random"
targets = ["strong", "weak"]
weights = [3, 7]
seed = 42
```

The configuration has two layers:

- `[targets.strong]` and `[targets.weak]` define models that Switchyard can call.
  Their `id` values are the model identifiers sent to the upstream provider.
- `[routes.ab_test]` defines the random route. The table name `ab_test` is local
  to the TOML file, while its `id`, `ab-test`, is the model name clients send to
  Switchyard.

## Control the traffic split

The route's `targets` list establishes the order:

```toml
targets = ["strong", "weak"]
```

The corresponding `weights` list assigns a relative weight to each target in
that same order:

```toml
weights = [3, 7]
```

The weights do not need to add up to `1`; Switchyard treats `3:7` as a 30%/70%
split. More generally:

- Every target name must be unique and refer to a target defined in the same
  TOML file.
- `weights` must contain one non-negative, finite number per target.
- Omitting `weights` gives every target equal weight.
- A weight of `0` disables that target. At least one weight must be greater than
  `0`.
- `seed` is optional. Set it to reproduce the same selection sequence for the
  same request order in tests or benchmarks. Omit it for normal traffic.

Selection is independent for each request, so short runs may not match the
configured proportions exactly. Separate turns in the same conversation can
therefore use different targets.

## Run the route

After [building the Rust server](../getting_started.md#install), export the
provider credential, validate the configuration, and start the release binary:

```bash
export OPENROUTER_API_KEY="your-openrouter-key"  # pragma: allowlist secret
./target/release/switchyard-server --config routes.toml --dry-run
./target/release/switchyard-server --config routes.toml \
  --host 127.0.0.1 --port 4000
```

Send a request using the route ID:

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ab-test","messages":[{"role":"user","content":"hello"}]}'
```

## Experimental LiteLLM integration

The package's
[developer guide](../../examples/experimental/litellm/README.md) is the
canonical documentation for integrating Switchyard with a Dockerized LiteLLM
gateway. Its source-installable client adapts libsy's dictionary contract and
demonstrates weighted random routing between model aliases backed by
OpenRouter. The guide covers setup, application usage, live end-to-end testing,
and a three-task Harbor smoke benchmark.
