# Evaluate Routing Locally

Use the standalone server to try routing configurations with your own model pool. It is for demos, evaluation, and personal use on single-user machines. **It is not a supported production deployment path.** To add routing to your gateway or application, use the [libsy quickstart](../getting_started.md).

## Install the server

You need Rust 1.96.1+, a native build toolchain, `curl`, and an OpenRouter API key. Install Rust using the [official instructions](https://rust-lang.org/tools/install/) and check [platform requirements](../../INSTALLATION.md#requirements). The v0.3.0 server is release-validated on Ubuntu 24.04, Linux x86_64.

Install the released server:

```bash
cargo install --locked switchyard-server --version 0.3.0
switchyard-server --help
```

For unreleased features, [build from main](../../crates/switchyard-server/README.md). Keep your configuration and binary on compatible versions.

## Configure your model pool

Create `routes.toml`:

```toml
schema_version = 1

[llm_clients.openrouter]
format = "openai_chat"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"

[targets.efficient]
id = "z-ai/glm-5.2"
llm_client = "openrouter"

[targets.capable]
id = "anthropic/claude-opus-4.8"
llm_client = "openrouter"

[routes.smart]
id = "switchyard"
type = "auto"
capable_target = "capable"
efficient_target = "efficient"
```

This uses the same Auto preset as the first libsy example: Stage routing with `efficient_first`, a confidence threshold of `0.5`, and no classifier call. The model pool also appears in our [Terminal-Bench 2.1 results](../benchmarks.md#terminal-bench-21). This small Auto example is not the benchmark configuration; use the linked benchmark profiles to reproduce their settings.

To use your own pool:

| Change | Where |
| --- | --- |
| Efficient and capable models | Replace `targets.efficient.id` and `targets.capable.id`. |
| Provider endpoint | Change `llm_clients.openrouter.base_url` and `api_key_env`. Use a matching `format`: `openai_chat`, `openai_responses`, or `anthropic_messages`. |
| Different providers for each model | Add another `llm_clients` entry and set each target's `llm_client`. |
| Public model name used by callers | Change `routes.smart.id` and the request's `model` together. |

Keep credentials in environment variables. For provider options, credential forwarding, and route types, see the [TOML reference](../reference/toml_schema.md) and [model selection guide](../routing_algorithms/model_selection.md).

## Run the server

Set `OPENROUTER_API_KEY` in your environment with access to the configured models. Then validate and run the configuration:

```bash
export OPENROUTER_API_KEY="your-openrouter-key"  # pragma: allowlist secret
switchyard-server --config routes.toml --dry-run
switchyard-server --config routes.toml --host 127.0.0.1 --port 4000
```

The dry run prints `server OK: switchyard`. It checks configuration and credential lookup, not whether the provider accepts the key.

In a second terminal, check local readiness:

```bash
curl -fsS http://127.0.0.1:4000/health
curl -fsS http://127.0.0.1:4000/v1/models
```

The model listing includes `switchyard`. These local checks make no provider requests.

## Verify routing is active

The next two requests call your providers and may incur charges. Start with a fresh server, or clear its counters before this sequence:

```bash
curl -fsS -X POST http://127.0.0.1:4000/v1/stats/reset
```

First, send a request without tool history:

```bash
curl -i http://127.0.0.1:4000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"switchyard","messages":[{"role":"user","content":"Say hello in one sentence."}]}'
```

The response header identifies the efficient model:

```text
x-model-router-selected-model: z-ai/glm-5.2
```

Next, save as `escalate.json`:

```json
{
  "model": "switchyard",
  "messages": [
    {"role": "user", "content": "Run the test suite."},
    {"role": "assistant", "content": null, "tool_calls": [
      {"id": "call_1", "type": "function",
       "function": {"name": "bash", "arguments": "{\"command\":\"pytest -q\"}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_1",
     "content": "MemoryError: cannot allocate memory\nKilled"},
    {"role": "user", "content": "Fix it."}
  ]
}
```

Send it:

```bash
curl -i http://127.0.0.1:4000/v1/chat/completions \
  -H 'Content-Type: application/json' -d @escalate.json
```

The critical error forces the capable model:

```text
x-model-router-selected-model: anthropic/claude-opus-4.8
```

Read the counters with `jq` installed:

```bash
curl -fsS http://127.0.0.1:4000/v1/stats | jq '.models | map_values(.calls)'
curl -fsS http://127.0.0.1:4000/v1/stats | jq '.algorithm_stats.stage_router.routing_decisions'
```

After exactly these two successful completions, expect one call per model:

```json
{"anthropic/claude-opus-4.8":1,"z-ai/glm-5.2":1}
```

The decision counters show one `fall_open` choice and one `override`. Here, `fall_open` means the request had no usable signals and used the efficient-first default. `override` means a hard error forced the capable tier. Additional requests change the totals. Counters reset on server restart or `POST /v1/stats/reset`.

For per-request records, restart with `--routing-log-file routing.jsonl`. Each record includes the selected model and token usage. Decision reasons are reported by the stats endpoint. These checks confirm routing behavior, not task quality or savings.

## Compare configurations

- [Benchmark results](../benchmarks.md) link to model pools, results, and runnable profiles. For Terminal-Bench 2.1, the [escalation profile](../../benchmark/routing-profiles/tb21-escalation-opus-glm-deepseek.toml) includes its judge and reasoning settings. Its comments explain the endpoint differences from the measured runs.
- [Benchmark setup](../../benchmark/README.md) explains how to compare routing against single-model baselines on the same tasks.
- [Routing overview](../routing_algorithms/overview.md) explains Auto, Task, Execution, and Composite. A TOML file can expose multiple routes through different route IDs.
- [Server reference](../../crates/switchyard-server/README.md) covers configuration, TLS, logging, and metrics.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Missing credential | Set the environment variable named by `api_key_env` in the server's terminal. Run `--dry-run` again. |
| Provider authentication or model error | Confirm the key grants access to the exact configured model IDs and endpoint. A successful dry run does not check this. |
| Connection refused | Check the server terminal and `curl -fsS http://127.0.0.1:4000/health`. |
| Every fresh chat selects the efficient model | Expected for this preset without tool signals. Use the escalation request above. |

Stop the server with Ctrl+C when finished.
