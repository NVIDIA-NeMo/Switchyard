# Switchyard Examples

Runnable examples for the Switchyard Python bindings and server.

| Path | What it shows |
|------|---------------|
| [`libsy.py`](libsy.py) | Drive a libsy routing algorithm stream from Python — build a weighted `algorithms.random` route, consume `Step.Decision` / `Step.CallModel` / `Step.Done` from `run_stream`, and answer model calls with a custom async client. |
| [`experimental/litellm/`](experimental/litellm/) | Experimental LiteLLM stage-router integration: benchmark route TOML, LiteLLM proxy config, compose stack, and tests. |
| [`prometheus/`](prometheus/) | Drop-in Prometheus + Alertmanager configuration with recording and alert rules for a Switchyard deployment. |

## Running the libsy example

The example only needs the `nemo-switchyard` package installed:

```bash
uv run python examples/libsy.py
```

It uses an in-file `EchoClient` that returns a fixed completion, so no provider
API keys are required. See [`examples/prometheus/README.md`](prometheus/README.md)
and [`examples/experimental/litellm/README.md`](experimental/litellm/README.md)
for the requirements of those examples.
