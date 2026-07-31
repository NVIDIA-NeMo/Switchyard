# Switchyard + LiteLLM

> **Experimental integration:** This example and its Python APIs are
> experimental and may change without notice.

Build on this package when you want to put Switchyard routing in front of a
LiteLLM gateway. It demonstrates the integration end to end, and its client,
Compose configuration, and example code are intended as a copyable starting
point for your own application.

LiteLLM provides the OpenAI-compatible gateway, model aliases, and OpenRouter
provider integration. Switchyard's Stage router makes the routing decision from
the coding agent's recent tool history.
`LiteLLMSyClient` connects Switchyard's normalized libsy requests to the
selected alias through LiteLLM's asynchronous Completion API and the
Dockerized gateway. Together, they let an application keep routing policy in
Switchyard while LiteLLM owns model access and sends Chat Completions inference
through OpenRouter.

## Request flow

```text
your application → libsy normalized request → Switchyard Stage router
                 → LiteLLMSyClient → LiteLLM async Completion API
                 → Dockerized gateway alias → OpenRouter Chat Completions
                 → selected model
```

The bundled router treats `strong` as the capable tier and `fast` as the
efficient tier. It uses the `efficient_first` picker, so an initial request or
an ambiguous turn falls open to `fast`. Decisive error-recovery signals, such as
a critical failed tool result, route the turn to `strong`. The client asks
LiteLLM's Completion API to call the selected alias through the gateway rather
than embedding a provider model ID in application code.

## Quick start

This example uses these pinned gateway aliases and image:

- `strong` maps to OpenRouter's `openai/gpt-5.6-sol` model.
- `fast` maps to OpenRouter's `moonshotai/kimi-k3` model.
- The Python client pins `litellm==1.92.0`.
- The gateway runs `ghcr.io/berriai/litellm:v1.92.0`.

### Prerequisites

You need Python 3.12 or later, [uv](https://docs.astral.sh/uv/), Docker Compose,
an `OPENROUTER_API_KEY`, and OpenRouter account access to both model IDs.

### Configure

From the repository root:

```bash
cd examples/experimental/litellm
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY.
```

### Install from the checkout

From the package directory:

```bash
uv sync
```

### Install the package from source in another uv environment

From the repository root:

```bash
LITELLM_ENV_DIR="$(mktemp -d "${TMPDIR:-/tmp}/switchyard-litellm-env.XXXXXX")"
uv venv "$LITELLM_ENV_DIR"
uv pip install --python "$LITELLM_ENV_DIR/bin/python" ./examples/experimental/litellm
"$LITELLM_ENV_DIR/bin/python" -c "from switchyard_litellm import LiteLLMSyClient"
rm -rf -- "$LITELLM_ENV_DIR"
```

### Start and inspect LiteLLM

From the package directory:

```bash
docker compose up -d --wait
docker compose ps
curl -fsS http://127.0.0.1:4000/health/liveliness
```

### Run the example

```bash
uv run python example.py
```

The bundled request includes an out-of-memory tool failure, so the Stage router
selects the capable `strong` alias. Remove the assistant tool call and tool
result to see the same router fall open to `fast`.

## Use in your application

After installing the package, adapt this program to your application's
normalized request shape and routing policy:

```python
import asyncio

from switchyard.libsy import LlmTarget, algorithms
from switchyard_litellm import LiteLLMSyClient


async def main() -> None:
    request = {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Fix the failing tests."}],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "call_1",
                        "name": "Bash",
                        "arguments": {"command": "pytest"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_call_id": "call_1",
                        "content": [
                            {
                                "type": "text",
                                "text": "fatal runtime error: out of memory",
                            }
                        ],
                        "is_error": True,
                    }
                ],
            },
        ],
        "reasoning": {"effort": "low"},
        "output": {"max_output_tokens": 64},
    }
    strong = LiteLLMSyClient("strong")
    fast = LiteLLMSyClient("fast")
    router = algorithms.stage_router(
        LlmTarget("strong", strong),
        LlmTarget("fast", fast),
        picker="efficient_first",
        confidence_threshold=0.5,
        recent_window=3,
    )
    try:
        decisions, response = await router.run(request)
        print(decisions)
        print(response)
    finally:
        await strong.aclose()
        await fast.aclose()


asyncio.run(main())
```

## Known limitations

This example intentionally supports buffered text and OpenAI-compatible
function-tool traffic. It translates normalized function definitions, tool
choices, assistant tool calls, and text-only tool results into gateway requests,
and normalizes returned function calls for `libsy`.
Media, instructions, non-text tool results, structured output, provider-specific
extensions, preserved raw payloads, and streaming fail explicitly.

The pinned LiteLLM release does not yet recognize Kimi K3's current
`reasoning_effort` support. `LiteLLMSyClient` forwards LiteLLM's
`allowed_openai_params` hint to the gateway so OpenRouter receives that
supported parameter. Recheck this compatibility hint when upgrading LiteLLM.

LiteLLM 1.92 enters its optional proxy MCP path before distinguishing ordinary
OpenAI function tools. `LiteLLMSyClient` disables that bridge for these calls so
the integration does not need LiteLLM's proxy dependencies. Recheck this pinned
compatibility behavior when upgrading LiteLLM.

## Test

From the package directory, run the offline tests without provider calls:

```bash
PYTHONPATH=. uv run --project . pytest tests -m "not e2e" -v
```

The E2E test starts its own LiteLLM gateway and makes two paid OpenRouter Chat
Completions calls through one Stage router. A no-signal request falls open to
`fast`; a critical failed-tool request routes to `strong`.
`SWITCHYARD_LITELLM_E2E=1` is the explicit spend opt-in:

```bash
SWITCHYARD_LITELLM_E2E=1 \
uv run --project . --env-file .env pytest tests/test_e2e.py -m e2e -v
```

## Run the optional three-task Harbor smoke benchmark

See the repository's [Harbor benchmark guide](../../../benchmark/README.md) for
the one-time Harbor patch and dataset preparation. This is a separate test
boundary: the package E2E tests `LiteLLMSyClient`, while Harbor tests
`Harbor → Switchyard server → LiteLLM → OpenRouter`.

From the repository root, start the example gateway:

```bash
docker compose --env-file examples/experimental/litellm/.env \
  -f examples/experimental/litellm/compose.yaml up -d --wait
```

Then run the three-task smoke benchmark in the foreground:

```bash
SWITCHYARD_DOCKER_NETWORK=switchyard-litellm \
bash benchmark/run-baseline.sh \
  --harbor-path benchmark/datasets/openthoughts-tblite-closed-book \
  --server-config examples/experimental/litellm/benchmark-route.toml \
  --task-list-file examples/experimental/litellm/benchmark-tasks.txt \
  --model litellm-stage \
  --agent codex \
  --reasoning-effort low \
  --n-tasks 3 \
  --n-concurrent 1 \
  --max-retries 0 \
  --port 4100 \
  --foreground
```

The native benchmark route uses the same efficient-first Stage policy but does
not use `LiteLLMSyClient`; Switchyard server translates requests directly to
the shared LiteLLM gateway. The gateway drops parameters unsupported by the
selected provider model, so the benchmark's reasoning effort reaches `strong`
without causing `fast` requests to fail. The sample runs `broken-python`,
`cosign-keyless-signing`, and `jq-data-processing`. Three autonomous coding-agent
tasks can make many paid model calls. Inspect the resulting run under
`benchmark/tb_runs/`, including
`run_manifest.json`, `harbor.log`, `routing_stats_final.json`, and
`jobs/<job>/<task>/agent/trajectory.json`.

The verified run used the benchmark runner's fallback Codex model template:
the task image's pinned Codex 0.144.5 requires `supports_reasoning_summaries`,
which a newer host-generated catalog may omit. If Harbor fails with that missing
field, use the following recovery procedure. It changes no repository source
and is only for that exact error.

Create a temporary `codex` shim that makes the host-only
`codex debug models --bundled` probe fail. The runner then uses its existing
`_fallback_codex_model_template()`:

```bash
CODEX_SHIM_DIR="$(mktemp -d "${TMPDIR:-/tmp}/switchyard-codex.XXXXXX")"
printf '#!/usr/bin/env bash\nexit 1\n' > "$CODEX_SHIM_DIR/codex"
chmod +x "$CODEX_SHIM_DIR/codex"
```

From the repository root, rerun the benchmark with the temporary directory at
the front of `PATH`:

```bash
PATH="$CODEX_SHIM_DIR:$PATH" \
SWITCHYARD_DOCKER_NETWORK=switchyard-litellm \
bash benchmark/run-baseline.sh \
  --harbor-path benchmark/datasets/openthoughts-tblite-closed-book \
  --server-config examples/experimental/litellm/benchmark-route.toml \
  --task-list-file examples/experimental/litellm/benchmark-tasks.txt \
  --model litellm-stage \
  --agent codex \
  --reasoning-effort low \
  --n-tasks 3 \
  --n-concurrent 1 \
  --max-retries 0 \
  --port 4100 \
  --foreground
```

After the run, remove only the shim and its empty temporary directory:

```bash
rm -f "$CODEX_SHIM_DIR/codex"
rmdir "$CODEX_SHIM_DIR"
```

## Stop

For the package-directory quick start:

```bash
docker compose down
```

After the repository-root Harbor workflow:

```bash
docker compose --env-file examples/experimental/litellm/.env \
  -f examples/experimental/litellm/compose.yaml down
```

## Troubleshooting

- Ensure `OPENROUTER_API_KEY` is present and not blank in `.env`.
- Confirm your OpenRouter account can access `openai/gpt-5.6-sol` and
  `moonshotai/kimi-k3`.
- If host port 4000 is occupied, set `LITELLM_PORT` to an unused port and use
  it in the health check, gateway URL, and `LiteLLMSyClient` `base_url`. The
  bundled `example.py` assumes port 4000 unless you edit it.
- If the container is unhealthy, from the repository root inspect:

  ```bash
  docker compose --env-file examples/experimental/litellm/.env \
    -f examples/experimental/litellm/compose.yaml logs litellm
  ```
- Harbor must use the Docker network named exactly `switchyard-litellm`.
- The benchmark uses Switchyard port 4100 to avoid LiteLLM's host port 4000.

## Security

The gateway is unauthenticated and intended only for local development. Its
port is bound to loopback only. Use LiteLLM authentication and its production
deployment guidance before exposing it remotely.

## References

- [OpenRouter Chat Completions](https://openrouter.ai/docs/api-reference/chat-completion)
- [GPT-5.6 Sol on OpenRouter](https://openrouter.ai/openai/gpt-5.6-sol)
- [Kimi K3 on OpenRouter](https://openrouter.ai/moonshotai/kimi-k3-20260715)
- [LiteLLM gateway quick start](https://docs.litellm.ai/docs/proxy/quick_start)
- [Switchyard Stage-router docs](../../../docs/routing_algorithms/stage_router_routing.md)
