# Getting Started with Switchyard

## Prerequisites

- Python 3.12 or later
- macOS, Linux, or Windows
- An API key for OpenRouter, OpenAI, Anthropic, or another OpenAI-compatible endpoint.
  To use OpenRouter, create an account at [openrouter.ai](https://openrouter.ai/)
  and generate a key from the [OpenRouter keys page](https://openrouter.ai/keys).

## Install

```bash
pip install "nemo-switchyard[cli,server]"
```

## Configure

Interactive setup saves your provider credentials and routing bundle to
`~/.config/switchyard/`. All paths below pick them up automatically at runtime.

```bash
switchyard configure
```

Or non-interactively with a routing-profile YAML:

```bash
export OPENROUTER_API_KEY="your-openrouter-key"  # pragma: allowlist secret

cat > routes.yaml <<'EOF'
defaults:
  api_key: ${OPENROUTER_API_KEY}
  base_url: https://openrouter.ai/api/v1
  format: openai

routes:
  smart:
    type: random_routing
    strong:
      model: openai/gpt-4o
    weak:
      model: openai/gpt-4o-mini
    strong_probability: 0.3
    fallback_target_on_evict: strong
EOF

switchyard --routing-profiles routes.yaml -- configure
```

> **Format default and caching.** Omitting `format:` from a tier silently defaults to `OPENAI` (Chat Completions) — not `AUTO`. For Claude/Anthropic/Bedrock tiers this is wrong: set `format: anthropic` explicitly. The native `/v1/messages` path preserves `cache_control`, which is what enables prompt caching. `format: openai` routes Claude through OpenAI-format translation that strips `cache_control`: the request still succeeds, but caching silently never engages and you pay full input price. Always use `format: openai` for NIM/non-Claude models and `format: anthropic` for Claude and Bedrock models. Use `format: auto` only when the upstream is genuinely unknown.

Inspect what was saved:

```bash
switchyard configure --show          # redacted snapshot
switchyard configure --show --check  # also probes GET /models
```

---

## Path A: Server mode

Serves the saved routing bundle as a long-running proxy. Any client that speaks
OpenAI Chat Completions, Anthropic Messages, or OpenAI Responses API can connect.

```bash
switchyard serve
```

Test with curl:

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "smart", "messages": [{"role": "user", "content": "hello"}]}'
```

**CI pattern:** run `switchyard --routing-profiles routes.yaml -- configure` in
your environment setup, then `switchyard serve` in your service start step. No
flags needed at serve time.

> **Override (dev / one-off work):** pass `--routing-profiles` to use a different
> bundle for a session without overwriting your saved config:
> ```bash
> switchyard --routing-profiles dev.yaml -- serve --port 4001
> ```

---

## Path B: Agent launcher

Starts a proxy and spawns a coding agent against it in one command. The proxy
shuts down when the agent exits. The live stats footer shows per-tier token usage.

```bash
switchyard launch claude      # Claude Code
switchyard launch codex       # Codex CLI
switchyard launch openclaw    # OpenClaw
```

Each launcher reads the routing bundle and provider credentials saved by
`switchyard configure`. See [Agent Launchers](guides/agent_launchers.md) for
supported harness versions, model requirements, and Claude Code `/model` picker
aliasing.

> **Override (dev / one-off work):** pass `--routing-profiles` (global switchyard
> flag) or `--model` (launcher flag) to use a different bundle or single model for
> a session without changing your saved config (the two are mutually exclusive):
> ```bash
> switchyard --routing-profiles dev.yaml -- launch claude
> switchyard launch claude --model openai/gpt-4o
> ```

---

## Routing profiles

All route types work with both [Path A](#path-a-server-mode) and
[Path B](#path-b-agent-launcher). Declare a type in your YAML, run
`switchyard --routing-profiles routes.yaml -- configure`, then `serve` or `launch` as above.

### Choose a route type

This guide used `random_routing` so you can get a working proxy quickly. Choose
another route type when the routing decision needs different inputs:

| Algorithm | Use it when | Config |
|---|---|---|
| [Random Routing](routing_algorithms/random_routing.md) | You need a fixed strong/weak split for A/B tests or baselines. | `random_routing` |
| [LLM Classifier Routing](routing_algorithms/llm_classifier_routing.md) | Request content should decide whether to use `weak` or `strong`. | `deterministic` |
| [Cascade Routing](routing_algorithms/cascade_routing.md) | Tool-result and progress signals should route most turns without an extra classifier call. | `cascade` |

LLM classifier routes can also enable
[Session Affinity (Sticky Routing)](routing_algorithms/sticky_routing.md) to pin
multi-turn conversations to one tier.

A single YAML file can declare multiple routes. Each route becomes a model id on
`GET /v1/models`; the first declared route is the launcher's initial model. See
[Routing Overview](routing_algorithms/overview.md) for route selection and the
strategy-specific pages for full examples and tuning notes.

---

## Path C: Python library

Embed Switchyard directly in your application without a separate proxy process:

```python
import asyncio
from switchyard import ChatRequest, PassthroughProfileConfig, ProfileSwitchyard

switchyard = ProfileSwitchyard(PassthroughProfileConfig(
    api_key="sk-or-...",  # pragma: allowlist secret
    base_url="https://openrouter.ai/api/v1",
).build())

async def chat(user_message: str) -> str:
    request = ChatRequest.openai_chat({
        "model": "openai/gpt-4o",
        "messages": [{"role": "user", "content": user_message}],
    })
    response = await switchyard.call(request)
    return response["choices"][0]["message"]["content"]

print(asyncio.run(chat("What is 2+2?")))
```

To host the chain as an HTTP server:

```python
import uvicorn
from switchyard import PassthroughProfileConfig, ProfileSwitchyard, build_switchyard_app

switchyard = ProfileSwitchyard(PassthroughProfileConfig(
    api_key="sk-or-...",  # pragma: allowlist secret
    base_url="https://openrouter.ai/api/v1",
).build())
uvicorn.run(build_switchyard_app(switchyard), port=4000)
```

---

## Troubleshooting

**No API key / auth error**

```bash
switchyard configure          # re-run interactive setup to update credentials
switchyard configure --show   # confirm what key source is in use
```

For launchers and verification, you can pass `--api-key` directly. For `serve`, put credentials in the routing-profile YAML or saved config.

```bash
switchyard launch claude --api-key sk-...
switchyard verify --api-key sk-...
```

**Connection refused**

Check health: `curl http://localhost:4000/health`

**Telemetry header opt-out**

Switchyard adds an `X-Switchyard-Version` header to outbound LLM calls for
release attribution. No request or response content is included. To disable:

```bash
export SWITCHYARD_TELEMETRY_OPT_OUT=1
```

**Development setup**

```bash
git clone https://github.com/NVIDIA-NeMo/Switchyard.git
cd Switchyard
uv sync
source .venv/bin/activate
uv run pytest tests/ -v
uv run ruff check .
uv run mypy switchyard
```

---

## Next steps

- [CLI Reference](cli_reference.md): full flag reference for every verb
- [Agent Launchers](guides/agent_launchers.md): Claude Code, Codex, and OpenClaw launcher details
- [Architecture](architecture.md): system context and end-to-end request flow
- [Routing Overview](routing_algorithms/overview.md): choose the right routing strategy
- [Random Routing](routing_algorithms/random_routing.md): fixed strong/weak split routing
- [LLM Classifier Routing](routing_algorithms/llm_classifier_routing.md): classifier-driven strong/weak routing
- [Cascade Routing](routing_algorithms/cascade_routing.md): picker layers, signal dimensions, calibration
- [Sticky Routing](routing_algorithms/sticky_routing.md): conversation-level route affinity
