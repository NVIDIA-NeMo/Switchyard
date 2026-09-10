# Use Verification-Gated Routing from an Agent

Verification-Gated Routing (VGR) runs in `switchyard-server` as an
OpenAI- and Anthropic-compatible proxy. An agent does not need a VGR plugin:
point the agent at Switchyard, select the VGR route ID as its model, and keep a
stable session ID across one conversation.

## 1. Configure the route

Create `routes.toml`. This example uses one OpenRouter client for both tiers,
but the local and cloud targets can use different clients and wire formats.

```toml
schema_version = 1

[llm_clients.openrouter]
format = "openai_chat"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"

[targets.local]
id = "openai/gpt-4o-mini"
llm_client = "openrouter"

[targets.cloud]
id = "openai/gpt-4o"
llm_client = "openrouter"

[routes.vgr]
id = "switchyard/vgr"
type = "vgr"
local_target = "local"
cloud_target = "cloud"
mode = "active"
active_approval = "prospective-validation-and-canary-approved"
deadline_seconds = 30
task_typing = true
local_supports_images = false

# Advertise capabilities to agents that discover models through /v1/models.
context_window = 128000
tool_calling = true
reasoning = true
vision = true
```

Active mode serves the readiness-gated VGR decision: a verified local answer
is committed, while missing or failed verification escalates to the cloud
tier. The exact `active_approval` attestation is required for the server to
accept active mode.

The TOML file names the environment variable containing the provider key. Do
not put the key itself in the file. See
[Verification-Gated Routing](routing_algorithms/vgr_routing.md) for all route
options and readiness requirements.

## 2. Start Switchyard

Build from a repository checkout:

```bash
export OPENROUTER_API_KEY="<provider-key>"

cargo build --locked --release -p switchyard-server
./target/release/switchyard-server --config routes.toml --dry-run
./target/release/switchyard-server \
  --config routes.toml \
  --host 127.0.0.1 \
  --port 4000 \
  --routing-log-file routing_requests.jsonl
```

Keep the server bound to loopback for local use. If other machines must reach
it, put it behind authentication and TLS; the agent-facing API is not an
internet-facing authentication boundary.

Confirm that the route is available:

```bash
curl --fail http://127.0.0.1:4000/health
curl --fail http://127.0.0.1:4000/v1/models
```

## 3. Point the agent at VGR

Use these values in any agent that accepts a custom OpenAI-compatible provider:

```text
Base URL: http://127.0.0.1:4000/v1
Model:    switchyard/vgr
API key:  any non-empty placeholder, if the agent requires one
```

For agents configured through OpenAI environment variables:

```bash
export OPENAI_BASE_URL="http://127.0.0.1:4000/v1"
export OPENAI_API_KEY="switchyard-local"
export OPENAI_MODEL="switchyard/vgr"
```

`switchyard-local` is only a placeholder for clients that reject an empty API
key. Upstream credentials remain in the Switchyard server environment unless
the route explicitly enables `forward_auth`.

For an Anthropic-native agent, use the Anthropic endpoint instead:

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:4000"
export ANTHROPIC_API_KEY="switchyard-local"
export ANTHROPIC_MODEL="switchyard/vgr"
```

Switchyard accepts all three supported agent-facing APIs:

| Agent protocol | Base URL | Endpoint |
|---|---|---|
| OpenAI Chat Completions | `http://127.0.0.1:4000/v1` | `/chat/completions` |
| OpenAI Responses | `http://127.0.0.1:4000/v1` | `/responses` |
| Anthropic Messages | `http://127.0.0.1:4000` | `/v1/messages` |

If the agent runs inside Docker, `127.0.0.1` refers to the agent container.
Use `http://host.docker.internal:4000` instead. On Linux, the container may
also need `--add-host host.docker.internal:host-gateway`.

## 4. Preserve conversation state

VGR retains a selected tier across tool-result continuations and re-evaluates
the next user turn. For this to work, every request in one conversation must
send the same header:

```text
x-switchyard-session-id: <stable-conversation-id>
```

Use a different value for each independent conversation. Without this header,
requests still route, but continuation affinity and session-level latching
cannot reliably carry across calls.

The agent must also resend the normal conversation history, including
assistant tool calls and their corresponding tool-result messages. VGR derives
tool evidence from that normalized transcript; it does not execute the tools
itself.

## 5. Verify routing

Send a direct smoke request before starting the agent:

```bash
curl --fail-with-body http://127.0.0.1:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-switchyard-session-id: setup-smoke-1" \
  -d '{
    "model": "switchyard/vgr",
    "messages": [
      {"role": "user", "content": "Reply with the word ready."}
    ]
  }'
```

Inspect:

- `x-model-router-selected-model` on the response for the serving target.
- `GET /v1/stats` for bounded VGR decision totals.
- `routing_requests.jsonl` for per-request session, tier, model, token, and VGR
  decision evidence.

## Troubleshooting

**The agent reports an unknown model**

Set its model to the route `id` (`switchyard/vgr` above), not either upstream
target ID. Confirm the route appears in `GET /v1/models`.

**The agent cannot connect**

Check `/health`, the configured port, and whether the agent is on the host or
inside a container. Use `host.docker.internal` from Docker.

**Switchyard reports a missing credential**

Export the variable named by `api_key_env` before starting the server, then
repeat `switchyard-server --config routes.toml --dry-run`.

**Tool continuations are re-evaluated**

Confirm that every request carries the same `x-switchyard-session-id` and that
the agent preserves assistant tool calls and tool-result messages in history.

**An attached image disappears before reaching Switchyard**

Agents such as Codex use `/v1/models` capability metadata. Set `vision = true`
only when the route can safely accept images. If the local tier is text-only,
keep `local_supports_images = false` so VGR bypasses it for image requests.
