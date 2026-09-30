# Use Switchyard with Oh My Pi

[Oh My Pi](https://github.com/can1357/oh-my-pi) (`omp`) is a coding agent based on pi.
It reads its model providers from `~/.omp/agent/models.yml`. Add Switchyard there and
`omp` sends every model call to `switchyard-server`. `omp` has no `--base-url` flag, and
setting `OPENAI_BASE_URL` does not change the address of its built-in `openai` provider.
This page was tested with Oh My Pi 18.1.21 against the
[Getting Started](../getting_started.md#server-path) server on `http://localhost:4000`
with route id `switchyard`.

## Configure

`~/.omp/agent/models.yml`:

```yaml
providers:
  switchyard:
    baseUrl: http://localhost:4000/v1
    api: openai-completions
    auth: none
    models:
      - id: switchyard
        name: Switchyard stage router
        reasoning: true
        input: [text, image]
        contextWindow: 200000
        maxTokens: 32000
```

- `auth: none` marks the provider as keyless. Switchyard ignores client keys unless an
  LLM client sets `forward_auth = true`. Without `auth: none`, `omp` refuses to send a
  request.
- `models[].id` must equal a route `id` from your TOML file. `contextWindow` and
  `maxTokens` set `omp`'s compaction limit and output cap. `reasoning: true` turns on the
  `--thinking` flag.
- `omp models find switchyard` shows the loaded entry.

### Or let omp list the routes

```yaml
providers:
  switchyard:
    baseUrl: http://localhost:4000/v1
    api: openai-completions
    auth: none
    discovery:
      type: openai-models-list
```

`omp` reads `GET /v1/models` and adds one model per route. For routes that declare
`context_window`, `omp` takes the context window from the `context_length` field that
the server reports. Other routes get `omp`'s default of 128K tokens. Discovered models
have no thinking controls and accept text only. List the models by hand when you need
`--thinking` or image input. `omp` caches the list for a day, so run `omp models refresh`
after you change the routes.

## Run

```bash
omp --model switchyard/switchyard
omp -p --model switchyard/switchyard "List the files in this directory."
```

`--thinking off|minimal|low|medium|high|xhigh` sets the reasoning level. Setting
`modelRoles.default: switchyard/switchyard` in `~/.omp/agent/config.yml` makes the route
the default model.

## Check the routing

The checks in [Use Switchyard with pi](pi.md#check-the-routing) work the same way for
`omp`. On the Chat Completions API, `omp` sends no session header for a custom provider,
so the routing log records `"session_id": null`. Routes with
`classify_trigger = "user_turn"` or `"new_session"`, advisor budgets, and the stage
router's `capable_hold_turns` then treat each request as its own session. If you need
per-session routing, use `anthropic-messages`. On that API `omp` sends the
`X-Claude-Code-Session-Id` header, which Switchyard reads as the session id.

## Which request API

| `api` | Endpoint | Use it when |
|---|---|---|
| `openai-completions` | `/v1/chat/completions` | Default. The targets use `format = "openai_chat"`, for example OpenRouter. |
| `openai-responses` | `/v1/responses` | The targets use `format = "openai_responses"`. `omp` sends `store: false` and the full history every turn. |
| `anthropic-messages` | `/v1/messages` | You need per-session routing. Set `baseUrl: http://localhost:4000` and keep `auth: none`. |

When the route's LLM client uses the same format as the request, Switchyard forwards the
body unchanged except for `model`. When the client uses another format, Switchyard
translates the request. Unlike pi, `omp` keeps the local model id `switchyard` on stored
messages, so the served target's name in the response does not affect its thinking
replay or compaction.

Port 4000 is also the default port for `omp`'s `litellm` provider and for
`omp auth-gateway`. If `LITELLM_API_KEY` is set, `omp` probes `http://localhost:4000/v1`
as a LiteLLM proxy. In that case, run Switchyard on another port or set
`LITELLM_BASE_URL`. Set `cost` on the model entry if you want `omp` to show a non-zero
cost.

### Claude targets behind an OpenAI-compatible gateway

The [pi guide's notes on Claude targets behind a gateway](pi.md#claude-targets-behind-an-openai-compatible-gateway)
apply to `omp` too. The target's LLM client `format`, not the `api` that `omp` uses,
decides which gateway endpoint Switchyard calls and so whether the gateway caches the
prompt. Prefer `format = "openai_chat"` or `"anthropic_messages"` for Claude targets,
because a gateway may not cache Claude prompts on `/v1/responses`. Thinking depends on
both the `format` and the `api`, as the next paragraph explains.

With thinking on, `omp` sends a reasoning effort on `openai-completions` and
`openai-responses`. Switchyard passes it to an `openai_chat` target as
`reasoning_effort`. Some gateways turn that field into a thinking setting that Claude
Opus 5.5 and Sonnet 5 reject with HTTP 400. Switchyard turns the effort into adaptive
thinking only when it translates a Chat Completions or Responses request for an
`anthropic_messages` target. When `omp` uses `anthropic-messages`, Switchyard forwards
the request to that target with `omp`'s own `thinking` settings unchanged.

A route that forwards the caller's key to an `anthropic_messages` client accepts
requests only on `/v1/messages`, and it cannot also forward the key to an
`openai_chat` or `openai_responses` client. So a route with an OpenAI-format target,
such as a GPT judge on `openai_responses`, cannot forward the caller's key to
both the judge and Claude targets on `anthropic_messages`. Choose one of two setups:

- Forward the key to every client, and keep the Claude targets on `openai_chat` with
  `omit_body_fields = ["reasoning_effort"]`. The Claude models then think at their
  default effort, and `--thinking` has no effect on them.
- Forward the key only to the GPT judge, and put the Claude targets on an
  `anthropic_messages` client that reads a key held by the server from `api_key_env`.
  Switchyard then turns the effort into adaptive thinking. The route accepts only
  `/v1/chat/completions` and `/v1/responses`, so use `openai-completions` or
  `openai-responses`.
