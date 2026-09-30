# Use Switchyard with pi

[pi](https://github.com/earendil-works/pi) reads its model providers from
`~/.pi/agent/models.json`. Add Switchyard there and pi sends every model call to
`switchyard-server`, which picks the target model for each call. pi does not read
`OPENAI_BASE_URL`, so a provider entry is how you point pi at a proxy. This
page was tested with pi 0.84.3 against the
[Getting Started](../getting_started.md#server-path) server on `http://localhost:4000`
with route id `switchyard`.

## Configure

`~/.pi/agent/models.json`:

```json
{
  "providers": {
    "switchyard": {
      "baseUrl": "http://localhost:4000/v1",
      "api": "openai-completions",
      "apiKey": "switchyard",
      "compat": {
        "supportsDeveloperRole": false,
        "sendSessionAffinityHeaders": true,
        "sessionAffinityFormat": "openrouter"
      },
      "models": [
        {
          "id": "switchyard",
          "name": "Switchyard stage router",
          "reasoning": true,
          "input": ["text", "image"],
          "contextWindow": 200000,
          "maxTokens": 32000
        }
      ]
    }
  }
}
```

- `models[].id` must equal a route `id` from your TOML file. Add one entry per route.
- `apiKey` is a placeholder. Switchyard ignores client keys unless an LLM client sets
  `forward_auth = true`. pi still needs some value here before it lists the model.
- `contextWindow` and `maxTokens` set pi's compaction limit and output cap. pi does not
  read these values from the server. Use the smallest context window among the route's
  targets.
- `reasoning: true` turns on the `--thinking` flag. pi then sends `reasoning_effort`.
- `supportsDeveloperRole: false` keeps the system prompt in the `system` role, which
  every upstream provider accepts.
- `sendSessionAffinityHeaders: true` with `sessionAffinityFormat: "openrouter"` makes pi
  send the `x-session-id` header. Switchyard reads that header as the session id. Routes
  with `classify_trigger = "user_turn"` or `"new_session"`, advisor budgets, the stage
  router's `capable_hold_turns`, and `GET /v1/routing/session-stats` all depend on the
  session id.

## Run

```bash
pi --provider switchyard --model switchyard
pi -p --provider switchyard --model switchyard "List the files in this directory."
```

`--thinking off|minimal|low|medium|high|xhigh` sets the reasoning level that pi sends.

## Check the routing

The command `curl -s localhost:4000/v1/stats | jq '.models | map_values({calls, prompt_tokens})'`
lists each target model with its call count. Every response carries the header
`x-model-router-selected-model`. With `--routing-log-file PATH`, the server writes one
record per call. In the record below, `session_id` is the value of pi's `x-session-id`
header:

```json
{"route_id":"switchyard","algorithm":"stage_router","model":"azure/anthropic/claude-haiku-4-5","session_id":"01a0caac-e421-7328-adaf-79d3440c0406","prompt_tokens":2659,"completion_tokens":97}
```

## Which request API

When the route's LLM client uses the same format as the request, Switchyard forwards the
body unchanged except for `model`. When the client uses another format, Switchyard
translates the request.

| `api` | Endpoint | Use it when |
|---|---|---|
| `openai-completions` | `/v1/chat/completions` | Default. The targets use `format = "openai_chat"`, for example OpenRouter. |
| `openai-responses` | `/v1/responses` | The targets use `format = "openai_responses"`. pi sends `store: false` and the full history every turn. Keep `sessionAffinityFormat: "openrouter"`. |
| `anthropic-messages` | `/v1/messages` | Do not use it with pi. See below. |

Do not use `anthropic-messages` with pi. Switchyard puts the served target's id in the
response `model` field, and pi's Anthropic client stores that id on the assistant
message. When routing picks another target on the next turn, pi treats the change as a
model switch: it drops thinking signatures and turns off overflow compaction. pi's OpenAI
clients keep the local id `switchyard` instead.

Set `cost` on the model entry if you want pi to show a non-zero cost.
[`benchmark/run-baseline.sh`](../../benchmark/README.md) runs Terminal-Bench tasks with
pi through Switchyard when you pass `--agent pi`.

### Claude targets behind an OpenAI-compatible gateway

Some gateways, such as a LiteLLM proxy, serve Claude models on `/v1/chat/completions`,
`/v1/responses`, and `/v1/messages` with one API key. The target's LLM client `format`
decides which endpoint Switchyard calls. That choice can change prompt caching and
thinking for Claude.

**Prompt caching.** On the LiteLLM gateway tested for this page, a repeated Claude prompt
was read from the cache on `/v1/chat/completions` and `/v1/messages`, but never on
`/v1/responses`. Every request to `/v1/responses` paid for the whole prompt again. Use
`format = "openai_chat"` or `"anthropic_messages"` for Claude targets. To check your
gateway, start the server with `--routing-log-file PATH` and send the same long prompt
twice. Claude does not cache short prompts. A prompt of at least 5,000 tokens is a safe
size; it is not Claude's exact minimum. Then read the records:

```bash
jq -c '{route_id, prompt_tokens, cached_tokens, cache_creation_tokens}' PATH
```

If the gateway caches the prompt, the first record shows it in `cache_creation_tokens`
and the second shows `cached_tokens` close to `prompt_tokens`. Writing the cache makes
the first request cost more, and each later request that reuses the prompt costs much
less. If the second record shows `"cached_tokens": 0`, the gateway billed the whole
prompt again.

**Thinking.** Claude Opus 5.5 and Sonnet 5 accept only adaptive thinking. Some gateways
turn the Chat Completions `reasoning_effort` field into Anthropic's older
`thinking: {type: "enabled"}`, and the model then returns HTTP 400:

```text
"thinking.type.enabled" is not supported for this model. Use "thinking.type.adaptive" and "output_config.effort" to control thinking behavior.
```

With `reasoning: true`, pi sends `reasoning_effort`, and Switchyard passes it to an
`openai_chat` target. You have two options:

- Use an `anthropic_messages` target. Switchyard turns the requested effort into
  `thinking: {type: "adaptive"}` and `output_config.effort`.
- Keep the `openai_chat` target and drop the field:

  ```toml
  [targets.claude]
  id = "claude-opus-5-5"
  llm_client = "gateway_chat"  # format = "openai_chat"
  omit_body_fields = ["reasoning_effort"]
  ```

  The request then succeeds, and the model thinks at its default effort. pi's
  `--thinking` level has no effect on this target.

**Forwarded keys.** Every LLM client in one route that sets `forward_auth = true` must
use the same API family: `openai_chat` and `openai_responses`, or `anthropic_messages`.
Otherwise the server does not start and prints
`route <name> cannot forward both Anthropic and OpenAI caller credentials`, where
`<name>` is the route's `[routes.<name>]` table key, not its `id`. A route that
forwards the caller's key to an `anthropic_messages` client also accepts requests only
on `/v1/messages`, and pi should not use that API. So when Switchyard forwards pi's key
to the Claude targets, use `openai_chat` Claude targets with `omit_body_fields`.

An `anthropic_messages` client that reads the key from `api_key_env` works with every
request API, as long as no other LLM client in the route sets `forward_auth = true`. If
one does, the forwarded client limits the route's endpoints. For example, a route with a
forwarded `openai_responses` client accepts only `/v1/chat/completions` and
`/v1/responses`, and returns HTTP 400 on `/v1/messages`. pi uses those APIs, so such a
route can forward pi's key to a GPT judge on `openai_responses` while its Claude targets
use `anthropic_messages` with a key that the server holds.
