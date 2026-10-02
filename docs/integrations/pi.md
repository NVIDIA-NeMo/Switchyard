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
  `forward_auth = true`. pi still needs some value here before it lists the model. To
  send your gateway key through a route that forwards it, see
  [Forwarded keys](#forwarded-keys).
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

## Claude through an LLM gateway

An LLM gateway, such as a LiteLLM proxy, can serve Claude on three endpoints:
`/v1/chat/completions`, `/v1/responses`, and `/v1/messages`. Switchyard calls the
endpoint that matches the `format` of the Claude target's LLM client, whichever `api` pi
uses. Give Claude targets an LLM client with `format = "anthropic_messages"`:

```toml
[llm_clients.gateway_claude]
format = "anthropic_messages"
base_url = "https://gateway.example.com"
api_key_env = "GATEWAY_API_KEY"

[targets.claude]
id = "claude-sonnet-5"  # the gateway's model ID
llm_client = "gateway_claude"
```

On the tested gateway, both OpenAI formats returned HTTP 400 when pi sent a thinking
level, and `openai_responses` never read the prompt cache:

| Claude LLM client `format` | Gateway endpoint | Prompt cache | pi's thinking level |
|---|---|---|---|
| `anthropic_messages` | `/v1/messages` | Read on repeated prompts | Works |
| `openai_chat` | `/v1/chat/completions` | Read on repeated prompts | HTTP 400 |
| `openai_responses` | `/v1/responses` | Never read | HTTP 400 |

[Prompt caching](#prompt-caching) and [Thinking](#thinking) explain both failures. To
use each developer's own gateway key instead of a key that the server holds, see
[Forwarded keys](#forwarded-keys).

### Prompt caching

pi sends the whole conversation on every turn. When the gateway reads the repeated part
from Claude's prompt cache, that part costs 0.1 times the input price on Claude Sonnet 5
and 0.05 times on Claude Opus 5.5, according to Anthropic's
[pricing page](https://platform.claude.com/docs/en/about-claude/pricing). The tested
gateway never read Claude prompts from the cache on `/v1/responses`, so every turn there
paid the full input price for the whole conversation.

To check your gateway, start the server with `--routing-log-file PATH` and send the same
prompt twice. Claude does not cache short prompts, so use a prompt of at least 5,000
tokens. Then read the records:

```bash
jq -c '{route_id, prompt_tokens, cached_tokens, cache_creation_tokens}' PATH
```

When caching works, the first record shows the prompt in `cache_creation_tokens`, and the
second shows `cached_tokens` close to `prompt_tokens`. If the second record shows
`"cached_tokens": 0`, the gateway read nothing from the cache.

#### Estimate the cost

The routing log records token counts, not prices. To estimate what a request cost,
multiply each count in its record by the matching price and add the results:

| Tokens in the record | Price |
|---|---|
| `prompt_tokens - cached_tokens - cache_creation_tokens` | Input |
| `cached_tokens` | Cache read |
| `cache_creation_tokens` | Cache write |
| `completion_tokens` | Output |

Write the prices to `prices.json` in USD per million tokens, and key each entry by the
record's `model` value. This example uses Anthropic's list prices for Claude Sonnet 5 on
2026-10-02:

```json
{
  "claude-sonnet-5": {"input": 2.00, "cache_read": 0.20, "cache_write": 2.50, "output": 10.00}
}
```

The example's `cache_write` price is for a 5-minute cache. A 1-hour cache costs 2 times
the input price instead of 1.25 times. The routing log does not say which cache the
gateway used. A gateway may also charge its own prices, so the result is an estimate, not
the gateway's bill. This command prints one estimated cost per record, or a warning for a
model that has no entry in `prices.json`:

```bash
jq -r --slurpfile prices prices.json '
  . as $r
  | ($prices[0][$r.model // ""]) as $p
  | if $p == null then
      "warning: no price for model \($r.model); add it to prices.json"
    else
      ((($r.prompt_tokens // 0) - ($r.cached_tokens // 0) - ($r.cache_creation_tokens // 0)) * $p.input
       + ($r.cached_tokens // 0) * $p.cache_read
       + ($r.cache_creation_tokens // 0) * $p.cache_write
       + ($r.completion_tokens // 0) * $p.output) / 1000000
      | "\($r.route_id) \($r.model) estimated $\(. * 1000000 | round / 1000000)"
    end' PATH
```

### Thinking

With `reasoning: true` on the model entry, pi sends a thinking level on every request,
even when you do not pass `--thinking`. Switchyard passes that level to an OpenAI-format
target in OpenAI form: `reasoning_effort` on `openai_chat` and `reasoning.effort` on
`openai_responses`. The tested gateway turned either field into Anthropic's older
`thinking: {type: "enabled"}`. Claude Opus 5.5 and Sonnet 5 refuse that form, so the
gateway returned HTTP 400:

```text
"thinking.type.enabled" is not supported for this model. Use "thinking.type.adaptive" and "output_config.effort" to control thinking behavior.
```

For an `anthropic_messages` target, Switchyard sends the level in the form that Claude
accepts: `thinking: {type: "adaptive"}` with `output_config.effort`. pi's thinking level
then takes effect.

If a Claude target must stay on an OpenAI format, remove the effort field from its
requests with [`omit_body_fields`](../reference/toml_schema.md#targetsname). Claude then
thinks at its default effort, and pi's thinking level has no effect on that target:

```toml
[targets.claude]
id = "claude-opus-5-5"
llm_client = "gateway_chat"  # format = "openai_chat"
omit_body_fields = ["reasoning_effort"]  # use "reasoning" on openai_responses
```

Switchyard applies the target's `extra_body` and `reasoning_effort` after the removal, so
either can set the field again.

### Forwarded keys

With `api_key_env`, every caller's requests use one gateway key that the server holds.
To use each developer's own gateway key instead, set `forward_auth = true` on the LLM
client. Then set pi's `apiKey` to `$` plus the name of the environment variable that
holds your key: `"apiKey": "$GATEWAY_API_KEY"`. Without the `$`, pi sends the name itself
as the key, and the gateway returns HTTP 401. Only standalone `switchyard-server` forwards
keys. The native Nemo Relay plugin rejects routes that use `forward_auth = true` (see
[Request Handling](nemo_relay.md#request-handling)).

Which request APIs a route accepts depends on its LLM clients that forward the key.
Clients with `api_key_env` do not count:

| Forwarding LLM clients in the route | Request APIs the route accepts |
|---|---|
| None | All three |
| Only OpenAI formats | `/v1/chat/completions` and `/v1/responses` |
| OpenAI formats and `anthropic_messages`, all on the same scheme, host, and port | `/v1/chat/completions` and `/v1/responses` |
| Only `anthropic_messages` | `/v1/messages`, which pi should not use (see [Which request API](#which-request-api)) |

For example, a classifier route can forward pi's key to its GPT judge on
`openai_responses` and to its Claude targets on `anthropic_messages` when both LLM clients
point at the same gateway. If they use different hosts, ports, or schemes, the server does
not start. If a route forwards the key only to Claude targets on `anthropic_messages`, pi
cannot use it. Put those targets on `openai_chat` with `omit_body_fields` (see
[Thinking](#thinking)), or let the server hold the key with `api_key_env`.
