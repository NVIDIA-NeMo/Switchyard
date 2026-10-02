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

## Claude targets behind an OpenAI-compatible gateway

Some gateways, such as a LiteLLM proxy, serve Claude models on `/v1/chat/completions`,
`/v1/responses`, and `/v1/messages` with one API key. Switchyard calls the endpoint that
matches the target's LLM client `format`, whatever `api` pi uses. On the gateway tested
for this page, the `format` decided whether Claude prompts were cached and whether pi's
`--thinking` level worked.

Choose the Claude LLM client by who holds the gateway key:

| Who holds the gateway key | Claude LLM client | Result |
|---|---|---|
| The server, through `api_key_env` | `format = "anthropic_messages"` | Prompt caching and pi's `--thinking` level both work. Every caller's Claude requests use the server-owned key. |
| pi sends it as `apiKey`, and the route also forwards it to an OpenAI-format LLM client on the same gateway, such as a GPT judge | `format = "anthropic_messages"` with `forward_auth = true`, on the same scheme, host, and port as that client | Prompt caching and pi's `--thinking` level both work, and every request uses pi's own key. |
| pi sends it as `apiKey`, and the route forwards it only to Claude targets | `format = "openai_chat"`, with `omit_body_fields = ["reasoning_effort"]` on the target | Prompt caching works. pi's `--thinking` level has no effect, and Claude thinks at its default effort. The omitted field stays out only if the target's `extra_body` and `reasoning_effort` do not set it again. |

Do not use `format = "openai_responses"` for Claude targets on such a gateway. The
gateway tested for this page never cached Claude prompts on `/v1/responses`, and it
returned HTTP 400 when thinking was on (see [Thinking](#thinking)). In every setup, a GPT
judge on `openai_responses` can still use pi's forwarded key (see
[Forwarded keys](#forwarded-keys)).

### Prompt caching

On the LiteLLM gateway tested for this page, a repeated Claude prompt was read from the
cache on `/v1/chat/completions` and `/v1/messages`, but never on `/v1/responses`. Every
`/v1/responses` request counted the whole prompt as uncached input.

To check your gateway, start the server with `--routing-log-file PATH` and send the same
prompt twice. Claude does not cache short prompts, so use a prompt of at least 5,000
tokens. In the tests for this page, prompts of about 5,000 tokens were cached on Claude
Opus 5.5 and Sonnet 5; the tests did not find Claude's exact minimum. Then read the
records:

```bash
jq -c '{route_id, prompt_tokens, cached_tokens, cache_creation_tokens}' PATH
```

If the gateway caches the prompt, the first record shows it in `cache_creation_tokens`
and the second shows `cached_tokens` close to `prompt_tokens`. If the second record shows
`"cached_tokens": 0`, the gateway read nothing from the cache, and the whole prompt
counts as uncached input again.

#### Estimate the cost

The routing log records token counts, not prices. To estimate what a request cost,
multiply each token count in its record by the matching price, and add the results:

| Tokens in the record | Price |
|---|---|
| Uncached input: `prompt_tokens - cached_tokens - cache_creation_tokens` | Input price |
| `cached_tokens` | Cache-read price |
| `cache_creation_tokens` | Cache-write price |
| `completion_tokens` | Output price |

Anthropic's [pricing page](https://platform.claude.com/docs/en/about-claude/pricing)
sets the cache-read price at 0.1 times the input price for most models, including
Claude Sonnet 5, but at 0.05 times for Claude Opus 5.5. It sets the cache-write price at
1.25 times the input price for a 5-minute cache, or 2 times for a 1-hour cache. The
routing log does not record which cache lifetime the gateway used, and a gateway may
charge its own prices, so the result is an estimate, not the gateway's bill.

Put your prices in a `prices.json` file, in USD per million tokens. Key each entry by the
`model` value from the routing log. The example below uses Anthropic's list prices for
Claude Sonnet 5 on 2026-10-02, with the 5-minute cache-write price. Check the current
pricing page or your gateway's prices before you rely on the result.

```json
{
  "claude-sonnet-5": {"input": 2.00, "cache_read": 0.20, "cache_write": 2.50, "output": 10.00}
}
```

The command below prints one estimated cost per record. For a record whose model has no
entry in `prices.json`, it prints a warning instead of a cost:

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

Claude Opus 5.5 and Sonnet 5 accept only adaptive thinking. With `reasoning: true`, pi
sends `reasoning_effort`, even when you do not pass `--thinking`, and Switchyard
passes that field unchanged to an `openai_chat` target. Some gateways, including the one
tested for this page, turn `reasoning_effort` into Anthropic's older
`thinking: {type: "enabled"}` and return HTTP 400:

```text
"thinking.type.enabled" is not supported for this model. Use "thinking.type.adaptive" and "output_config.effort" to control thinking behavior.
```

An `openai_responses` target fails the same way. Switchyard sends the effort to it as
`reasoning.effort`, and the gateway returns the same error.

You have two options:

- Use an `anthropic_messages` LLM client. Switchyard turns pi's effort into
  `thinking: {type: "adaptive"}` and `output_config.effort`, so pi's `--thinking` level
  still applies.
- Keep the OpenAI-format LLM client and remove the effort field from requests to the
  target with [`omit_body_fields`](../reference/toml_schema.md#targetsname). Use the
  field name of the target's format: `reasoning_effort` on `openai_chat`, or `reasoning`
  on `openai_responses`. Switchyard applies the target's `extra_body` and
  `reasoning_effort` after `omit_body_fields`, so this works only if neither sets the
  field again.

  ```toml
  [targets.claude]
  id = "claude-opus-5-5"
  llm_client = "gateway_chat"  # format = "openai_chat"
  omit_body_fields = ["reasoning_effort"]
  ```

  The request then succeeds, and Claude thinks at its default effort. pi's `--thinking`
  level has no effect on this target.

### Forwarded keys

An LLM client with `forward_auth = true` sends the caller's key to the gateway. An LLM
client with `api_key_env` sends a server-owned key, which the server reads from an
environment variable (see
[`[llm_clients.<name>]`](../reference/toml_schema.md#llm_clientsname)). To forward pi's
key, replace the `apiKey` placeholder with the name of an environment variable that holds
your gateway key, with a leading `$`: `"apiKey": "$GATEWAY_API_KEY"`. Without the `$`,
pi sends the name itself as the key.

Only standalone `switchyard-server` forwards keys. The native Nemo Relay plugin rejects
routes that use `forward_auth = true` (see
[Request Handling](nemo_relay.md#request-handling)).

The LLM clients that forward the key decide which request APIs a route accepts:

- If they all use OpenAI formats (`openai_chat` or `openai_responses`), the route accepts
  `/v1/chat/completions` and `/v1/responses`, the two APIs pi uses.
- If they all use `anthropic_messages`, the route accepts only `/v1/messages`, and pi
  should not use that API (see [Which request API](#which-request-api)).
- If they mix the two, the route accepts `/v1/chat/completions` and `/v1/responses`. The
  server starts only if all of them use the same scheme, host, and port in `base_url`,
  as clients on one gateway do. Otherwise it prints an error that starts with
  `route <name> cannot forward both Anthropic and OpenAI caller credentials`. `<name>` is
  the route's `[routes.<name>]` table key, not its `id`.

So to forward pi's key and keep pi's `--thinking` level, the route needs an OpenAI-format
LLM client that also forwards the key on the same gateway, such as a GPT judge on
`openai_responses`. Then put the Claude targets on an `anthropic_messages` client with
`forward_auth = true` (see the example under
[`[llm_clients.<name>]`](../reference/toml_schema.md#llm_clientsname)).

A route without such a client cannot do both. To forward pi's key to Claude, put the
Claude targets on `openai_chat` with `omit_body_fields`. To keep `--thinking`, give them
an `anthropic_messages` client with `api_key_env`. Clients with `api_key_env` never limit
the request APIs: if no LLM client in the route sets `forward_auth = true`, the route
accepts every request API.
