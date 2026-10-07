# Display Switchyard session costs

Switchyard can price each observed call using the model that served it. Its session
report includes the answer and the classifier or judge calls used to route it.
The example displays below show this report in Claude Code, pi, and Oh My Pi.

## Configure rates

Start the server with a routing log and an optional pricing file:

```bash
switchyard-server --config routes.toml --routing-log-file routing.jsonl --pricing-file prices.json
```

The JSON file maps the served model IDs in the routing log to USD per million
tokens. These are illustrative rates; supply your gateway's rates:

```json
{
  "upstream/efficient": {
    "input": 1,
    "output": 5,
    "cache_read": 0.1,
    "cache_write": 2,
    "cache_write_1h": 3
  },
  "upstream/capable": {
    "input": 10,
    "output": 20,
    "cache_read": 0.2,
    "cache_write": 3
  }
}
```

The four base rates are required. `cache_write_1h` is optional. The server rejects
unknown fields, duplicate model IDs, empty or padded IDs, and rates outside
0 to 10000. Explicit zero rates mean free usage. Missing model rates mean unknown
cost. Model IDs must identify a single rate table across gateways; use distinct
served IDs when the same model has different gateway rates. Rates load at startup.
Restart to load changes; saved records keep the cost calculated when the call
completed.

A valid upstream `usage.cost` charge in USD takes precedence over an estimate,
even when it is zero. Without a charge, Switchyard multiplies the separate input,
cache-read, cache-write, and output token counts by the configured rates.
Reasoning tokens are already part of output usage and are not added again.
When the provider reports one-hour cache-write tokens, their rate is required.
When it does not report cache duration, the estimate uses `cache_write` for all
writes. Supply rates that match the requested tier; this table does not apply
long-context or service-tier surcharges automatically.

You can omit `--pricing-file` to record provider-reported charges alone.
Translation preserves charges and reported one-hour cache-write counts across
Chat Completions, Anthropic Messages, and Responses. Translated Chat Completions
responses emit cache writes as `prompt_tokens_details.cache_write_tokens`, which pi and Oh My Pi read.
The decoder also accepts the older `cache_creation_tokens` field. Same-format
raw response replay preserves the upstream JSON.

## Read a session report

The caller must send a session header that Switchyard recognizes on its model
requests, such as `x-switchyard-session-id: my-session`. Read that session with:

```bash
curl 'http://localhost:4000/v1/routing/session-stats?session_id=my-session'
```

The response adds `cost`, `answer_cost`, and `routing_cost`, plus `cost` for each
model. Each total has `known_usd`, `total_usd`, `unknown_calls`, `estimated_calls`,
and `provider_reported_calls`. `total_usd` is null if any observed call has no
known cost. `known_usd` remains the subtotal of priced calls. Old records without
cost fields contribute unpriced calls. The routing log stores each priced call's
`cost` as an object, such as `{ "usd": 0.01, "source": "estimated" }`. Its `source`
is `"estimated"` or `"provider_reported"`; an unpriced call has `"cost": null`.

Totals cover recorded calls. Failed answer, classifier, and stream calls that
reach the server's observation code are unpriced. Internal HTTP retries,
cancelled requests, and streams abandoned before observation completes can
leave charges unrecorded. The report cannot establish the provider's bill.
Session totals include all recorded branches and compaction calls carrying the
same session ID; a CLI's active-branch total can cover a different set of calls.
The pricing table stays in memory for the routing log writer's lifetime and does
not grow per call. Its size follows the pricing file. The log grows with recorded
calls; this option adds no rotation or size limit. Separate replica logs produce
separate totals. If a log append fails, the server warns and continues serving;
missed records are not recovered automatically. The session endpoint scans the
log on each read. The displays fetch on agent
completion, rather than polling in the background.

## Claude Code

Use the Python example as a status-line command. Set the repository path and the
server address in Claude Code's settings:

```json
{
  "statusLine": {
    "type": "command",
    "command": "python3 /path/to/Switchyard/examples/costs/claude_statusline.py --base-url http://localhost:4000"
  }
}
```

The command reads `session_id` from Claude Code's status-line JSON input and
prints the Switchyard total, including routing cost. If you already have a
status-line command, call the example from that command to retain your display.
You can also run it from a shell with `--session-id my-session`.

Claude Code's native cost display computes prices locally. Its managed
[`modelPricing`](https://code.claude.com/docs/en/settings-reference#modelpricing)
setting supports model and gateway-alias rate tables in version 2.1.242 or later.
Claude ignores that setting in ordinary user settings and `--settings`.
A fixed rate under a routing alias cannot describe differently priced targets or
include hidden judge calls. The Switchyard status line displays a separate total;
it does not change Claude Code's native cost or budget enforcement.

## pi

Load the example extension with your existing Switchyard provider configuration:

```bash
pi -e /path/to/Switchyard/examples/costs/pi.ts --provider switchyard --model switchyard
```

The extension sends `x-switchyard-session-id` and shows a separate footer after
each agent turn. Set `SWITCHYARD_BASE_URL` when the server is elsewhere. Keep the
helper `display.ts` beside `pi.ts` when copying the example.

pi's OpenAI adapters price the local model entry from its `cost` table. The
response's served model ID or `usage.cost` does not replace that table. A fixed
rate is useful for a fixed target; a router alias needs a separate display for
per-call target pricing. Keep the OpenAI request API described in the
[pi guide](pi.md#which-request-api).

## Oh My Pi

Load the example with an `anthropic-messages` Switchyard provider:

```bash
omp -e /path/to/Switchyard/examples/costs/omp.ts --model switchyard/switchyard
```

Oh My Pi sends a session header on that API, so the footer can look up the
session report. The current custom-provider OpenAI adapters send no session
header; this example cannot show their session totals. Use the configuration in
[the Oh My Pi guide](oh_my_pi.md#which-request-api). Keep `display.ts` beside
`omp.ts`, and set `SWITCHYARD_BASE_URL` for another server address.

Oh My Pi's custom `switchyard` provider uses configured model rates for its
native cost. Its supported OpenRouter integration can instead use `usage.cost`.
Returning the same field to a provider named `switchyard` does not activate that
behavior. The footer leaves native cost and budget enforcement unchanged.

## Codex

Codex's token display does not accept a dollar rate from Switchyard. Use the
session report or the Python command with an explicitly supplied session ID.
Per-request pricing still helps compare routed workloads even when the caller
has no native dollar display.
