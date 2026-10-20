# Model Selection

Switchyard routes each request to one of the models you configure. The models
you put in each role change results more than any threshold does. This page says
what each role needs, how to check a model before you route to it, which pairs
have measured results, and the mistakes we have seen in real runs.

## Roles

Each strategy names its models by role. The TOML keys differ by route type:

| Route `type` | Serves most turns | Takes the hard turns | Decides |
|---|---|---|---|
| `auto`, `stage_router`, `composite` | `efficient_target` | `capable_target` | Tool-result signals. `composite` adds a classifier. |
| `plan_execute` | `efficient_target`, after the first edit | `capable_target`, while planning | The first file edit. |
| `llm_classifier` (capability, escalation) | `weak_target` | `strong_target` | The judge at `classifier_target`. |
| `llm_classifier` (custom) | Your target names | Your target names | The judge in `models.judge`. |
| `advisor` | `executor_target` | `advisor_target` reviews. It does not serve turns. | The gate rules. |

A role is not a property of a model. The same model can be the capable tier in
one route and the efficient tier in another. See
[Core Concepts](../core_concepts.md#routing-algorithms).

What each role needs:

- **Capable, strong, or advisor.** Your best model on this workload. Every
  escalation lands here. Routing cannot solve a task this model cannot solve.
- **Efficient, weak, or executor.** Serves most turns. It must call the agent's
  tools correctly, follow the output format, hold the agent's context, and still
  complete tasks. Judge it by the tasks it completes on your workload, not by its
  price.
- **Judge or classifier.** Returns schema-valid JSON on every call, and does so
  quickly. See
  [Judge model compatibility](llm_classifier_routing.md#judge-model-compatibility).

## Check the efficient model before you route

Run these checks before you measure routing. Add a `passthrough` route for the
efficient model to the file from
[Getting Started](../getting_started.md#server-path), start the server with
`--routing-log-file routing.jsonl`, and send the checks to that route:

```toml
[routes.efficient_only]
id = "efficient-only"
type = "passthrough"
target = "weak"
```

### 1. Tool calling

Send a request with one tool and read the call back:

```bash
curl -s http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "efficient-only",
    "messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
    "tools": [{"type": "function", "function": {
      "name": "get_weather",
      "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
  }' | jq '.choices[0].message.tool_calls[0].function.arguments | fromjson'
```

Pass: the output is a JSON object with a `city` key. Fail: `null`, or a
`fromjson` error. Then run your agent against this route on one short task.
Models that pass with one tool can produce malformed calls once the agent sends
its full tool list. The agent reports those as invalid tool calls, or it repeats
the same step.

### 2. Structured output

Judges, and some agents, need JSON that matches a schema:

```bash
curl -s http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "efficient-only",
    "messages": [{"role": "user", "content": "Is 7 a prime number?"}],
    "response_format": {"type": "json_schema", "json_schema": {"name": "verdict", "strict": true,
      "schema": {"type": "object", "properties": {"prime": {"type": "boolean"}},
                 "required": ["prime"], "additionalProperties": false}}}
  }' | jq '.choices[0].message.content | fromjson'
```

Pass: `{"prime": true}`. An HTTP 400 means the provider does not offer JSON
Schema mode for this model. For a judge, set
`response_format_type = "json_object"` on the `llm_classifier` route. See
[Judge model compatibility](llm_classifier_routing.md#judge-model-compatibility).

### 3. Context window

The efficient model must hold the agent's largest prompt. Find that size after
a run through the passthrough route:

```bash
jq -s 'map(.prompt_tokens) | max' routing.jsonl
```

Compare it with the context length your provider lists for the model.
OpenRouter reports it as `context_length` in `GET https://openrouter.ai/api/v1/models`.
Set the route's `context_window` to the smaller of the two tiers' limits so
clients that read it compact in time. The server does not enforce it. See the
[route keys](../reference/toml_schema.md#routesname).

### 4. Reasoning

A model that reasons at length on every turn is slow and costly in the
efficient role. Read the reasoning share from the log:

```bash
jq -c '{model, completion_tokens, reasoning_tokens}' routing.jsonl
```

Providers that do not report reasoning tokens leave the field at `0`. When
reasoning tokens dominate on simple turns, set `reasoning_effort` on the target
for `openai_chat` and `openai_responses` clients, or use the provider's own
control through `extra_body`. See [Reasoning effort](#reasoning-effort).

### 5. Wire format

`format` on the LLM client must match the endpoint. `--dry-run` checks the file
only. It does not contact the provider. Send one request to the passthrough
route and confirm an HTTP 200 with a `choices` array. The same model can also
behave differently on different endpoints. See
[Pin the provider and endpoint](#pin-the-provider-and-endpoint).

## Measured pairs

See [Benchmarks](../benchmarks.md) for consolidated Terminal-Bench 2.1 results
and config links.

These pairs have results from checked-in profiles or a public report. Each row
describes one pair on one workload. The thresholds and results do not transfer
to another pair or another workload. The profile headers give the serving stack,
agent version, and scoring rule for each run, and absolute results vary by
serving stack.

| Efficient tier | Capable tier | Strategy | Workload | Result | Single-model anchors |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B, also the judge | Claude Opus 4.7 | `llm_classifier` custom, [balanced](../../benchmark/routing-profiles/tau2-telecom-custom-opus-qwen-balanced.toml) and [aggressive](../../benchmark/routing-profiles/tau2-telecom-custom-opus-qwen-aggressive.toml) | tau2-bench telecom, 114 tasks | 0.903 ± 0.071 solve rate with 45% of turns on the weak tier. 0.891 ± 0.029 with about 85%. | Within noise of the strong tier, per the profile headers |
| Nemotron 3.5 Lightning | Claude Opus 4.8, with Gemini 3.1 Flash-Lite as judge | `llm_classifier` escalation, `confirmations = 2` | LangChain Deep Agents suite, 145 tasks, 5 runs ([report](https://www.langchain.com/blog/switchyard-agent-routing-benchmark)) | 80.0% at $3.00 per run, with 7% of calls on Opus | Opus alone 86.0% at $11.45. Nemotron alone 77.7% at $0.72. |

What the rows show about model choice:

- The gap between the two single models sets the room for routing. Compare the
  fixed-model baselines with the routing results, as in the [Terminal-Bench 2.1 results](../benchmarks.md#terminal-bench-21).
- When the efficient model is close to the capable model, escalation keeps most
  of the accuracy at a fraction of the cost. On the LangChain suite the two
  single models were 8 points apart. Routing kept 80 of the 86 points at about
  a quarter of the cost.
- The judge's cost counts. In the LangChain runs the judge took 21.2% of spend.
  Choose a judge that is small and fast, and include it in the cost per run.

Other checked-in configurations:

- [`tb-lite-llm-classifier-opus-kimi-gemini.toml`](../../benchmark/server-configs/tb-lite-llm-classifier-opus-kimi-gemini.toml)
  (Claude Opus 4.7 strong, Kimi K2.7 code weak, Gemini 3.5 Flash judge) is a
  smoke test. It has no published result.

## Counter-examples

These cases came from users' runs. Each describes one pair on one workload.

- **The efficient model was too weak.** A first stage-router run scored below
  the single capable model. The efficient tier failed tasks the capable model
  solves, and the signals escalated too late to recover them. Check: run the
  [fixed arms](#check-for-an-inverted-pair) first. When the efficient-only arm
  is far below the capable-only arm, routing has to escalate a large share of
  turns to close the gap, and the savings shrink.
- **The pair was inverted.** On one coding benchmark the model in the efficient
  tier solved more tasks than the model in the capable tier. Every escalation
  then cost more and solved less. Check:
  [the inversion check](#check-for-an-inverted-pair).
- **The tool list broke the efficient model.** A 4-billion-parameter local model
  called tools correctly in a small test and produced malformed calls once the
  agent sent its full list of 27 tools. Check: [tool calling](#1-tool-calling)
  with the agent itself, not only with a one-tool request.
- **Switching cost more than it saved.** A pair matched the single capable
  model's accuracy but cost more. Each switch between models lost the prompt
  cache, so the next turn paid full price for the whole prompt, and the
  efficient model kept too little of the traffic to make up for it. Check:
  `cached_tokens` in the routing log on the turn after a model change, and cost
  per run for the routed arm against the fixed arms, not per-token prices.

## Check for an inverted pair

Symptom: the routed run scores below the efficient-only run, or `GET /v1/stats`
shows most turns on the capable tier while results do not improve.

Detection: run both models alone on the same tasks. Two `passthrough` routes in
one file do this:

```toml
[routes.capable_only]
id = "capable-only"
type = "passthrough"
target = "strong"

[routes.efficient_only]
id = "efficient-only"
type = "passthrough"
target = "weak"
```

Run your agent against each route ID on the same set of tasks, at least 20.
Count solved tasks per arm. Read token use per arm from the routing log:

```bash
jq -s 'group_by(.route_id) | map({route: .[0].route_id, calls: length,
  prompt_tokens: (map(.prompt_tokens) | add),
  completion_tokens: (map(.completion_tokens) | add)})' routing.jsonl
```

Multiply the totals by each model's prices to get cost per arm. The pi
integration page shows [one way to do that](../integrations/pi.md#estimate-the-cost).

Fix: if the efficient-only arm solves at least as many tasks as the capable-only
arm, swap the two roles, or serve the efficient model alone with `passthrough`.
If both arms solve the same tasks, routing has little to gain on this workload.
The room for routing is the set of tasks only the capable model solves.

## Pin the provider and endpoint

A model ID does not fix how the model is served. Three things move results
between runs.

**Provider.** Through an aggregator, one model ID can be served by several
providers with different quantization, context length, and tool-call support.
Symptom: the same configuration gives different results on different days, and
the routing log's `model` field does not change. Detection: the aggregator's
response metadata or dashboard shows which provider served each call. Fix: pin
the provider in the target's `extra_body`. On OpenRouter, `order` names the
providers to try and `allow_fallbacks = false` stops it from trying others:

```toml
[targets.weak]
id = "qwen/qwen3-35b-a3b"
llm_client = "openrouter"
extra_body = { provider = { order = ["your-provider"], allow_fallbacks = false } }
```

Use the provider slug from the model's page on OpenRouter.

**Endpoint and wire format.** The same model can behave differently on
`openai_chat`, `openai_responses`, and `anthropic_messages`. One measured case
is Claude behind an LLM gateway. On `anthropic_messages` the thinking level
worked and the prompt cache was read. On the two OpenAI formats a thinking level
returned HTTP 400, and `openai_responses` never read the cache. See
[Claude through an LLM gateway](../integrations/pi.md#claude-through-an-llm-gateway).
Record `format` and `base_url` with every result.

**Streamed token counts on `/v1/messages`.** When a client uses Anthropic
Messages and the selected target is on an OpenAI format, the streamed
`message_start` event reports `input_tokens: 0`. The counts the upstream
reports arrive in the final `message_delta` event. Symptom: the client shows zero prompt tokens, or a
context meter that never moves, for turns served by those targets. Detection:
compare the client's numbers with `prompt_tokens` in the routing log, which
records the upstream's counts. Fix: read usage from `message_delta`, or use the
routing log and `GET /v1/stats` for accounting.

### Reasoning effort

Effort changes a model's results and cost as much as the model choice does.
Record it with every result. Three rules apply:

- On `openai_responses` the effort field is `reasoning.effort`. A top-level
  `reasoning_effort` key in `extra_body` does not set it. Use the target's
  `reasoning_effort` key instead. It writes `reasoning.effort` on
  `openai_responses` and `reasoning_effort` on `openai_chat`, replacing the value
  the client sent. The DeepSWE profiles above use it to run both tiers at `max`.
- `reasoning_effort` is rejected on `anthropic_messages` clients. The server does
  not start:

    ```text
    target weak reasoning_effort is only supported on openai_chat and openai_responses clients
    ```

    For those targets the effort the client sends is passed through in
    Anthropic's form.

- One route cannot hold the same model at two efforts. Two targets with the same
  model ID on one client but different `reasoning_effort` or `extra_body` fail
  to load:

    ```text
    targets strong and strong_max both name model openai/gpt-4o on llm client openrouter but with different reasoning_effort, extra_body, or omit_body_fields; one target per model id is kept, so give each its own model id or llm client
    ```

    Separate routes can use separate `llm_clients` entries for the same
    endpoint. See the [target keys](../reference/toml_schema.md#targetsname).

## Related

- [Routing Overview](overview.md): choose a strategy and read what Auto resolves to.
- [Verify routing is active](../getting_started.md#verify-routing-is-active): see the served model and the decision counts.
- [Stage Router](stage_router_routing.md#tuning-confidence_threshold): tune the threshold once the pair is right.
- [Escalation-Router Routing](escalation_router_routing.md#judge-model-compatibility): judge requirements for escalation routes.
- [Benchmarks](../../benchmark/README.md): how the measured runs were made.
