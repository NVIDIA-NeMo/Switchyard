# Routing Overview

The `switchyard-server` binary loads a native TOML deployment. Each route's `id`
becomes a model ID available through OpenAI Chat Completions, Anthropic Messages,
and OpenAI Responses requests.

Use this page to choose a routing strategy, then open its detailed page for
configuration and tuning. For the vocabulary these pages use, see
[Core Concepts](../core_concepts.md).

## Choose a strategy

Start with **Auto**. Choose Task or Execution when you want more control over
how requests move between an efficient model and a capable one.

| Choice | Use it when | Route `type` |
|---|---|---|
| **[Auto](#auto)** | You want Switchyard's recommended preset. | `auto` |
| **[Task](llm_classifier_routing.md)** | You want an LLM to judge which model can handle the task. | `llm_classifier` |
| **[Execution](stage_router_routing.md)** | You want tool results and agent progress to guide each request. | `stage_router` |

Task uses the LLM classifier's `capability` mode. Its `classify_trigger` setting
controls whether the judge runs for each request, each user turn, or once per
session. Execution uses the stage router, which reads tool-result history and
can optionally call an LLM judge. The TOML configuration keys are unchanged.

Whichever strategy you choose, the two models you put behind it matter more than
any threshold. See [Model Selection](model_selection.md). To tune cost against
quality, follow the [tuning loop](../tuning/index.md).

### What the names mean

| Name | Route `type` | What it does |
|---|---|---|
| Auto | `auto` | The Execution preset with fixed settings. |
| Task | `llm_classifier`, `mode = "capability"` | A judge reads the task before the turn is served and picks the strong or weak model. |
| Execution | `stage_router` | Tool results from the last few turns pick the capable or efficient model for each turn. |
| Composite | `composite` | A judge picks the starting tier at each user turn. Execution runs the turns in between. |
| Plan/Execute | `plan_execute` | The capable model plans. The first file edit hands the session to the efficient model. |
| Escalation | `llm_classifier`, `mode = "escalation"` | The weak model serves until a judge sees it struggle on consecutive turns. Then the session latches to the strong model. |
| Custom | `llm_classifier`, `mode = "custom"` | Your own judge prompt and schema choose among any number of targets. |
| Advisor Gate | `advisor` | One executor serves every turn. A stronger advisor reviews its plans and "done" claims. |
| Verification-Gated | `vgr` | A local model drafts. Verified answers stay local, and everything else goes to the cloud model. |
| Prefill | `prefill_router` | A classifier you train on the prompt's hidden states picks the target. |
| Random | `random` | A fixed traffic split. |
| Fixed | `passthrough` | One target, no decision. |
| Sub-agent | `subagents` table | A separate policy for delegated sub-agent work on `passthrough`, `stage_router`, and `composite` routes. |

### Compare the strategies

| Route `type` | Decides from | Extra model calls | Switches | Session ID | When it cannot decide | Measured | Start from |
|---|---|---|---|---|---|---|---|
| `auto` | Tool results in the request | None | Every request | Optional. Keeps the capable hold. | Efficient tier | The preset of the measured `stage_router` settings, with a shorter window. See [Auto](#auto). | The [Getting Started](../getting_started.md#server-path) route file |
| `stage_router` | Tool results in the request | Optional judge on ambiguous turns | Every request | Optional. Keeps the capable hold. | Picker default | [DeepSWE v1.1 profile](../../benchmark/routing-profiles/deepswe-v11-stage-router-luna-sol.toml) | That profile |
| `llm_classifier`, capability | A judge's view of the task | One judge call per request, user turn, or session | Per trigger | Optional. Needed to retain a verdict. | Strong target | [Smoke-test config](../../benchmark/server-configs/tb-lite-llm-classifier-opus-kimi-gemini.toml) only, no published result | That config |
| `llm_classifier`, escalation | A judge's view of the finished weak turn | One judge call per unlatched turn | Latches once. Reversible with de-escalation. | Required for `confirmations` above 1 | An unreadable verdict keeps the current tier. A failed judge call stops the request in the server. | [Public report](https://www.langchain.com/blog/switchyard-agent-routing-benchmark) and the [v0.2.0 profile](../../benchmark/routing-profiles/tb21-escalation-opus-glm-deepseek.toml) | That profile |
| `llm_classifier`, custom | Your judge's verdict | One judge call per trigger | Per trigger | Optional. Needed to retain a verdict. | `default_target` | [tau2 profiles](../../benchmark/routing-profiles/tau2-telecom-custom-opus-qwen-balanced.toml) | That profile |
| `composite` | A judge at each user turn, tool results in between | One judge call per user turn or session | Every request, with a per-user-turn default | Required, or `message_hash_fallback` | Last tier, else efficient | Not benchmarked | [Composite Routing](composite_routing.md#configuration) |
| `plan_execute` | The first edit or write tool call | None | Once per session | Recommended. Latches the handoff. | Stays on the planner | [DeepSWE v1.1 profile](../../benchmark/routing-profiles/deepswe-v11-plan-execute-luna-sol.toml) | That profile |
| `advisor` | The executor's terminal turn | One advisor review per gated turn, up to `max_reviews` | Never. One executor serves. | Scopes the review budget | An unreadable verdict lets the turn through. A failed advisor call stops the request in the server. | [DeepSWE v1.1 profile](../../benchmark/routing-profiles/deepswe-v11-advisor-gate-luna-sol.toml) | That profile |
| `vgr` | Evidence about a local candidate | Verifier and judge calls | Per final answer | See [the page](vgr_routing.md) | Cloud tier | Not benchmarked. Planned for 0.4.0. | [`examples/vgr-local-servers.toml`](../../examples/vgr-local-servers.toml) |
| `prefill_router` | Hidden states of the prompt | Encoder inference, no provider call | Every request | Not required | Checkpoint output | Experimental. No checkpoint is supplied. | [Schema example](../reference/toml_schema.md#prefill_router) |
| `random` | Weights | None | Every request | Not required | Not applicable | Utility | [Random Routing](random_routing.md) |
| `passthrough` | Nothing | None | Never | Not required | Not applicable | Utility | [Schema](../reference/toml_schema.md#passthrough) |
| `noop` | Nothing | None | Never | Not required | Not applicable | Smoke test | [Schema](../reference/toml_schema.md#noop) |

Every route streams the served model's reply. Three hold a turn back first:
escalation buffers the weak reply until the judge rules on it, `advisor` holds
a gated turn until the review finishes, and `vgr` holds a local candidate until
it is verified. Task judges before the turn is served, which adds the judge's
latency to every judged request.

"Measured" links the checked-in profile or public report the result comes from.
The `capable_first` picker and `prefill_router` are experimental. See
[Model Selection](model_selection.md#measured-pairs) for the numbers.

### When routing will not help

Each strategy page lists the cases it does not fit:
[stage router](stage_router_routing.md#when-not-to-use-stage-router) and
[escalation](escalation_router_routing.md#when-not-to-use-escalation-routing).
These cases apply across strategies:

- **Requests carry no session ID.** Escalation with `confirmations` above 1
  never latches. Composite cannot retain the tier. Task with `user_turn` or
  `new_session` judges every request. Plan/Execute needs the first edit to stay
  in the request history. Execution keeps no capable hold. Send
  `x-switchyard-session-id`, or use a strategy that decides from the request
  alone.
- **Both tiers resolve to the same model, or the efficient model is the stronger
  one.** Routing then only adds cost. Run the
  [inverted-pair check](model_selection.md#check-for-an-inverted-pair) first.
- **Chat-only traffic on Auto or Execution.** With no tool results the signals
  abstain and every turn lands on the efficient tier. Use Task or Composite,
  which judge the request itself.
- **Responses continuations by ID.** A client that sends `previous_response_id`
  with new input, and no history, hides the tool calls that Execution and
  Plan/Execute read. See
  [Responses continuations by ID](llm_classifier_routing.md#responses-continuations-by-id)
  and the
  [Plan/Execute history requirement](plan_execute_routing.md#responses-api-history-requirement).
- **A judge that cannot return schema-valid JSON.** Task then sends every request
  to the strong target, Execution falls back to the picker default, and
  escalation never latches. When the provider rejects the judge request
  outright, escalation turns fail with that error instead. Run the
  [structured output check](model_selection.md#2-structured-output) and see
  [Judge model compatibility](llm_classifier_routing.md#judge-model-compatibility).
- **A judge on every tool step.** Task's default `classify_trigger = "every_request"`
  judges every tool continuation. An agent that makes twenty tool calls causes
  twenty-one judge calls for one task. Use `user_turn` with a session ID, or
  Execution, which reads the tool results without a model call. See
  [When the judge runs](llm_classifier_routing.md#when-the-judge-runs).
- **Sub-agent traffic without identity.** A `subagents` policy applies only when
  the harness sends sub-agent identity. Otherwise delegated requests follow the
  parent route, and the server logs a warning. See
  [Sub-Agent-Aware Routing](subagent_routing.md).

### Start here

Three starting points. Each names a strategy, a file in this repository to copy,
and the one dial to move first.

1. **A coding agent, and cost comes first.** Start with `auto`. When you need to
   tune, copy the
   [DeepSWE stage profile](../../benchmark/routing-profiles/deepswe-v11-stage-router-luna-sol.toml)
   and move `confidence_threshold` as described in
   [Tuning `confidence_threshold`](stage_router_routing.md#tuning-confidence_threshold).
2. **Long sessions where the cheap model should run until it visibly struggles.**
   Use escalation. Copy the
   [escalation profile](../../benchmark/routing-profiles/tb21-escalation-opus-glm-deepseek.toml),
   send a session ID, and move `confirmations` as described in
   [Tuning options](escalation_router_routing.md#tuning-options).
3. **One executor you want to keep, with a stronger model as a quality gate.**
   Use the advisor gate. Copy the
   [DeepSWE advisor profile](../../benchmark/routing-profiles/deepswe-v11-advisor-gate-luna-sol.toml)
   and move `max_reviews` together with `gate_min_tool_results` as described in
   [Tuning](advisor_gate_routing.md#tuning).

For multi-turn support traffic with few tool results, the
[tau2 custom-classifier profiles](../../benchmark/routing-profiles/tau2-telecom-custom-opus-qwen-balanced.toml)
are the measured starting point. Their dial is the judge rubric in `prompt`.

### Auto

Auto is a preset, not a separate routing algorithm. An `auto` route builds a
stage router with these settings. It does not compare strategies at runtime.

| Setting | Value | What it does |
|---|---|---|
| `picker` | `efficient_first` | The efficient tier serves a turn when the tool signals are not confident. |
| `confidence_threshold` | `0.5` | How strongly the tool signals must agree before they pick a tier on their own. |
| `recent_turn_window` | `3` | How many recent tool results the signals are computed over. |
| `capable_hold_turns` | `2` | How many turns stay on the capable tier after an escalation. A clean test pass ends the hold early. |
| Classifier | none | No LLM judge is called. |

Auto accepts no tuning keys. To change a setting, write the equivalent
`stage_router` route shown in the [TOML schema](../reference/toml_schema.md#auto)
and edit it there. Choose the two models with the
[model selection guide](model_selection.md).

Only the `efficient_first` picker has published results. The `capable_first`
picker is [experimental](stage_router_routing.md#pickers).

#### When Auto does not route

Auto decides from tool results in the request. In these cases every request goes
to one tier:

- **No tool results in the request.** Chat-only requests, and the first request
  of an agent session, carry no tool history. The signals abstain and the
  efficient tier serves the turn. This is by design. `GET /v1/stats` counts these
  turns under `algorithm_stats.stage_router.routing_decisions.fall_open`. See
  [Verify routing is active](../getting_started.md#verify-routing-is-active).
- **Both tiers name the same target.** The server accepts a route whose
  `capable_target` and `efficient_target` are the same target. Every request then
  goes to that one model, and `GET /v1/stats` lists a single entry under `.models`.
- **No session ID on the request.** The capable hold is kept per session. A
  request without a session ID gets fresh state, so a hold set by one request is
  gone on the next. Requests with a session ID but no agent ID share one hold
  within that session.

A critical error, a repeated identical failure, or a context-compaction marker in
the tool history still sends the turn to the capable tier. See
[decision sources](stage_router_routing.md#decision-sources).

Auto uses the settings of the published stage-router results, with two
differences: those runs set `recent_turn_window = 5` and added a handoff note.
Every published result comes from a coding agent whose requests carry tool
results. The [model selection guide](model_selection.md#measured-pairs) lists the
measured model pairs.

## More options

These options remain available when you need a different routing policy.

| Strategy | Use it when | Route `type` |
|---|---|---|
| [Plan/Execute](plan_execute_routing.md) | Use a capable model to inspect and plan, then switch to an efficient model after the first file mutation. | `plan_execute` |
| [Composite](composite_routing.md) | Combine Task and Execution. A classifier sets the stage router's default tier. | `composite` |
| [Escalation](escalation_router_routing.md) | Start on the efficient model and escalate when an LLM judge detects trouble. | `llm_classifier` with `mode = "escalation"` |
| [Custom](llm_classifier_routing.md#custom-multi-target-routing) | Route among two or more models using your own classification schema and rules. | `llm_classifier` with `mode = "custom"` |
| [Advisor Gate](advisor_gate_routing.md) | Keep one executor model and have a stronger advisor review its plans and completion claims. | `advisor` |
| [Sub-Agent-Aware Routing](subagent_routing.md) | Delegated sub-agents should use a separate routing policy from the parent agent. | `passthrough`, `stage_router`, or `composite` with `subagents` |
| [Random Routing](random_routing.md) | You need a fixed traffic split for A/B tests, baselines, or cost experiments. | `random` |
| [Verification-Gated Routing](vgr_routing.md) | Serve a local attempt only after bounded verification, and send the request to the capable model otherwise. | `vgr` |
| [Fixed Model](#direct-model-routes) | Send every request to one target without a routing decision. | `passthrough` |

### Experimental

[Prefill Router](../reference/toml_schema.md#prefill_router) uses a
checkpoint-backed classifier. It is experimental in v0.3.0. Switchyard does not
provide or support a checkpoint, exporter, or compatible encoder assets.

## Common route shape

A deployment has three layers. LLM clients describe how to reach a provider,
targets name models on those clients, and routes decide which target serves a
request:

```toml
schema_version = 1

[llm_clients.openrouter]
format = "openai_chat"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"

[targets.strong]
id = "openai/gpt-4o"
llm_client = "openrouter"

[targets.weak]
id = "openai/gpt-4o-mini"
llm_client = "openrouter"

[routes.fast]
id = "fast"
type = "passthrough"
target = "weak"

[routes.smart]
id = "smart"
type = "random"
targets = ["strong", "weak"]
weights = [3, 7]
```

Clients send a route's `id` (`fast` or `smart`) as the request's model ID. The
table name in `[routes.smart]` is local to the file; only `id` is visible to
clients. A single deployment can serve multiple routes on the same host and port.

Credentials stay outside the file: `api_key_env` names an environment variable
that is read at startup. Omit it for an upstream that needs no credential, such
as a local model server.

The examples use model IDs from the
[OpenRouter model catalog](https://openrouter.ai/api/v1/models). Select IDs
available to your account before deploying; catalog availability can change.

## Direct model routes

A `passthrough` route registers one target under one model ID with no routing
decision:

```toml
[routes.fast]
id = "fast"
type = "passthrough"
target = "weak"
```

Use a routing strategy instead when requests must be split or classified across
targets. There is no catalog auto-discovery, so to expose several upstream
models, add one `passthrough` route per model.

## Self-hosted targets

Any target can point at an OpenAI-compatible model server you operate. For
example, start a local vLLM server:

```bash
vllm serve ./my-rl-qwen --served-model-name my-rl-qwen --port 8000
```

Then declare it as its own LLM client and target. The client needs no
`api_key_env` when the server does not require a credential:

```toml
[llm_clients.local_vllm]
format = "openai_chat"
base_url = "http://localhost:8000/v1"

[targets.local]
id = "my-rl-qwen"
llm_client = "local_vllm"

[routes.local]
id = "local"
type = "passthrough"
target = "local"
```

Switchyard does not start or manage the model server; it only sends requests to
the configured endpoint.

## Run a deployment

After installing the Rust server, as described in
[Getting Started](../getting_started.md#install-the-server), export the provider
credential, validate the configuration, and start the binary:

```bash
export OPENROUTER_API_KEY="your-openrouter-key"  # pragma: allowlist secret
switchyard-server --config routes.toml --dry-run
switchyard-server --config routes.toml \
  --host 127.0.0.1 --port 4000
```

Send a request using a route ID:

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"smart","messages":[{"role":"user","content":"hello"}]}'
```

For the complete TOML schema and every route option, refer to the
[TOML Schema](../reference/toml_schema.md).
