# Plan/Execute with Stage

Plan/execute uses [Stage](stage_router_routing.md) to start a coding task on a
capable model. Stage can switch to efficient when tool results show progress,
then keep selecting that tier for the rest of the session. Configure the route
with `picker = "capable_first"`, `confidence_threshold = 0.0`, a planning prompt,
and `efficient_hold_turns = -1`.

## Configuration

Save this as `plan-execute.toml`. Replace the provider URL and model IDs with
your own, and set `OPENAI_API_KEY`.

```toml
schema_version = 1

[llm_clients.provider]
format = "openai_responses"
base_url = "https://example.com/v1"
api_key_env = "OPENAI_API_KEY"

[targets.planner]
id = "provider/capable-model"
llm_client = "provider"

[targets.executor]
id = "provider/efficient-model"
llm_client = "provider"

[routes.plan_execute]
id = "switchyard/plan-execute"
type = "stage_router"
capable_target = "planner"
efficient_target = "executor"
picker = "capable_first"
confidence_threshold = 0.0
capable_hold_turns = 0
efficient_hold_turns = -1
capable_system_prompt = """
Inspect the task and relevant code, then form a concrete implementation plan before modifying any files. Use read-only tools as needed. Once the plan is complete, implement and check the solution. Another model may take over to finish the remaining work.
"""
```

```bash
switchyard-server --config plan-execute.toml --port 4000
```

Send requests to `switchyard/plan-execute` with the full conversation and a
stable `x-switchyard-session-id` header from the first planning request onward.
The route serves OpenAI Chat Completions, Anthropic Messages, and OpenAI
Responses, including streaming responses.

## Planning and handoff

An initial request without tool history uses the capable target with the
planning prompt. Read-only inspection normally stays on that target. Stage
scores recent tool results; clean writes and edits move the score toward
efficient. At threshold `0`, any negative score selects efficient. A zero score
selects capable.

Recovery signals can keep the task on capable before handoff. A failed edit
does not always switch models, and critical errors, repeated failures, or
context compaction select capable. Setting `capable_hold_turns = 0` lets the next
request be scored without waiting through Stage's default recovery hold.

After a switch from capable to efficient, `efficient_hold_turns = -1` keeps
selecting efficient until the session state is cleared. Later errors, compaction,
and new user turns do not end this hold. The planning prompt goes only to the
capable target; the efficient target receives the conversation and tool results.
If the first request already selects efficient, no switch has occurred and no
hold starts. Keep the same session ID from the initial planning request.

Model-call fallback is separate from this routing choice. After an eligible
provider or context-window failure, Switchyard can try the other completion
target, even while a hold is active. Each target receives its own configured
prompt. See [Context-Window Handling](../operations/context_window.md).

## Settings and limits

| Setting | Value here | Effect |
|---|---|---|
| `picker` | `capable_first` | Stage selects capable when tool signals cannot choose a tier. |
| `confidence_threshold` | `0.0` | Stage selects a tier for every nonzero signal score. |
| `capable_hold_turns` | `0` | Stage scores each request before handoff. |
| `efficient_hold_turns` | `-1` | Stage keeps selecting efficient after a switch until session state is cleared. |
| `capable_system_prompt` | The value is a planning instruction. | Stage adds the instruction to requests served by the capable target. |
| `tool_semantics.mutate` | The list is optional. | Stage treats these custom tools as writes or edits when scoring tool results. |

For example, add a custom file-writing tool:

```toml
[routes.plan_execute.tool_semantics]
mutate = ["persist_source_file"]
```

The [Stage guide](stage_router_routing.md#knob-reference) lists the remaining
settings and tool matching rules. The configuration above has no
published benchmark results; the historical Plan/Execute results used the
removed implementation.

Each Stage route stores its session state in memory. Agent IDs have separate
tier holds within a session. A request with `session_final: true` clears the
whole session, including other agents' holds; the server accepts this through
`x-switchyard-session-final: true`. Send it only when that shared session is
complete.

Stage checks hourly for sessions inactive for at least one hour and removes
them. A server restart also clears state, and replicas do not share it. Route
each session to the same server instance. Stage has no fixed limit on the
number of sessions or bytes retained; memory use grows with retained sessions
and agent IDs. Without a session ID, requests do not retain the efficient hold.

## Migration from `plan_execute`

The standalone `type = "plan_execute"` route and Rust `PlanExecute` API have
been removed. Configurations with that route type or its obsolete settings
fail to load. Keep your route ID and target names, and replace the route
settings with the Stage configuration above. Rust callers can construct
`StageRouter` with `StageRouterConfig` instead.

| Old setting | Stage migration |
|---|---|
| `planning_prompt` | Use `capable_system_prompt`. Supply the instruction explicitly. |
| `tool_semantics.mutate` | Keep the list. Stage includes these tools in its score; a matching tool alone no longer forces a handoff. |
| `handoff_prompt` | Optionally use `handoff_notes.deescalation_note`. Stage adds the note when tool signals select efficient, even if the tier stays the same. Requests routed by a hold receive no note. |
| `planner_reasoning_as_text` | Remove it. Stage does not convert planner reasoning into text. Supply any required plan as ordinary conversation text. |
| `execution_trigger` and `judge` | Remove both settings for the configuration above. An optional Stage `classifier` runs only when tool signals cannot choose a tier; it does not reproduce the old `execution_trigger = "judge"` handoff. |

Stage's efficient hold lasts until session state is cleared, including across new
user turns. The old judged route restarted planning on each user turn. Stage
also replaces the old limit of 4,096 retained identities with the session
lifetime described above. Previously, a final request cleared only the
requesting agent's state. Stage clears every agent's state in the shared session.

## Responses API history requirement

Plan/execute needs the conversation history to score tool results and hand the
task to the executor. Continuing with only `previous_response_id` or a provider
`conversation` ID and new input is not supported for this handoff.

For example, a `function_call_output` contains the result and call ID, but not
the tool name. Without the earlier `write_file` call, the router cannot tell
that the result belongs to a write and can stay on the planner.

Send the conversation history in `input`, including earlier tool calls and
their results. Omit `previous_response_id` and `conversation` so routing can
switch models between turns.

When embedding `libsy`, supply that history in `Request.llm_request.messages`
before routing. `libsy` does not fetch it from the provider. A stable session ID
retains an active hold but cannot recover tool activity missing from the history.
