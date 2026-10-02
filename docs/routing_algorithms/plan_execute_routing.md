# Plan/Execute Routing

Plan/execute routing uses a capable model to inspect and plan a coding task,
then switches to an efficient model after the first file mutation. It does not
make a classifier call.

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
type = "plan_execute"
capable_target = "planner"
efficient_target = "executor"
```

Read-only inspection stays on the capable target with a planning instruction.
The first edit or write routes the full trajectory to the efficient target and
latches that choice by session ID. A failed edit still triggers the handoff.
Without a session ID, the first mutation must remain in the request history.

## Responses API history requirement

Plan/execute needs the conversation history to detect edits and hand the task to
the executor. Continuing with only `previous_response_id` or a provider
`conversation` ID and new input is not supported for this handoff.

For example, a `function_call_output` contains the result and call ID, but not
the tool name. Without the earlier `write_file` call, the router cannot tell that
the result belongs to an edit and can stay on the planner.

Send the conversation history in `input`, including earlier tool calls and their
results. Omit `previous_response_id` and `conversation` so routing can switch
models between turns.

When embedding `libsy`, your application must supply that history in
`Request.llm_request.messages` before routing. `libsy` does not fetch it from the
provider. A stable session ID remembers a detected handoff, but cannot detect an
edit missing from the request history.

## Optional settings

| Key | Behavior |
|---|---|
| `planning_prompt` | Replaces the built-in planning instruction. |
| `handoff_prompt` | Adds an instruction to the handoff request. |
| `planner_reasoning_as_text` | Converts visible planner reasoning summaries to assistant text at handoff. |

Use a stable session ID with handoff processing or history compaction.
