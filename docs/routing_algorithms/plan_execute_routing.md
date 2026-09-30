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

Optional settings:

| Key | Behavior |
|---|---|
| `planning_prompt` | Replaces the built-in planning instruction. |
| `handoff_prompt` | Adds an instruction to the handoff request. |
| `planner_reasoning_as_text` | Converts visible planner reasoning summaries to assistant text at handoff. |

Use a stable session ID with handoff processing or history compaction.
