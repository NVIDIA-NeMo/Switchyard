# Plan/Execute Routing

Plan/execute routing starts a coding task on a capable model and switches to an
efficient model after the first file mutation. It makes no classifier call.

Before the transition, Switchyard prepends a planning system instruction to the
outbound request. Read-only inspection and planning tool calls stay on the
capable target. An edit or write tool call anywhere in the normalized
conversation moves the request to the efficient target and the planning
instruction is no longer added. The efficient model receives the caller's full
conversation, including the capable model's plan and tool history.

The transition is latched by session ID when one is available, so later context
compaction cannot move that session back into planning. Without a session ID,
Switchyard determines the phase from the conversation on every request; callers
must therefore retain the first mutation in the history to keep the efficient
target selected. A request marked as the session's final request uses the latch
for that request and then releases it.

## Configure the route

```toml
schema_version = 1

[llm_clients.openrouter]
format = "openai_chat"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"

[targets.planner]
id = "anthropic/claude-opus-4.7"
llm_client = "openrouter"

[targets.executor]
id = "moonshotai/kimi-k2.7-code"
llm_client = "openrouter"

[routes.plan_execute]
id = "switchyard/plan-execute"
type = "plan_execute"
capable_target = "planner"
efficient_target = "executor"
```

`planning_prompt` optionally replaces the built-in planning instruction:

```toml
[routes.plan_execute]
id = "switchyard/plan-execute"
type = "plan_execute"
capable_target = "planner"
efficient_target = "executor"
planning_prompt = "Inspect the task and write a concrete plan before editing."
```

The prompt must contain non-whitespace text. It is inserted ahead of caller
system and developer instructions only during planning. Switchyard does not add
it to conversation messages, so the handoff removes the planning constraint
without deleting or rewriting any caller-owned trajectory.

## Transition signals

The route reuses the stage router's provider-neutral tool-signal extraction. It
recognizes dedicated edit and write tools such as `apply_patch`, `Edit`,
`Write`, and `write_file`, plus common file-mutating shell commands issued
through Claude Code, Codex, and other coding-agent shells. Read, search, test,
and planning tools do not trigger the handoff.

Because the switch is based on the recorded tool call, a failed first edit still
begins execution. This keeps the phase boundary deterministic and lets the
efficient model diagnose and retry the attempted change from the inherited
history.

## Benchmarking

Use the same route ID and task set for every run. Compare it with passthrough
routes for each target to measure quality, input/output tokens, time to first
token, and end-to-end latency. A stable session ID is recommended for long tasks
that may compact their history.

See [Soak Testing](../operations/soak_test.md) for the local scenario backend
and routing benchmark workflow. The repository also includes
`benchmark/server-configs/tb-lite-plan-execute-opus-kimi.toml` for Harbor Terminal-Bench Lite.
