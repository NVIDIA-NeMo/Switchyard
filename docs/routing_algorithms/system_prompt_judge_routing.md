# System-Prompt-Judge Routing

`system_prompt_judge` serves every caller-visible turn with one target, but first
asks a judge model whether to inject one hidden system prompt from a text DB.

Use it for small experiments where a reviewer should choose from a fixed set of
interventions without exposing those interventions to the served agent unless
one is selected.

## Configuration

```toml
[targets.primary]
id = "model/agent"
llm_client = "upstream"

[targets.judge]
id = "model/judge"
llm_client = "upstream"

[routes.agent]
id = "switchyard/agent"
type = "system_prompt_judge"
target = "primary"
judge_target = "judge"
db_path = "actions.txt"
```

`db_path` is resolved relative to the TOML file when the server loads the
deployment from disk. Absolute paths also work.

The action DB is a plain text file with one `[action_id]` section per prompt:

```text
[compile_failure]
Focus on the exact compiler error before editing code.

[stuck_loop]
Stop repeating the same command. Make a new hypothesis and test it.
```

For each request, the judge sees the conversation plus the hidden action DB and
returns:

```json
{"action":"compile_failure"}
```

or:

```json
{"action":"none"}
```

If the judge chooses a known action, Switchyard prepends that action's text as a
system instruction on the call to `target`. If the judge fails, returns invalid
JSON, returns `none`, or selects an unknown action, the route fails open and
sends the original request to `target`.

## Route keys

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `target` | Yes | — | Target that serves the caller-visible request. |
| `judge_target` | Yes | — | Target used to choose one action id or `none`. Not a routing destination. |
| `db_path` | Yes | — | Text DB containing `[action_id]` prompt sections. Relative paths resolve next to the TOML file. |
| `max_output_tokens` | No | `64` | Maximum completion tokens for the judge verdict. |
