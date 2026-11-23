# Apply a decision model to a list-selection call

Use `decision_select` when an agent already sends a chat request that means “choose from this list.” An operator configures a rule for that call. Each time it matches, Switchyard turns the task and choices into a batch of decision questions, calls the decision model, and returns the chosen names to the agent. Other chat requests go to the usual LLM.

## Follow one request

Suppose a music agent asks which API it needs to find a song. It sends this ordinary chat request to the gateway.

```json
{
  "model": "music-agent",
  "messages": [
    {"role": "system", "content": "Choose names"},
    {"role": "user", "content": "Catalog\nmusic:\n  buy: Purchase a track\n  search: Find songs"},
    {"role": "user", "content": "Task: find a song. List names."}
  ]
}
```

**What Switchyard sends.** The rule recognizes `Choose names`, reads the catalog, and extracts `find a song` from the last message. On a System One backend the task goes into `state`. Each API name and description becomes a yes/no question. Both questions travel in the same HTTP request.

```json
{
  "model": "decision-model",
  "state": "find a song",
  "questions": {
    "q0": {"type": "noul", "instructions": "Which names are relevant?\nConsider one option only: music.buy (Purchase a track)\nDoes this option belong to the answer?"},
    "q1": {"type": "noul", "instructions": "Which names are relevant?\nConsider one option only: music.search (Find songs)\nDoes this option belong to the answer?"}
  }
}
```

**What the agent gets back.** Suppose the model returns the following scores. `noul` is the System One field for the probability of a yes answer.

```json
{"answers": {"q0": {"type": "noul", "noul": 0.1}, "q1": {"type": "noul", "noul": 0.9}}}
```

With `top_k = 1`, Switchyard maps the winning ID `q1` back to `music.search`. The agent receives this content in its usual chat response.

```json
{"choices": [{"message": {"role": "assistant", "content": "music.search"}}]}
```

## Configure it for your agent

Start from a real selection request. Copy its system instruction and the markers around its catalog and task. Set `question` to the instruction you want the decision model to apply. Point `target` at the regular chat target and `decision_target` at a configured decision target.

```toml
[routes.music]
id = "music-agent"
type = "decision_select"
target = "writer"
decision_target = "selector"

[routes.music.rule]
enabled = true
system_instruction = "Choose names"
catalog_prefix = "Catalog\n"
task_prefix = "Task: "
task_suffix = ". List names."
question = "Which names are relevant?"
shortlist = 4
top_k = 1
```

The incoming request supplies the choices on every call. `shortlist` limits how many choices the model scores and `top_k` limits how many names the gateway returns. With multiple winners, the assistant text has one name per line. A mismatch or failed selection uses the regular chat target. To turn the rule off, set `enabled = false` and reload the route.

The [AppWorld example](../../examples/appworld/README.md) applies the same feature to API selection before the agent writes Python. Its [complete TOML](../../examples/appworld/routes.toml) includes the client and model targets. See the [configuration reference](../reference/toml_schema.md#decision_select) for the full matching and fallback rules.
