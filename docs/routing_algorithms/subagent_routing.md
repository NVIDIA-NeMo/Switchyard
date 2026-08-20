# Sub-Agent-Aware Routing

Sub-agent-aware routing leaves parent-agent traffic with its configured routing
algorithm while routing delegated sub-agent work separately. It is available on
`passthrough` and `stage_router` routes through the optional `subagents` table.

```toml
schema_version = 1

[llm_clients.openrouter]
format = "openai_chat"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"

[targets.parent]
id = "anthropic/claude-sonnet-5"
llm_client = "openrouter"

[targets.classifier]
id = "openai/gpt-4.1-mini"
llm_client = "openrouter"

[targets.worker]
id = "openai/gpt-5.4-mini"
llm_client = "openrouter"

[targets.reviewer]
id = "anthropic/claude-opus-5"
llm_client = "openrouter"

[routes.agent]
id = "agent"
type = "passthrough"
target = "parent"
context_window = 400000
tool_calling = true
reasoning = true

[routes.agent.subagents]
type = "llm_classifier"
mode = "custom"
classifier_target = "classifier"
targets = ["worker", "reviewer"]
default_target = "worker"
classify_trigger = "new_session"
max_output_tokens = 64
prompt = """
Select exactly one target for the delegated task.

- Select "reviewer" for code review, critique, auditing, or correctness analysis.
- Select "worker" for implementation, research, explanation, and other delegated work.

Return only JSON matching the response schema.
"""
response_schema = '''
{
  "type": "object",
  "properties": {
    "target": {"type": "string", "enum": ["worker", "reviewer"]}
  },
  "required": ["target"],
  "additionalProperties": false
}
'''
policy = { type = "target_selector", selector = "/target" }
```

Set `OPENROUTER_API_KEY`, save the configuration as `routes.toml`, and validate it
before starting the server:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."  # pragma: allowlist secret
switchyard-server --config routes.toml --dry-run
switchyard-server --config routes.toml --port 4000
```

The parent always uses `parent`. For a delegated request, the classifier sees
the prompt supplied by the parent and selects one configured target. With
`classify_trigger = "new_session"`, Switchyard reuses that decision for later
requests from the same `session + agent` identity. Use `every_request` to
classify each delegated request. `user_turn` is not supported for sub-agent
routing. Harness-maintenance requests continue through the parent route.

To use Stage Router for parent traffic, replace the `[routes.agent]` table in the
example with the following. The nested `[routes.agent.subagents]` classifier is
unchanged.

```toml
[routes.agent]
id = "agent"
type = "stage_router"
capable_target = "reviewer"
efficient_target = "worker"
picker = "efficient_first"
confidence_threshold = 0.7
context_window = 400000
tool_calling = true
reasoning = true
```

Clients must still request the route ID (`agent` above). An explicit model name
that is not registered as a route is rejected before sub-agent classification.
`message_hash_fallback` is not supported for sub-agent routing because affinity
requires harness-provided child identity.

To send every delegated sub-agent request to one fixed target without calling a
classifier, replace the `subagents` table above with:

```toml
[routes.agent.subagents]
type = "passthrough"
target = "worker"
```
