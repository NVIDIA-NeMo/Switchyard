# Context-Window Handling

When an upstream rejects a request because the prompt exceeds the model's
context window, Switchyard drops that target and calls another target on the
same route, repeating until a call succeeds or every target has been tried. If
the request carries `x-switchyard-session-id` or a recognized coding-agent
session header such as `x-claude-code-session-id`, the target remains excluded
for the rest of that session. Without a session header, the fallback still
applies to the current request, but the overflow is not remembered.
After truncating or resetting context, clients should use a new session ID;
reusing the old ID preserves its target exclusions.

## What counts as an overflow

Switchyard treats an upstream reply as a context overflow only when the status
is HTTP 400 **and** the body identifies a context-length error: an `error.code`
of `context_length_exceeded`, or a message containing a phrase such as
`maximum context length` or `prompt is too long`. An overflow reported any other
way — HTTP 413 or 422, for example — is not recognized as one. The request
fails on the spot with the upstream's status code, and the upstream's body is
returned inside Switchyard's error message.

## Configuration

There is no eviction key. A route falls through whenever it has more than one
target, so the route below needs nothing beyond its two tiers:

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

[routes.stage]
id = "switchyard/stage"
type = "stage_router"
capable_target = "strong"
efficient_target = "weak"
picker = "efficient_first"
confidence_threshold = 0.5
```

Response headers report where the request actually landed:

```text
x-model-router-selected-model: openai/gpt-4o-mini
x-model-router-rationale: openai/gpt-4o exceeded its context window; fell back to openai/gpt-4o-mini
```

## When every target overflows

Once no untried target is left — including a single-target `passthrough` route,
which has no alternative from the start — the request fails with HTTP 400 and an
error `code` of `context_length_exceeded`.
