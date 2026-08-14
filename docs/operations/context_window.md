# Context-Window Handling

When an upstream rejects a request because the prompt exceeds the model's
context window, Switchyard calls the remaining targets on the same route in
configured order, stopping when one answers or every target has been tried.
Fallback applies only to the current request. An overflow is not remembered
across turns, so the route may select and try the same target again on the next
request.

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

The response body, `x-model-router-selected-model` header, usage metrics, and
routing log name the candidate that actually served the request, including when
the client falls through from the algorithm's first choice.

## When every target overflows

Once no untried target is left — including a single-target `passthrough` route,
which has no alternative from the start — the request fails with HTTP 400 and an
error `code` of `context_length_exceeded`.
