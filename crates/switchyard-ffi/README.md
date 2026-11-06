# C routing interface

An experimental C interface to libsy. The host supplies eligible models and
executes model and decision calls. The interface uses opaque handles and
NUL-terminated JSON strings. Ownership and concurrency rules are in
[`include/switchyard.h`](include/switchyard.h).

## JSON payloads

| Function | Input |
| --- | --- |
| `sy_router_new` | `{algorithm, targets}` or `{configured: {toml, route_id}}` |
| `sy_router_target_models` | Configured router handle; returns an owned target-to-`{client,id}` map |
| `sy_run_new` | `{request, models?, subagent_models?, metadata?, required_labels?}` |
| `sy_run_respond` | `{response, served_model?}` or `{error}` |
| `sy_decode_request`, `sy_decode_response` | `{format, body}` |
| `sy_encode_request` | `{format, request, model}` |

`request` and model responses use Switchyard's normalized protocol JSON.
Raw routers require `models`; `models` and `subagent_models` map category names to ordered model-ID lists.
Optional metadata fields are `session_id`, `agent_id`, `is_subagent`,
`is_delegated_work`, `session_final`, and `task_id`.

Configured routers load one TOML route through `AlgorithmFactory::build_configured`
without provider clients or credential reads. They check `required_labels` before
returning a run and reject host-supplied model groups. Omitted/null labels supply
no assessment; `[]` supplies an explicitly empty assessment. Task retention belongs
to the reused router handle. Raw routers reject non-null labels, including `[]`.
Configured construction rejects mixed raw settings and is supported on native and WASM builds.

`sy_router_target_models(router, &out_json)` returns owned JSON such as
`{"private_fast":{"client":"approved","id":"provider/private-fast"}}`.
Keys match emitted LLM target names or Decision provider IDs. The map includes all
potential alternatives, not just currently eligible targets. Raw handles reject
discovery. Free the output with `sy_string_free`.

These additions keep ABI generation 1: existing C signatures, opaque handles and
raw payloads are unchanged. Discovery adds a symbol, so rebuild the native library
or WASM guest and host before using it. Configured construction also needs a newer library;
older libraries reject its JSON because it has no raw `algorithm` field. Hosts
must not send labels through a legacy raw constructor, whose older implementation
would ignore unknown fields.

For model replies, `served_model` identifies the target that answered, which
can differ from both the first candidate and the provider's response model name.
Decision replies contain a normalized `DecisionResponse`; `served_model` does
not apply to them. Error replies contain a string.

Each accepted read delivers one step or an error through its callback:

- `call_model`: `{type, id, models, request}`
- `call_decision`: `{type, id, model, request}`
- `done`: `{type, selected_model_ids, request, fallback_request, candidate_requests, response, metadata}`

A callback error or `done` ends the run. Outcomes retain ordered targets, the
prepared request, an optional buffered answer, and optional routing evidence.
The host executes the answer call when `response` is null.
Streamed routing answers are collected into a buffered response before `done`.
Cancellation interrupts collection; stream errors end the run with a callback error.
Use `candidate_requests[model_id]` when present. Otherwise, use `fallback_request`
for candidates after the first when it is non-null.
Content-policy denials must not advance to those fallback candidates.

Wire formats are `openai_chat`, `openai_responses`, and `anthropic_messages`.
The translation functions use the existing Switchyard codecs.
