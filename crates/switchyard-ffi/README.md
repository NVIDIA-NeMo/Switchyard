# C routing interface

An experimental C interface to libsy. The host supplies eligible models and
executes model and decision calls. The interface uses opaque handles and
NUL-terminated JSON strings. Ownership and concurrency rules are in
[`include/switchyard.h`](include/switchyard.h).

## JSON payloads

| Function | Input |
| --- | --- |
| `sy_router_new` | `{algorithm, targets}`: an `AlgorithmSpec` and target-alias-to-model-ID map |
| `sy_run_new` | `{request, models, subagent_models?, metadata?}` |
| `sy_decide_new` | `{format, request, target_formats?, models, subagent_models?, metadata?}` |
| `sy_run_respond` | `{response, served_model?}` or `{error}` |
| `sy_decode_request`, `sy_decode_response` | `{format, body}` |
| `sy_encode_request` | `{format, request, model}` |

`request` and model responses use Switchyard's normalized protocol JSON.
`models` and `subagent_models` map category names to ordered model-ID lists.
Optional metadata fields are `session_id`, `agent_id`, `is_subagent`,
`is_delegated_work`, and `session_final`.

For model replies, `served_model` identifies the target that answered, which
can differ from both the first candidate and the provider's response model name.
Decision replies contain a normalized `DecisionResponse`; `served_model` does
not apply to them. Error replies contain a string.

Each accepted read delivers one step or an error through its callback:

- `call_model`: `{type, id, models, request}`
- `call_decision`: `{type, id, model, request}`
- `done`: `{type, selected_model_ids, request, fallback_request, response, metadata}`

A callback error or `done` ends the run. Outcomes retain ordered targets, the
prepared request, an optional buffered answer, and optional routing evidence.
The host executes the answer call when `response` is null.
Streamed routing answers are collected into a buffered response before `done`.
Cancellation interrupts collection; stream errors end the run with a callback error.
When `fallback_request` is non-null, use it for candidates after the first.
Content-policy denials must not advance to those fallback candidates.

Wire formats are `openai_chat`, `openai_responses`, and `anthropic_messages`.
The translation functions use the existing Switchyard codecs.

## Provider-shaped routing

`sy_decide_new` accepts a provider request. `format` names its wire format and
is also the default for outgoing calls. `target_formats` can override it per
target ID. The host retains control of model eligibility and provider calls.

Use the same read, reply, cancellation, and cleanup functions as `sy_run_new`.
Model steps replace `request` with ordered `requests: [{model, format, request}]`.
Successful model replies contain provider JSON; identify `served_model` when
there is more than one candidate. Typed decision calls keep their existing
request and reply protocol.

The terminal outcome keeps its normalized request, optional answer, and evidence,
and adds the prepared candidate `requests`. Algorithm-produced answers remain
normalized. This interface does not serve the final provider request.

Invalid model replies return an error without consuming the call ID. Request
preparation errors end the run rather than following the algorithm's provider-error
policy. Ordinary `{error}` replies still follow that policy. Candidate requests
are prepared eagerly, including any large payloads.
