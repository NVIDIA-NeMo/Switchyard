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
- `done`: `{type, selected_model_ids, request, response, metadata}`

A callback error or `done` ends the run. Outcomes retain ordered targets, the
prepared request, an optional buffered answer, and optional routing evidence.
The host executes the answer call when `response` is null.

Wire formats are `openai_chat`, `openai_responses`, and `anthropic_messages`.
The translation functions use the existing Switchyard codecs.
