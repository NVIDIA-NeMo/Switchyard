# C translation interface

An experimental C interface to Switchyard's request and response codecs. The
interface uses NUL-terminated JSON strings. Ownership rules are in
[`include/switchyard.h`](include/switchyard.h).

## JSON payloads

| Function | Input |
| --- | --- |
| `sy_decode_request`, `sy_decode_response` | `{format, body}` |
| `sy_encode_request` | `{format, request, model}` |

`request` and model responses use Switchyard's normalized protocol JSON.
`sy_encode_request` applies the selected model before encoding the request.

Wire formats are `openai_chat`, `openai_responses`, and `anthropic_messages`.
The translation functions use the existing Switchyard codecs.

Fallible calls return NULL on success or an owned error string. Free every
returned error or JSON string with `sy_string_free`.
