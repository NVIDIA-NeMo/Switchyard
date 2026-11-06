# Self-Delegation Routing

Self-delegation routing lets the efficient model decide each turn for itself.
The efficient model, called the student, starts every reply with one route
line: `{"route":"local"}` or `{"route":"cloud"}`. A local reply is served as it
came. The cloud line is a sentinel: the route calls the capable model with the
client's request, unchanged, and serves that reply behind the sentinel. No
judge, no extra prompt, and no session state are involved.

Use it with a student that was trained to emit the route line, for example a
model fine-tuned on traces collected through this same route. Without such a
student every reply is invalid and the route serves it as it came.

## How it works

1. The route calls the student with the client's request. Both upstream calls
   ask for a buffered reply even when the client streams, so the whole reply is
   available for the check below.
2. The route reads the reply's text and counts its tool calls:
   - Any reply with tool calls is **local** and is served as it came, whatever
     its first line says. Nothing executes twice.
   - A reply whose whole text is `{"route":"cloud"}` is **cloud**.
   - With `lenient_cloud` (the default), the sentinel inside a code fence, with
     single quotes, with extra spacing, or with text after it is also cloud. The
     trace labels it `cloud_lenient`.
   - A reply that would end the client's turn for the wrong reason is
     **invalid**: an empty reply, a local line with nothing after it, a second
     route line, or prose without a route line and without tool calls. The
     `on_invalid` policy asks the student again, then serves the last reply or
     escalates.
3. On cloud, the route calls the capable model with the same request. If the
   capable model starts its text with a route line, that line is removed and
   the trace records `imitated_route_line`. The served text is
   `{"route":"cloud"}`, a line break, and the capable model's text. The served
   usage is the sum of both calls.
4. With `trace_jsonl` set, the route appends one JSON object per decision.

The client sees one answer per request. A streaming client receives the
buffered answer as a stream. The student's reply is held back until the check
finishes, and a cloud decision adds the capable model's latency.

Failures never change a label. A student call that fails returns its error to
the client, which retries on its own, unless `on_student_error = "escalate"`.
A capable call that fails after a cloud decision returns its error to the
client; the next attempt starts again with the student.

## The route line

The contract is the same for serving and for collection:

| Reply | Meaning |
|---|---|
| `{"route":"local"}` then text or tool calls | The student's own step. Served as it came. |
| `{"route":"cloud"}` and nothing else | Delegate this turn. The capable model's step is served behind the line. |

The route line stays visible to the client by default. Both Claude Code and
Codex render it, keep it in the transcript, and send it back in the history, so
the student sees its earlier decisions and the delegated steps on later turns.
`hide_route = true` removes the line from served text; see
[Hiding the route line](#hiding-the-route-line).

The trace also records a strict `valid` flag for every student reply: the text
starts with a route line, a cloud line has nothing after it and no tool calls,
a local line has a tool call or text after it, and no other route line appears.
The export tooling uses this flag; the gateway's own decision is the looser
classification above.

## Configuration

```toml
schema_version = 1

[llm_clients.local_vllm]
format = "openai_chat"
base_url = "http://127.0.0.1:8000/v1"

[llm_clients.inference_hub]
format = "openai_chat"
base_url = "https://inference-api.nvidia.com/v1"
api_key_env = "NVIDIA_API_KEY"

[targets.student]
id = "edgewise-9b"
llm_client = "local_vllm"

[targets.strong]
id = "nvidia/moonshotai/kimi-k3"
llm_client = "inference_hub"

[routes.edgewise]
id = "switchyard/edgewise"
type = "self_delegation"
efficient_target = "student"
capable_target = "strong"
trace_jsonl = "/var/tmp/edgewise-serve-trace.jsonl"
```

| Key | Default | Meaning |
|---|---|---|
| `efficient_target` | required | The student. In collection mode, the teacher. |
| `capable_target` | unset | The strong model called on the cloud sentinel. Unset selects collection mode: the sentinel is served as it came. |
| `trace_jsonl` | unset | File that receives one JSON object per decision. Use one file per server process. A failed write is logged and never fails the request. |
| `hide_route` | `false` | Remove the route line from served text. |
| `lenient_cloud` | `true` | Treat a damaged sentinel as cloud at serving. |
| `on_invalid.retry` | `1` | How many times an invalid reply is asked again, with the same request. |
| `on_invalid.action` | `"serve"` | `serve` the last invalid reply as it came, or `escalate` to the capable model. `escalate` needs `capable_target`. |
| `on_student_error` | `"error"` | `error` returns a failed student call to the client; `escalate` calls the capable model instead. `escalate` needs `capable_target`. |

Both targets answer requests, so neither is a judge-only target. A
`system_prompt` on `efficient_target` is prepended to the student call, because
the route declares that target as its answer target. A `system_prompt` on a
different `capable_target` has no effect: the capable model receives the
client's request unchanged.

Three example files are checked in under
[`examples/edgewise/`](../../examples/edgewise/): serving, collection, and an
evaluation arm whose `capable_target` is the student itself, so the student
answers its own sentinel.

## Collection mode

Collection runs the same route with the teacher as `efficient_target`, no
`capable_target`, and the labeling rubric as the teacher target's
`system_prompt`. The rubric is the only prompt the gateway adds, and it is
added in the client layer below the algorithm, so the request in the trace is
the neutral request the student will later be trained on.

```toml
[targets.teacher]
id = "nvidia/moonshotai/kimi-k3"
llm_client = "inference_hub"
system_prompt = """
Start every reply with exactly one line, {"route":"local"} or {"route":"cloud"},
then the step.
"""

[routes.edgewise_collect]
id = "switchyard/edgewise-collect"
type = "self_delegation"
efficient_target = "teacher"
trace_jsonl = "/var/tmp/edgewise-collect-trace.jsonl"
```

In collection mode a cloud reply is served as it came, so the teacher must put
its step after the cloud line. The export tooling applies its own, stricter
rules to teacher replies.

## Hiding the route line

With `hide_route = true` the served text loses its leading route line and the
line break after it, and a delegated step is served without the sentinel. A
local reply that was only the route line is served without a text block,
because an empty text block is rejected by the Anthropic request format on the
client's next turn. The trace keeps the unstripped reply either way.

Hiding changes what the student sees in its history on later turns. Use it
only with a student trained for history without route lines.

## The trace

Each line of `trace_jsonl` is one JSON object. The request is recorded as the
student engine receives it: the OpenAI Chat Completions body of the neutral
request with the student's provider model, and without `stream`. Replies use
the OpenAI Chat spelling for `tool_calls` and `finish_reason`. Usage counts are
null when the provider did not report them.

```json
{
  "schema_version": "native-v1",
  "ts": "2026-11-05T17:20:31.512Z",
  "route": "switchyard/edgewise",
  "mode": "serve",
  "hide_route": false,
  "lenient_cloud": true,
  "session_id": "fb46caae-eac6-4f5f-83fd-8fc8f5743abb",
  "agent_id": null,
  "turn_id": null,
  "task_id": null,
  "correlation_id": null,
  "wire_format": "anthropic_messages",
  "is_compaction": false,
  "has_compaction": false,
  "request": {"model": "edgewise-9b", "messages": ["..."], "tools": ["..."]},
  "student": {
    "target": "student",
    "attempts": [{
      "content": "{\"route\":\"cloud\"}",
      "tool_calls": [],
      "finish_reason": "stop",
      "usage": {"input_tokens": 1830, "output_tokens": 8},
      "latency_ms": 412,
      "classification": "cloud",
      "valid": true,
      "error": null
    }]
  },
  "decision": "cloud",
  "strong": {
    "target": "strong",
    "content": "I will edit the parser.",
    "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": "{\"cmd\":\"ls\"}"}}],
    "finish_reason": "tool_calls",
    "usage": {"input_tokens": 1790, "output_tokens": 61},
    "latency_ms": 2210,
    "imitated_route_line": false,
    "fallback": false,
    "error": null
  },
  "served": {"by": "strong", "content": "{\"route\":\"cloud\"}\nI will edit the parser.", "tool_calls": 1, "stripped_route_line": false}
}
```

| Field | Meaning |
|---|---|
| `route` | The model name the client sent, which is the route's `id`. |
| `mode` | `serve` with a `capable_target`, `collect` without one. |
| `session_id`, `agent_id`, `turn_id`, `task_id`, `correlation_id` | Correlation IDs resolved from the client's headers, or null. |
| `wire_format` | The client's API: `anthropic_messages`, `openai_responses`, `openai_chat`, or null. |
| `is_compaction`, `has_compaction` | The request performs a compaction, or its history carries a compaction marker. |
| `student.attempts[]` | Every student call, in order. `classification` is `local`, `cloud`, `cloud_lenient`, or `invalid`, and null when the call failed; `error` holds the failure message. |
| `decision` | The classification of the last student reply, with `cloud_lenient` collapsed to `cloud`. `invalid` when every student call failed. |
| `strong` | The capable model's call, or null when it was not made. `content` is its text before the merge. `fallback` is true when the call came from `on_invalid` or `on_student_error` rather than from a cloud decision. |
| `served` | What the client received, or null when the request failed. `tool_calls` is a count. `stripped_route_line` is true when `hide_route` removed or withheld a route line. |

The routing log of the reference server records both calls with their usage
as usual; the trace is the only place that holds request and reply content.

## When not to use self-delegation routing

- **The efficient model emits no route line.** Every reply is then invalid.
  The `on_invalid` policy asks again and serves or escalates, but nothing is
  ever delegated by decision. Use a judge-based route such as
  [Task](llm_classifier_routing.md) or [Escalation](escalation_router_routing.md).
- **Replies must start streaming before the model finishes.** The student's
  reply is buffered before the check. Set the client's request watchdog with
  that in mind.
- **Text is graded.** The route line is the first line of every served text.
  Tasks graded on repository or terminal state are unaffected; a grader that
  reads the final message needs `hide_route` or has to skip the first line.
