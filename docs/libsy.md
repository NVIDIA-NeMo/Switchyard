# libsy Python API

The `switchyard.libsy` API embeds Switchyard's Rust-owned routing algorithms
directly in a Python process. Use it when a host application needs to choose a
model itself instead of sending traffic through Switchyard's HTTP proxy or YAML
route bundles.

`Algorithm.decide()` selects one configured target and returns its routing
metadata without executing that selected target. It does not expose an HTTP
endpoint.

## Decide without running the selected model

Targets are fixed when an algorithm is constructed. A call to `decide()` takes
only the normalized request and optional request headers:

```python
import asyncio

from switchyard.libsy import LlmTarget, algorithms


request = {
    "model": "auto",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Explain this traceback"},
            ],
        }
    ],
}


async def main() -> None:
    router = algorithms.random(
        [LlmTarget("fast"), LlmTarget("quality")],
        weights=[3, 1],
        seed=42,
    )

    decision = await router.decide(request)
    print(decision)


asyncio.run(main())
```

The seeded example returns:

```python
{
    "selected_model": "fast",
    "reasoning": "random routing selected target 'fast'",
    "routing_tier": None,
}
```

All three keys are always present. `reasoning` and `routing_tier` are `None`
when the algorithm does not assign them.

The `fast` and `quality` targets do not need clients because `decide()` never
executes either candidate.

## Normalized requests

`run()` and `decide()` accept libsy's normalized request dictionary, not an
OpenAI or Anthropic wire payload. In particular, every message's `content` is a
list of typed blocks:

```python
request = {
    "model": "auto",
    "messages": [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Summarize this change"}],
        }
    ],
}
```

Do not pass a bare content string such as `"content": "Summarize this change"`.
Provider-specific request conversion belongs in the client or host integration
that surrounds libsy.

## Available algorithms

The Python bindings support the same five algorithm types as Switchyard's Rust
server:

| Factory | Selection |
|---|---|
| `algorithms.noop()` | Select the inbound request model, or `switchyard/noop` when absent. |
| `algorithms.passthrough(target)` | Always select one configured target. |
| `algorithms.random(targets, *, weights=None, seed=None)` | Choose from one or more targets using optional relative weights. |
| `algorithms.llm_classifier(...)` | Ask a judge whether the efficient or capable target should handle the request. |
| `algorithms.stage_router(...)` | Route coding-agent turns from tool signals, with an optional LLM classifier fallback. |

The generic Rust `FallThrough` composition is not exposed as a Python factory.

### Passthrough

```python
router = algorithms.passthrough(LlmTarget("model-a"))
decision = await router.decide(request)
```

Passthrough decisions have `reasoning=None` and `routing_tier=None`.

### Weighted random

Weights are relative and follow target order. A zero weight disables a target;
at least one weight must be positive. Supplying a seed makes the shared
selection sequence reproducible:

```python
router = algorithms.random(
    [
        LlmTarget("efficient"),
        LlmTarget("capable"),
    ],
    weights=[3, 1],
    seed=42,
)
```

Calls to `run()` and `decide()` on the same router consume the same random
sequence.

## LLM classifier

The LLM classifier is decision-only with respect to the selected candidate,
but it may call its judge as an auxiliary routing operation. Give the judge a
client and leave the efficient and capable candidates clientless:

```python
from collections.abc import Mapping


class JudgeClient:
    async def call(
        self,
        request: Mapping[str, object],
    ) -> Mapping[str, object]:
        # Call a provider here and return a normalized aggregate response.
        ...


router = algorithms.llm_classifier(
    judge=LlmTarget("judge", JudgeClient()),
    efficient=LlmTarget("fast"),
    capable=LlmTarget("strong"),
    base_threshold=0.5,
    min_confidence=0.6,
    capability_elevated_floor=0.75,
    session_affinity=True,
    message_hash_fallback=False,
    recent_turn_window=3,
)

decision = await router.decide(request)
```

`base_threshold` is required. The other controls are:

| Argument | Default | Meaning |
|---|---:|---|
| `min_confidence` | `0.0` | Minimum judge confidence that permits the efficient route. |
| `capability_elevated_floor` | `None` | Higher solve-probability floor for uncertain, unmatched, or unsupported tasks. `None` reuses `base_threshold`. |
| `session_affinity` | `False` | Retain the first assignment for subsequent requests in the same session. |
| `message_hash_fallback` | `False` | Derive affinity from the first user message when session metadata is absent. Requires `session_affinity=True`. |
| `recent_turn_window` | `None` | Include the opening task and this many recent conversation turns in the judge request. `None` judges only the newest user message. |

Classifier decisions use `routing_tier="weak"` for the efficient target and
`routing_tier="strong"` for the capable target when their target names differ.
A judge failure or malformed verdict follows the existing fail-open policy and
selects the capable target.

The selected efficient or capable client is never accessed by `decide()`.

## Stage router

The stage router first scores normalized coding-agent tool calls and results. A
decisive signal selects the capable or efficient target without any model call.
If the signals are ambiguous, the router can call an optional judge before
falling open to the tier selected by `picker`.

```python
stage_request = {
    "model": "auto",
    "messages": [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Fix the failing build"}],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "id": "call-1",
                    "name": "Bash",
                    "arguments": {"command": "cargo test"},
                }
            ],
        },
        {
            "role": "tool",
            "content": [
                {
                    "type": "tool_result",
                    "tool_call_id": "call-1",
                    "content": [
                        {
                            "type": "text",
                            "text": "fatal runtime error: out of memory",
                        }
                    ],
                    "is_error": True,
                }
            ],
        },
    ],
}

router = algorithms.stage_router(
    capable=LlmTarget("strong"),
    efficient=LlmTarget("fast"),
    picker="efficient_first",
    confidence_threshold=0.5,
)

decision = await router.decide(stage_request)
```

This request selects `strong` with `routing_tier="strong"` and does not require
a client on either candidate.

The full factory exposes the configuration that affects the Rust StageRouter:

```python
router = algorithms.stage_router(
    capable=LlmTarget("strong"),
    efficient=LlmTarget("fast"),
    picker="efficient_first",
    confidence_threshold=0.5,
    recent_turn_window=4,
    handoff_escalation_note="Continue the failed diagnosis.",
    handoff_deescalation_note="The build is healthy again.",
    handoff_only_on_wrong_signal_escalation=True,
    capable_system_prompt="Solve difficult failures.",
    efficient_system_prompt="Handle routine work.",
    judge=LlmTarget("judge", JudgeClient()),
    classifier_base_threshold=0.5,
    classifier_min_confidence=0.6,
    classifier_capability_elevated_floor=0.75,
    classifier_recent_turn_window=3,
)
```

The arguments behave as follows:

| Argument | Default | Meaning |
|---|---:|---|
| `picker` | Required | `"capable_first"` falls open to the capable tier; `"efficient_first"` falls open to the efficient tier. |
| `confidence_threshold` | Required | Minimum tool-signal confidence needed for a decisive signal route. |
| `recent_turn_window` | `None` | Number of recent tool results scored. `None` uses the Rust default. |
| `handoff_escalation_note` | `None` | Note appended when handing a qualifying turn to the capable tier. |
| `handoff_deescalation_note` | `None` | Optional note appended when handing work back to the efficient tier. Requires an escalation note. |
| `handoff_only_on_wrong_signal_escalation` | `True` | Restrict the escalation note to signal-driven escalation instead of an ambiguous capable default. |
| `capable_system_prompt` | `None` | System prompt applied when `run()` executes the capable target. |
| `efficient_system_prompt` | `None` | System prompt applied when `run()` executes the efficient target. |
| `judge` | `None` | Optional auxiliary judge target used only when tool signals abstain. |
| `classifier_base_threshold` | `None` | Solve-probability threshold for the optional judge. Must be supplied together with `judge`. |
| `classifier_min_confidence` | `0.0` | Minimum confidence accepted from the optional judge. |
| `classifier_capability_elevated_floor` | `None` | Higher judge threshold for uncertain, unmatched, or unsupported tasks. |
| `classifier_recent_turn_window` | `None` | Conversation window sent to the optional judge. |

Handoff notes and tier prompts affect the request executed by `run()`.
`decide()` still processes the same routing state and metadata, but returns
before executing the selected target. StageRouter does not expose classifier
session affinity because its judge is a per-turn fallback inside the signal
cascade.

## Request headers and session affinity

Pass headers separately from the normalized request:

```python
decision = await router.decide(
    request,
    headers={"x-switchyard-session-id": "conversation-42"},
)
```

Headers are normalized into libsy metadata in the same way as an HTTP host.
With classifier session affinity enabled, a retained assignment can avoid a
later judge call.

## `run()` versus `decide()`

| Method | Selected target executed? | Result |
|---|---|---|
| `await algorithm.decide(request, headers=None)` | No | One decision dictionary. |
| `await algorithm.run(request, headers=None)` | Yes, except for no-op | `(decision_trace, normalized_response)`. |

Every target reachable through `run()` needs a client. A target used only as a
candidate for `decide()` can omit it:

```python
target = LlmTarget("fast")
```

Decision dictionaries in both `decide()` results and `run()` traces contain
`selected_model`, `reasoning`, and `routing_tier`.

## Errors and limitations

- Invalid target objects, algorithm configuration, and normalized request
  shapes raise `TypeError` or `ValueError` at the Python boundary.
- Managed execution failures raise `switchyard.libsy.LibsyError`.
- An LLM classifier needs a client on its judge target. Its efficient and
  capable targets do not need clients for `decide()`.
- A stage router needs a client only on its optional judge target for
  `decide()`; its capable and efficient candidates can remain clientless.
- `decide()` returns the initial route. It cannot predict context-window
  fallback because that requires executing a target and observing an overflow.
- Candidate sets cannot be replaced per request.
- Decision streaming is not exposed.
- A custom Rust algorithm must implement the decision task contract before
  `decide()` can drive it.

## Related documentation

- [Routing Overview](routing_algorithms/overview.md)
- [Random Routing](routing_algorithms/random_routing.md)
- [LLM Classifier Routing](routing_algorithms/llm_classifier_routing.md)
- [Core Concepts](core_concepts.md)
