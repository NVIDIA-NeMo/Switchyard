# Design: `@route` decorator — pure-Python port of the libsy optimizer interfaces

**Date:** 2026-07-04
**Status:** Approved (design), pending implementation plan
**Branch:** `feat/libsy-routers-poc`

## Problem

`crates/libsy` defines a new set of Rust interfaces for LLM traffic routing — the
`AgentApiOptimizer` pattern: a per-session stateful optimizer driven by a
`feed(input) → optimize() → Decision{ModelInference | Return}` loop, minted by an
`AgentApiOptAlgorithm` factory. Two concrete algorithms exist:

- `rand.rs` — `RandomRouterAlgorithm`: weighted random selection over N targets.
- `llm_class.rs` — `LlmClassifierAlgorithm`: a genuinely multi-round router that
  first runs a classifier model call, then routes to a strong/weak model based on
  the parsed score, then returns control.

libsy is a **pure-Rust crate with no Python bindings**. We want a Python
`@route` decorator — in the spirit of litellm's unified `completion()` entry point
— that wraps a user's async "make an API call to a model" function and drives it
through one of these routing algorithms, rewriting the model (and, for multi-round
routers, the prompt) per the optimizer's decisions.

## Goals

- A pure-Python port of the libsy optimizer interfaces and the two algorithms,
  mirroring the Rust design 1:1 in behavior.
- A `@route(algorithm, ...)` decorator that transparently drives the wrapped
  async function through the optimizer loop.
- Correct support for **multi-step** routers (the LLM classifier), where the
  wrapped function is invoked more than once per user call.
- Faithful port of the Rust tests to pytest.

## Non-Goals

- No pyo3 / Rust bindings. This is a Python-only exploration.
- No sync-function support. Async only (matches the Rust async traits 1:1).
- No integration into the production `switchyard` chain, endpoints, or CLI.
- No streaming responses. Requests/responses are single-shot.
- No new dependencies. `litellm` is referenced only in docstrings/examples; the
  decorator does not import it.

## Decisions (from brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Substrate | Pure-Python port | libsy has no bindings; fastest POC, mirrors Rust 1:1. |
| Function contract | Configurable adapters with litellm-shaped defaults | Works with any fn shape; common case needs zero config. |
| Sync/async | Async only | Cleanest 1:1 mapping to the Rust async traits. |
| Optimizer lifetime | One optimizer instance per decorated call | Natural decorator mapping of the Rust "one instance per session". |

## Architecture

A self-contained new subpackage `switchyard/lib/agentapi/`, mirroring the
`crates/libsy` module split. It does not depend on the rest of the `switchyard`
package and nothing in the package depends on it (POC isolation).

```
switchyard/lib/agentapi/
├── __init__.py        # public exports: route, RandomRouter, WeightedModel,
│                      #   LlmClassifier, ChatRequest, ChatResponse, EnrichmentData,
│                      #   AgentApiOptimizer, AgentApiOptAlgorithm, Decision, OptInput
├── chat.py            # ChatRequest, ChatResponse, EnrichmentData
├── optimizer.py       # OptInput, Decision, OptimizerResponse, abstract
│                      #   AgentApiOptimizer + AgentApiOptAlgorithm
├── rand.py            # WeightedModel, RandomRoutingDecision, RandomRouter
├── llm_class.py       # ClassifierTier, ClassifierRoutingDecision, LlmClassifier
└── decorator.py       # route(...)
```

### Ported interface (1:1 with `lib.rs`)

Naming note: the working tree of `lib.rs` is mid-rename
(`ChatRequest` → `AgentApitRequest`, a typo; the router modules still reference
`ChatRequest`, so the crate does not currently compile). This port uses the
coherent pre-rename names `ChatRequest` / `ChatResponse` and does **not**
propagate the typo.

**`chat.py`** — plain dataclasses:

```python
@dataclass
class ChatRequest:
    prompt: str
    model: str

@dataclass
class ChatResponse:
    completion: str

@dataclass
class EnrichmentData:
    session_id: str | None = None
    agent_id: str | None = None
    task_id: str | None = None
    correlation_id: str | None = None
    extra_metadata: dict[str, str] | None = None
```

**`optimizer.py`** — the input/decision unions and the abstract roles.

`OptInput` is a tagged union. Mirroring `AgentApiOptInput::{Request, Response,
Metadata}`. Modeled as a small closed set of dataclasses under a `OptInput` base:

```python
class OptInput: ...
@dataclass
class RequestInput(OptInput):   request: ChatRequest
@dataclass
class ResponseInput(OptInput):  response: ChatResponse
@dataclass
class MetadataInput(OptInput):  metadata: dict[str, str]
```

> Faithfulness note: the Rust `AgentApiOptInput::Response` variant currently wraps
> a `ChatRequest` (the tests stuff the completion text into its `prompt` field).
> The Python port cleans this up by using `ChatResponse.completion`, which is the
> evident intent. This is the one intentional deviation from the Rust source, and
> it is documented in the module docstring.

`OptimizerResponse` and `Decision`:

```python
@dataclass
class OptimizerResponse:
    requests: list[ChatRequest]
    enrichment_data: list[EnrichmentData] = field(default_factory=list)
    decision_reasoning: str | None = None
    decision_info: object | None = None   # concrete `D` per algorithm; no generics

class Decision: ...
@dataclass
class ModelInference(Decision):  response: OptimizerResponse
@dataclass
class Return(Decision):          pass
```

`AgentApiOptimizer` (abstract base) and `AgentApiOptAlgorithm` (factory):

```python
class AgentApiOptimizer(abc.ABC):
    async def feed(self, input: OptInput, enrichment: EnrichmentData) -> None: ...
    @abc.abstractmethod
    async def optimize(self) -> Decision: ...

class AgentApiOptAlgorithm(abc.ABC):
    @abc.abstractmethod
    def optimizer(self) -> AgentApiOptimizer: ...
```

`feed` has a no-op default (matching the Rust default trait method). `optimize` is
made **abstract** in Python — a deliberate tightening of the Rust trait, which
ships a default `optimize` returning an empty `ModelInference`. Both concrete
algorithms override it, so requiring the override surfaces an incomplete
optimizer at construction time rather than silently no-op'ing. The `D` generic is
dropped — Python attaches `decision_info` as a plain object.

### Algorithm: `rand.py` — `RandomRouter`

Ports `RandomRouterAlgorithm` + `RandomRouter`.

- `WeightedModel(model: str, weight: float)`.
- `RandomRouter(models: list[WeightedModel], rng_seed: int | None = None)` — the
  factory (`AgentApiOptAlgorithm`). `optimizer()` mints a fresh per-session
  optimizer. Uses `random.Random(seed)` when a seed is given, else
  `random.Random()`.
- The per-session optimizer holds `models`, `rng`, `pending_request`, `completed`.
  - `feed(RequestInput)` → buffer the request.
  - `feed(ResponseInput)` → mark completed.
  - `optimize()` when completed → `Return`.
  - `optimize()` with a pending request → weighted draw: sum finite positive
    weights; error if the sum ≤ 0 ("random router has no target with positive
    weight"); draw in `[0, total)`, walk cumulative weights, fall back to the last
    positively-weighted target on floating-point overshoot. Rewrite
    `request.model`, return `ModelInference` with a `RandomRoutingDecision`
    (`selected_model`, `draw`, `total_weight`) as `decision_info`.
  - `optimize()` before any request was fed → error ("optimize called before a
    request was fed").

### Algorithm: `llm_class.py` — `LlmClassifier`

Ports `LlmClassifierAlgorithm` + `LlmClassifierRouter`, including the phase
machine.

- `ClassifierTier` enum (`STRONG` / `WEAK`) with a stable `as_str()`.
- `ClassifierRoutingDecision(score: float | None, threshold, tier, selected_model)`.
- `LlmClassifier(classifier_model, strong_model, weak_model, threshold)` — the
  factory. `optimizer()` mints a per-session optimizer starting in
  `AwaitingRequest`.
- Phases: `AwaitingRequest → Classify → AwaitingScore → Route → AwaitingResponse
  → Done` (mirrors the Rust `Phase` enum exactly).
- `CLASSIFIER_PROMPT_PREAMBLE` matches the Rust literal's **resolved** value (the
  Rust source uses a `\`-newline continuation with leading indentation that
  collapses to single spaces), i.e. the exact string:
  `"Rate how strongly this request needs a frontier model. Reply with a single strong-win-rate score in [0, 1]:\n"`.
- Behavior mirrors `llm_class.rs`:
  - `feed(RequestInput)` → buffer request, phase `Classify`.
  - `feed(ResponseInput)` in `AwaitingScore` → parse score
    (`float(text.strip())`, `None` on failure), phase `Route`. In
    `AwaitingResponse` → phase `Done`. Any other phase → error ("classifier router
    received a response outside a pending model call").
  - `optimize()` in `Classify` → emit `ChatRequest(classifier_model, preamble +
    user_prompt)`, phase `AwaitingScore`.
  - `optimize()` in `Route` → decide tier: `score >= threshold` → strong;
    `score < threshold` → weak; `score is None` → **default strong** (keep traffic
    flowing when the classifier output was unusable). Rewrite model, phase
    `AwaitingResponse`, return `ModelInference` with `ClassifierRoutingDecision`.
  - `optimize()` in `Done` → `Return`.
  - `optimize()` in `AwaitingRequest` / `AwaitingScore` / `AwaitingResponse` →
    the corresponding "called before … was fed" errors.

### The decorator: `decorator.py`

```python
def route(
    algorithm: AgentApiOptAlgorithm,
    *,
    get_prompt: Callable[[Mapping], str] = _default_get_prompt,
    apply_request: Callable[[dict, ChatRequest], None] = _default_apply_request,
    get_completion: Callable[[object], str] = _default_get_completion,
) -> Callable:
    ...
```

`route` returns a decorator that wraps an **async** function and returns an async
wrapper. The wrapper, per call:

1. Binds the call's args to a mutable `kwargs` view (via `inspect.signature`,
   normalizing positional args to keyword form so adapters see a uniform mapping).
2. Mints a fresh optimizer: `optimizer = algorithm.optimizer()`.
3. Extracts the initial prompt (`get_prompt(kwargs)`) and model (`kwargs["model"]`)
   and feeds `RequestInput(ChatRequest(prompt, model))` with empty
   `EnrichmentData`.
4. **Round-agnostic drive loop:**

   ```python
   last_resp = None
   while True:
       decision = await optimizer.optimize()
       if isinstance(decision, Return):
           break
       for req in decision.response.requests:
           apply_request(kwargs, req)          # sets model AND prompt for this round
           last_resp = await fn(**kwargs)
           completion = get_completion(last_resp)
           await optimizer.feed(ResponseInput(ChatResponse(completion)),
                                EnrichmentData())
   return last_resp
   ```

5. Returns `last_resp` — the wrapped fn's return value from the **last** model
   call before `Return`. For the classifier this is the routed call's response;
   the earlier classifier call's response is consumed internally as the score.

**Why this supports multi-step routers:** the loop makes no assumption about how
many `ModelInference` rounds occur. Each `ModelInference` carries full
`ChatRequest`s (model + prompt); `apply_request` applies the *entire* request to
the call each round, so the classifier round correctly sends the classifier
prompt to the classifier model, and the routed round sends the original prompt to
the routed model. A 1-round router (`rand`) and an N-round router (`llm_class`)
drive through identical code.

**Adapters — litellm-shaped defaults (zero config for the common case):**

- `_default_get_prompt(kwargs)` → `kwargs["messages"][-1]["content"]`; falls back
  to `kwargs["prompt"]` when there is no `messages` key.
- `_default_apply_request(kwargs, req)` → sets `kwargs["model"] = req.model`; if
  `messages` is present, replaces the **last user message's content** with
  `req.prompt` (preserving system messages / earlier history); otherwise sets
  `kwargs["prompt"] = req.prompt`.
- `_default_get_completion(resp)` → `resp.choices[0].message.content`; falls back
  to `resp` itself when it is a `str`.

All three are overridable to support arbitrary function shapes.

`functools.wraps` preserves the wrapped function's identity. If the wrapped
function is not a coroutine function, `route` raises `TypeError` at decoration
time (async-only contract).

## Data flow

```
caller: await chat(model="auto", messages=[{user: "..."}])
  → wrapper binds args → kwargs
  → feed(RequestInput(ChatRequest(prompt, "auto")))
  → loop:
      optimize() → ModelInference(reqs)     # rand: 1 round; classifier: 2 rounds
        → apply_request(kwargs, req); resp = await chat(**kwargs)
        → feed(ResponseInput(ChatResponse(completion(resp))))
      optimize() → Return
  → return last resp
```

## Error handling

- Optimizer state errors (optimize-before-feed, no-positive-weight, response out
  of phase) propagate as `ValueError` (Python analog of the Rust
  `Box<dyn Error>`), with messages matching the Rust strings.
- Decorating a non-coroutine function raises `TypeError` at decoration time.
- Adapter failures (e.g. missing `model` kwarg, unparseable response shape)
  surface as the natural `KeyError` / `AttributeError`; the POC does not wrap
  these.

## Testing

Ported to pytest (`asyncio_mode = "auto"`, no explicit markers), against a mock
async wrapped function, under `tests/`:

**`tests/test_agentapi_rand.py`** (ports `rand.rs` tests):
- all weight on one model always selects it
- selection frequencies track weights over 20k draws (with fixed seed)
- decision reports draw within total weight
- returns to agent after a mocked response is fed
- optimize-before-feed errors
- no-positive-weight errors
- factory mints independent deterministic optimizers (same seed → same first draw)

**`tests/test_agentapi_llm_class.py`** (ports `llm_class.rs` tests):
- score ≥ threshold routes strong; score < threshold routes weak
- score exactly at threshold routes strong
- unparseable score defaults to strong (score `None`)
- optimize-before-feed errors
- optimize-before-classifier-response errors

**`tests/test_agentapi_decorator.py`** (new, decorator end-to-end):
- rand: wrapped fn called once; returned response is the routed call's; model was
  rewritten to the weighted target.
- classifier: wrapped fn called twice (classifier + routed); the classifier round
  receives the classifier model and preamble prompt; the returned response is the
  routed round's; the routed round receives the original prompt.
- custom adapters: a non-litellm function shape drives correctly via overridden
  `get_prompt` / `apply_request` / `get_completion`.
- decorating a sync function raises `TypeError`.

## Validation

- `uv run ruff check .` — zero errors.
- `uv run mypy switchyard` — clean (strict).
- `uv run pytest tests/test_agentapi_*.py -v` — green.

## Public API

Exported from `switchyard/lib/agentapi/__init__.py`. Whether these also surface
from the top-level `switchyard/__init__.py` `__all__` is deferred to the
implementation plan (POC isolation argues for keeping them subpackage-local
initially).
