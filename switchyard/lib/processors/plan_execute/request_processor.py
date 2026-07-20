# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM-backed planning request processor."""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from switchyard.lib.llm_client import OpenAILLMClient
from switchyard.lib.processors.message_condensing import strip_markdown_fence
from switchyard.lib.processors.plan_execute.plan import (
    CTX_PLANNER_DECISION,
    PlannerDecision,
)
from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.stats_accumulator import StatsAccumulator
from switchyard_rust.core import ChatRequest, ChatRequestType

log = logging.getLogger(__name__)

DEFAULT_INITIAL_PLANNER_SYSTEM_PROMPT = """\
You are drafting the agent's opening thought before it tackles a
task.  The conversation summary you've been given is the request the
agent is about to handle.  **You have not seen the actual codebase
or filesystem** — your job is to outline how the agent should
approach the work, not to prescribe specific file paths or commands
based on assumed structure.

Output exactly one JSON object with two fields:

  {"plan_needed": boolean, "plan_text": string}

# Fast path — no plan

For trivial requests (greetings, single one-line questions,
clarifications, anything answerable in one short turn), emit exactly:

  {"plan_needed": false, "plan_text": ""}

Stop generating immediately.  Do not produce any further tokens.

# Slow path — emit a plan

When the task involves multi-step work (debugging, refactoring,
tool orchestration, multi-stage analysis), set plan_needed=true and
populate plan_text with the plan content.

plan_text will be **prepended to the agent's next response** — the
agent will literally start its response with your text and continue
from where you end.  So write the plan as the agent's own opening
reasoning, in first-person ("I'll start by ...", "My approach:",
"Let me think this through:").

## What the plan should contain

Use this structure (markdown headers are fine but optional — what
matters is the substance):

* **Context** — one to two sentences on what the task actually
  asks for and what success looks like.

* **Approach** — the strategy you'd take, framed as exploration-
  first.  Start with "I'll first explore the structure to find …"
  or "I need to understand X before changing anything," then
  describe what comes next.  Don't prescribe specific paths or
  filenames you'd be guessing at — leave that for the agent to
  discover.

* **Steps** — an ordered outline, but framed as *what to investigate
  or attempt*, not exact-path-X-do-Y instructions.  Example: ✔
  "Locate the failing test and read its assertions" rather than ✘
  "Edit /app/tests/test_x.py line 42".

* **Risks / unknowns** (optional) — short list of things you're
  uncertain about that the agent should pay attention to (e.g.
  "Project layout is unknown — agent should verify before assuming
  Python vs Node.js"; "Test framework may be pytest or unittest").

* **Verification** — how the agent will know it's done.  Frame as
  expectations to validate, not commands to run literally.

## What NOT to include

* **Do not list specific file paths** unless the request itself
  named them.  You're guessing; the agent will discover the real
  layout.  Bad: "Edit /app/src/main.py line 42".  Good: "Locate
  the main entry point and adjust the handler."
* **Do not prescribe specific shell commands** unless the request
  named them.  The agent will choose appropriate commands once it
  sees the environment.
* **Do not produce long markdown documentation** — keep the plan
  tight, scannable, ~300-500 tokens.  This is the agent's *opening
  thought*, not a spec.

End with a short transition sentence so the agent rolls naturally
into execution: "Let me start by exploring the workspace." or "I'll
begin with step 1." or similar.

# Do NOT

- Wrap the JSON in markdown code fences (no ``` around the output)
- Include `<plan>` tags or any wrapping markers in plan_text — the
  harness adds them
- Emit commentary or chain-of-thought outside the JSON object
"""

DEFAULT_REVISION_PLANNER_SYSTEM_PROMPT = """\
You are a software architect reassessing an ongoing implementation.
The agent has been working on a task using an existing plan, and you
are deciding whether mid-stream revision is warranted.

The conversation summary you've been given starts with the prior
plan (under "Prior plan:"), then the full conversation context
including the agent's actions and any tool results.

Output exactly one JSON object with two fields:

  {"plan_needed": boolean, "plan_text": string}

# Default — no revision

**The dominant case is plan_needed=false.**  Most mid-task turns are
steady-state continuation — the existing plan is still good and the
agent should keep working without interruption.  Emit exactly:

  {"plan_needed": false, "plan_text": ""}

Stop generating immediately.

# Three reasons to revise

Only set plan_needed=true when one of these applies:

1. **New information contradicts the plan** — the agent discovered the
   file layout differs from what was assumed, a dependency is missing,
   the test framework is different than the plan presumed.
2. **A tool error reveals a different problem** — the failure makes
   clear the original approach won't work as planned.
3. **The user redirected** — the latest user message changes the
   task or constrains the approach in a way the prior plan doesn't
   cover.

Do **not** revise just to:
- Restate the current plan with cosmetic edits
- Add steps for work the agent has already completed
- Refine wording without substance
- "Improve" a plan that's working fine

# Emitting a revision

If you revise, plan_text will be **prepended to the agent's next
response** with a `<plan>` wrapper.  Therefore:

* Briefly acknowledge what changed (one short sentence).
* Update the structured plan (Context / Approach / Steps / Critical
  Files / Verification) — preserve continuity with the prior plan;
  only revise what actually changed.
* Use first-person voice.
* End with a transition like "Resuming with the revised plan." or
  "Continuing from here." so the agent rolls back into execution.

# Do NOT

- Wrap the JSON in markdown code fences
- Include `<plan>` tags or wrapping markers in plan_text — the
  harness adds them
- Emit commentary outside the JSON object
"""


class PlanningTriggerMode(str, Enum):
    """How :class:`PlanningRequestProcessor` decides whether to run."""

    DISABLED = "disabled"
    ALWAYS = "always"


class PlanningError(RuntimeError):
    """Raised when the planner cannot produce a valid decision."""


class PlanningConfig(BaseModel):
    """Configuration for :class:`PlanningRequestProcessor`.

    Planning is intentionally independent from classifier routing signals.
    ``trigger_mode=ALWAYS`` runs the planner LLM for every request; the
    planner itself then decides whether the request warrants a plan (see
    :class:`PlannerDecision`).  ``DISABLED`` skips the planner call
    entirely, leaving requests untouched and burning no planner tokens.
    """

    model_config = ConfigDict(frozen=True)

    model: str = Field(min_length=1)
    api_key: str | None = None
    base_url: str | None = None
    timeout_s: float | None = Field(default=None, gt=0.0)
    max_request_chars: int = Field(default=16_000, ge=256)
    trigger_mode: PlanningTriggerMode = PlanningTriggerMode.ALWAYS
    inject_plan: bool = True
    fail_open: bool = True
    initial_system_prompt: str = Field(
        default=DEFAULT_INITIAL_PLANNER_SYSTEM_PROMPT,
        min_length=1,
    )
    """Planner system prompt used on the **first** turn of a task —
    when the executor conversation has no prior assistant message and
    the planner is drafting the opening anchor.  The default voice is
    "I'll approach this as follows..." (initial intent)."""

    revision_system_prompt: str = Field(
        default=DEFAULT_REVISION_PLANNER_SYSTEM_PROMPT,
        min_length=1,
    )
    """Planner system prompt used on **subsequent** turns of a task —
    when there is already a prior assistant message and the planner is
    deciding whether mid-task replanning is warranted.  The default
    strongly biases toward ``plan_needed=false`` (fast steady-state)
    and only emits a revision when something material changed.  The
    voice when it does revise is "Pausing to reassess..."."""
    #: Hint vLLM upstreams (NVIDIA Inference Hub uses vLLM under LiteLLM)
    #: to skip chain-of-thought when generating the planner JSON by
    #: sending ``extra_body={"chat_template_kwargs": {"enable_thinking":
    #: False}}``.  Required for DeepSeek V4 Flash / V4 Pro: without it,
    #: those models misroute structured-JSON output into
    #: ``reasoning_content`` and leave ``content`` empty, so the planner
    #: returns nothing parseable.
    #:
    #: Defaults to ``False`` because non-DeepSeek upstreams reject the
    #: unknown ``chat_template_kwargs`` field — verified empirically:
    #: Anthropic Opus on Bedrock 400s with ``"chat_template_kwargs: Extra
    #: inputs are not permitted"``.  Callers using a DeepSeek planner
    #: must set this to ``True`` explicitly.
    disable_reasoning: bool = False

    #: Sampling temperature for the planner call.  ``0.0`` (default)
    #: gives deterministic JSON output on most upstreams.  Set to
    #: ``None`` to omit the field entirely — required for Bedrock-hosted
    #: Anthropic models, which 400 with ``"temperature is deprecated for
    #: this model"`` when the field is present at all.
    #: :class:`~switchyard.lib.profiles.PlanExecuteProfileConfig`
    #: auto-detects Anthropic-family planners (via
    #: :func:`is_anthropic_model`) and sets this to ``None`` so Opus 4.7
    #: on Bedrock can serve as the planner.
    temperature: float | None = 0.0

    #: ``response_format`` override sent on the planner call.  Default
    #: ``{"type": "json_object"}`` constrains the decoder to valid JSON on
    #: most upstreams.  Anthropic-via-LiteLLM (Bedrock Opus) has no native
    #: ``response_format``: LiteLLM rewrites ``json_object`` into a *forced
    #: tool call* to coerce Anthropic's Messages API into emitting JSON, so
    #: the planner output arrives in ``tool_calls`` rather than ``content``.
    #: :class:`OpenAIChatPlannerClient` reads the tool-call arguments when
    #: ``content`` is empty (see :func:`_tool_call_arguments`), so the forced
    #: call makes JSON *validity* a hard guarantee rather than a system-prompt
    #: hope — there is no need to drop the field for Anthropic planners.  Set
    #: to ``None`` to omit it entirely.
    response_format: dict[str, Any] | None = Field(
        default_factory=lambda: {"type": "json_object"},
    )

    #: Per-call HTTP headers forwarded on every planner completion.
    #: Mirrors :attr:`~switchyard.lib.backends.llm_target.LlmTarget.extra_headers`
    #: for the planner's own LLM call, which goes through a separate
    #: client (not :class:`LlmTarget`).  Typical use: pin
    #: ``X-Inference-Priority: batch`` on benchmark deployments so the
    #: planner's calls land on the same benchmarking gateway as the
    #: executor backend and inherit the relaxed proxy timeout.
    #: Unknown headers are silently ignored upstream, so passing them
    #: on non-NIH endpoints is benign — except on NIH's Bedrock proxy,
    #: where ``X-Inference-Priority`` triggers an alternate IAM-bound
    #: route; the benchmark entrypoints gate this header on a
    #: ``_NIH_OSS_PROVIDERS`` allowlist.
    extra_headers: dict[str, str] | None = None

    #: Throttle the planner LLM call to roughly every ``cadence_n``
    #: assistant turns.  Default ``2`` empirically beat both ``1``
    #: (every turn, expensive + noisy) and ``4`` (too sparse) on the
    #: TB-Lite Nemotron-Nano sweep.
    #:
    #: Turn counting is implicit: the processor scans
    #: ``request.body["messages"]`` for ``role: "assistant"`` entries
    #: and fires when ``len(assistant_messages) % cadence_n == 0``.  So
    #: the first turn always plans (zero assistant messages, ``0 % N
    #: == 0``); subsequent fires land on turns ``N+1``, ``2N+1``, ...
    #:
    #: Skipped turns short-circuit with a synthetic "no plan needed"
    #: decision — no LLM call, no plan injection, the request flows to
    #: the executor unmodified.  Use in tandem with ``inject_plan=True``
    #: when the executor benefits from a fresh plan periodically but
    #: not every turn (e.g. weak open-source executors where the
    #: planner is expensive frontier compute).  Set to ``1`` to
    #: restore the original every-turn behaviour.
    cadence_n: int = Field(default=2, ge=1)


@dataclass(frozen=True)
class PlannerCompletion:
    """Raw planner LLM output paired with token-usage metadata.

    Mirrors :class:`switchyard.lib.processors.llm_classifier.ClassifierCompletion`:
    a thin wrapper that lets :class:`PlanningRequestProcessor` consume
    both the JSON payload (which it parses into :class:`PlannerDecision`)
    and the upstream's usage report (which it emits in the per-request
    audit line so planner-cost can be reconstructed post-hoc).

    ``usage`` is the raw OpenAI-SDK usage object (or any duck-typed
    equivalent exposing ``prompt_tokens`` / ``completion_tokens`` /
    optional ``prompt_tokens_details.cached_tokens``).  It may be
    ``None`` when the upstream omits the usage block (some self-hosted
    OpenAI-compat servers do).
    """

    content: str
    usage: Any | None = None


class PlannerClient(Protocol):
    """Protocol for the underlying LLM used by the planning processor."""

    async def plan(
        self,
        *,
        model: str,
        system_prompt: str,
        request_summary: str,
    ) -> PlannerCompletion:
        """Return the JSON planner output plus its usage block."""
        ...


class OpenAIChatPlannerClient:
    """OpenAI-chat-compatible implementation of :class:`PlannerClient`."""

    def __init__(
        self,
        client: OpenAILLMClient,
        *,
        disable_reasoning: bool = True,
        temperature: float | None = 0.0,
        response_format: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._client = client
        self._disable_reasoning = disable_reasoning
        self._temperature = temperature
        # ``response_format=None`` means "omit the field from the
        # request"; pass an explicit dict (typically
        # ``{"type": "json_object"}``) to send it.  PlanningConfig
        # supplies the strict-JSON default; direct OpenAIChatPlannerClient
        # callers must opt in.
        self._response_format = response_format
        self._extra_headers = extra_headers

    async def plan(
        self,
        *,
        model: str,
        system_prompt: str,
        request_summary: str,
    ) -> PlannerCompletion:
        extra_body: dict[str, Any] | None = None
        if self._disable_reasoning:
            # Same vLLM ``enable_thinking=False`` hint the classifier
            # path uses — see ``PlanningConfig.disable_reasoning``.
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

        # Build the messages for the planner call.  For Anthropic-based
        # planners (e.g. Bedrock-Opus) we structure the user message as
        # two content blocks with ``cache_control={type: ephemeral}``
        # marking the boundary between the stable prefix and the
        # varying tail.  Anthropic caches everything from the start of
        # the prompt through the marker — so the cached prefix
        # includes the system prompt AND the stable portion of the
        # request_summary.  The cached prefix needs to exceed
        # **1024 tokens** to hit Anthropic's minimum-cacheable-block
        # threshold; the planner system prompt alone is below that, so
        # marking just the system block wouldn't cache anything.
        # Pairing it with the stable terminus-2-system + first-user
        # prefix (typically ~3K tokens of the request_summary) reliably
        # clears the threshold on TB-Lite-class tasks.
        #
        # ``ANTHROPIC_CACHE_PREFIX_CHARS`` slices the request_summary
        # at a fixed character offset that should land *past* terminus-2's
        # original system + first user message for typical Harbor
        # invocations.  If the request_summary is shorter than the
        # threshold (small smoke tests, etc.), we fall back to no
        # splitting since the cache won't fire anyway.
        if is_anthropic_model(model) and len(request_summary) > _ANTHROPIC_CACHE_PREFIX_CHARS:
            user_content: Any = [
                {
                    "type": "text",
                    "text": request_summary[:_ANTHROPIC_CACHE_PREFIX_CHARS],
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "text",
                    "text": request_summary[_ANTHROPIC_CACHE_PREFIX_CHARS:],
                },
            ]
        else:
            user_content = request_summary

        # Build kwargs conditionally so each ``None`` field omits
        # itself from the request entirely.  Required for Anthropic-via-
        # LiteLLM (Bedrock Opus) paths, where ``temperature`` triggers
        # a 400 and ``response_format=json_object`` triggers a tool-use
        # scaffold that routes the actual output into ``tool_calls``
        # rather than ``content``.  See the matching PlanningConfig
        # fields for the empirical history.
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "extra_body": extra_body,
        }
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        if self._response_format is not None:
            kwargs["response_format"] = self._response_format
        if self._extra_headers is not None:
            kwargs["extra_headers"] = self._extra_headers

        result = await self._client.acompletion(**kwargs)
        return PlannerCompletion(
            content=_completion_content(result),
            usage=getattr(result, "usage", None),
        )


def is_anthropic_model(model: str) -> bool:
    """True if ``model`` is an Anthropic-based upstream (direct or via
    NIH's LiteLLM-Bedrock shim).  Used to gate ``cache_control``
    markers — Anthropic Messages API requires explicit cache markers
    for ephemeral prompt caching, while OpenAI-compatible upstreams
    auto-detect cache prefixes.
    """
    m = model.lower()
    return "anthropic" in m or "claude" in m


#: Character offset at which the planner's user content is split so
#: the first chunk carries ``cache_control={type: ephemeral}`` and
#: the second chunk does not.  At ~4 chars/token this is ~2000 tokens
#: of user prefix — added to the ~700-1000 tokens of system prompt
#: that precede it, the cached prefix is ~2845 tokens, comfortably
#: above Anthropic's 1024-token minimum cacheable-prefix size.
#:
#: This cut needs to land *inside* the stable portion of the request
#: — for terminus-2-driven Harbor invocations, the first 8K chars of
#: the JSON-dumped request body cover the request envelope plus
#: terminus-2's system prompt plus the original first-user task
#: statement.  All three are stable across every turn of a single
#: task, so on turns 2..N the prefix matches turn 1's cache write
#: and Anthropic returns a cache hit.  Going larger risks extending
#: into the variable per-turn dialog and breaking cache hits; going
#: smaller still caches but saves fewer tokens.  On TB-Lite-class
#: tasks with ~7800-token average prompts, this captures ~36% of
#: each call's input as cacheable.
#:
#: Shorter request_summaries (smoke tests, single-turn requests where
#: the request body is itself < this cut) fall back to a plain
#: string with no split — the cache wouldn't fire anyway since
#: there's no later turn to repeat against.
_ANTHROPIC_CACHE_PREFIX_CHARS = 8000


class PlanningRequestProcessor:
    """Generate and optionally inject an execution plan before backend dispatch.

    The planner LLM returns a :class:`PlannerDecision` that either
    contains a typed :class:`ExecutionPlan` (``plan_needed=True``) or
    declines to plan (``plan_needed=False``).  When the planner declines
    the request body is left untouched; the decision itself is still
    stamped on ctx for observability.
    """

    def __init__(
        self,
        config: PlanningConfig,
        *,
        client: PlannerClient | None = None,
        stats_accumulator: StatsAccumulator | None = None,
    ) -> None:
        self._config = config
        self._client = client or OpenAIChatPlannerClient(
            OpenAILLMClient(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=config.timeout_s,
            ),
            disable_reasoning=config.disable_reasoning,
            temperature=config.temperature,
            response_format=config.response_format,
            extra_headers=config.extra_headers,
        )
        #: When set, planner success + failure counts roll into the
        #: ``planner`` bucket on ``/v1/routing/stats``.  Without it the
        #: planner only emits stderr audit lines and is invisible to the
        #: stats endpoint.  Mirrors the classifier's optional
        #: ``stats_accumulator`` pattern.
        self._stats_accumulator = stats_accumulator

    async def process(self, ctx: ProxyContext, request: ChatRequest) -> ChatRequest:
        if not self._should_plan():
            return request

        # Cadence throttle: skip planner LLM call when this turn isn't
        # at a cadence boundary.  Emits a "skipped" audit line so the
        # observer can see how often the cadence elided a call.  The
        # request flows to the executor unchanged.
        if self._config.cadence_n > 1:
            body = request.body
            messages = body.get("messages", []) if isinstance(body, dict) else []
            n_assistant = sum(
                1 for m in messages
                if isinstance(m, dict) and m.get("role") == "assistant"
            )
            if n_assistant % self._config.cadence_n != 0:
                print(
                    f"planner_decision={{"
                    f"\"skipped\": true, "
                    f"\"reason\": \"cadence\", "
                    f"\"n_assistant\": {n_assistant}, "
                    f"\"cadence_n\": {self._config.cadence_n}"
                    f"}}",
                    flush=True,
                )
                return request

        # Pick initial vs revision system prompt based on whether the
        # conversation already contains a ``<plan>...</plan>`` block
        # in any prior assistant turn.  This is more reliable than
        # "any prior assistant turn" because the executor may have
        # responded once or twice before the planner started running
        # (no prior plan); we only want the revision prompt once an
        # actual plan exists in history.  See
        # :func:`_latest_plan_in_history` for the extraction logic.
        prior_plan = _latest_plan_in_history(request)
        is_revision = prior_plan is not None
        system_prompt = (
            self._config.revision_system_prompt
            if is_revision
            else self._config.initial_system_prompt
        )

        request_summary = _summarize_request(
            request,
            max_chars=self._config.max_request_chars,
        )
        if is_revision:
            # Show the planner the prior plan inline at the top of the
            # request summary so it can decide whether to revise.
            # Keeps the system prompt cache-friendly (static) while
            # giving the revision planner the context it needs.  The
            # revision system prompt mentions "Prior plan:" so the
            # planner knows what to look for here.
            request_summary = (
                f"Prior plan:\n{prior_plan}\n\n---\n\n{request_summary}"
            )
        completion: PlannerCompletion | None = None
        started_at = time.perf_counter()
        try:
            completion = await self._client.plan(
                model=self._config.model,
                system_prompt=system_prompt,
                request_summary=request_summary,
            )
            decision = parse_planner_decision(completion.content)
        except Exception as exc:
            if not self._config.fail_open:
                raise PlanningError("planner failed to produce a valid decision") from exc
            log.warning(
                "PlanningRequestProcessor: planner failed; continuing without plan: %s",
                exc,
            )
            # Record the failure into the planner bucket on the stats
            # accumulator (if wired) so ``/v1/routing/stats`` reports
            # ``planner.total_errors`` and the per-model ``errors``
            # counter — silent fail-opens otherwise hide the failure
            # rate from the benchmark observer.  Mirrors the
            # classifier's fail-open branch in
            # :class:`LLMClassifierRequestProcessor`.
            if self._stats_accumulator is not None:
                await self._stats_accumulator.record_planner_error(self._config.model)
            # Emit a failure-shaped audit line so the same grep that
            # tallies plan_needed counts also surfaces fail-open
            # invocations.  Differentiates ``call`` failures (upstream
            # HTTP error before any content) from ``parse`` failures
            # (upstream returned content but it didn't validate as a
            # PlannerDecision); the latter case includes a truncated
            # preview of the raw content so you can iterate the
            # planner system prompt without re-running the smoke.
            failure_payload: dict[str, Any] = {
                "failed": True,
                "stage": "parse" if completion is not None else "call",
                "error": str(exc)[:200],
            }
            if completion is not None:
                failure_payload["content_preview"] = completion.content[:500]
            sys.stderr.write(
                f"planner_decision={json.dumps(failure_payload, sort_keys=True)}\n"
            )
            sys.stderr.flush()
            return request

        # Stamp the decision unconditionally so audit logs can see why
        # the planner declined to plan when ``plan_needed=False``.
        ctx.metadata[CTX_PLANNER_DECISION] = decision

        # Record planner-side spend into the stats accumulator's
        # ``planner`` bucket (if wired).  Mirrors the classifier's
        # ``record_classifier_usage`` call.  Latency captures the full
        # planner call: HTTP roundtrip + JSON parse + decision
        # validation.  When the accumulator isn't wired (e.g. unit
        # tests with a fake client), this branch is a no-op and the
        # audit line below remains the only observability path.
        if self._stats_accumulator is not None:
            await self._record_planner_call(
                usage=completion.usage,
                latency_ms=(time.perf_counter() - started_at) * 1000,
            )

        # One-line audit dump of the planner decision + per-call usage,
        # written to stderr directly (not via the logging module) so it
        # lands in the captured server log regardless of uvicorn's
        # logger config — which does not pick up non-uvicorn package
        # loggers.  Mirrors the ``classifier_signals=...`` pattern in
        # ``LLMClassifierRequestProcessor``.  Grep ``planner_decision=``
        # to tally plan_needed distribution; cross-reference the
        # ``planner`` bucket on ``/v1/routing/stats`` for aggregate
        # spend + failure-rate numbers.
        audit_payload = _build_audit_payload(
            decision, completion.usage, is_revision=is_revision,
        )
        sys.stderr.write(
            f"planner_decision={json.dumps(audit_payload, sort_keys=True)}\n"
        )
        sys.stderr.flush()

        if not decision.plan_needed:
            log.debug(
                "PlanningRequestProcessor: planner declined to plan "
                "(is_revision=%s); leaving request unmodified.",
                is_revision,
            )
            return request

        # ``PlannerDecision._validate_consistency`` guarantees
        # ``plan_text`` is non-empty whenever ``plan_needed`` is True.
        # Wrap in ``<plan>...</plan>`` markers so the executor sees an
        # explicit, delimited block during prefill generation.  The
        # plan is never echoed back to the client — terminus-2 sees
        # only the executor's continuation.
        wrapped = wrap_plan(decision.plan_text)
        if self._config.inject_plan:
            _inject_plan(request, wrapped)
        return request

    def _should_plan(self) -> bool:
        return self._config.trigger_mode is PlanningTriggerMode.ALWAYS

    async def _record_planner_call(
        self,
        *,
        usage: Any,
        latency_ms: float,
    ) -> None:
        """Extract token counts from the SDK ``usage`` object and record them.

        Records into the planner bucket on :class:`StatsAccumulator` so
        the planner model's token spend doesn't merge with the
        same-named executor backend (e.g. when running V4-Pro-as-planner
        + V4-Pro-as-executor for self-planning sweeps).  Latency is
        recorded too so the planner-side latency histogram shows up on
        the snapshot's ``planner.models`` block.  Mirrors
        :meth:`switchyard.lib.processors.llm_classifier.LLMClassifierRequestProcessor._record_classifier_call`.
        """
        assert self._stats_accumulator is not None
        prompt = 0
        completion_tokens = 0
        cached = 0
        if usage is not None:
            prompt = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            ptd = getattr(usage, "prompt_tokens_details", None)
            if ptd is not None:
                cached = getattr(ptd, "cached_tokens", 0) or 0
        await self._stats_accumulator.record_planner_usage(
            model=self._config.model,
            prompt_tokens=prompt,
            completion_tokens=completion_tokens,
            cached_tokens=cached,
            latency_ms=latency_ms,
        )


def _build_audit_payload(
    decision: PlannerDecision,
    usage: Any | None,
    *,
    is_revision: bool,
) -> dict[str, Any]:
    """Build the JSON dict emitted as the ``planner_decision=…`` audit line.

    Pure function so it's trivially testable and the call site stays
    short.  Tolerates a missing or partial ``usage`` block (some
    OpenAI-compat upstreams omit fields) — missing token counts are
    reported as ``0`` rather than ``null`` so the field is always
    numeric and downstream sums don't have to special-case ``None``.

    Audit shape matches the slim two-field :class:`PlannerDecision`:
    ``plan_text_len`` and (when non-empty) a truncated
    ``plan_text_preview`` substitute for the old structured
    ``plan_step_count`` / ``plan_confidence`` fields.  Reading the
    audit log feels more like a transcript of the planner's drafted
    thoughts than a metrics dump.
    """
    payload: dict[str, Any] = {
        "plan_needed": decision.plan_needed,
        "is_revision": is_revision,
        "plan_text_len": len(decision.plan_text),
    }
    if decision.plan_text:
        # Truncated preview keeps the audit line scannable while still
        # capturing enough content for "did the planner produce
        # something coherent?" inspection.
        payload["plan_text_preview"] = decision.plan_text[:200]

    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0
    if usage is not None:
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
        if prompt_tokens_details is not None:
            cached_tokens = int(getattr(prompt_tokens_details, "cached_tokens", 0) or 0)
    payload["prompt_tokens"] = prompt_tokens
    payload["completion_tokens"] = completion_tokens
    payload["cached_tokens"] = cached_tokens
    return payload


def parse_planner_decision(raw: str) -> PlannerDecision:
    """Parse the planner LLM's JSON output into :class:`PlannerDecision`.

    Tolerates a leading ```` ```json ```` fence (some non-strict modes
    emit one despite instructions).  Any other deviation from the
    schema raises :class:`PlanningError`.
    """
    stripped = strip_markdown_fence(raw)
    try:
        return PlannerDecision.model_validate_json(stripped)
    except ValidationError as exc:
        raise PlanningError("planner JSON did not match PlannerDecision") from exc


def _summarize_request(
    request: ChatRequest,
    *,
    max_chars: int,
) -> str:
    body = getattr(request, "body", {})
    payload = {
        "request_type": request.request_type.value,
        "body": body,
    }
    text = json.dumps(payload, default=str, ensure_ascii=False, sort_keys=True)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 32] + "...<truncated>"


def _latest_plan_in_history(request: ChatRequest) -> str | None:
    """Return the content of the most recent ``<plan>...</plan>`` block
    in any assistant message in the outbound conversation, or ``None``
    if no plan has been emitted yet for this task.

    The presence of such a block in the conversation means the planner
    has previously emitted a plan that the response-side prepend
    stamped into the executor's content — and terminus-2 has been
    echoing it back ever since.  When found, the planner uses the
    revision system prompt; when absent, it uses the initial prompt.
    Walks ``messages`` in reverse so the latest plan (after any
    revisions) is what's returned.

    Covers all three wire formats by inspecting the same logical
    "messages" array.  Anthropic message content can be either a
    string or a list of content blocks; both are handled.
    """
    body = getattr(request, "body", None)
    if not isinstance(body, dict):
        return None

    if request.request_type is ChatRequestType.OPENAI_CHAT:
        messages = body.get("messages")
    elif request.request_type is ChatRequestType.ANTHROPIC:
        messages = body.get("messages")
    elif request.request_type is ChatRequestType.OPENAI_RESPONSES:
        raw_input = body.get("input")
        messages = raw_input if isinstance(raw_input, list) else None
    else:
        return None

    if not isinstance(messages, list):
        return None

    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        text = _assistant_message_text(msg.get("content"))
        if not text:
            continue
        plan = _extract_plan_block(text)
        if plan is not None:
            return plan
    return None


def _assistant_message_text(content: Any) -> str:
    """Flatten an assistant message's content into a single string.

    Handles the two common shapes: a plain string (OpenAI Chat) or a
    list of content blocks (Anthropic Messages, Responses API).  For
    block lists, concatenate any text-type block contents.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                # Anthropic-style text block.
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                # OpenAI-style content array also uses {"type": "text", "text": ...}.
                elif isinstance(block.get("text"), str):
                    parts.append(block["text"])
        return "\n".join(parts)
    return ""


def _extract_plan_block(text: str) -> str | None:
    """Extract the content between the most recent ``<plan>`` and
    ``</plan>`` markers in ``text``, or ``None`` if no complete block
    is present."""
    open_idx = text.rfind(PLAN_OPEN_MARKER)
    if open_idx < 0:
        return None
    close_idx = text.find(PLAN_CLOSE_MARKER, open_idx + len(PLAN_OPEN_MARKER))
    if close_idx < 0:
        return None
    return text[open_idx + len(PLAN_OPEN_MARKER) : close_idx].strip()


#: Open marker that wraps the prefilled plan in the executor's
#: request.  Used by :func:`_latest_plan_in_history` to detect prior
#: plans in conversation history for revision-mode prompting (a
#: latent path — the chain no longer echoes plans back to the client,
#: so prior plans only appear when an outside caller pre-stamps them).
#: Symmetric open + close form a clean extractable block.
PLAN_OPEN_MARKER = "<plan>"
PLAN_CLOSE_MARKER = "</plan>"


def wrap_plan(plan_text: str) -> str:
    """Wrap raw plan content in the ``<plan>...</plan>`` markers.

    The prefill uses the wrapped form so the executor sees an
    explicit, delimited block during generation.  The markers are
    also what :func:`_latest_plan_in_history` searches for when an
    upstream pre-stamps a plan into history.
    """
    return f"{PLAN_OPEN_MARKER}\n{plan_text}\n{PLAN_CLOSE_MARKER}"


def _inject_plan(request: ChatRequest, wrapped_plan: str) -> None:
    """Append the wrapped plan as an assistant prefill at the tail of
    the outbound conversation.

    The upstream model treats the trailing assistant content as a
    prefix and continues from where it ends.  Wrapping the plan in
    ``<plan>...</plan>`` markers gives the executor an explicit,
    delimited block to anchor its generation.

    **Executor contract**: the executor's upstream MUST accept
    conversations ending on an ``assistant`` role (assistant prefill).
    OpenAI Chat / native Anthropic / OSS-on-NIH vLLM all qualify.
    NIH's LiteLLM-Bedrock-Anthropic shim does NOT — Bedrock-hosted
    Anthropic models cannot serve as the executor under this
    primitive.
    """
    body = getattr(request, "body", None)
    if not isinstance(body, dict):
        return

    prefill_msg = {"role": "assistant", "content": wrapped_plan}

    if request.request_type is ChatRequestType.OPENAI_CHAT:
        messages = body.get("messages")
        if not isinstance(messages, list):
            messages = []
            body["messages"] = messages
        messages.append(prefill_msg)
    elif request.request_type is ChatRequestType.ANTHROPIC:
        messages = body.get("messages")
        if not isinstance(messages, list):
            messages = []
            body["messages"] = messages
        messages.append(prefill_msg)
    elif request.request_type is ChatRequestType.OPENAI_RESPONSES:
        raw_input = body.get("input")
        if isinstance(raw_input, list):
            raw_input.append(prefill_msg)
        elif isinstance(raw_input, str):
            body["input"] = [
                {"role": "user", "content": raw_input},
                prefill_msg,
            ]
        else:
            body["input"] = [prefill_msg]
    else:
        return
    request.replace_body(body)


def _completion_content(result: Any) -> str:
    choices = getattr(result, "choices", None)
    if not choices:
        raise PlanningError("planner completion had no choices")

    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content

    # Anthropic-via-LiteLLM (Bedrock) fallback: Anthropic's Messages API has
    # no native ``response_format``, so LiteLLM coerces
    # ``response_format={"type": "json_object"}`` into a forced tool call.
    # The planner JSON then arrives in ``tool_calls[0].function.arguments``
    # and ``content`` is empty.  Read it from there so the forced-tool path
    # yields the same JSON string the OpenAI-native path returns in
    # ``content`` — ``parse_planner_decision`` validates the shape either way.
    tool_arguments = _tool_call_arguments(message)
    if tool_arguments is not None:
        return tool_arguments

    raise PlanningError("planner completion had empty content")


def _tool_call_arguments(message: Any) -> str | None:
    """Return the JSON arguments string of the first tool call, or ``None``.

    LiteLLM rewrites ``response_format={"type": "json_object"}`` into a forced
    tool call for Anthropic-on-Bedrock, placing the model's JSON in
    ``tool_calls[0].function.arguments`` rather than ``content``.  This reads
    that argument string so :func:`_completion_content` can fall back to it.
    """
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        return None
    function = getattr(tool_calls[0], "function", None)
    arguments = getattr(function, "arguments", None)
    if isinstance(arguments, str) and arguments.strip():
        return arguments
    return None


__all__ = [
    "DEFAULT_INITIAL_PLANNER_SYSTEM_PROMPT",
    "DEFAULT_REVISION_PLANNER_SYSTEM_PROMPT",
    "OpenAIChatPlannerClient",
    "PlannerClient",
    "PlannerCompletion",
    "PlanningConfig",
    "PlanningError",
    "PlanningRequestProcessor",
    "PlanningTriggerMode",
    "is_anthropic_model",
    "parse_planner_decision",
]
