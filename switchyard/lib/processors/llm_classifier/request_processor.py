# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request processor that asks an LLM to extract routing signals."""

import json
import logging
import sys
import time
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from switchyard.lib.llm_client import OpenAILLMClient
from switchyard.lib.processors._structured_output import build_response_format
from switchyard.lib.processors.llm_classifier.signals import (
    CTX_DETERMINISTIC_ROUTE_SIGNALS,
    RouteDecision,
    RouteSignals,
    RouteTier,
)
from switchyard.lib.processors.message_condensing import strip_markdown_fence, trim_messages
from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.session_affinity import SessionAffinity
from switchyard.lib.stats_accumulator import StatsAccumulator
from switchyard_rust.core import ChatRequest

log = logging.getLogger(__name__)

DEFAULT_CLASSIFIER_SYSTEM_PROMPT = """\
You are a routing classifier for an LLM proxy.

Inspect the incoming request summary and return exactly one JSON object matching
this schema:

{
  "task_type": one of ["chat", "summarization", "extraction", "translation",
    "coding", "debugging", "math", "planning", "creative_writing",
    "agentic_task", "research", "data_analysis", "other"],
  "complexity": one of ["simple", "medium", "complex", "reasoning"],
  "reasoning_depth": one of ["none", "light", "multi_step", "deep"],
  "tool_planning_required": boolean,
  "precision_requirement": one of ["low", "medium", "high"],
  "context_dependency": one of ["latest_message", "conversation",
    "external_context"],
  "structured_output_risk": one of ["low", "medium", "high"],
  "recommended_tier": one of ["simple", "medium", "complex", "reasoning"],
  "confidence": number between 0 and 1,
  "reason_code": one of ["simple_qa", "summarization", "extraction",
    "translation", "coding_simple", "coding_complex", "debugging",
    "math_reasoning", "tool_agentic", "long_context", "structured_output",
    "creative_generation", "research_synthesis", "ambiguous", "other"],
  "abstain": boolean
}

Do not include markdown, commentary, or chain-of-thought. Use "abstain": true
and reason_code "ambiguous" when the request summary is insufficient.
"""

DEFAULT_MAX_REQUEST_CHARS = 16_000


class LLMClassifierError(RuntimeError):
    """Raised when the classifier LLM cannot produce valid route signals."""


@dataclass(frozen=True)
class ClassifierCompletion:
    """Result of a single classifier LLM call.

    Carries the raw ``content`` (a JSON string to be parsed into
    :class:`RouteDecision`) and the raw SDK ``usage`` object so the
    processor can record token spend into a
    :class:`~switchyard.lib.stats_accumulator.StatsAccumulator` when one
    is wired in. ``usage`` is intentionally typed as ``Any`` — the
    OpenAI-Python SDK's ``CompletionUsage`` shape is what we have today;
    custom client implementations may return any duck-typed object with
    the same attribute set.
    """

    content: str
    usage: Any | None = None


class LLMClassifierConfig(BaseModel):
    """Configuration for :class:`LLMClassifierRequestProcessor`.

    The classifier call is OpenAI-chat-compatible by default. Tests and custom
    hosts can inject any object implementing :class:`LLMClassifierClient`.
    """

    model_config = ConfigDict(frozen=True)

    model: str = Field(min_length=1)
    api_key: str | None = None
    base_url: str | None = None
    timeout_s: float | None = Field(default=None, gt=0.0)
    max_request_chars: int = Field(default=DEFAULT_MAX_REQUEST_CHARS, ge=256)
    fail_open: bool = True
    fallback_recommended_tier: RouteTier = RouteTier.MEDIUM

    recent_turn_window: int = Field(default=0, ge=0)
    """Number of trailing messages to keep in the classifier summary
    on top of the system + first-user anchors.

    ``0`` (default) preserves the historical "system + first user +
    last user" slice — small, cache-friendly, but leaves the
    classifier blind to recent assistant tool calls and tool results,
    which is exactly the state needed to estimate
    ``tool_call_count_estimate`` and identify DEBUG / EXPLORATION
    turn types. The classifier ends up guessing from a terse "Continue"
    last-user echo and tends to over-escalate.

    Set ``4`` for agent-loop traffic on a long-context classifier
    (DeepSeek V4 Flash 1M, GPT-5.2 1M, etc.). That window catches the
    last assistant turn + its tool result + the current user message,
    giving the classifier visibility into what the agent has actually
    been doing. The growing prefix is cache-friendly: each turn
    appends to a stable prefix the upstream can prompt-cache, so the
    extra tokens cost ~2% of their list price on cache-discounted
    backends (V4 Flash sees ~98% discount on cached input).

    Wider windows (8, 16) help further on dense multi-step debug
    sessions but plateau quickly; 4 is the sweet spot for TB-style
    workloads."""
    system_prompt: str = Field(
        default=DEFAULT_CLASSIFIER_SYSTEM_PROMPT,
        min_length=1,
    )
    structured_output_mode: Literal["json_schema", "json_object"] = "json_schema"
    """Wire-level structured-output mechanism.

    ``json_schema`` (default) sends the pydantic ``signal_schema`` to the
    backend with ``strict: true``, constraining the decoder to schema-valid
    output. Downgrade to ``json_object`` for OpenAI-compatible backends that
    do not advertise Structured Outputs support.
    """

    max_completion_tokens: int = Field(default=4096, ge=64)
    """Upper bound on ``max_tokens`` for the classifier completion call.

    Has to fit **both** internal reasoning content (for chain-of-thought
    models like DeepSeek V4 Flash / R1) **and** the structured JSON
    output. Setting this too low silently truncates the model
    mid-reasoning so the actual content comes back empty, the JSON
    parse fails, and the classifier abstains — fallback to default
    tier. 4096 is large enough for typical 200-500-token reasoning +
    300-token JSON without ever binding the cap; for non-reasoning
    classifiers (Nemotron-3-super-v3 etc.) the model only emits ~300
    tokens, so the higher ceiling is a no-op for them. Lower to
    1024-ish if you're paying for outputs by the token and your
    classifier is non-reasoning, but never below ~600 (small JSON
    schemas need at least that)."""

    extra_headers: dict[str, str] | None = None
    """Per-call HTTP headers forwarded on every classifier completion.

    Mirrors :attr:`~switchyard.lib.backends.llm_target.LlmTarget.extra_headers`
    for the classifier's own LLM call, which goes through a separate
    client (not :class:`LlmTarget`). Typical use: pin
    ``X-Inference-Priority: batch`` on benchmark deployments so the
    classifier's calls land on the same benchmarking gateway as the
    routed tier backends and inherit the relaxed proxy timeout.
    Unknown headers are silently ignored upstream, so passing them on
    non-NIH endpoints is benign."""

    dump_signals_to_stderr: bool = True
    """Emit one ``classifier_signals={...}`` JSON line to ``sys.stderr``
    per classifier call.

    Defaults ``True`` for the benchmark-server entrypoint, where the
    server's stderr is captured to a per-run log file and the lines are
    grepped post-hoc to tally tier-selection distributions. **Disabled
    by callers that share stderr with an interactive TUI** (e.g. the
    ``switchyard launch claude`` / ``launch codex`` paths (LLM-classifier
    routing is their default), where the proxy runs in
    the same process as the spawned coding agent and bare stderr writes
    bleed into the agent's terminal rendering). Toggled off in
    :class:`DeterministicRoutingFactory` for that reason."""

    disable_reasoning: bool = True
    """Pass ``chat_template_kwargs.enable_thinking=False`` via
    ``extra_body`` so reasoning models (DeepSeek V4 Flash / V4 Pro,
    R1-style) skip chain-of-thought and emit the structured JSON
    response directly.

    **Why this defaults True:** the classifier's job is fast
    structured-JSON extraction. When reasoning models are combined
    with strict ``response_format``, the JSON output gets misrouted
    into ``reasoning_content`` while ``content`` stays empty — a known
    vLLM regression (vllm-project/vllm#41132). The classifier then
    abstains (empty content fails JSON parse) and every request
    fallback-routes to the default tier. Disabling reasoning
    eliminates the misroute, restores ``content``, and keeps the
    classifier's wall-time tax low (10 vs 200+ completion tokens on
    a typical request).

    **Compatibility:** ``chat_template_kwargs`` is a vLLM-side hint
    on the NVIDIA Inference Hub. Non-reasoning models ignore the
    field as a no-op; OpenAI-direct backends are not known to reject
    unknown ``extra_body`` keys, so the default is safe. Set to
    ``False`` if a specific upstream rejects it, or if you
    deliberately want reasoning during classification."""


class LLMClassifierClient(Protocol):
    """Protocol for the underlying LLM used by the classifier processor."""

    async def classify(
        self,
        *,
        model: str,
        system_prompt: str,
        request_summary: str,
    ) -> ClassifierCompletion:
        """Return the classifier's content + raw usage from one LLM call.

        Content must be a JSON string that validates as the configured
        :class:`RouteDecision` subclass. ``usage`` can be ``None`` for
        in-memory test doubles that don't simulate token spend.
        """
        ...


class OpenAIChatLLMClassifierClient:
    """OpenAI-chat-compatible implementation of :class:`LLMClassifierClient`.

    Defaults to strict Structured Outputs (``response_format`` of type
    ``json_schema`` with ``strict: true``), which constrains the model's
    decoder to emit schema-valid JSON. Falls back to ``json_object`` mode
    when the configured backend does not advertise Structured Outputs.
    """

    def __init__(
        self,
        client: OpenAILLMClient,
        *,
        signal_schema: type[BaseModel] | None = None,
        structured_output_mode: Literal["json_schema", "json_object"] = "json_schema",
        max_completion_tokens: int = 4096,
        disable_reasoning: bool = True,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        if structured_output_mode == "json_schema" and signal_schema is None:
            raise ValueError(
                "OpenAIChatLLMClassifierClient requires signal_schema when "
                "structured_output_mode='json_schema'",
            )
        self._client = client
        self._signal_schema = signal_schema
        self._structured_output_mode = structured_output_mode
        self._max_completion_tokens = max_completion_tokens
        self._disable_reasoning = disable_reasoning
        self._extra_headers = extra_headers
        self._response_format = build_response_format(signal_schema, structured_output_mode)

    async def classify(
        self,
        *,
        model: str,
        system_prompt: str,
        request_summary: str,
    ) -> ClassifierCompletion:
        extra_body: dict[str, Any] | None = None
        if self._disable_reasoning:
            # Hint to vLLM (NVIDIA Inference Hub uses vLLM under LiteLLM)
            # to skip chain-of-thought for this call. Without it, DeepSeek
            # V4 Flash / V4 Pro misroute structured-JSON output into
            # ``reasoning_content`` and leave ``content`` empty — see
            # ``LLMClassifierConfig.disable_reasoning`` for the full
            # explanation.
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

        result = await self._client.acompletion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request_summary},
            ],
            temperature=0,
            # Wide enough to fit reasoning_content + JSON for chain-of-thought
            # classifiers (DeepSeek V4 Flash, R1-style) when reasoning is
            # enabled. Default 4096 is generous and never binds for
            # non-reasoning models that emit only ~300 tokens of JSON.
            max_tokens=self._max_completion_tokens,
            response_format=self._response_format,
            extra_body=extra_body,
            extra_headers=self._extra_headers,
        )
        return ClassifierCompletion(
            content=_completion_content(result),
            usage=getattr(result, "usage", None),
        )


class LLMClassifierRequestProcessor:
    """Extract semantic routing signals by calling an underlying LLM.

    The processor leaves the inbound request unchanged. It validates the
    classifier's JSON output as :class:`RouteSignals` and stores it under
    ``ctx.metadata[CTX_DETERMINISTIC_ROUTE_SIGNALS]`` for a later deterministic
    picker to consume.
    """

    def __init__(
        self,
        config: LLMClassifierConfig,
        *,
        client: LLMClassifierClient | None = None,
        signal_schema: type[RouteDecision] = RouteSignals,
        stats_accumulator: StatsAccumulator | None = None,
        affinity: SessionAffinity | None = None,
    ) -> None:
        self._config = config
        self._signal_schema = signal_schema
        self._stats_accumulator = stats_accumulator
        # Shared tier-pin store. Once a conversation is pinned, its tier is
        # fixed, so the classifier verdict would be ignored — skip the LLM call.
        self._affinity = affinity
        self._client = client or OpenAIChatLLMClassifierClient(
            OpenAILLMClient(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=config.timeout_s,
                # Disable the OpenAI SDK's default 2-retry budget on the
                # classifier path. With slow / cold-start upstreams (e.g.
                # DeepSeek V4 Flash on NVIDIA Inference Hub) the SDK's
                # exponential-backoff retries compound to >60s wall time on
                # what's meant to be a 30s ceiling. Our own ``fail_open``
                # fallback is the right surface for "classifier didn't
                # answer in time" — let it fire cleanly at ``timeout_s``.
                max_retries=0,
            ),
            signal_schema=signal_schema,
            structured_output_mode=config.structured_output_mode,
            max_completion_tokens=config.max_completion_tokens,
            disable_reasoning=config.disable_reasoning,
            extra_headers=config.extra_headers,
        )

    async def process(self, ctx: ProxyContext, request: ChatRequest) -> ChatRequest:
        if self._affinity is not None and (await self._affinity.pinned(ctx, request)) is not None:
            # Already pinned: the selector reuses the pin and ignores any
            # verdict, so skip the LLM call — classify once per task, not per turn.
            return request

        request_summary = _summarize_request(
            request,
            max_chars=self._config.max_request_chars,
            recent_turn_window=self._config.recent_turn_window,
        )
        started_at = time.perf_counter()
        try:
            completion = await self._client.classify(
                model=self._config.model,
                system_prompt=self._config.system_prompt,
                request_summary=request_summary,
            )
            signals = parse_route_decision(completion.content, self._signal_schema)
        except Exception as exc:
            if not self._config.fail_open:
                raise LLMClassifierError(
                    "LLM classifier failed to produce valid route signals",
                ) from exc
            log.warning(
                "LLMClassifierRequestProcessor: classifier failed; "
                "stamping abstain signals: %s",
                exc,
            )
            if self._stats_accumulator is not None:
                # Record the failure into the classifier bucket so
                # ``/v1/routing/stats`` reports ``classifier.total_errors``
                # alongside ``classifier.total_requests`` — otherwise a
                # silent fail-open hides the failure rate from the
                # benchmark observer.  Mirrors the success-branch
                # ``_record_classifier_call`` below.
                await self._stats_accumulator.record_classifier_error(self._config.model)
            signals = self._signal_schema.make_abstain(
                self._config.fallback_recommended_tier,
            )
            _fail_open_exc: Exception | None = exc
        else:
            _fail_open_exc = None
            prompt_tokens, completion_tokens, cached_tokens = _classifier_token_counts(
                completion.usage,
            )
            # Record usage so the intake sink emits the classifier sub-call as its
            # own record; the routed-turn record never sees this call.
            ctx.record_submodel_call(
                model=self._config.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
                router_type="deterministic",
                routed_to="classifier",
            )
            if self._stats_accumulator is not None:
                await self._record_classifier_call(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cached_tokens=cached_tokens,
                    latency_ms=(time.perf_counter() - started_at) * 1000,
                )

        ctx.metadata[CTX_DETERMINISTIC_ROUTE_SIGNALS] = signals
        # One-line dump of the extracted signals + the deterministic
        # policy_tier the selector will route on. Written to stderr
        # directly (not via the logging module) so it lands in the
        # captured server log regardless of uvicorn's logger config,
        # which does not pick up non-uvicorn package loggers. Grep
        # ``classifier_signals=`` to tally the distribution of
        # turn_type / scope / tool_count / recommended_tier per run.
        # Disabled by interactive launcher callers — see
        # :attr:`LLMClassifierConfig.dump_signals_to_stderr`.
        if self._config.dump_signals_to_stderr:
            try:
                signals_payload: dict[str, Any] = signals.model_dump(mode="json")
            except Exception:
                signals_payload = {"abstain": True}
            signals_payload["policy_tier"] = signals.policy_tier().value
            if _fail_open_exc is not None:
                signals_payload["fail_open"] = True
                signals_payload["error"] = str(_fail_open_exc)[:200]
            sys.stderr.write(
                f"classifier_signals={json.dumps(signals_payload, sort_keys=True)}\n"
            )
            sys.stderr.flush()
        return request

    async def _record_classifier_call(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int,
        latency_ms: float,
    ) -> None:
        """Record classifier token spend into the ``StatsAccumulator``.

        Records into the classifier bucket so the classifier model's token
        spend doesn't merge with the same-named backend tier (default TB-lite
        config has classifier and weak both pointing at Nemotron-3-Super-v3).
        Latency is recorded too so the per-request classifier-tax line shows up
        on the snapshot's classifier-models block.
        """
        assert self._stats_accumulator is not None
        await self._stats_accumulator.record_classifier_usage(
            model=self._config.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            latency_ms=latency_ms,
        )


def _classifier_token_counts(usage: Any) -> tuple[int, int, int]:
    """Return ``(prompt, completion, cached)`` tokens from an SDK usage object."""
    if usage is None:
        return 0, 0, 0
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    ptd = getattr(usage, "prompt_tokens_details", None)
    cached = (getattr(ptd, "cached_tokens", 0) or 0) if ptd is not None else 0
    return prompt, completion, cached


def parse_route_decision(
    raw: str,
    schema: type[RouteDecision] = RouteSignals,
) -> RouteDecision:
    """Parse classifier JSON output against the given decision schema."""
    stripped = strip_markdown_fence(raw)
    try:
        return schema.model_validate_json(stripped)
    except ValidationError as exc:
        raise LLMClassifierError(
            f"classifier JSON did not match {schema.__name__}",
        ) from exc


def parse_route_signals(raw: str) -> RouteSignals:
    """Parse classifier JSON output into :class:`RouteSignals` (legacy alias)."""
    decision = parse_route_decision(raw, RouteSignals)
    assert isinstance(decision, RouteSignals)
    return decision


def _summarize_request(
    request: ChatRequest, *, max_chars: int, recent_turn_window: int = 0
) -> str:
    body = getattr(request, "body", {})
    summary_body = (
        _condense_body(body, recent_turn_window=recent_turn_window)
        if isinstance(body, dict)
        else body
    )
    payload = {
        "request_type": request.request_type.value,
        "body": summary_body,
    }
    text = json.dumps(payload, default=str, ensure_ascii=False, sort_keys=True)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 32] + "...<truncated>"


# Bulk fields that we synthesize ourselves below — copy everything else from
# the raw body so the classifier still sees model, temperature,
# reasoning_effort, etc.
_BULK_BODY_FIELDS = ("messages", "tools", "input", "tool_choice")


def _condense_body(
    body: dict[str, Any], *, recent_turn_window: int = 0
) -> dict[str, Any]:
    """Strip non-routing-signal bulk before serializing for the classifier.

    Keeps only the slice that reliably carries routing signal:

    * system / developer messages (and Anthropic's top-level ``system``);
    * tool *names* + *descriptions* (drops parameter / input_schema bodies);
    * the **first** user message (task framing on agent harnesses that
      bundle the task into ``role="user"`` rather than a ``system`` field);
    * the last ``recent_turn_window`` messages (assistant tool calls,
      tool results, intermediate user turns) — ``0`` (default) keeps
      only the last user message; raise on long-context classifiers
      that need visibility into recent agent activity.

    Drops bulk that doesn't carry signal: prior assistant turns outside
    the window, tool-result messages outside the window, and full tool
    parameter schemas. These dominate prompt size on multi-turn agent
    traffic (TB tasks, Claude Code) and contribute almost nothing to a
    tier decision relative to their token cost.
    """
    out: dict[str, Any] = {k: v for k, v in body.items() if k not in _BULK_BODY_FIELDS}

    tools = body.get("tools")
    if isinstance(tools, list):
        out["tools"] = [_condense_tool(t) for t in tools]

    messages = body.get("messages")
    if isinstance(messages, list):
        out["messages"] = trim_messages(messages, recent_turn_window=recent_turn_window)

    raw_input = body.get("input")
    if isinstance(raw_input, list):
        out["input"] = trim_messages(raw_input, recent_turn_window=recent_turn_window)
    elif isinstance(raw_input, str):
        out["input"] = raw_input

    return out


def _condense_tool(tool: Any) -> Any:
    """Drop tool parameter schemas; keep ``name`` + ``description`` only.

    Handles both OpenAI Chat (``{"type": "function", "function": {...}}``)
    and Anthropic (``{"name", "description", "input_schema"}``).
    """
    if not isinstance(tool, dict):
        return tool
    fn = tool.get("function")
    if isinstance(fn, dict):
        slim_fn = {k: v for k, v in fn.items() if k != "parameters"}
        return {**{k: v for k, v in tool.items() if k != "function"}, "function": slim_fn}
    return {k: v for k, v in tool.items() if k != "input_schema"}


def _completion_content(result: Any) -> str:
    choices = getattr(result, "choices", None)
    if not choices:
        raise LLMClassifierError("classifier completion had no choices")

    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    raise LLMClassifierError("classifier completion had empty content")


__all__ = [
    "DEFAULT_CLASSIFIER_SYSTEM_PROMPT",
    "ClassifierCompletion",
    "LLMClassifierClient",
    "LLMClassifierConfig",
    "LLMClassifierError",
    "LLMClassifierRequestProcessor",
    "OpenAIChatLLMClassifierClient",
    "parse_route_decision",
    "parse_route_signals",
]
