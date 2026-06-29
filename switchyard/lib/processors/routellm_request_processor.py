# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RouteLLM request processor — classifier-driven tier selection.

Picks ``"strong"`` or ``"weak"`` based on a classifier score against a
threshold and stamps the result into ``ctx.metadata[CTX_ROUTELLM_TIER]``
and ``ctx.selected_target`` for downstream ``MultiLlmBackend`` dispatch. Mirrors
:class:`RandomRoutingRequestProcessor`'s shape; the only difference
is the picking logic (classifier score vs. weighted coin).

Classifier weights are owned by the processor instance. Shared model
lifecycle belongs in a profile or explicit runtime owner, not in a
process-wide cache hidden behind the processor API.

The classifier object is duck-typed — anything exposing
``calculate_strong_win_rate(prompt: str) -> float`` works. In production
this is a ``routellm.controller.Controller``'s router; tests can pass a
fake.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

from switchyard.lib import metrics, spans
from switchyard.lib.proxy_context import CTX_ROUTER_NAME
from switchyard_rust.components import set_stats_route_label
from switchyard_rust.core import ChatRequestType, request_type_matches

if TYPE_CHECKING:
    from switchyard.lib.profiles.routellm import RouteLLMConfig
    from switchyard.lib.proxy_context import ProxyContext
    from switchyard_rust.core import ChatRequest

log = logging.getLogger(__name__)

# Compatibility metadata key; Rust ``MultiLlmBackend`` dispatches from
# ``ProxyContext.selected_target``.
CTX_ROUTELLM_TIER = "_routellm_tier"

class _ClassifierProtocol(Protocol):
    """Duck type for the routellm router objects we score with."""

    def calculate_strong_win_rate(self, prompt: str) -> float: ...


class RouteLLMRequestProcessor:
    """Score the request and pick a tier; stamp the choice into ``ctx.metadata``.

    Args:
        config: The full :class:`RouteLLMConfig` — both tiers, threshold,
            router type, classifier model.
        classifier: Pre-built classifier (Python escape hatch for
            tests / non-routellm classifiers). Production callers leave this
            ``None`` and let ``startup()`` load the configured classifier.
    """

    def __init__(
        self,
        config: RouteLLMConfig,
        *,
        classifier: _ClassifierProtocol | None = None,
    ) -> None:
        self._config = config
        self._injected = classifier is not None
        self._classifier: _ClassifierProtocol | None = classifier

    async def startup(self) -> None:
        if self._injected:
            return
        self._classifier = await self._load_classifier()

    async def shutdown(self) -> None:
        if self._injected:
            return
        if self._classifier is not None:
            await self._unload_classifier(self._classifier)
        self._classifier = None

    async def process(self, ctx: ProxyContext, request: ChatRequest) -> ChatRequest:
        if self._classifier is None:
            # Defensive — startup() should have run. Default to strong
            # so traffic still flows; log loud so the misuse is visible.
            log.warning(
                "RouteLLMRequestProcessor.process called before startup — "
                "defaulting to 'strong' tier",
            )
            self._stamp(ctx, "strong")
            return request

        prompt = _extract_user_prompt(request)
        if prompt is None:
            log.info("RouteLLM: no user prompt extracted, defaulting to strong")
            self._stamp(ctx, "strong")
            return request

        score = float(self._classifier.calculate_strong_win_rate(prompt))
        tier = "strong" if score >= self._config.threshold else "weak"
        self._stamp(ctx, tier, score=score)
        log.info(
            "RouteLLM: score=%.4f threshold=%.4f -> %s",
            score, self._config.threshold, tier,
        )
        return request

    def _stamp(self, ctx: ProxyContext, tier: str, score: float | None = None) -> None:
        target = self._config.strong if tier == "strong" else self._config.weak
        ctx.metadata[CTX_ROUTELLM_TIER] = tier
        set_stats_route_label(ctx, tier)
        ctx.selected_target = target.id
        ctx.selected_model = target.model
        ctx.metadata[CTX_ROUTER_NAME] = "routellm"
        # ``score=None`` marks a default-to-strong fallback (no classifier run).
        with spans.route_decision_span(
            router="routellm",
            tier=tier,
            selected_model=target.model,
            selected_target=target.id,
            score=score,
            threshold=self._config.threshold,
        ):
            pass
        metrics.record_routing_decision(
            router="routellm",
            source="classifier" if score is not None else "default",
            tier=tier,
        )

    # ------------------------------------------------------------------
    # Classifier load / unload — overridable for non-routellm classifiers.
    # ------------------------------------------------------------------

    async def _load_classifier(self) -> _ClassifierProtocol:
        # Local import keeps the routellm[serve] dependency optional —
        # only paid for when this processor's startup actually runs.
        from routellm.controller import Controller  # pyright: ignore[reportMissingImports]

        kwargs: dict[str, Any] = {
            "routers": [self._config.router_type],
            "strong_model": self._config.strong.model,
            "weak_model": self._config.weak.model,
        }
        # Override the router's default checkpoint when the caller
        # supplied one:
        #   config={router_type: {"checkpoint_path": <model id>}}
        # The ``random`` router has no checkpoint to override.
        if self._config.classifier_model and self._config.router_type != "random":
            kwargs["config"] = {
                self._config.router_type: {
                    "checkpoint_path": self._config.classifier_model,
                },
            }
        controller = Controller(**kwargs)
        return controller.routers[self._config.router_type]  # type: ignore[no-any-return]

    async def _unload_classifier(self, value: _ClassifierProtocol) -> None:  # noqa: ARG002
        # routellm classifiers don't expose a teardown; rely on Python
        # GC + the refcount to evict from the cache. If a future
        # classifier needs explicit unloading, override this method.
        return None


def _extract_user_prompt(request: ChatRequest) -> str | None:
    """Extract the latest user-turn text from a Rust-backed ``ChatRequest``.

    Dispatches on the request format tag. Returns ``None`` when no user
    text is found; the processor falls back to the strong tier in that
    case (defensive default).

    The inner message-list iteration still uses ``.get`` because the
    elements are heterogeneous TypedDicts (user, assistant, tool, ...);
    only user-role messages are guaranteed to carry ``content`` of the
    shapes we care about, so we filter on role first and probe the
    ``content`` value defensively.
    """
    body = request.body
    if not isinstance(body, dict):
        return None
    if request_type_matches(request, ChatRequestType.OPENAI_CHAT) or request_type_matches(
        request, ChatRequestType.ANTHROPIC,
    ):
        return _last_user_text_from_messages(body.get("messages", []))
    if request_type_matches(request, ChatRequestType.OPENAI_RESPONSES):
        raw_input = body.get("input")
        if isinstance(raw_input, str):
            return raw_input or None
        if isinstance(raw_input, list):
            return _last_user_text_from_messages(raw_input)
        return None
    return None


def _last_user_text_from_messages(messages: list[Any]) -> str | None:
    """Return the concatenated text of the last user-role message, or None."""
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content or None
        if isinstance(content, list):
            parts = [
                block["text"]
                for block in content
                if isinstance(block, dict) and isinstance(block.get("text"), str)
            ]
            if parts:
                return "\n".join(parts)
    return None
