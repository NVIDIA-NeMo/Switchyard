# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end installation verifier for the launchers.

Mirrors the ``launch_claude`` / ``launch_codex`` / ``launch_openclaw``
flow but, instead of dropping the user into an interactive TUI, runs a
sequenced checklist and reports pass/fail for each step.  Four modes:

* :func:`verify_proxy` — backend + chain only.  No harness binary
  needed.  Catches credential / network / translation drift in seconds.
* :func:`verify_claude` — proxy + spawn ``claude -p "..."``.  Full
  end-to-end through Claude Code's CLI.
* :func:`verify_codex` — proxy + spawn ``codex exec "..."``.  Full
  end-to-end through Codex's CLI (with the same ``-c`` provider
  overrides ``launch_codex`` uses, so this validates the wire shape too).
* :func:`verify_openclaw` — proxy + spawn ``openclaw agent --local
  --message ... --json`` against a transient OpenClaw workspace
  identical to ``launch_openclaw``'s, so any drift in the JSON5
  config schema surfaces here.  Note ``verify_openclaw`` uses the
  one-shot ``agent`` subcommand because it has to exit cleanly without
  a TTY; ``launch_openclaw`` uses the interactive ``chat`` subcommand.

Designed to double as our offline e2e production gate: a pytest test
that shells out to ``switchyard verify <mode>`` and asserts
``returncode == 0`` protects the user-facing UX automatically.

The checklist runner intentionally writes to ``sys.stdout`` rather than
a logger — verify is *the* command users run to confirm their install,
and a checklist pinned to print semantics renders cleanly in CI logs,
plain terminals, and ``capture_output=True`` subprocess captures.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import uvicorn

from switchyard.cli.launchers.claude_code_launcher import (
    _build_claude_switchyard,
    _find_claude_binary,
)
from switchyard.cli.launchers.codex_cli_launcher import (
    _build_switchyard as _build_codex_switchyard,
)
from switchyard.cli.launchers.codex_cli_launcher import (
    _find_codex_binary,
    _provider_overrides,
)
from switchyard.cli.launchers.openclaw_launcher import (
    _build_switchyard as _build_openclaw_switchyard,
)
from switchyard.cli.launchers.openclaw_launcher import (
    _find_openclaw_binary,
    _openclaw_env,
    _openclaw_model_display_name,
    _remove_openclaw_workspace,
    _write_openclaw_workspace,
)
from switchyard.lib.backends.backend_format_resolver import (
    probe_anthropic_messages_support,
)
from switchyard.lib.route_table import SwitchyardApp
from switchyard.lib.route_table_builders import build_single_model_table
from switchyard.server.switchyard_app import build_switchyard_app

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Same readiness-poll deadline launch_claude / launch_codex use; verify
# is the same hot path so a divergent timeout would be a footgun.
_READY_TIMEOUT_S = 10.0
_SHUTDOWN_JOIN_S = 3.0
_BACKEND_PROBE_TIMEOUT_S = 10.0
_ROUNDTRIP_TIMEOUT_S = 60.0
# Per-harness invocation budget.  ``claude -p`` and ``codex exec`` should
# both finish "say ok" in ~1-3s on a healthy stack; 60s leaves enough
# headroom for cold-start + slow network without making a hung CI job
# wait forever.
_HARNESS_TIMEOUT_S = 60.0

# Single-word reply prompt — deterministic, cheap (one input/output
# token), and easy to assert on (any non-empty response passes).  The
# explicit "Reply with the single word: ok" phrasing nudges chatty
# models toward a tight reply.
_SMOKE_PROMPT = "Reply with the single word: ok"

_EXIT_OK = 0
_EXIT_FAIL = 1
_EXIT_BINARY_NOT_FOUND = 127


# ---------------------------------------------------------------------------
# Checklist runner
# ---------------------------------------------------------------------------


@dataclass
class _VerifyState:
    """Cross-step state bag for the orchestrators.

    Promoted to a dataclass instead of a ``dict`` so each field carries
    its own type — the orchestrators are closure-heavy by design (each
    checklist step is a thunk so test mocks stay simple), and a dict
    forces every cross-step pull through a ``cast`` or an ``Any``
    annotation (banned by project policy ``disallow_any_explicit``).

    All fields are ``Optional`` because the same bag is reused across
    the three modes — ``codex_bin`` is unset on the claude path,
    ``claude_bin`` is unset on the codex path, and ``server`` /
    ``thread`` / ``port`` are unset until the proxy-start step runs.
    """

    server: uvicorn.Server | None = None
    thread: threading.Thread | None = None
    port: int | None = None
    claude_bin: str | None = None
    codex_bin: str | None = None
    openclaw_bin: str | None = None
    openclaw_workspace: str | None = None


@dataclass
class _StepResult:
    """One row in the verify checklist.

    ``passed`` doubles as the binary outcome and the determinant of the
    final exit code: any False ⇒ overall FAIL.  ``detail`` is the
    success annotation (e.g. "200 OK, 142ms") or the failure reason
    (e.g. "ConnectionError: nodename nor servname provided").
    """

    name: str
    passed: bool
    detail: str
    elapsed_ms: float


@dataclass
class _Checklist:
    """Stdout-rendered ``[N/M] step... OK/FAIL`` checklist.

    Owns three things:

    1. Step counter — every :meth:`run` call advances it and prints the
       header before the step body executes.
    2. Result accumulator — :attr:`results` lets callers assemble a
       summary block at the end (counts, total elapsed, mode label).
    3. Halt-on-fail policy — once a step records ``passed=False``, all
       subsequent :meth:`run` calls become no-ops that record a
       ``"skipped (prior step failed)"`` row.  This keeps the output
       readable when the first step fails (no point trying to spawn
       ``claude`` if backend connectivity is down).
    """

    title: str
    total: int
    results: list[_StepResult] = field(default_factory=list)
    _idx: int = 0
    _halted: bool = False

    def _emit(self, line: str) -> None:
        # Single ``print`` per line so subprocess capture stays
        # well-ordered.  ``flush=True`` forces output even when the
        # caller has buffered stdout (pytest's ``-s`` redirected
        # captures, etc.).
        print(line, flush=True)

    def run(self, name: str, body: Callable[[], str]) -> _StepResult:
        """Execute one step and record / print its outcome.

        ``body`` is a thunk that returns a detail string on success and
        raises any exception on failure (the exception's class name +
        message become the failure detail).  Returning a non-string is
        tolerated and coerced to ``str()``.
        """
        self._idx += 1
        self._emit(f"[{self._idx}/{self.total}] {name}...")

        if self._halted:
            result = _StepResult(
                name=name,
                passed=False,
                detail="skipped (prior step failed)",
                elapsed_ms=0.0,
            )
            self.results.append(result)
            self._emit(f"        SKIP {result.detail}")
            return result

        start = time.monotonic()
        try:
            detail = body()
        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000.0
            reason = f"{type(e).__name__}: {e}"
            result = _StepResult(
                name=name, passed=False, detail=reason, elapsed_ms=elapsed_ms,
            )
            self.results.append(result)
            self._emit(f"        FAIL ({reason}, {elapsed_ms:.0f}ms)")
            self._halted = True
            return result

        elapsed_ms = (time.monotonic() - start) * 1000.0
        result = _StepResult(
            name=name, passed=True, detail=str(detail), elapsed_ms=elapsed_ms,
        )
        self.results.append(result)
        # Empty-detail steps (e.g. "Tearing down proxy") get just OK + ms;
        # filled-in details get the annotation in parens.
        if detail:
            self._emit(f"        OK ({detail}, {elapsed_ms:.0f}ms)")
        else:
            self._emit(f"        OK ({elapsed_ms:.0f}ms)")
        return result

    def summarize(self, mode: str, model: str, total_elapsed_ms: float) -> int:
        """Print the final ``verify <mode>: PASS/FAIL`` line and return exit code.

        Halt-injected SKIP rows count as failures for the exit code;
        the user already saw the FAIL that triggered the halt, so the
        summary just reports overall pass/fail without re-listing the
        skips.
        """
        passed = all(r.passed for r in self.results)
        verdict = "PASS" if passed else "FAIL"
        self._emit("")
        self._emit(
            f"verify {mode}: {verdict} (model={model}, {total_elapsed_ms:.0f}ms)"
        )
        return _EXIT_OK if passed else _EXIT_FAIL


# ---------------------------------------------------------------------------
# Local helpers (duplicated from launch_*.py — trivial, intentional)
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    """Bind to ``127.0.0.1:0``, let the OS pick, return the chosen port.

    Duplicated from ``launch_claude._find_free_port`` per the same
    "deliberate near-duplicate" pattern the launchers themselves use —
    keeps verify independent of any private symbol there so the
    launcher path stays untouched as a hot-fix concern.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


def _wait_ready(server: uvicorn.Server, timeout_s: float = _READY_TIMEOUT_S) -> bool:
    """Poll ``server.started`` until True or the timeout elapses."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if server.started:
            return True
        time.sleep(0.05)
    return False


def _spawn_proxy_thread(
    switchyard: SwitchyardApp,
    port: int,
) -> tuple[uvicorn.Server, threading.Thread]:
    """Run uvicorn in a background daemon thread, return (server, thread).

    ``log_level="critical"`` is intentional — verify is a structured
    checklist and any uvicorn output that interleaves with the
    checklist breaks the layout.  In particular, ``force_exit=True``
    during :func:`_teardown` cancels in-flight lifespan handlers,
    which uvicorn surfaces as an ``asyncio.CancelledError`` traceback
    at ERROR level.  That traceback is harmless (graceful-shutdown
    path) but visually noisy; CRITICAL hides it without suppressing
    real chain failures (those propagate through the HTTP response,
    not uvicorn's loggers).  ``log_config=None`` keeps uvicorn from
    overwriting the level we already set in
    :func:`_silence_chatty_loggers`.
    """
    app = build_switchyard_app(switchyard)
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="critical",
        log_config=None,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(
        target=server.run, name="verify-proxy", daemon=True,
    )
    thread.start()
    return server, thread


# ---------------------------------------------------------------------------
# Step bodies
# ---------------------------------------------------------------------------


def _check_credentials(
    api_key: str | None,
    base_url: str | None,
    key_source: str | None = None,
) -> str:
    """Step body: validate that resolution produced something usable.

    ``base_url`` is allowed to be empty — the launchers fall through to
    the OpenAI SDK default.  ``api_key`` is required because every
    backend we ship against demands one; failing here saves a 401
    later.

    ``key_source`` (e.g. ``"$OPENROUTER_API_KEY"`` or
    ``"secrets.json[openrouter]"``) is an optional human label that says
    *where* the key came from.  Surfaced in the step's success detail
    so the user knows which source they're verifying — and so the
    backend-reach step can name the source in a 401 hint without
    re-doing the resolution itself.
    """
    if not api_key:
        raise RuntimeError(
            "no API key resolved (set --api-key, OPENROUTER_API_KEY, "
            "NVIDIA_API_KEY, OPENAI_API_KEY, or secrets/secrets.json)"
        )
    base_url_label = base_url or "<openai default>"
    source_suffix = f" from {key_source}" if key_source else ""
    return f"api_key=*****{source_suffix}, base_url={base_url_label}"


def _check_backend_reachable(
    base_url: str,
    api_key: str,
    key_source: str | None = None,
) -> str:
    """Step body: HTTP GET ``{base_url}/models`` with the user's key.

    Picks ``/models`` because it's the cheapest authenticated endpoint
    every OpenAI-compatible backend exposes — it doesn't consume
    tokens, it doesn't require a request body, and the response shape
    is tiny.

    Status-code branches surface actionable hints rather than a bare
    ``"HTTP <code>"``:

    * ``200`` → success.
    * ``401`` → key was rejected.  This is the most common
      "credential failure" case once a key has been resolved; we name
      the source (``key_source``, e.g. ``$OPENROUTER_API_KEY``) so the
      user knows exactly which variable / file to fix, and call out
      that the backend itself rejected the key (vs. the proxy or a
      network issue).  ``key_source`` falls back to a generic
      ``"the resolved key"`` label when the caller didn't track it.
    * ``404`` → ``/models`` route missing.  Almost always a
      ``--base-url`` typo — surface that hypothesis directly so the
      user looks at the URL, not the key.
    * other 4xx / 5xx → generic, with the status code in the
      message so a follow-up curl is one ``grep`` away.

    A 4xx still counts as "backend reachable" structurally — the TCP
    handshake completed and the server replied — but verify reports
    it as a failure because the user almost certainly cares about
    "backend reachable AND responds successfully".
    """
    url = f"{base_url.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    start = time.monotonic()
    with httpx.Client(timeout=_BACKEND_PROBE_TIMEOUT_S) as client:
        resp = client.get(url, headers=headers)
    elapsed_ms = (time.monotonic() - start) * 1000.0
    if resp.status_code == 200:
        return f"GET /models 200, {elapsed_ms:.0f}ms"

    source_label = key_source or "the resolved key"
    if resp.status_code == 401:
        raise RuntimeError(
            f"HTTP 401 — the backend rejected {source_label}.  "
            f"The key resolved successfully but is invalid, expired, "
            f"or doesn't have access to {url}.  Double-check {source_label} "
            f"or rotate it."
        )
    if resp.status_code == 404:
        raise RuntimeError(
            f"HTTP 404 — {url} not found.  Likely a --base-url / "
            f"OPENROUTER_BASE_URL / NVIDIA_BASE_URL / OPENAI_BASE_URL typo (the path should "
            f"end in /v1).",
        )
    raise RuntimeError(
        f"GET {url} returned HTTP {resp.status_code}"
    )


async def _check_anthropic_probe(base_url: str, api_key: str) -> str:
    """Step body: probe whether the upstream speaks ``/v1/messages`` natively.

    Outcome is informational: we want the user to know which chain
    they'll get (native passthrough vs OpenAI translation) so the rest
    of the diagnostics make sense.  A False result isn't a failure —
    it's the documented fallback.
    """
    is_native = await probe_anthropic_messages_support(
        base_url=base_url, api_key=api_key,
    )
    return (
        "native passthrough"
        if is_native
        else "translation (backend lacks /v1/messages)"
    )


def _start_proxy(
    switchyard: SwitchyardApp, port: int | None,
) -> tuple[uvicorn.Server, threading.Thread, int]:
    """Step helper: spin up a proxy and wait for ``server.started``.

    Returns ``(server, thread, resolved_port)`` so the caller can hold
    the lifecycle handles for the round-trip + teardown steps.  Raises
    on readiness timeout — the runner converts that to a FAIL row.
    """
    resolved = port if port is not None else _find_free_port()
    server, thread = _spawn_proxy_thread(switchyard, resolved)
    if not _wait_ready(server):
        # Best-effort teardown so we don't leak a half-started uvicorn.
        server.should_exit = True
        thread.join(timeout=_SHUTDOWN_JOIN_S)
        raise RuntimeError(
            f"proxy did not become ready within {_READY_TIMEOUT_S:.1f}s"
        )
    return server, thread, resolved


def _proxy_roundtrip(port: int, model: str) -> str:
    """Step body: send one ``/v1/chat/completions`` request through the proxy.

    Picks Chat Completions (not Responses, not Anthropic Messages)
    because every chain we build serves it as the canonical
    inbound, and the response shape is the smallest of the three.

    Verify's contract is *"did the chain correctly translate, forward,
    and return a structurally valid response?"* — not *"did the model
    say something specific?"* — so the assertions are:

    * HTTP 200
    * Response has ``choices`` and ``usage`` (chain plumbed both ends)
    * Generation actually happened (``completion_tokens > 0``)
    * Visible text *is* surfaced when the model produced any —
      ``content`` becomes the reply preview in the success line

    Reasoning-only truncation is treated as a pass:  models like
    ``nvidia/qwen/qwen3.5-*`` and the GPT-5-reasoning family
    consume their entire output-token budget on internal reasoning
    when the budget is small, and the visible-text block comes back
    empty.  The chain still worked end-to-end (200 + usage + non-zero
    output tokens), so we report the result with a
    ``"no visible text (reasoning-only)"`` annotation rather than
    failing.  Earlier production e2e suites
    (``test_passthrough_anthropic_e2e.py``) already adopted this
    forgiveness pattern; verify follows it for consistency.

    ``max_tokens=2048`` is large enough that most non-reasoning
    models will surface visible text, but small enough that the
    one-line smoke prompt can't burn meaningful cost even on
    pricier backends.
    """
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": _SMOKE_PROMPT}],
        "max_tokens": 2048,
    }
    start = time.monotonic()
    with httpx.Client(timeout=_ROUNDTRIP_TIMEOUT_S) as client:
        resp = client.post(url, json=body)
    elapsed_ms = (time.monotonic() - start) * 1000.0
    if resp.status_code != 200:
        # Show a snippet of the body to help debug (NVIDIA error
        # responses usually carry a JSON ``message`` field worth
        # surfacing).  Truncate aggressively to keep output readable.
        snippet = resp.text[:200].replace("\n", " ")
        raise RuntimeError(
            f"POST {url} returned HTTP {resp.status_code}: {snippet}"
        )
    data = resp.json()

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(
            f"chat completion returned no choices; "
            f"response.keys={list(data.keys())}"
        )

    usage = data.get("usage") or {}
    in_toks = usage.get("prompt_tokens", 0)
    out_toks = usage.get("completion_tokens", 0)
    if not out_toks:
        # Zero output tokens means the model didn't generate
        # anything — that's a real chain or upstream issue worth
        # surfacing as a failure (vs the reasoning-truncation case
        # below where output tokens *did* get generated, just not
        # surfaced as text).
        raise RuntimeError(
            f"chat completion produced no output tokens; usage={usage}"
        )

    content = (choices[0].get("message", {}).get("content") or "").strip()
    if content:
        reply_preview = content[:40].replace("\n", " ")
        return (
            f"reply={reply_preview!r}, {elapsed_ms:.0f}ms, "
            f"in={in_toks} out={out_toks} tokens"
        )

    # Reasoning-only truncation: the chain worked, the model
    # generated tokens, but they were all internal reasoning that
    # didn't surface as visible text.  Show the finish_reason so
    # the user sees the cause rather than a mysterious "no text"
    # success.
    finish_reason = choices[0].get("finish_reason", "?")
    return (
        f"no visible text (finish_reason={finish_reason}, reasoning-only), "
        f"{elapsed_ms:.0f}ms, in={in_toks} out={out_toks} tokens"
    )


def _locate_claude() -> str:
    """Step body: find the ``claude`` binary or raise.

    Reuses :func:`launch_claude._find_claude_binary` so the search
    paths stay in lockstep with the actual launcher (otherwise verify
    could pass while launch fails, or vice versa, which is the worst
    possible outcome for a "did I install everything correctly"
    check).
    """
    bin_path = _find_claude_binary()
    if bin_path is None:
        raise RuntimeError(
            "claude binary not found on PATH or in known fallback "
            "locations (~/.claude/local/claude, ~/.local/bin/claude). "
            "Install with `curl -fsSL https://claude.ai/install.sh | bash`"
        )
    return bin_path


def _locate_codex() -> str:
    """Step body: find the ``codex`` binary or raise.

    Sibling of :func:`_locate_claude` for the codex path — same
    "search paths must match the launcher" argument.
    """
    bin_path = _find_codex_binary()
    if bin_path is None:
        raise RuntimeError(
            "codex binary not found on PATH or in known fallback "
            "locations (~/.npm-global/bin/codex, ~/.local/bin/codex). "
            "Install with `npm install -g @openai/codex`"
        )
    return bin_path


def _locate_openclaw() -> str:
    """Step body: find the ``openclaw`` binary or raise.

    Sibling of :func:`_locate_codex` for the openclaw path — same
    "search paths must match the launcher" argument.
    """
    bin_path = _find_openclaw_binary()
    if bin_path is None:
        raise RuntimeError(
            "openclaw binary not found on PATH or in known fallback "
            "locations (~/.npm-global/bin/openclaw, ~/.local/bin/openclaw, "
            "~/.nvm/versions/node/*/bin/openclaw). "
            "Install with `npm install -g openclaw@latest`"
        )
    return bin_path


def _run_claude_prompt(claude_bin: str, port: int, model: str) -> str:
    """Step body: spawn ``claude -p "..."`` against the proxy, capture output.

    Uses the same env-injection contract as ``launch_claude``
    (``ANTHROPIC_BASE_URL`` / ``ANTHROPIC_AUTH_TOKEN`` /
    ``ANTHROPIC_API_KEY=""`` / ``ANTHROPIC_MODEL`` /
    ``ANTHROPIC_SMALL_FAST_MODEL``) so any drift in that contract
    surfaces here rather than in production.

    Flags:
        ``-p``                              non-interactive prompt mode
        ``--output-format json``            structured envelope so we can
                                             distinguish "empty result"
                                             (reasoning truncation, OK)
                                             from "missing result"
                                             (real failure)
        ``--max-turns 1``                   one round-trip only — verify
                                             isn't testing agentic loops
        ``--dangerously-skip-permissions``  required for non-interactive
                                             so claude doesn't block on a
                                             permission prompt that has no
                                             stdin to answer it

    The JSON envelope is the contract: claude prints a single
    ``{"type": "result", ..., "result": "..."}`` object on stdout.
    We treat:

    * ``returncode != 0`` → real failure (claude or the chain blew up)
    * exit 0, valid JSON, ``result`` non-empty → success with a reply
      preview
    * exit 0, valid JSON, ``result`` empty/missing → reasoning-only
      truncation (same forgiveness as :func:`_proxy_roundtrip` —
      surface ``num_turns`` / ``stop_reason`` if present so the user
      sees the cause)
    * exit 0 but stdout isn't valid JSON → real failure (claude
      changed its envelope shape, or something corrupted stdout)
    """
    env_overrides = {
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
        "ANTHROPIC_AUTH_TOKEN": "switchyard",
        "ANTHROPIC_API_KEY": "",
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_SMALL_FAST_MODEL": model,
    }
    env = os.environ.copy()
    env.update(env_overrides)

    cmd = [
        claude_bin, "-p", _SMOKE_PROMPT,
        "--output-format", "json",
        "--max-turns", "1",
        "--dangerously-skip-permissions",
    ]

    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True,
            timeout=_HARNESS_TIMEOUT_S, check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"claude did not exit within {_HARNESS_TIMEOUT_S:.0f}s"
        ) from e
    elapsed_ms = (time.monotonic() - start) * 1000.0

    if proc.returncode != 0:
        stderr_snippet = (proc.stderr or "").strip()[:200].replace("\n", " ")
        raise RuntimeError(
            f"claude exited {proc.returncode}: {stderr_snippet}"
        )

    stdout = (proc.stdout or "").strip()
    if not stdout:
        raise RuntimeError(
            "claude exited 0 but produced no stdout (expected JSON envelope)"
        )

    # Parse the envelope.  A 0-exit with non-JSON stdout means
    # claude's output contract changed under us — surface that as
    # a real failure rather than ignoring the malformed body.
    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError as e:
        snippet = stdout[:200].replace("\n", " ")
        raise RuntimeError(
            f"claude produced non-JSON stdout: {snippet} ({e})"
        ) from e

    if not isinstance(envelope, dict):
        raise RuntimeError(
            f"claude JSON envelope wasn't an object: {type(envelope).__name__}"
        )

    result_text = envelope.get("result") or ""
    if isinstance(result_text, str):
        result_text = result_text.strip()
    else:
        # ``result`` exists but isn't a string — pre-2.x claude
        # used a list of content blocks; the contract drift is
        # worth surfacing rather than coercing.
        raise RuntimeError(
            f"claude 'result' field had unexpected type: {type(result_text).__name__}"
        )

    if result_text:
        preview = result_text[:40].replace("\n", " ")
        return f"reply={preview!r}, {elapsed_ms:.0f}ms"

    # Reasoning-only truncation case — the chain worked end-to-end
    # (claude got a response) but the model's visible-text block was
    # empty.  Same forgiveness as :func:`_proxy_roundtrip`: report
    # what we *do* know (envelope fields claude surfaces) so the user
    # sees this is a model-budget artifact, not a chain bug.
    num_turns = envelope.get("num_turns", "?")
    stop_reason = (
        envelope.get("stop_reason")
        or envelope.get("subtype")
        or "?"
    )
    return (
        f"no visible reply (num_turns={num_turns}, stop_reason={stop_reason}, "
        f"reasoning-only), {elapsed_ms:.0f}ms"
    )


def _run_openclaw_prompt(
    openclaw_bin: str,
    workspace: str,
    model: str,
) -> str:
    """Step body: spawn ``openclaw agent`` against the proxy with a smoke prompt.

    Uses the same env-injection contract as ``launch_openclaw`` —
    ``OPENCLAW_STATE_DIR`` / ``OPENCLAW_HOME`` / ``OPENCLAW_CONFIG_PATH``
    point openclaw at a transient workspace with a ``models.providers.
    switchyard`` block declaring the proxy as a custom provider, and
    ``SWITCHYARD_API_KEY`` is set to the opaque placeholder the JSON5
    config references.

    ``openclaw agent`` is a non-interactive one-shot turn (``--local``
    binds it to the embedded agent runtime that reads our transient
    ``openclaw.json``).  Flags:

    * ``--message`` — the smoke prompt.  Required: ``openclaw agent``
      refuses to read from stdin.
    * ``--session-key`` — required session selector.  We use a
      per-PID smoke key so concurrent verifies don't collide.
    * ``--json`` — emit a structured envelope to stdout so we can pull
      ``finalAssistantVisibleText`` deterministically.

    Outcome handling matches :func:`_run_codex_prompt`:

    * exit non-zero → real failure (openclaw or the chain blew up);
      include stderr snippet
    * exit 0 + non-empty visible reply → success with reply preview
    * exit 0 + empty ``finalAssistantVisibleText`` → reasoning-only
      truncation (same forgiveness as the codex / proxy-roundtrip
      steps).
    """
    env = _openclaw_env(workspace=workspace)
    session_key = f"smoke-{os.getpid()}-{int(time.time())}"
    cmd = [
        openclaw_bin,
        "agent",
        "--local",
        "--session-key", session_key,
        "--message", _SMOKE_PROMPT,
        "--json",
    ]

    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True,
            timeout=_HARNESS_TIMEOUT_S, check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"openclaw did not exit within {_HARNESS_TIMEOUT_S:.0f}s"
        ) from e
    elapsed_ms = (time.monotonic() - start) * 1000.0

    if proc.returncode != 0:
        stderr_snippet = (proc.stderr or "").strip()[:200].replace("\n", " ")
        raise RuntimeError(
            f"openclaw exited {proc.returncode}: {stderr_snippet}"
        )

    reply = _extract_openclaw_visible_text(proc.stdout or "")
    if reply:
        preview = reply[:40].replace("\n", " ")
        return (
            f"reply={preview!r}, model={_openclaw_model_display_name(model)}, "
            f"{elapsed_ms:.0f}ms"
        )

    # Reasoning-only / no visible-text reply.
    return (
        f"no visible reply (openclaw exited 0 with empty "
        f"finalAssistantVisibleText, reasoning-only), {elapsed_ms:.0f}ms"
    )


def _extract_openclaw_visible_text(stdout: str) -> str:
    """Pull the assistant's user-visible reply out of ``openclaw agent --json``.

    The envelope schema has shifted between openclaw releases — the
    visible reply has lived at the top level and nested under wrapper
    keys like ``data`` / ``result`` in different versions.  We walk the
    JSON depth-first for the first ``finalAssistantVisibleText`` key,
    then fall back to ``finalAssistantRawText`` if the visible text was
    suppressed (for instance, when openclaw stripped a thinking tag).

    Returns ``""`` when JSON parsing fails, when the key isn't found,
    or when the matched value is whitespace-only.  Callers map ``""``
    to the reasoning-only / empty-reply branch.
    """
    text = stdout.strip()
    if not text:
        return ""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # Plain-text fallback for an openclaw release that prints
        # something non-JSON under --json (or an error preamble before
        # the envelope).  Treat the whole stdout as the preview so we
        # still surface something.
        return text
    for key in ("finalAssistantVisibleText", "finalAssistantRawText"):
        found = _find_first_key(payload, key)
        if isinstance(found, str) and found.strip():
            return found.strip()
    return ""


def _find_first_key(payload: Any, target: str) -> Any:
    """Depth-first lookup of ``target`` in nested dicts and lists.

    Returns the first match's value, or ``None`` if not found.
    """
    if isinstance(payload, dict):
        if target in payload:
            return payload[target]
        for value in payload.values():
            found = _find_first_key(value, target)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_first_key(item, target)
            if found is not None:
                return found
    return None


def _run_codex_prompt(codex_bin: str, port: int, model: str) -> str:
    """Step body: spawn ``codex exec "..."`` against the proxy.

    Uses the same six ``-c`` provider-config overrides as
    ``launch_codex`` (via :func:`_provider_overrides`) so any wire-
    contract drift caught here matches what the launcher would hit.

    Flags:
        ``exec <prompt>``  codex's non-interactive subcommand
        ``-m <model>``     pin the model on the codex side (status line
                            display); the proxy's request processor
                            rewrites it as a safety net.

    Codex's ``exec`` mode requires ``OPENAI_API_KEY`` to be set per
    its provider-config validation — value is opaque (proxy injects
    the real upstream credential at call time).

    Outcome handling matches :func:`_run_claude_prompt`:

    * exit non-zero → real failure (codex or the chain blew up);
      include stderr snippet
    * exit 0 + non-empty stdout → success with reply preview
    * exit 0 + empty stdout → reasoning-only truncation (same
      forgiveness as the claude / proxy-roundtrip steps).  Codex's
      ``exec`` mode doesn't emit a structured envelope we can
      inspect for ``num_turns`` / ``stop_reason``, so we don't get
      to be as precise as ``_run_claude_prompt`` here — but
      tolerating the empty case keeps reasoning-model verifies
      passing instead of false-failing.
    """
    overrides = _provider_overrides(port)

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = "switchyard"

    cmd = [codex_bin, *overrides, "-m", model, "exec", _SMOKE_PROMPT]

    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True,
            timeout=_HARNESS_TIMEOUT_S, check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"codex did not exit within {_HARNESS_TIMEOUT_S:.0f}s"
        ) from e
    elapsed_ms = (time.monotonic() - start) * 1000.0

    if proc.returncode != 0:
        stderr_snippet = (proc.stderr or "").strip()[:200].replace("\n", " ")
        raise RuntimeError(
            f"codex exited {proc.returncode}: {stderr_snippet}"
        )
    stdout = (proc.stdout or "").strip()
    if stdout:
        preview = stdout[:40].replace("\n", " ")
        return f"reply={preview!r}, {elapsed_ms:.0f}ms"

    # Reasoning-only truncation case.  See :func:`_run_claude_prompt`
    # for the broader rationale; codex doesn't expose ``num_turns``
    # or ``stop_reason`` on stdout in ``exec`` mode, so the
    # annotation is sparser.
    return (
        f"no visible reply (codex exited 0 with empty stdout, "
        f"reasoning-only), {elapsed_ms:.0f}ms"
    )


def _teardown(server: uvicorn.Server, thread: threading.Thread) -> str:
    """Step body: signal uvicorn to exit, join the thread.

    Sets both ``should_exit`` (graceful) and ``force_exit`` (drop
    in-flight connections) because uvicorn's default
    ``timeout_graceful_shutdown=None`` blocks shutdown until every
    keep-alive connection closes, and clients like Claude Code can
    leave a connection lingering after the response stream
    completes.  Without ``force_exit`` the join here would hang for
    minutes on a streaming workload.

    A timed-out join is reported as an OK with a "shutdown slow"
    annotation, *not* a failure: the proxy thread is a daemon, so it
    dies when the verify process exits regardless.  Failing here
    would mask the meaningful signal (the rest of the chain worked
    end-to-end) behind a uvicorn shutdown quirk that doesn't affect
    a single-shot CLI.  A genuinely-stuck shutdown still surfaces in
    the annotation so a developer scanning verify output sees it.
    """
    server.should_exit = True
    server.force_exit = True
    thread.join(timeout=_SHUTDOWN_JOIN_S)
    if thread.is_alive():
        return (
            f"shutdown slow (>{_SHUTDOWN_JOIN_S:.1f}s, daemon thread will "
            f"exit when verify process does)"
        )
    return ""


# ---------------------------------------------------------------------------
# Mode orchestrators
# ---------------------------------------------------------------------------


def _silence_chatty_loggers() -> None:
    """Quiet down dependency loggers so the checklist stays readable.

    Two tiers:

    * WARNING for ``switchyard`` / ``httpx`` / ``openai`` /
      ``anthropic`` — same as the launchers, keeps INFO chatter out
      of the user's checklist while letting WARNING / ERROR surface
      real problems (probe timeout, upstream 5xx, etc).

    * CRITICAL for ``uvicorn*`` / ``starlette*`` — strictly
      noisier.  When ``force_exit=True`` cancels in-flight lifespan
      handlers during teardown, starlette / uvicorn log a
      ``CancelledError`` traceback at ERROR level even though it's
      a normal part of the shutdown path and harmless to verify's
      contract.  Bumping to CRITICAL hides that traceback without
      hiding genuine HTTP / framework errors during the active
      steps (those propagate via the chain itself, not the server's
      own loggers).  The launchers don't need this because they
      run uvicorn for the lifetime of the parent process — the
      shutdown only fires on process exit, where stderr noise
      doesn't matter.
    """
    for noisy in ("switchyard", "httpx", "openai", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    for very_noisy in (
        "uvicorn", "uvicorn.access", "uvicorn.error",
        "starlette", "starlette.routing",
    ):
        logging.getLogger(very_noisy).setLevel(logging.CRITICAL)


def verify_proxy(
    *,
    model: str,
    base_url: str,
    api_key: str,
    port: int | None = None,
    timeout: float | None = None,
    key_source: str | None = None,
) -> int:
    """Run the proxy-only verify checklist; return exit code.

    Six steps:

    1. Resolve credentials
    2. Reach backend
    3. Probe ``/v1/messages`` (informational)
    4. Start proxy
    5. Round-trip a chat completion through the chain
    6. Tear down proxy

    Designed to be the cheapest possible "is my install good?" check
    — no harness binary needed, just credentials + network + the
    proxy thread.  Fast enough to run as a K8s readiness probe.

    ``key_source`` is an optional human label naming where the
    ``api_key`` came from (e.g. ``"$OPENROUTER_API_KEY"`` or
    ``"secrets.json[openrouter]"``).  Surfaced in step 1's success
    annotation and in step 2's 401 hint so a credential mishap names
    the exact knob the user has to turn.
    """
    _silence_chatty_loggers()
    overall_start = time.monotonic()
    checklist = _Checklist(title="verify proxy", total=6)

    checklist.run(
        "Resolving credentials",
        lambda: _check_credentials(api_key, base_url, key_source),
    )
    checklist.run(
        "Reaching backend",
        lambda: _check_backend_reachable(base_url, api_key, key_source),
    )
    checklist.run(
        "Probing /v1/messages support",
        lambda: asyncio.run(_check_anthropic_probe(base_url, api_key)),
    )

    # Steps 4-6 share the proxy lifecycle through ``state`` — each
    # step is a closure so it can mutate dataclass fields without
    # ``nonlocal``.  Closures-as-step-bodies is the design: keeps the
    # checklist runner generic and the test mocks simple.
    state = _VerifyState()

    def _start() -> str:
        switchyard = _build_claude_switchyard(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        server, thread, resolved_port = _start_proxy(switchyard, port)
        state.server = server
        state.thread = thread
        state.port = resolved_port
        return f"127.0.0.1:{resolved_port}"

    checklist.run("Starting proxy", _start)

    def _roundtrip() -> str:
        assert state.port is not None  # set by _start; halt-on-fail prevents reaching here otherwise
        return _proxy_roundtrip(state.port, model)

    checklist.run("Round-tripping chat completion", _roundtrip)

    def _shutdown() -> str:
        if state.server is None or state.thread is None:
            return "(no proxy started)"
        return _teardown(state.server, state.thread)

    checklist.run("Tearing down proxy", _shutdown)

    total_ms = (time.monotonic() - overall_start) * 1000.0
    return checklist.summarize("proxy", model, total_ms)


def verify_claude(
    *,
    model: str,
    base_url: str,
    api_key: str,
    port: int | None = None,
    timeout: float | None = None,
    key_source: str | None = None,
) -> int:
    """Run the full ``claude`` e2e verify checklist; return exit code.

    Eight steps — the proxy-mode six plus the harness-specific two
    (locate binary + spawn-and-validate).

    See :func:`verify_proxy` for the ``key_source`` contract.
    """
    _silence_chatty_loggers()
    overall_start = time.monotonic()
    checklist = _Checklist(title="verify claude", total=8)

    checklist.run(
        "Resolving credentials",
        lambda: _check_credentials(api_key, base_url, key_source),
    )
    checklist.run(
        "Reaching backend",
        lambda: _check_backend_reachable(base_url, api_key, key_source),
    )
    checklist.run(
        "Probing /v1/messages support",
        lambda: asyncio.run(_check_anthropic_probe(base_url, api_key)),
    )

    state = _VerifyState()

    def _start() -> str:
        switchyard = _build_claude_switchyard(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        server, thread, resolved_port = _start_proxy(switchyard, port)
        state.server = server
        state.thread = thread
        state.port = resolved_port
        return f"127.0.0.1:{resolved_port}"

    checklist.run("Starting proxy", _start)

    def _locate() -> str:
        bin_path = _locate_claude()
        state.claude_bin = bin_path
        return bin_path

    checklist.run("Locating claude binary", _locate)

    def _roundtrip() -> str:
        # Lightweight smoke first via the proxy directly — catches
        # backend / chain failures *before* spawning claude, so a
        # claude failure is unambiguously a claude / env issue
        # rather than an upstream issue.
        assert state.port is not None
        return _proxy_roundtrip(state.port, model)

    checklist.run("Round-tripping chat completion", _roundtrip)

    def _spawn() -> str:
        assert state.claude_bin is not None
        assert state.port is not None
        return _run_claude_prompt(state.claude_bin, state.port, model)

    checklist.run("Spawning claude with proxy env", _spawn)

    def _shutdown() -> str:
        if state.server is None or state.thread is None:
            return "(no proxy started)"
        return _teardown(state.server, state.thread)

    checklist.run("Tearing down proxy", _shutdown)

    total_ms = (time.monotonic() - overall_start) * 1000.0
    return checklist.summarize("claude", model, total_ms)


def verify_codex(
    *,
    model: str,
    base_url: str,
    api_key: str,
    port: int | None = None,
    timeout: float | None = None,
    key_source: str | None = None,
) -> int:
    """Run the full ``codex`` e2e verify checklist; return exit code.

    Seven steps — like ``verify claude`` minus the Anthropic probe
    (codex always uses the OpenAI translation chain, no native
    passthrough path to detect).

    See :func:`verify_proxy` for the ``key_source`` contract.
    """
    _silence_chatty_loggers()
    overall_start = time.monotonic()
    checklist = _Checklist(title="verify codex", total=7)

    checklist.run(
        "Resolving credentials",
        lambda: _check_credentials(api_key, base_url, key_source),
    )
    checklist.run(
        "Reaching backend",
        lambda: _check_backend_reachable(base_url, api_key, key_source),
    )

    state = _VerifyState()

    def _start() -> str:
        chain = _build_codex_switchyard(
            model, api_key, base_url, timeout,
        )
        switchyard = build_single_model_table(model, chain)
        server, thread, resolved_port = _start_proxy(switchyard, port)
        state.server = server
        state.thread = thread
        state.port = resolved_port
        return f"127.0.0.1:{resolved_port}"

    checklist.run("Starting proxy", _start)

    def _locate() -> str:
        bin_path = _locate_codex()
        state.codex_bin = bin_path
        return bin_path

    checklist.run("Locating codex binary", _locate)

    def _roundtrip() -> str:
        assert state.port is not None
        return _proxy_roundtrip(state.port, model)

    checklist.run("Round-tripping chat completion", _roundtrip)

    def _spawn() -> str:
        assert state.codex_bin is not None
        assert state.port is not None
        return _run_codex_prompt(state.codex_bin, state.port, model)

    checklist.run("Spawning codex with proxy provider", _spawn)

    def _shutdown() -> str:
        if state.server is None or state.thread is None:
            return "(no proxy started)"
        return _teardown(state.server, state.thread)

    checklist.run("Tearing down proxy", _shutdown)

    total_ms = (time.monotonic() - overall_start) * 1000.0
    return checklist.summarize("codex", model, total_ms)


def verify_openclaw(
    *,
    model: str,
    base_url: str,
    api_key: str,
    port: int | None = None,
    timeout: float | None = None,
    key_source: str | None = None,
) -> int:
    """Run the full ``openclaw`` e2e verify checklist; return exit code.

    Eight steps — the proxy-only five plus harness-specific three
    (locate binary + write transient workspace + spawn-and-validate).

    See :func:`verify_proxy` for the ``key_source`` contract.
    """
    _silence_chatty_loggers()
    overall_start = time.monotonic()
    checklist = _Checklist(title="verify openclaw", total=8)

    checklist.run(
        "Resolving credentials",
        lambda: _check_credentials(api_key, base_url, key_source),
    )
    checklist.run(
        "Reaching backend",
        lambda: _check_backend_reachable(base_url, api_key, key_source),
    )

    state = _VerifyState()

    def _start() -> str:
        # OpenClaw always speaks OpenAI Chat Completions to the proxy,
        # so we always build the OpenAI translation chain — no probe
        # required (matches launch_openclaw).
        switchyard = _build_openclaw_switchyard(
            model, api_key, base_url, timeout,
        )
        server, thread, resolved_port = _start_proxy(switchyard, port)
        state.server = server
        state.thread = thread
        state.port = resolved_port
        return f"127.0.0.1:{resolved_port}"

    checklist.run("Starting proxy", _start)

    def _locate() -> str:
        bin_path = _locate_openclaw()
        state.openclaw_bin = bin_path
        return bin_path

    checklist.run("Locating openclaw binary", _locate)

    def _workspace() -> str:
        assert state.port is not None
        # Build a single-model catalog matching the one launch_openclaw
        # would emit, so the verify path exercises the same JSON5
        # schema.
        entries = [(
            model,
            f"{_openclaw_model_display_name(model)} (Switchyard)",
            f"Routed through Switchyard to {model}.",
        )]
        workspace = _write_openclaw_workspace(
            port=state.port,
            entries=entries,
            primary_model_id=f"switchyard/{model.lstrip('/')}",
        )
        state.openclaw_workspace = workspace
        return workspace

    checklist.run("Writing transient openclaw workspace", _workspace)

    def _roundtrip() -> str:
        assert state.port is not None
        return _proxy_roundtrip(state.port, model)

    checklist.run("Round-tripping chat completion", _roundtrip)

    def _spawn() -> str:
        assert state.openclaw_bin is not None
        assert state.openclaw_workspace is not None
        return _run_openclaw_prompt(
            state.openclaw_bin, state.openclaw_workspace, model,
        )

    checklist.run("Spawning openclaw with proxy workspace", _spawn)

    def _shutdown() -> str:
        if state.server is None or state.thread is None:
            return "(no proxy started)"
        result = _teardown(state.server, state.thread)
        _remove_openclaw_workspace(state.openclaw_workspace)
        return result

    checklist.run("Tearing down proxy", _shutdown)

    total_ms = (time.monotonic() - overall_start) * 1000.0
    return checklist.summarize("openclaw", model, total_ms)


__all__ = [
    "verify_proxy",
    "verify_claude",
    "verify_codex",
    "verify_openclaw",
]
