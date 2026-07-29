# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``switchyard.server.verify``.

Covers the three top-level orchestrators (``verify_proxy``,
``verify_claude``, ``verify_codex``), the ``_Checklist`` runner that
produces their pass/fail output, and each step body in isolation
(credential resolution, backend reachability, probe shape, proxy
round-trip, harness binary lookup, harness invocation, lifecycle
teardown).

Real network, uvicorn, and ``subprocess.run`` are all mocked — these
tests don't open a port or hit the wire.  The end-to-end integration
tests live in ``tests/offline_production_tests/v2/test_verify_e2e.py``
and are gated on a real NVIDIA API key + the harness binaries.

Mirrors the structure of :mod:`tests.test_launch_claude_v2` /
:mod:`tests.test_launch_codex_v2` for consistency: same fixture
patterns, same monkeypatch path strings, same per-section comment
banners.  Keeps the verify tests easy to read alongside their
launcher counterparts.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock

import httpx
import pytest

from switchyard.server import verify as verify_mod
from switchyard.server.verify import (
    _check_backend_reachable,
    _check_credentials,
    _Checklist,
    _locate_claude,
    _locate_codex,
    _proxy_roundtrip,
    _run_claude_prompt,
    _run_codex_prompt,
    _teardown,
    verify_claude,
    verify_codex,
    verify_proxy,
)

# ---------------------------------------------------------------------------
# _Checklist — the runner that powers the pass/fail UX
# ---------------------------------------------------------------------------


class TestChecklist:
    """Stdout rendering, halt-on-fail policy, and summary-line semantics."""

    def test_happy_path_three_ok_steps_returns_zero(self, capsys):
        cl = _Checklist(title="t", total=3)
        cl.run("a", lambda: "detail-a")
        cl.run("b", lambda: "detail-b")
        cl.run("c", lambda: "")  # detail-less ⇒ "OK (Nms)" form
        rc = cl.summarize("proxy", "model-x", 12.5)

        assert rc == 0
        out = capsys.readouterr().out
        assert "[1/3] a..." in out
        assert "OK (detail-a" in out
        assert "[2/3] b..." in out
        assert "[3/3] c..." in out
        assert "verify proxy: PASS (model=model-x" in out

    def test_first_failure_halts_subsequent_steps(self, capsys):
        cl = _Checklist(title="t", total=3)
        cl.run("a", lambda: "ok-a")

        def _boom() -> str:
            raise RuntimeError("nope")
        cl.run("b", _boom)

        sentinel: dict[str, bool] = {"called": False}
        def _later() -> str:
            sentinel["called"] = True
            return "should-not-render"
        cl.run("c", _later)

        rc = cl.summarize("proxy", "m", 1.0)

        assert rc == 1
        assert sentinel["called"] is False, (
            "Step c body must be skipped once an earlier step has failed; "
            "otherwise the verifier would mask the root-cause failure "
            "behind a cascade of follow-up failures."
        )
        out = capsys.readouterr().out
        assert "FAIL (RuntimeError: nope" in out
        assert "SKIP skipped (prior step failed)" in out
        assert "verify proxy: FAIL" in out

    def test_results_record_passed_and_detail(self, capsys):  # noqa: ARG002
        cl = _Checklist(title="t", total=2)
        cl.run("a", lambda: "ok-a-detail")
        cl.run("b", lambda: (_ for _ in ()).throw(ValueError("bad")))

        assert len(cl.results) == 2
        assert cl.results[0].passed is True
        assert cl.results[0].detail == "ok-a-detail"
        assert cl.results[1].passed is False
        assert "ValueError: bad" in cl.results[1].detail


# ---------------------------------------------------------------------------
# _check_credentials
# ---------------------------------------------------------------------------


class TestCheckCredentials:
    """Pure function: missing key raises, valid key produces a redacted label."""

    def test_missing_api_key_raises(self):
        with pytest.raises(RuntimeError, match="no API key resolved"):
            _check_credentials(None, "https://example.com/v1")

    def test_empty_api_key_raises(self):
        with pytest.raises(RuntimeError, match="no API key resolved"):
            _check_credentials("", "https://example.com/v1")

    def test_valid_credentials_returns_label(self):
        out = _check_credentials("sk-abc", "https://example.com/v1")
        # Key must be redacted — verify is run by users in shared
        # terminals (CI logs, screenshots, slack pastes).  A leaked
        # key in our own diagnostic is the worst possible outcome.
        assert "sk-abc" not in out
        assert "*****" in out
        assert "https://example.com/v1" in out

    def test_missing_base_url_falls_back_to_label(self):
        out = _check_credentials("sk-abc", None)
        assert "openai default" in out

    def test_key_source_surfaced_in_success_detail(self):
        """When the caller knows where the key came from, the success
        annotation should name the source.  This is the contract that
        lets the user spot which env var / file is in play *before* a
        downstream failure forces them to hunt.
        """
        out = _check_credentials(
            "sk-abc", "https://example.com/v1", key_source="$NVIDIA_API_KEY",
        )
        assert "from $NVIDIA_API_KEY" in out
        # Source gets prefixed by " from " so it reads naturally next
        # to the redacted key:  "api_key=***** from $NVIDIA_API_KEY".
        assert "api_key=***** from $NVIDIA_API_KEY" in out

    def test_key_source_omitted_when_caller_doesnt_track_it(self):
        """Direct API callers can omit ``key_source`` — no source label
        should appear, and the rest of the detail should still render.
        """
        out = _check_credentials("sk-abc", "https://example.com/v1")
        assert " from " not in out
        assert "*****" in out


# ---------------------------------------------------------------------------
# _check_backend_reachable
# ---------------------------------------------------------------------------


class TestCheckBackendReachable:
    """HTTP semantics: 200 OK returns detail, non-200 raises with status code."""

    def test_200_returns_detail(self, monkeypatch):
        captured: dict = {}

        class _FakeResp:
            status_code = 200
            text = ""

        class _FakeClient:
            def __init__(self, *_, **kwargs):
                captured["timeout"] = kwargs.get("timeout")

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def get(self, url, headers):
                captured["url"] = url
                captured["headers"] = headers
                return _FakeResp()

        monkeypatch.setattr(httpx, "Client", _FakeClient)

        out = _check_backend_reachable(
            "https://api.example.com/v1", "sk-abc",
        )
        assert "GET /models 200" in out
        assert captured["url"] == "https://api.example.com/v1/models"
        assert captured["headers"]["Authorization"] == "Bearer sk-abc"

    def _install_fake_status(self, monkeypatch, status_code: int, body: str = ""):
        """Helper: pin ``httpx.Client.get`` to return a fixed status code.

        Lets each status-code branch test stay self-contained without
        repeating the four-method fake-client boilerplate.
        """
        class _FakeResp:
            def __init__(self, code: int, text: str) -> None:
                self.status_code = code
                self.text = text

        class _FakeClient:
            def __init__(self, *_, **__):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def get(self, url, headers):  # noqa: ARG002
                return _FakeResp(status_code, body)

        monkeypatch.setattr(httpx, "Client", _FakeClient)

    def test_401_names_the_key_source(self, monkeypatch):
        """The 401 branch must call out (a) that the *backend* rejected
        the key, (b) which source the key came from, and (c) what to
        do about it.  All three are user-facing requirements — a bare
        "HTTP 401" forces the user to hunt for which knob to turn.
        """
        self._install_fake_status(monkeypatch, 401, '{"error": "invalid"}')

        with pytest.raises(RuntimeError) as exc_info:
            _check_backend_reachable(
                "https://api.example.com/v1", "sk-bad",
                key_source="$NVIDIA_API_KEY",
            )
        msg = str(exc_info.value)
        assert "HTTP 401" in msg
        assert "$NVIDIA_API_KEY" in msg
        assert "rejected" in msg.lower()
        # Actionable hint must be present — the user's next move is
        # rotate-or-replace, not "investigate".
        assert "rotate" in msg.lower() or "double-check" in msg.lower()

    def test_401_uses_generic_label_when_source_unknown(self, monkeypatch):
        """Direct callers may not know where the key came from — fall
        back to a neutral phrase rather than a literal ``"None"``.
        """
        self._install_fake_status(monkeypatch, 401)
        with pytest.raises(RuntimeError) as exc_info:
            _check_backend_reachable(
                "https://api.example.com/v1", "sk-bad",
            )
        msg = str(exc_info.value)
        assert "HTTP 401" in msg
        assert "None" not in msg
        assert "the resolved key" in msg

    def test_404_hints_at_base_url_typo(self, monkeypatch):
        """404 means the URL is wrong, not the key.  The hint must
        steer the user to the URL.  Common case: ``--base-url`` was
        set without the ``/v1`` suffix or with a typo elsewhere.
        """
        self._install_fake_status(monkeypatch, 404)
        with pytest.raises(RuntimeError) as exc_info:
            _check_backend_reachable(
                "https://api.example.com/typo", "sk-abc",
                key_source="$NVIDIA_API_KEY",
            )
        msg = str(exc_info.value)
        assert "HTTP 404" in msg
        assert "typo" in msg.lower()
        assert "/v1" in msg
        # Source label is irrelevant for a 404 (it's a URL problem,
        # not a credential problem) — must NOT mislead the user by
        # mentioning the key source.
        assert "$NVIDIA_API_KEY" not in msg

    def test_other_4xx_falls_through_to_generic(self, monkeypatch):
        """Status codes outside 401/404 use the generic "HTTP <code>"
        message — keeps the surface predictable and doesn't pretend
        to diagnose codes we haven't validated specific hints for.
        """
        self._install_fake_status(monkeypatch, 403)
        with pytest.raises(RuntimeError) as exc_info:
            _check_backend_reachable("https://api.example.com/v1", "sk-abc")
        assert "HTTP 403" in str(exc_info.value)

    def test_strips_trailing_slash_in_url(self, monkeypatch):
        captured: dict = {}

        class _FakeResp:
            status_code = 200
            text = ""

        class _FakeClient:
            def __init__(self, *_, **__):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def get(self, url, headers):  # noqa: ARG002
                captured["url"] = url
                return _FakeResp()

        monkeypatch.setattr(httpx, "Client", _FakeClient)

        _check_backend_reachable("https://api.example.com/v1/", "sk-abc")
        # Must not produce ``//models`` from ``base_url='/v1/'`` + ``/models``.
        assert captured["url"] == "https://api.example.com/v1/models"


# ---------------------------------------------------------------------------
# _proxy_roundtrip
# ---------------------------------------------------------------------------


class TestProxyRoundtrip:
    """Validates the synthetic chat-completions probe sent through the proxy."""

    def _install_fake_response(self, monkeypatch, payload, status_code=200):
        """Helper: pin ``httpx.Client.post`` to return a fixed JSON body.

        Centralizes the four-method fake-client boilerplate so each
        scenario test reads as a single ``payload`` dict.  ``status_code``
        is overridable for the non-200 case.
        """
        class _FakeResp:
            def __init__(self, code, body):
                self.status_code = code
                self.text = ""
                self._body = body

            def json(self):
                return self._body

        class _FakeClient:
            def __init__(self, *_, **kwargs):
                # SWITCH-857: the loopback verify roundtrip must bypass env
                # proxies, so verify.py builds this client with trust_env=False.
                # Assert it here so dropping that kwarg fails the suite.
                assert kwargs.get("trust_env") is False, (
                    "verify roundtrip must use httpx.Client(trust_env=False) to "
                    "bypass env proxies on the 127.0.0.1 probe"
                )

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def post(self, url, json):  # noqa: ARG002
                return _FakeResp(status_code, payload)

        monkeypatch.setattr(httpx, "Client", _FakeClient)

    def test_happy_path_with_visible_text_returns_reply_preview(
        self, monkeypatch,
    ):
        self._install_fake_response(monkeypatch, {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1},
        })

        out = _proxy_roundtrip(54321, "test-model")
        assert "reply='ok'" in out
        assert "in=10" in out
        assert "out=1" in out

    def test_non_200_raises_with_body_snippet(self, monkeypatch):
        self._install_fake_response(
            monkeypatch, {}, status_code=500,
        )
        with pytest.raises(RuntimeError, match="HTTP 500"):
            _proxy_roundtrip(54321, "test-model")

    def test_reasoning_only_truncation_is_tolerated(self, monkeypatch):
        """Reasoning models (Qwen, GPT-5-reasoning) at small max_tokens
        consume the entire budget on internal reasoning and surface
        empty visible text.  The chain still worked end-to-end —
        verify must report this as a pass with an annotation, not as
        a failure that would mask a real chain bug.

        Critical contract: completion_tokens > 0 (model *did*
        generate) + content empty + finish_reason="length" → PASS
        with ``"no visible text (reasoning-only)"`` detail.
        """
        self._install_fake_response(monkeypatch, {
            "choices": [{
                "message": {"content": ""},
                "finish_reason": "length",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2048},
        })

        out = _proxy_roundtrip(54321, "test-model")
        assert "no visible text" in out
        assert "reasoning-only" in out
        assert "finish_reason=length" in out
        # Token counts must still appear so the user sees the chain
        # *did* shuffle tokens through — that's the positive signal.
        assert "in=10" in out
        assert "out=2048" in out

    def test_zero_output_tokens_raises(self, monkeypatch):
        """``completion_tokens == 0`` is a real failure: the upstream
        returned a structurally valid response but never generated
        anything, so verify can't claim "all good".  Distinct from
        the reasoning-truncation case (which has nonzero output
        tokens, just no visible text).
        """
        self._install_fake_response(monkeypatch, {
            "choices": [{"message": {"content": ""}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 0},
        })

        with pytest.raises(RuntimeError, match="no output tokens"):
            _proxy_roundtrip(54321, "test-model")

    def test_no_choices_raises(self, monkeypatch):
        """A 200 response with no ``choices`` array means the upstream
        replied but the chain didn't produce a usable shape — has
        happened with badly-configured upstreams that 200 a JSON
        error blob.  Surface it loudly.
        """
        self._install_fake_response(monkeypatch, {
            "id": "bad", "object": "chat.completion", "usage": {},
        })

        with pytest.raises(RuntimeError, match="no choices"):
            _proxy_roundtrip(54321, "test-model")


# ---------------------------------------------------------------------------
# _locate_claude / _locate_codex
# ---------------------------------------------------------------------------


class TestLocateBinaries:
    """Reuses the launcher's binary-finder so resolution stays in lockstep."""

    def test_locate_claude_present(self, monkeypatch):
        monkeypatch.setattr(verify_mod, "_find_claude_binary", lambda: "/x/claude")
        assert _locate_claude() == "/x/claude"

    def test_locate_claude_missing_raises_with_install_hint(self, monkeypatch):
        monkeypatch.setattr(verify_mod, "_find_claude_binary", lambda: None)
        with pytest.raises(RuntimeError, match="claude binary not found"):
            _locate_claude()

    def test_locate_codex_present(self, monkeypatch):
        monkeypatch.setattr(verify_mod, "_find_codex_binary", lambda: "/x/codex")
        assert _locate_codex() == "/x/codex"

    def test_locate_codex_missing_raises_with_install_hint(self, monkeypatch):
        monkeypatch.setattr(verify_mod, "_find_codex_binary", lambda: None)
        with pytest.raises(RuntimeError, match="codex binary not found"):
            _locate_codex()


# ---------------------------------------------------------------------------
# _run_claude_prompt / _run_codex_prompt
# ---------------------------------------------------------------------------


class TestRunClaudePrompt:
    """Subprocess invocation: env injection, exit-code handling, timeout,
    and the ``--output-format json`` envelope contract.
    """

    @staticmethod
    def _envelope(result: str | None = "ok", **extra) -> str:
        """Build a claude ``--output-format json`` envelope for tests.

        Mirrors the real shape: a single JSON object with ``type``,
        ``subtype``, ``result``, plus optional ``num_turns`` /
        ``stop_reason`` fields claude surfaces in the empty-result
        case.  Centralized so test data stays consistent.
        """
        import json as _json
        body = {
            "type": "result",
            "subtype": "success",
            "result": result,
        }
        body.update(extra)
        return _json.dumps(body)

    def test_happy_path_returns_reply_preview(self, monkeypatch):
        captured: dict = {}

        def fake_run(cmd, env, capture_output, text, timeout, check):  # noqa: ARG001
            captured["cmd"] = cmd
            captured["env"] = env
            return subprocess.CompletedProcess(
                cmd, returncode=0, stdout=self._envelope("ok"), stderr="",
            )
        monkeypatch.setattr(subprocess, "run", fake_run)

        out = _run_claude_prompt("/fake/claude", 54321, "test-model")
        assert "reply='ok'" in out

        # Env-var contract must match launch_claude — drift here means
        # the verifier validates a shape Claude Code doesn't actually
        # accept in production.
        env = captured["env"]
        assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:54321"
        assert env["ANTHROPIC_AUTH_TOKEN"] == "switchyard"
        assert env["ANTHROPIC_API_KEY"] == ""
        assert env["ANTHROPIC_MODEL"] == "test-model"
        assert env["ANTHROPIC_SMALL_FAST_MODEL"] == "test-model"

        # Argv contract: -p prompt, JSON output, --max-turns 1, skip
        # permissions.  The output-format pin is what enables the
        # reasoning-truncation tolerance below — drift here would
        # break that diagnostic.
        cmd = captured["cmd"]
        assert cmd[0] == "/fake/claude"
        assert "-p" in cmd
        assert "--output-format" in cmd
        assert "json" in cmd
        assert "--max-turns" in cmd
        assert "--dangerously-skip-permissions" in cmd

    def test_non_zero_exit_surfaces_stderr(self, monkeypatch):
        def fake_run(cmd, **kwargs):  # noqa: ARG001
            return subprocess.CompletedProcess(
                cmd, returncode=1, stdout="", stderr="boom",
            )
        monkeypatch.setattr(subprocess, "run", fake_run)

        with pytest.raises(RuntimeError, match=r"claude exited 1.*boom"):
            _run_claude_prompt("/fake/claude", 54321, "m")

    def test_empty_stdout_raises(self, monkeypatch):
        """Exit 0 with literally no stdout means claude didn't even
        emit its envelope — that's a real failure, distinct from
        the "envelope present but ``result`` empty" case below.
        """
        def fake_run(cmd, **kwargs):  # noqa: ARG001
            return subprocess.CompletedProcess(
                cmd, returncode=0, stdout="   \n", stderr="",
            )
        monkeypatch.setattr(subprocess, "run", fake_run)

        with pytest.raises(RuntimeError, match="exited 0 but produced no stdout"):
            _run_claude_prompt("/fake/claude", 54321, "m")

    def test_empty_result_field_is_tolerated(self, monkeypatch):
        """Reasoning models burn the budget on internal reasoning and
        surface an empty ``result`` field.  Same contract as the
        proxy roundtrip step: chain worked end-to-end → PASS with
        a "reasoning-only" annotation rather than failing.
        """
        def fake_run(cmd, **kwargs):  # noqa: ARG001
            return subprocess.CompletedProcess(
                cmd, returncode=0,
                stdout=self._envelope(
                    result="", num_turns=1, stop_reason="length",
                ),
                stderr="",
            )
        monkeypatch.setattr(subprocess, "run", fake_run)

        out = _run_claude_prompt("/fake/claude", 54321, "qwen-reasoning")
        assert "no visible reply" in out
        assert "reasoning-only" in out
        # Surface the envelope hints so the user sees *why* the
        # reply is empty — it's a model-budget artifact, not a
        # chain bug.
        assert "num_turns=1" in out
        assert "stop_reason=length" in out

    def test_invalid_json_stdout_raises(self, monkeypatch):
        """Exit 0 with stdout that isn't valid JSON means claude's
        output contract changed — surface as a real failure rather
        than swallowing the malformed body.
        """
        def fake_run(cmd, **kwargs):  # noqa: ARG001
            return subprocess.CompletedProcess(
                cmd, returncode=0, stdout="just plain text, not json",
                stderr="",
            )
        monkeypatch.setattr(subprocess, "run", fake_run)

        with pytest.raises(RuntimeError, match="non-JSON stdout"):
            _run_claude_prompt("/fake/claude", 54321, "m")

    def test_timeout_raises_runtime_error(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
        monkeypatch.setattr(subprocess, "run", fake_run)

        with pytest.raises(RuntimeError, match="claude did not exit within"):
            _run_claude_prompt("/fake/claude", 54321, "m")


class TestRunCodexPrompt:
    """Sibling of TestRunClaudePrompt.  Codex's ``exec`` mode doesn't
    expose a structured envelope, so the reasoning-only-truncation
    detection here is coarser (exit 0 + empty stdout = tolerate)
    than in :class:`TestRunClaudePrompt`.
    """

    def test_happy_path_returns_reply_preview(self, monkeypatch):
        captured: dict = {}

        def fake_run(cmd, env, capture_output, text, timeout, check):  # noqa: ARG001
            captured["cmd"] = cmd
            captured["env"] = env
            return subprocess.CompletedProcess(
                cmd, returncode=0, stdout="ok\n", stderr="",
            )
        monkeypatch.setattr(subprocess, "run", fake_run)

        out = _run_codex_prompt("/fake/codex", 54321, "test-model")
        assert "reply='ok'" in out

        # OPENAI_API_KEY placeholder is required for codex's provider
        # config validation; matches launch_codex's contract.
        assert captured["env"]["OPENAI_API_KEY"] == "switchyard"

        # Argv contract: codex_bin, six -c overrides (12 entries),
        # -m model, exec, prompt.  Provider overrides must include
        # the proxy URL with the chosen port.
        cmd = captured["cmd"]
        assert cmd[0] == "/fake/codex"
        assert "exec" in cmd
        assert "-m" in cmd
        assert "test-model" in cmd
        assert any('base_url="http://127.0.0.1:54321/v1"' in v for v in cmd)

    def test_non_zero_exit_surfaces_stderr(self, monkeypatch):
        def fake_run(cmd, **kwargs):  # noqa: ARG001
            return subprocess.CompletedProcess(
                cmd, returncode=2, stdout="", stderr="codex error",
            )
        monkeypatch.setattr(subprocess, "run", fake_run)

        with pytest.raises(RuntimeError, match=r"codex exited 2.*codex error"):
            _run_codex_prompt("/fake/codex", 54321, "m")

    def test_empty_stdout_is_tolerated(self, monkeypatch):
        """Exit 0 with empty stdout = reasoning-only truncation.
        Codex doesn't emit an envelope so we can't be precise about
        ``num_turns`` / ``stop_reason``, but the chain still ran
        end-to-end (exit 0).  Must not false-fail here.
        """
        def fake_run(cmd, **kwargs):  # noqa: ARG001
            return subprocess.CompletedProcess(
                cmd, returncode=0, stdout="   \n", stderr="",
            )
        monkeypatch.setattr(subprocess, "run", fake_run)

        out = _run_codex_prompt("/fake/codex", 54321, "qwen-reasoning")
        assert "no visible reply" in out
        assert "reasoning-only" in out

    def test_timeout_raises_runtime_error(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
        monkeypatch.setattr(subprocess, "run", fake_run)

        with pytest.raises(RuntimeError, match="codex did not exit within"):
            _run_codex_prompt("/fake/codex", 54321, "m")


# ---------------------------------------------------------------------------
# _teardown
# ---------------------------------------------------------------------------


class TestTeardown:
    """Lifecycle: signal exit (both graceful + force), join with timeout,
    soft-warn rather than fail loud when the shutdown is slow.
    """

    def test_clean_shutdown_returns_empty_detail(self):
        server = MagicMock()
        server.should_exit = False
        server.force_exit = False
        thread = MagicMock()
        thread.is_alive.return_value = False

        out = _teardown(server, thread)
        assert out == ""
        # Both flags must be set: uvicorn's default
        # timeout_graceful_shutdown=None blocks shutdown until all
        # keep-alive connections close, so streaming clients (Claude
        # Code, codex) can hold the proxy hostage without
        # ``force_exit``.  Drift here would silently regress the
        # streaming-shutdown contract.
        assert server.should_exit is True
        assert server.force_exit is True
        thread.join.assert_called_once()

    def test_slow_shutdown_returns_warning_not_failure(self):
        """When uvicorn doesn't drain in time, the teardown step must
        return a soft warning rather than raising — verify is a
        single-shot CLI, the proxy thread is a daemon, and so a
        lingering thread dies on process exit anyway.  Failing here
        would mask the real signal (the verify chain worked end-to-
        end) behind a uvicorn shutdown quirk.
        """
        server = MagicMock()
        thread = MagicMock()
        thread.is_alive.return_value = True

        out = _teardown(server, thread)
        # Soft warning text — the user should still see that
        # shutdown was slow so a real shutdown bug isn't fully
        # silenced, but the step doesn't fail the verify run.
        assert "shutdown slow" in out
        assert "daemon thread" in out


# ---------------------------------------------------------------------------
# Top-level orchestrators (with externals mocked)
# ---------------------------------------------------------------------------


def _make_fake_server(started: bool = True) -> MagicMock:
    server = MagicMock()
    server.started = started
    server.should_exit = False
    return server


def _stub_spawn_proxy_returning(server: MagicMock):
    def _inner(switchyard, port):  # noqa: ARG001
        thread = MagicMock()
        thread.is_alive.return_value = False
        return server, thread
    return _inner


def _stub_httpx_responses(monkeypatch, *, models_status=200, chat_status=200,
                         chat_content="ok"):
    """Wire ``httpx.Client`` to return canned responses for ``GET /models``
    and ``POST /v1/chat/completions``.  Routing is by URL substring so
    one stub covers both verify steps in a single fixture.
    """

    class _FakeResp:
        def __init__(self, status, body=None):
            self.status_code = status
            self.text = ""
            self._body = body or {}

        def json(self):
            return self._body

    class _FakeClient:
        def __init__(self, *_, **__):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def get(self, url, headers):  # noqa: ARG002
            return _FakeResp(models_status)

        def post(self, url, json):  # noqa: ARG002
            body = {
                "choices": [{"message": {"content": chat_content}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            }
            return _FakeResp(chat_status, body)

    monkeypatch.setattr(httpx, "Client", _FakeClient)


@pytest.fixture
def _mock_verify_externals(monkeypatch):
    """Mock probe + chain construction so verify_* never hit the wire.

    Individual tests still install their own ``subprocess.run`` /
    ``httpx.Client`` mocks because each one tests a different mode-
    specific code path; this fixture just covers the parts that are
    identical across all three orchestrators.
    """
    async def _probe_false(**_):
        return False
    def _build_switchyard(*_args, **_kwargs):
        return MagicMock()
    monkeypatch.setattr(verify_mod, "probe_anthropic_messages_support", _probe_false)
    monkeypatch.setattr(verify_mod, "_build_claude_switchyard", _build_switchyard)
    monkeypatch.setattr(
        verify_mod, "_build_codex_switchyard", lambda *a, **kw: MagicMock(),
    )
    monkeypatch.setattr(
        verify_mod, "_spawn_proxy_thread",
        _stub_spawn_proxy_returning(_make_fake_server(started=True)),
    )
    monkeypatch.setattr(verify_mod, "_find_free_port", lambda: 54321)


class TestVerifyProxyOrchestrator:
    """End-to-end smoke for the proxy-only mode with all externals mocked."""

    def test_happy_path_returns_zero(
        self, monkeypatch, capsys, _mock_verify_externals,
    ):
        _stub_httpx_responses(monkeypatch)

        rc = verify_proxy(
            model="test-model",
            base_url="https://example.com/v1",
            api_key="sk-test",
            port=None,
            timeout=None,
        )

        assert rc == 0
        out = capsys.readouterr().out
        # All six steps must report — the count check is the
        # contract that the runner didn't short-circuit silently.
        assert "[1/6]" in out
        assert "[6/6]" in out
        assert "verify proxy: PASS" in out

    def test_unreachable_backend_returns_one_and_skips_rest(
        self, monkeypatch, capsys, _mock_verify_externals,
    ):
        _stub_httpx_responses(monkeypatch, models_status=503)

        rc = verify_proxy(
            model="test-model",
            base_url="https://example.com/v1",
            api_key="sk-test",
            port=None,
            timeout=None,
        )

        assert rc == 1
        out = capsys.readouterr().out
        assert "FAIL (RuntimeError" in out
        assert "verify proxy: FAIL" in out
        # Subsequent steps must be skipped — ensures the user sees
        # ONE root-cause failure, not a cascade of derived ones.
        assert "SKIP" in out

    def test_missing_api_key_fails_at_step_1(
        self, monkeypatch, capsys, _mock_verify_externals,
    ):
        _stub_httpx_responses(monkeypatch)

        rc = verify_proxy(
            model="test-model",
            base_url="https://example.com/v1",
            api_key="",
            port=None,
            timeout=None,
        )

        assert rc == 1
        out = capsys.readouterr().out
        # Failure at step 1 means everything else gets SKIP.
        assert "[1/6] Resolving credentials..." in out
        assert "FAIL" in out


class TestVerifyClaudeOrchestrator:
    """End-to-end smoke for ``verify claude`` with all externals mocked."""

    def test_happy_path_returns_zero(
        self, monkeypatch, capsys, _mock_verify_externals,
    ):
        _stub_httpx_responses(monkeypatch)
        monkeypatch.setattr(verify_mod, "_find_claude_binary", lambda: "/x/claude")

        # Claude's --output-format json envelope shape — see
        # :class:`TestRunClaudePrompt._envelope` for the canonical
        # builder.  Inlined here because this test only needs a
        # single happy-path body and importing across test classes
        # would couple them.
        import json as _json
        envelope = _json.dumps({
            "type": "result", "subtype": "success", "result": "ok",
        })

        def fake_run(cmd, **kwargs):  # noqa: ARG001
            return subprocess.CompletedProcess(
                cmd, returncode=0, stdout=envelope, stderr="",
            )
        monkeypatch.setattr(subprocess, "run", fake_run)

        rc = verify_claude(
            model="test-model",
            base_url="https://example.com/v1",
            api_key="sk-test",
            port=None,
            timeout=None,
        )

        assert rc == 0
        out = capsys.readouterr().out
        assert "[8/8]" in out
        assert "verify claude: PASS" in out

    def test_missing_claude_binary_returns_one(
        self, monkeypatch, capsys, _mock_verify_externals,
    ):
        _stub_httpx_responses(monkeypatch)
        monkeypatch.setattr(verify_mod, "_find_claude_binary", lambda: None)

        sentinel = {"called": False}
        def fake_run(*_args, **_kwargs):
            sentinel["called"] = True
            return subprocess.CompletedProcess([], returncode=0)
        monkeypatch.setattr(subprocess, "run", fake_run)

        rc = verify_claude(
            model="test-model",
            base_url="https://example.com/v1",
            api_key="sk-test",
            port=None,
            timeout=None,
        )

        assert rc == 1
        # claude must NOT have been spawned — locating fails first.
        assert sentinel["called"] is False
        out = capsys.readouterr().out
        assert "Locating claude binary" in out
        assert "claude binary not found" in out


class TestVerifyCodexOrchestrator:
    """End-to-end smoke for ``verify codex`` with all externals mocked."""

    def test_happy_path_returns_zero(
        self, monkeypatch, capsys, _mock_verify_externals,
    ):
        _stub_httpx_responses(monkeypatch)
        monkeypatch.setattr(verify_mod, "_find_codex_binary", lambda: "/x/codex")

        def fake_run(cmd, **kwargs):  # noqa: ARG001
            return subprocess.CompletedProcess(
                cmd, returncode=0, stdout="ok\n", stderr="",
            )
        monkeypatch.setattr(subprocess, "run", fake_run)

        rc = verify_codex(
            model="test-model",
            base_url="https://example.com/v1",
            api_key="sk-test",
            port=None,
            timeout=None,
        )

        assert rc == 0
        out = capsys.readouterr().out
        # codex skips the Anthropic probe step (only 7 total) — pin
        # the count so the test catches if the orchestrator drifts to
        # claude's 8-step shape.
        assert "[7/7]" in out
        assert "[8/8]" not in out
        assert "verify codex: PASS" in out

    def test_missing_codex_binary_returns_one(
        self, monkeypatch, capsys, _mock_verify_externals,
    ):
        _stub_httpx_responses(monkeypatch)
        monkeypatch.setattr(verify_mod, "_find_codex_binary", lambda: None)

        rc = verify_codex(
            model="test-model",
            base_url="https://example.com/v1",
            api_key="sk-test",
            port=None,
            timeout=None,
        )

        assert rc == 1
        out = capsys.readouterr().out
        assert "Locating codex binary" in out
        assert "codex binary not found" in out


# ---------------------------------------------------------------------------
# CLI handlers — credential resolution + arg forwarding
# ---------------------------------------------------------------------------


def _isolate_credentials(monkeypatch, *, secrets_exists: bool = False,
                        secrets_content: dict | None = None) -> None:
    """Wipe credentials from the test environment.

    Drops every env var the verify resolver inspects and points
    ``DEFAULT_SECRETS_FILE`` at a controllable temp file.  Tests
    that want to exercise specific source combinations can then
    set just the env var(s) they care about.

    The default ``secrets_exists=False`` makes the file appear
    missing — most "no key" tests want this.  ``secrets_content``
    populates the file when ``secrets_exists=True``.
    """
    for var in (
        "OPENROUTER_API_KEY", "NVIDIA_API_KEY", "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY", "OPENROUTER_BASE_URL", "NVIDIA_BASE_URL",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(var, raising=False)

    import json as _json
    import tempfile

    fake_path = (
        tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
        if secrets_exists
        else "/tmp/__verify_test_no_such_file_should_not_exist__.json"
    )
    if secrets_exists:
        from pathlib import Path as _Path
        _Path(fake_path).write_text(_json.dumps(secrets_content or {}))

    from pathlib import Path as _Path

    monkeypatch.setattr(
        "switchyard.cli.switchyard_cli.DEFAULT_SECRETS_FILE",
        _Path(fake_path),
    )
    # ``load_secrets`` reads from ``DEFAULT_SECRETS_FILE`` in the
    # ``server_util`` module too — patch both so the diagnostic and
    # the underlying resolver see the same picture.
    monkeypatch.setattr(
        "switchyard.server.server_util.DEFAULT_SECRETS_FILE",
        _Path(fake_path),
    )


# ---------------------------------------------------------------------------
# _diagnose_credential_resolution
# ---------------------------------------------------------------------------


class TestDiagnoseCredentialResolution:
    """The waterfall walker that powers verify's "why no key?" UX.

    Every test must assert (a) the resolved ``api_key`` value,
    (b) the ``key_source`` label, and (c) that ``attempts`` contains
    a row for *every* source we considered (so the no-key error
    can list them all).  Together those three pin both the success
    contract (right source wins) and the failure UX (no source is
    silently skipped).
    """

    def _make_args(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        port: int | None = None,
    ):
        """Build a minimal ``argparse.Namespace`` matching what the
        verify CLI parser produces.  ``port`` is included because
        the underlying ``resolve_config_with_secrets`` helper
        mutates it from secrets when present.
        """
        import argparse as _argparse
        return _argparse.Namespace(
            api_key=api_key, base_url=base_url, port=port,
        )

    def test_no_sources_returns_none_with_full_attempt_list(self, monkeypatch):
        from switchyard.cli.switchyard_cli import (
            _diagnose_credential_resolution,
        )
        _isolate_credentials(monkeypatch)

        api_key, _base_url, key_source, attempts = _diagnose_credential_resolution(
            self._make_args(),
        )

        assert api_key is None
        assert key_source is None
        # Sources verified in resolution order: --api-key, env vars, then
        # secrets.json provider sections. Pin the order so a future refactor
        # can't silently drop a source.
        labels = [a.label for a in attempts]
        assert labels == [
            "--api-key",
            "$OPENROUTER_API_KEY",
            "$NVIDIA_API_KEY",
            "$OPENAI_API_KEY",
            "$ANTHROPIC_API_KEY",
            "secrets.json[openrouter.api_key]",
            "secrets.json[nvidia.api_key]",
        ]
        # Every attempt's status must explain *why* it didn't yield a
        # key — empty / missing / not-set — so the tabulated error
        # message has something useful per row.
        for attempt in attempts:
            assert attempt.has_value is False
            assert attempt.status, (
                f"empty status for {attempt.label!r} would render as a "
                f"blank row in the user-facing error table"
            )

    def test_cli_api_key_wins(self, monkeypatch):
        from switchyard.cli.switchyard_cli import (
            _diagnose_credential_resolution,
        )
        _isolate_credentials(monkeypatch)
        # Plant a competing env var to verify CLI wins per the
        # waterfall order — this is the same priority the launcher
        # uses, so a regression here would be a behavior change.
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-from-env")

        api_key, _base_url, key_source, _ = _diagnose_credential_resolution(
            self._make_args(api_key="sk-from-cli"),
        )

        assert api_key == "sk-from-cli"
        assert key_source == "--api-key"

    def test_env_var_wins_when_no_cli(self, monkeypatch):
        from switchyard.cli.switchyard_cli import (
            _diagnose_credential_resolution,
        )
        _isolate_credentials(monkeypatch)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-from-openrouter")
        monkeypatch.setenv("NVIDIA_API_KEY", "sk-from-nvidia")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-openai")

        api_key, _, key_source, _ = _diagnose_credential_resolution(
            self._make_args(),
        )

        assert api_key == "sk-from-openrouter"  # pragma: allowlist secret  # pragma: allowlist secret
        assert key_source == "$OPENROUTER_API_KEY"

    def test_base_url_matches_selected_env_provider(self, monkeypatch):
        from switchyard.cli.switchyard_cli import (
            _diagnose_credential_resolution,
        )
        _isolate_credentials(monkeypatch)
        monkeypatch.setenv("NVIDIA_API_KEY", "sk-from-nvidia")
        monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.test/v1")

        api_key, base_url, key_source, _ = _diagnose_credential_resolution(
            self._make_args(),
        )

        assert api_key == "sk-from-nvidia"
        assert key_source == "$NVIDIA_API_KEY"
        assert base_url == "https://inference-api.nvidia.com/v1"

    def test_secrets_file_wins_when_no_cli_or_env(self, monkeypatch):
        from switchyard.cli.switchyard_cli import (
            _diagnose_credential_resolution,
        )
        _isolate_credentials(
            monkeypatch,
            secrets_exists=True,
            secrets_content={
                "openrouter": {
                    "api_key": "sk-from-secrets",
                    "base_url": "https://from-secrets/v1",
                },
            },
        )

        api_key, base_url, key_source, _ = _diagnose_credential_resolution(
            self._make_args(),
        )

        assert api_key == "sk-from-secrets"
        assert key_source == "secrets.json[openrouter.api_key]"
        assert base_url == "https://from-secrets/v1"

    def test_secrets_section_missing_status(self, monkeypatch):
        """secrets.json is present but the ``openrouter`` section isn't.
        The diagnostic must distinguish "file not found" from
        "section missing" — they're different fixes.
        """
        from switchyard.cli.switchyard_cli import (
            _diagnose_credential_resolution,
        )
        _isolate_credentials(
            monkeypatch,
            secrets_exists=True,
            secrets_content={"some_other_section": {}},
        )

        api_key, _, _, attempts = _diagnose_credential_resolution(
            self._make_args(),
        )
        assert api_key is None
        secrets_row = next(a for a in attempts if "secrets.json" in a.label)
        assert "missing" in secrets_row.status

    def test_secrets_section_present_field_missing_status(self, monkeypatch):
        """``openrouter`` section exists but has no ``api_key`` field —
        another distinct failure mode requiring its own status string.
        """
        from switchyard.cli.switchyard_cli import (
            _diagnose_credential_resolution,
        )
        _isolate_credentials(
            monkeypatch,
            secrets_exists=True,
            secrets_content={"openrouter": {"base_url": "https://x/v1"}},
        )

        api_key, _, _, attempts = _diagnose_credential_resolution(
            self._make_args(),
        )
        assert api_key is None
        secrets_row = next(a for a in attempts if "secrets.json" in a.label)
        assert "field missing" in secrets_row.status

    def test_empty_env_var_reported_distinctly(self, monkeypatch):
        """An env var that's been *set* to the empty string is a
        common subtle bug (typically a shell rc that exports the
        var unconditionally).  The status must say ``"empty (0
        chars)"`` rather than ``"not set"`` so the user sees the
        difference.
        """
        from switchyard.cli.switchyard_cli import (
            _diagnose_credential_resolution,
        )
        _isolate_credentials(monkeypatch)
        monkeypatch.setenv("OPENROUTER_API_KEY", "")

        _, _, _, attempts = _diagnose_credential_resolution(
            self._make_args(),
        )
        openrouter_row = next(a for a in attempts if a.label == "$OPENROUTER_API_KEY")
        assert "empty" in openrouter_row.status

    def test_set_env_var_reports_char_count(self, monkeypatch):
        """A set env var should advertise its length — useful for
        spotting truncated keys without ever exposing the secret.
        """
        from switchyard.cli.switchyard_cli import (
            _diagnose_credential_resolution,
        )
        _isolate_credentials(monkeypatch)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-12345")

        _, _, _, attempts = _diagnose_credential_resolution(
            self._make_args(),
        )
        openrouter_row = next(a for a in attempts if a.label == "$OPENROUTER_API_KEY")
        # 8 chars = len("sk-12345").  The actual key value MUST NOT
        # appear in the status — leaking secrets in our own
        # diagnostic is the worst possible outcome.
        assert "8 chars" in openrouter_row.status
        assert "sk-12345" not in openrouter_row.status


# ---------------------------------------------------------------------------
# _format_credential_failure
# ---------------------------------------------------------------------------


class TestFormatCredentialFailure:
    """The tabulated user-facing error.  Pin the parts that matter:
    every source label appears, each row has its status, and a
    concrete "to fix" pointer is at the bottom.
    """

    def test_all_attempts_appear_in_output(self):
        from switchyard.cli.switchyard_cli import (
            _CredentialAttempt,
            _format_credential_failure,
        )
        attempts = [
            _CredentialAttempt("--api-key", "not provided", False),
            _CredentialAttempt("$OPENROUTER_API_KEY", "not set", False),
            _CredentialAttempt("$OPENAI_API_KEY", "empty (0 chars)", False),
            _CredentialAttempt("secrets.json[openrouter.api_key]", "file not found", False),
        ]

        out = _format_credential_failure(attempts)

        for attempt in attempts:
            assert attempt.label in out, (
                f"missing row for {attempt.label!r} — user can't see "
                f"that source was checked"
            )
            assert attempt.status in out
        # "To fix" pointer must include both the env-var path and the
        # secrets-file path so the user has two concrete next moves.
        assert "export OPENROUTER_API_KEY" in out
        assert "secrets.json" in out


# ---------------------------------------------------------------------------
# CLI handlers — credential resolution + arg forwarding
# ---------------------------------------------------------------------------


class TestVerifyCliHandlers:
    """``_cmd_verify_*`` must surface a clear error when secrets are absent
    and forward parsed args correctly when they are present.
    """

    def test_missing_credentials_raises_systemexit_with_tabulated_help(
        self, monkeypatch,
    ):
        """The error must list every source we tried + a concrete "to
        fix" pointer.  This is the contract that lets a fresh user
        see exactly which knob to turn instead of paging through
        docs to figure out what env var to set.
        """
        from switchyard.cli.switchyard_cli import (
            _build_parser,
            _cmd_verify,
        )
        _isolate_credentials(monkeypatch)

        parser = _build_parser()
        args = parser.parse_args(["verify"])

        with pytest.raises(SystemExit) as exc_info:
            _cmd_verify(args)

        msg = str(exc_info.value)
        assert "no API key resolved" in msg
        # Every source must be named in the table — silently skipping
        # any of them would leave the user guessing.
        for source in (
            "--api-key", "$OPENROUTER_API_KEY", "$NVIDIA_API_KEY",
            "$OPENAI_API_KEY", "$ANTHROPIC_API_KEY", "secrets.json",
        ):
            assert source in msg, f"missing source {source!r} in error"
        assert "To fix" in msg

    def test_happy_path_forwards_args_to_verify_proxy(self, monkeypatch):
        """The CLI must forward all six verify-proxy args (model /
        api_key / base_url / port / timeout / key_source) so a
        regression silently dropping any of them surfaces here.
        """
        from switchyard.cli.switchyard_cli import (
            _build_parser,
            _cmd_verify,
        )
        _isolate_credentials(monkeypatch)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-resolved")
        monkeypatch.setenv("OPENROUTER_BASE_URL", "https://upstream/v1")

        captured: dict = {}
        def fake_verify_proxy(**kwargs):
            captured.update(kwargs)
            return 0
        monkeypatch.setattr(
            "switchyard.server.verify.verify_proxy",
            fake_verify_proxy,
        )

        parser = _build_parser()
        args = parser.parse_args([
            "verify",
            "--model", "custom-model",
            "--port", "4444",
            "--timeout", "30",
        ])

        with pytest.raises(SystemExit) as exc_info:
            _cmd_verify(args)
        assert exc_info.value.code == 0

        assert captured["model"] == "custom-model"
        assert captured["api_key"] == "sk-resolved"
        assert captured["base_url"] == "https://upstream/v1"
        assert captured["port"] == 4444
        assert captured["timeout"] == 30.0
        # Source label must be forwarded — this is what powers the
        # 401 hint in step 2.  Pin the exact label so a future
        # rename of the source string is caught here, not in
        # production.
        assert captured["key_source"] == "$OPENROUTER_API_KEY"
