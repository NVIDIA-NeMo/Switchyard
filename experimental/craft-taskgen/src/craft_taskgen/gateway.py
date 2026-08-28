# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gateway env-builder shared by `claude_cli.py` (subprocess `claude -p`) and
`llm_judge.py` (direct litellm calls).

Gateway-only policy: every pipeline LLM call — whether a `claude -p` session or
a direct prompt/response — must route through the NVIDIA LiteLLM gateway. The
enforcement point is `build_gateway_env()` (and its read-only credential
accessors below); they raise `RuntimeError` rather than silently falling back
to OAuth or to api.anthropic.com.

This module exposes three surfaces:

- `build_gateway_env(model)` — full subprocess env for `claude -p`. Pins every
  sub-agent alias to the gateway model, disables non-essential traffic, and
  removes any `CLAUDE_CODE_OAUTH_TOKEN`. Used by `claude_cli.run_claude*`.
- `get_anthropic_gateway_creds()` — `(api_key, base_url)` for the Anthropic
  gateway route. Used internally by `build_gateway_env()` for the `claude -p`
  subprocess path.
- `get_openai_gateway_creds()` — `(api_key, base_url)` for the OpenAI-compatible
  gateway route. Used by `llm_judge` for *every* direct-API dispatch: the
  wrapper normalizes every model (including Anthropic-family Opus) to an
  `openai/`-prefixed litellm route so the gateway sees a uniform transport.
  Consequently `OPENAI_API_KEY` and `OPENAI_BASE_URL` must be set for any
  pipeline run, regardless of which model the step invokes.

Previously located in `claude_cli.py::_build_gateway_env`; factored out so the
direct-API wrapper can reuse the same guardrail.
"""

from __future__ import annotations

import os

# Only these host environment variables are forwarded to the `claude -p`
# subprocess. Copying the full os.environ would expose unrelated host secrets
# (DB passwords, cloud creds, tokens) to the agent and its tools
# (CWE-201 / CWE-522). Anything Claude/node genuinely needs at runtime is
# either listed here or set explicitly in build_gateway_env().
_ENV_PASSTHROUGH = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "TERM",
        "TZ",
        "TMPDIR",
        "TMP",
        "TEMP",
        "LANG",
        "LANGUAGE",
        "NODE_EXTRA_CA_CERTS",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    }
)
_ENV_PASSTHROUGH_PREFIXES = ("LC_", "XDG_")


def _filtered_host_env() -> dict[str, str]:
    """Return host env vars from a curated allowlist (see _ENV_PASSTHROUGH)."""
    return {
        k: v
        for k, v in os.environ.items()
        if k in _ENV_PASSTHROUGH or k.startswith(_ENV_PASSTHROUGH_PREFIXES)
    }


def get_anthropic_gateway_creds() -> tuple[str, str]:
    """Return `(api_key, base_url)` for the Anthropic-family gateway route.

    Raises `RuntimeError` if either `ANTHROPIC_API_KEY` or `ANTHROPIC_BASE_URL`
    is missing from the environment. OAuth fallback is forbidden.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "").strip()
    if not api_key or not base_url:
        raise RuntimeError(
            "Gateway-only policy: ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL "
            "must both be set (typically via .env). OAuth fallback is forbidden."
        )
    return api_key, base_url


def get_openai_gateway_creds() -> tuple[str, str]:
    """Return `(api_key, base_url)` for the OpenAI-compatible gateway route.

    Used by `llm_judge` for every direct-API dispatch — the wrapper normalizes
    all models (Anthropic-family included) to an `openai/` litellm route so
    the gateway sees a uniform transport. Raises `RuntimeError` if either
    `OPENAI_API_KEY` or `OPENAI_BASE_URL` is missing.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    if not api_key or not base_url:
        raise RuntimeError(
            "Gateway-only policy: OPENAI_API_KEY and OPENAI_BASE_URL must both "
            "be set (typically via .env) for OpenAI-route litellm calls."
        )
    return api_key, base_url


def build_gateway_env(model: str) -> dict[str, str]:
    """Construct the env for `claude -p` that forces NVIDIA gateway routing.

    Mirrors Harbor's recipe (harbor/agents/installed/claude_code.py): set
    ANTHROPIC_* creds + model, pin every sub-agent alias to the gateway model,
    suppress non-essential traffic that could leak to api.anthropic.com.

    Fails loudly if gateway credentials or model are missing — OAuth fallback
    is forbidden on this machine.
    """
    api_key, base_url = get_anthropic_gateway_creds()
    if not model:
        raise RuntimeError(
            "Gateway-only policy: `claude -p` must be called with an explicit "
            "gateway model name (set profile.llm_step_model)."
        )

    env = _filtered_host_env()
    env["ANTHROPIC_API_KEY"] = api_key
    env["ANTHROPIC_BASE_URL"] = base_url
    env["ANTHROPIC_MODEL"] = model

    # Pin every sub-agent alias to the gateway model so background tasks
    # can't slip to Anthropic direct (matches Harbor claude_code.py:935-939).
    env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = model
    env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = model
    env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = model
    env["CLAUDE_CODE_SUBAGENT_MODEL"] = model

    # Cut off telemetry + attribution + non-essential traffic that could
    # route to api.anthropic.com (matches Harbor's patch).
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    env["CLAUDE_CODE_ATTRIBUTION_HEADER"] = "0"
    env["CLAUDE_CODE_ENABLE_TELEMETRY"] = "0"

    # Remove any OAuth token — even if it's set, we want it unusable.
    env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
    return env
