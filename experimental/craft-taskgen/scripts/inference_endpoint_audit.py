#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit a local or gateway-hosted OpenAI-compat endpoint.

Tells you, per request-body parameter, what the endpoint actually
honors versus silently ignores. Runs against any /v1/chat/completions
endpoint: vLLM (Qwen3 / MiniMax / Nemotron), NVIDIA gateway (Claude
routes), direct Anthropic/OpenAI, etc.

Default mode is a `reasoning_effort` sweep across
{none, low, medium, high, xhigh} — two HTTP calls per level — followed
by a summary table. Non-monotonic reasoning output across levels is
your cue that the endpoint is ignoring the parameter.

Per-level probe (two HTTP calls, pure-python urllib):

  1. non-stream: emits probes A (message fields) and B (usage fields)
     from the SAME response. Probe A checks for `message.reasoning`
     / `message.reasoning_content` — the Anthropic gateway uses
     `reasoning_content`; vLLM with `--reasoning-parser qwen3` /
     `minimax_m2` / `nemotron_v3` uses `reasoning` or
     `reasoning_content`. Probe B checks for
     `usage.completion_tokens_details.reasoning_tokens` — needed if
     opencode's `tokens.reasoning` metric is to be non-zero. vLLM
     typically omits this; the NVIDIA gateway emits it for Claude
     routes. This gap is structural and unfixable client-side.

  2. stream: probe C — do SSE frames carry `delta.reasoning` or
     `delta.reasoning_content`? `@ai-sdk/openai-compatible@2.0.41`
     reads both at `dist/index.mjs:712` and emits
     `reasoning-start / reasoning-delta / reasoning-end` parts
     downstream. If stream carries it but opencode's trajectory shows
     0 reasoning events, the drop is on opencode's side
     (sst/opencode#16963, #19988).

Also supports `--sampling-defaults qwen3|minimax|nemotron` to apply
the family's paper-recommended sampling params and verify the endpoint
accepts them.

Usage:
  # Default: sweep all effort levels + summary table
  scripts/inference_endpoint_audit.py \\
      --base-url http://localhost:9000/v1 \\
      --model model \\
      --api-key EMPTY

  # Single effort level (skip sweep)
  scripts/inference_endpoint_audit.py \\
      --base-url "$OPENAI_BASE_URL" \\
      --model aws/anthropic/bedrock-claude-opus-4-6 \\
      --reasoning-effort high

  # Apply family sampling defaults while sweeping effort
  scripts/inference_endpoint_audit.py \\
      --base-url http://localhost:9000/v1 \\
      --model model \\
      --sampling-defaults minimax

Exit code 0 if probes ran (regardless of results). Non-zero only on
connection errors or bad arguments.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


def _load_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _resolve_auth(args: argparse.Namespace) -> tuple[str, str]:
    base = args.base_url or os.environ.get("VLLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    key = args.api_key or os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not (base and key):
        env_file = _load_env_file(Path(".env"))
        base = base or env_file.get("OPENAI_BASE_URL") or env_file.get("OPENAI_API_BASE")
        key = key or env_file.get("OPENAI_API_KEY") or env_file.get("VLLM_API_KEY") or "EMPTY"
    if not base:
        print("ERROR: no base URL (pass --base-url or set VLLM_BASE_URL / OPENAI_BASE_URL)", file=sys.stderr)
        sys.exit(2)
    return base.rstrip("/"), key


def _post(base: str, key: str, body: dict) -> bytes:
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "content-type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        return e.read()


def _body(args: argparse.Namespace, **overrides) -> dict:
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
    }
    if args.reasoning_effort:
        body["reasoning_effort"] = args.reasoning_effort
    if args.temperature is not None:
        body["temperature"] = args.temperature
    if args.top_p is not None:
        body["top_p"] = args.top_p
    if args.top_k is not None:
        # top_k is standard on vLLM OpenAI-compat but not part of the OpenAI
        # spec. Strict gateways may 400 on this; vLLM accepts it at top level.
        body["top_k"] = args.top_k
    body.update(overrides)
    return body


def probe_non_stream(args, base, key) -> tuple[dict, dict]:
    """Single non-stream request; extract both message fields and usage."""
    raw = _post(base, key, _body(args))
    try:
        d = json.loads(raw)
    except Exception:
        err = {"error": "non-json", "raw_preview": raw[:400].decode("utf-8", "replace")}
        return (
            {"probe": "A_non_stream", **err},
            {"probe": "B_usage", "error": "non-json"},
        )
    if "error" in d:
        err = {"error": d["error"]}
        return (
            {"probe": "A_non_stream", **err},
            {"probe": "B_usage", **err},
        )

    msg = d.get("choices", [{}])[0].get("message", {})
    message_result = {
        "probe": "A_non_stream",
        "message_keys": sorted(msg.keys()),
        "has_reasoning": "reasoning" in msg,
        "has_reasoning_content": "reasoning_content" in msg,
        "reasoning_len": len(msg.get("reasoning") or ""),
        "reasoning_content_len": len(msg.get("reasoning_content") or ""),
        "content_len": len(msg.get("content") or ""),
        "reasoning_preview": (msg.get("reasoning") or msg.get("reasoning_content") or "")[:200],
    }

    usage = d.get("usage") or {}
    details = usage.get("completion_tokens_details")
    usage_result = {
        "probe": "B_usage",
        "usage": usage,
        "has_completion_tokens_details": isinstance(details, dict),
        "reasoning_tokens": details.get("reasoning_tokens") if isinstance(details, dict) else None,
    }
    return message_result, usage_result


def probe_c_stream(args, base, key) -> dict:
    raw = _post(
        base,
        key,
        _body(args, stream=True, stream_options={"include_usage": True}),
    )
    text = raw.decode("utf-8", errors="replace")
    delta_reasoning = 0
    delta_reasoning_content = 0
    delta_thinking = 0
    delta_content = 0
    reasoning_chars = 0
    other_delta_keys: dict[str, int] = {}
    samples: list[dict] = []
    final_usage = None
    for line in text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        try:
            chunk = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        if "usage" in chunk and chunk["usage"]:
            final_usage = chunk["usage"]
        for ch in chunk.get("choices") or []:
            delta = ch.get("delta") or {}
            for k in delta:
                if k not in ("role", "content", "reasoning", "reasoning_content", "thinking"):
                    other_delta_keys[k] = other_delta_keys.get(k, 0) + 1
            if "reasoning" in delta:
                delta_reasoning += 1
                val = delta.get("reasoning")
                if isinstance(val, str):
                    reasoning_chars += len(val)
                if len(samples) < 2:
                    samples.append({"delta_subset": {"reasoning": val}})
            if "reasoning_content" in delta:
                delta_reasoning_content += 1
                val = delta.get("reasoning_content")
                if isinstance(val, str):
                    reasoning_chars += len(val)
                if len(samples) < 2:
                    samples.append({"delta_subset": {"reasoning_content": val}})
            if "thinking" in delta:
                delta_thinking += 1
                if len(samples) < 2:
                    samples.append({"delta_subset": {"thinking": delta.get("thinking")}})
            if delta.get("content"):
                delta_content += 1
    return {
        "probe": "C_stream",
        "delta_reasoning_frames": delta_reasoning,
        "delta_reasoning_content_frames": delta_reasoning_content,
        "delta_thinking_frames": delta_thinking,
        "delta_content_frames": delta_content,
        "reasoning_chars": reasoning_chars,
        "other_delta_keys": other_delta_keys,
        "final_usage": final_usage,
        "sample_frames": samples,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit an OpenAI-compat endpoint for reasoning + sampling plumbing (vLLM / gateway).",
    )
    parser.add_argument(
        "--base-url", help="e.g. http://localhost:9000/v1 or the gateway $OPENAI_BASE_URL"
    )
    parser.add_argument("--model", required=True, help="model id to probe")
    parser.add_argument("--api-key", help="defaults to VLLM_API_KEY / OPENAI_API_KEY / .env")
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh"],
        default=None,
        help="run ONE probe at this reasoning_effort level (skip the sweep). "
        "'none' sends no reasoning_effort field. Default: sweep all levels. "
        "Matches EFFORT_LEVELS in craft_taskgen.baselines (plus 'none' for the "
        "unset case specific to probing).",
    )
    # Default prompt must be hard enough that a capable model genuinely needs
    # to reason, not just emit a memorized fact. Simple arithmetic is NOT
    # adequate — GPT-4-tier models answer `127*83` instantly with zero
    # reasoning steps, which makes the probe's reasoning-frame count
    # artificially low and can disguise a broken parser as a working one.
    # This prompt (AIME-style, 2024 II #5) has enough branching that even
    # frontier models produce multi-hundred-token reasoning traces.
    parser.add_argument(
        "--prompt",
        default=(
            "Let x, y, z be positive real numbers satisfying "
            "log_2(x / (yz)) = 1/2, log_2(y / (xz)) = 1/3, and "
            "log_2(z / (xy)) = 1/4. "
            "Find the value of |log_2(x^4 y^3 z^2)|. "
            "Show your reasoning step by step and give a final numeric answer."
        ),
        help="user prompt for the probe (default: AIME-style log-system problem)",
    )
    # 8k default — hard reasoning problems (AIME-style) can easily produce
    # 3-5k reasoning tokens before a conclusion. 2k gets truncated, making
    # it impossible to tell if the model finished or was cut off.
    parser.add_argument("--max-tokens", type=int, default=8000)
    # Sampling parameters. None → don't send the field (server default).
    # For Qwen3 and MiniMax, these are the real knobs the model honors
    # (reasoning_effort is a no-op on those). Useful to set when smoke-
    # testing whether the launcher's auto-exported defaults actually
    # reach the endpoint.
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--sampling-defaults",
        choices=["qwen3", "minimax", "nemotron"],
        help="shorthand for the launcher's family defaults. qwen3: T=0.6 p=0.95 k=20. "
        "minimax: T=1.0 p=0.95 k=40. nemotron: T=1.0 p=0.95 (no top_k). "
        "Individual --temperature/--top-p/--top-k flags win over these.",
    )
    args = parser.parse_args(argv)

    # Apply --sampling-defaults only for fields not explicitly overridden.
    _FAMILY_DEFAULTS: dict[str, dict[str, float | int | None]] = {
        "qwen3": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
        "minimax": {"temperature": 1.0, "top_p": 0.95, "top_k": 40},
        # Nemotron-3-Super card: T=1.0 p=0.95 "across all tasks". No top_k.
        "nemotron": {"temperature": 1.0, "top_p": 0.95, "top_k": None},
    }
    if args.sampling_defaults:
        defaults = _FAMILY_DEFAULTS[args.sampling_defaults]
        if args.temperature is None and defaults["temperature"] is not None:
            args.temperature = defaults["temperature"]
        if args.top_p is None and defaults["top_p"] is not None:
            args.top_p = defaults["top_p"]
        if args.top_k is None and defaults["top_k"] is not None:
            args.top_k = defaults["top_k"]

    base, key = _resolve_auth(args)

    sampling_str = (
        f"T={args.temperature if args.temperature is not None else '<default>'}"
        f" p={args.top_p if args.top_p is not None else '<default>'}"
        f" k={args.top_k if args.top_k is not None else '<default>'}"
    )
    print(f"# Model: {args.model}")
    print(f"# Sampling: {sampling_str}")
    print(f"# Prompt: {args.prompt[:100]}{'...' if len(args.prompt) > 100 else ''}")
    print()

    summary: list[dict] = []

    def run_once(effort: str | None) -> None:
        args.reasoning_effort = effort
        label = effort or "<unset>"
        print(f"# Probing {base} effort={label}")
        print()
        ns, us = probe_non_stream(args, base, key)
        st = probe_c_stream(args, base, key)
        for result in (ns, us, st):
            result["effort"] = label
            print(json.dumps(result, indent=2))
            print()
        summary.append(
            {
                "effort": label,
                "non_stream_reasoning_chars": (
                    ns.get("reasoning_len", 0) + ns.get("reasoning_content_len", 0)
                ),
                "usage_reasoning_tokens": us.get("reasoning_tokens"),
                "stream_reasoning_frames": (
                    st.get("delta_reasoning_frames", 0) + st.get("delta_reasoning_content_frames", 0)
                ),
                "stream_reasoning_chars": st.get("reasoning_chars", 0),
            }
        )

    if args.reasoning_effort is not None:
        # Single-effort mode: one run, no summary table.
        run_once(None if args.reasoning_effort == "none" else args.reasoning_effort)
    else:
        # Default: sweep all effort levels. Lets you see at a glance
        # whether the endpoint actually honors reasoning_effort.
        for effort in (None, "low", "medium", "high", "xhigh"):
            run_once(effort)
        print("# Summary (reasoning activity vs effort level)")
        print(f"# model={args.model}")
        header = (
            f"# {'effort':<10} {'ns_chars':>10} {'usage_rtok':>12} {'stream_frames':>15} {'stream_chars':>14}"
        )
        print(header)
        for row in summary:
            print(
                f"# {row['effort']:<10} "
                f"{row['non_stream_reasoning_chars']:>10} "
                f"{str(row['usage_reasoning_tokens']):>12} "
                f"{row['stream_reasoning_frames']:>15} "
                f"{row['stream_reasoning_chars']:>14}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
