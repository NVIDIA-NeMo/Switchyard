#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimum-viable ReAct agent, with traffic routed by the agentapi `@route` decorator.

A small Thought/Action/Observation loop over a single tool (a calculator). Every
model call goes through one function, `call_model`, which is wrapped with
`@route(...)` — so routing across models is a one-line drop-in. The decorator
rewrites the target model per the chosen algorithm before each call, and
`call_model` prints the model it was actually routed to.

Configuration (env):
    OPENAI_API_KEY            required — key for the OpenAI-compatible endpoint
    OPENAI_BASE_URL           optional — endpoint base url (default OpenAI)
    SWITCHYARD_WEAK_MODEL     optional — cheap/default model  (default gpt-4o-mini)
    SWITCHYARD_STRONG_MODEL   optional — frontier model       (default gpt-4o)
    SWITCHYARD_DEMO_ROUTER    optional — "rand" (default) or "llm_class"

Usage:
    export OPENAI_API_KEY="sk-..."
    python examples/react_agent.py "What is 17 * 24, then add 100?"
"""

from __future__ import annotations

import ast
import asyncio
import operator
import os
import re
import sys
from pathlib import Path
from typing import Any

# Add package to path for development (not needed when installed via pip).
sys.path.insert(0, str(Path(__file__).parent.parent))

import litellm

from switchyard.lib.agentapi import (
    AgentApiOptAlgorithm,
    LlmClassifier,
    RandomRouter,
    WeightedModel,
    route,
)

API_KEY = os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
WEAK_MODEL = os.environ.get("SWITCHYARD_WEAK_MODEL", "gpt-4o-mini")
STRONG_MODEL = os.environ.get("SWITCHYARD_STRONG_MODEL", "gpt-4o")

SYSTEM_PROMPT = """\
You are a reasoning agent. Solve the task using EXACTLY this format:

Thought: <your reasoning>
Action: calculator
Action Input: <a Python arithmetic expression, e.g. 12 * (3 + 4)>

You will then receive an "Observation:" line with the tool result. Repeat
Thought/Action/Action Input as many times as needed. When you have the answer,
reply with:

Thought: <reasoning>
Final Answer: <the answer>

The only tool is `calculator`, which evaluates one arithmetic expression."""


def build_router() -> AgentApiOptAlgorithm:
    """Pick the routing algorithm from env; defaults to a weighted random split."""
    kind = os.environ.get("SWITCHYARD_DEMO_ROUTER", "rand")
    if kind == "llm_class":
        # Multi-round: an LLM classifier scores each request, then routes to the
        # strong or weak model. Each ReAct step therefore makes two model calls.
        return LlmClassifier(
            classifier_model=WEAK_MODEL,
            strong_model=STRONG_MODEL,
            weak_model=WEAK_MODEL,
            threshold=0.5,
        )
    # Single-round: send ~75% of traffic to the weak model, ~25% to the strong one.
    return RandomRouter([WeightedModel(WEAK_MODEL, 3.0), WeightedModel(STRONG_MODEL, 1.0)])


ROUTER = build_router()


@route(ROUTER)
async def call_model(model: str, messages: list[dict]) -> Any:
    """Make one chat-completion call. `@route` rewrites `model` before we run."""
    print(f"    -> routed to {model}")
    return await litellm.acompletion(
        model=model,
        messages=messages,
        api_key=API_KEY,
        api_base=BASE_URL,
        temperature=0.0,
    )


# Safe arithmetic evaluator: walk a parsed AST rather than calling eval().
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def calculator(expr: str) -> str:
    """Evaluate a single arithmetic expression, e.g. "17 * 24"."""
    return str(_eval(ast.parse(expr, mode="eval").body))


def run_tool(name: str, arg: str) -> str:
    """Dispatch a tool call, returning an Observation string for the model."""
    if name == "calculator":
        try:
            return calculator(arg)
        except (ValueError, KeyError, SyntaxError, ZeroDivisionError, TypeError) as exc:
            return f"calculator error: {exc}"
    return f"unknown tool: {name}"


_FINAL_RE = re.compile(r"Final Answer:\s*(.*)", re.DOTALL)
_ACTION_RE = re.compile(r"Action:\s*(.+)")
_INPUT_RE = re.compile(r"Action Input:\s*(.+)")


async def react(question: str, max_steps: int = 6) -> str:
    """Run the ReAct loop until a Final Answer or the step budget is exhausted."""
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    for step in range(1, max_steps + 1):
        print(f"[step {step}]")
        response = await call_model(model="auto", messages=messages)
        text = response.choices[0].message.content
        print("\n".join(f"    {line}" for line in text.strip().splitlines()))

        final = _FINAL_RE.search(text)
        if final:
            return final.group(1).strip()

        messages.append({"role": "assistant", "content": text})
        action = _ACTION_RE.search(text)
        action_input = _INPUT_RE.search(text)
        if action and action_input:
            observation = run_tool(action.group(1).strip(), action_input.group(1).strip())
        else:
            observation = "no valid Action found; follow the Thought/Action/Action Input format"
        print(f"    Observation: {observation}")
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    return "(max steps reached without a final answer)"


async def main() -> None:
    if not API_KEY:
        print("Set OPENAI_API_KEY (and optionally OPENAI_BASE_URL) to run this example.")
        return
    question = " ".join(sys.argv[1:]) or "What is 17 * 24, then add 100?"
    print(f"Router:   {type(ROUTER).__name__}")
    print(f"Question: {question}\n")
    answer = await react(question)
    print(f"\nFinal Answer: {answer}")


if __name__ == "__main__":
    asyncio.run(main())
