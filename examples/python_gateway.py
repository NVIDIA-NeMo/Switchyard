#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Use TOML routing and target request settings with gateway-owned model clients.

The host owns credentials, transport, and retries. Callbacks take prepared provider JSON
and return JSON or an async iterator of raw SSE bytes; this example uses buffered JSON.
Runner returns bodies or events without upstream headers or structured terminal HTTP
status. This example does not start an HTTP server.
"""

import asyncio
import json
from typing import Any

from switchyard.runner import Runner, UpstreamError, WireFormat

CONFIG = """
schema_version = 1
[llm_clients]
gateway = { format = 'openai_chat' }
[targets]
judge = { id = 'judge', llm_client = 'gateway' }
fast = { id = 'fast', llm_client = 'gateway', system_prompt = 'Be concise.' }
capable = { id = 'capable', llm_client = 'gateway', system_prompt = 'Check your answer.' }
[routes.coding]
id = 'switchyard/coding'
type = 'llm_classifier'
classifier_target = 'judge'
weak_target = 'fast'
strong_target = 'capable'
base_threshold = 0.5
"""


async def gateway_client(body: dict[str, Any]) -> dict[str, Any]:
    """Replace this fake with the gateway's own client and credentials."""
    model = body["model"]
    print("Gateway called:", model)
    if model == "fast":
        raise UpstreamError(503, {"error": {"message": "fast is unavailable"}})
    text = (
        json.dumps(
            {
                "crux": "bounded task",
                "primary_rule": "SUP-1",
                "capability_boundary": "supported",
                "p_solve": 0.9,
            }
        )
        if model == "judge"
        else "The gateway's capable model answered after fast failed."
    )
    return {
        "id": f"chatcmpl-{model}",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


async def main() -> None:
    """Load once, then accept provider-format requests on each gateway call."""
    runner = Runner.from_toml(source=CONFIG, clients={"gateway": gateway_client})
    response = await runner.execute(
        {
            "model": "switchyard/coding",
            "messages": [{"role": "user", "content": "Help fix this test."}],
        },
        format=WireFormat.OPENAI_CHAT,
        headers={"x-switchyard-session-id": "example-task"},
    )
    if not isinstance(response, dict):
        raise TypeError("Expected a buffered response")
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
