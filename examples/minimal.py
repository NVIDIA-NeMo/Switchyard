#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Minimal Switchyard Example

This example demonstrates how to use Switchyard as a Python library
to route LLM requests through a backend.

Usage:
    export OPENROUTER_API_KEY="sk-or-..."
    python examples/minimal.py
"""

import asyncio
import sys
from pathlib import Path

# Add package to path for development (not needed when installed via pip)
sys.path.insert(0, str(Path(__file__).parent.parent))

from switchyard import ChatRequest
from switchyard.cli.route_bundle import load_route_bundle_table


async def main() -> None:
    """Run a minimal Switchyard example."""

    routes = load_route_bundle_table(Path(__file__).with_name("route.yaml"))
    switchyard = routes.lookup_switchyard("fast-kimi")

    print("=" * 60)
    print("Switchyard Minimal Example")
    print("=" * 60)

    # Create a chat request
    request = ChatRequest.openai_chat({
            "model": "fast-kimi",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ],
        "max_tokens": 100,
    })

    print(f"Sending request to {request.body['model']}...")

    # Call the LLM through the switchyard
    response = await switchyard.call(request)

    print("\nResponse:")
    print(f"  Content: {response['choices'][0]['message']['content']}")
    print(f"  Tokens: {response['usage']['total_tokens']}")

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
