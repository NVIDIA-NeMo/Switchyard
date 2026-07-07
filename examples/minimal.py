#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Minimal Switchyard Example

This example demonstrates how to use Switchyard as a Python library
to route LLM requests through a backend.

Configuration (in priority order):
    1. Environment variables: OPENAI_API_KEY, OPENAI_BASE_URL
    2. Secrets file: secrets/secrets.json

Usage:
    export OPENAI_API_KEY="sk-..."
    python examples/minimal.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add package to path for development (not needed when installed via pip)
sys.path.insert(0, str(Path(__file__).parent.parent))

from switchyard import ChatRequest, SwitchyardRecipes


async def main():
    """Run a minimal Switchyard example."""

    # Get API key from environment or secrets file
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        print("Set it with: export OPENAI_API_KEY='sk-...'")
        return

    # Create a single-model route that forwards to OpenAI
    switchyard = SwitchyardRecipes.passthrough_recipe(
        api_key=api_key,
        base_url="https://api.openai.com/v1",
    )

    print("=" * 60)
    print("Switchyard Minimal Example")
    print("=" * 60)

    # Create a chat request
    request = ChatRequest.openai_chat({
        "model": "gpt-4o-mini",
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
    print(f"  Content: {response.body['choices'][0]['message']['content']}")
    print(f"  Tokens: {response.body['usage']['total_tokens']}")

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
