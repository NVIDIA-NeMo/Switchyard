# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP endpoint modules (and their SSE helpers).

Note: HTTP endpoint classes require fastapi (install with [server] extra).
They are lazily loaded to avoid hard dependency on fastapi for library-only users.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from switchyard.lib.endpoints.anthropic_messages_endpoint import (
        AnthropicMessagesEndpoint,
    )
    from switchyard.lib.endpoints.models_endpoint import ModelsEndpoint
    from switchyard.lib.endpoints.openai_chat_endpoint import (
        OpenAIChatEndpoint,
    )
    from switchyard.lib.endpoints.responses_endpoint import ResponsesEndpoint

__all__ = [
    "AnthropicMessagesEndpoint",
    "ModelsEndpoint",
    "OpenAIChatEndpoint",
    "ResponsesEndpoint",
]


def __getattr__(name: str) -> Any:
    """Lazy load HTTP endpoint classes that require fastapi."""
    if name == "AnthropicMessagesEndpoint":
        from switchyard.lib.endpoints.anthropic_messages_endpoint import (
            AnthropicMessagesEndpoint,
        )
        return AnthropicMessagesEndpoint
    elif name == "OpenAIChatEndpoint":
        from switchyard.lib.endpoints.openai_chat_endpoint import (
            OpenAIChatEndpoint,
        )
        return OpenAIChatEndpoint
    elif name == "ModelsEndpoint":
        from switchyard.lib.endpoints.models_endpoint import ModelsEndpoint
        return ModelsEndpoint
    elif name == "ResponsesEndpoint":
        from switchyard.lib.endpoints.responses_endpoint import (
            ResponsesEndpoint,
        )
        return ResponsesEndpoint
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
