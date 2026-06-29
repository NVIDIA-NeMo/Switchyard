# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concrete request-side and response-side processor implementations."""

from typing import TYPE_CHECKING, Any

from switchyard.lib.processors.model_rewrite_request_processor import (
    ModelRewriteRequestProcessor,
)

if TYPE_CHECKING:
    from switchyard.lib.processors.intake_client import IntakeClient
    from switchyard.lib.processors.intake_payload_builder import IntakePayloadBuilder
    from switchyard.lib.processors.intake_request_processor import IntakeRequestProcessor
    from switchyard.lib.processors.intake_response_processor import IntakeResponseProcessor

__all__ = [
    "IntakeClient",
    "IntakePayloadBuilder",
    "IntakeRequestProcessor",
    "IntakeResponseProcessor",
    "ModelRewriteRequestProcessor",
]


def __getattr__(name: str) -> Any:
    """Lazy load intake processors to avoid circular imports."""
    if name == "IntakeClient":
        from switchyard.lib.processors.intake_client import IntakeClient
        return IntakeClient
    elif name == "IntakePayloadBuilder":
        from switchyard.lib.processors.intake_payload_builder import (
            IntakePayloadBuilder,
        )
        return IntakePayloadBuilder
    elif name == "IntakeRequestProcessor":
        from switchyard.lib.processors.intake_request_processor import (
            IntakeRequestProcessor,
        )
        return IntakeRequestProcessor
    elif name == "IntakeResponseProcessor":
        from switchyard.lib.processors.intake_response_processor import (
            IntakeResponseProcessor,
        )
        return IntakeResponseProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
