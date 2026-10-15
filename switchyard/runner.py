# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run TOML-configured routes using the gateway's Python model clients."""

from typing import TYPE_CHECKING

from switchyard_rust.runner import Runner, UpstreamError
from switchyard_rust.wire_format import __getattr__ as __getattr__

if TYPE_CHECKING:
    from switchyard_rust.wire_format import WireFormat as WireFormat

__all__ = ["Runner", "UpstreamError", "WireFormat"]
