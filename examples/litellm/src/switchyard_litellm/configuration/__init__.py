# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load deployer-owned Switchyard routing plugin configuration."""

from .loader import (
    CONFIG_ENV,
    RoutingPlugin,
    load_routing_plugin,
    load_routing_plugin_from_environment,
)

__all__ = [
    "CONFIG_ENV",
    "RoutingPlugin",
    "load_routing_plugin",
    "load_routing_plugin_from_environment",
]
