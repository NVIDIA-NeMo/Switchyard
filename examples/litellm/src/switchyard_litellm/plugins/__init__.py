# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Candidate-bound Switchyard routing plugins for LiteLLM."""

from .random_routing_plugin import RandomRoutingPlugin
from .stage_routing_plugin import StageRoutingPlugin
from .switchyard_routing_plugin import SwitchyardRoutingPlugin

__all__ = ["RandomRoutingPlugin", "StageRoutingPlugin", "SwitchyardRoutingPlugin"]
