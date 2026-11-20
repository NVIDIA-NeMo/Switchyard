# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Candidate-bound Switchyard routing plugins for LiteLLM."""

from .classifier_routing_plugin import ClassifierRoutingPlugin
from .judge import JudgeClient, LiteLLMJudge, is_judge_call, serving_judge_call
from .random_routing_plugin import RandomRoutingPlugin
from .stage_routing_plugin import StageRoutingPlugin
from .switchyard_routing_plugin import SwitchyardRoutingPlugin

__all__ = [
    "ClassifierRoutingPlugin",
    "JudgeClient",
    "LiteLLMJudge",
    "RandomRoutingPlugin",
    "StageRoutingPlugin",
    "SwitchyardRoutingPlugin",
    "is_judge_call",
    "serving_judge_call",
]
