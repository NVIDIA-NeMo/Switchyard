# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expose the environment-configured plugin object imported by LiteLLM."""

from .loader import load_routing_plugin_from_environment

ROUTING_PLUGIN = load_routing_plugin_from_environment()

__all__ = ["ROUTING_PLUGIN"]
