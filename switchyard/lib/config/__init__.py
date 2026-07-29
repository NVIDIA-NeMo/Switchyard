# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration models shared by profile and processor implementations."""

from switchyard.lib.config.intake_sink_config import (
    IntakeQueueFullPolicy,
    IntakeSinkConfig,
)

__all__ = [
    "IntakeQueueFullPolicy",
    "IntakeSinkConfig",
]
