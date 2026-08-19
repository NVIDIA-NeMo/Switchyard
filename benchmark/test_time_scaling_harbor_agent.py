# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor mini-SWE-agent adapter that saves the final repository patch."""

from harbor.agents.installed.mini_swe_agent import MiniSweAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

PATCH_COMMAND = (
    "mkdir -p /logs/artifacts && "
    "git -C /testbed add -A && "
    "git -C /testbed diff --cached --binary --no-ext-diff "
    "> /logs/artifacts/patch.diff"
)


class TestTimeScalingMiniSweAgent(MiniSweAgent):  # type: ignore[misc]
    """Run mini-SWE-agent and save the final repository patch."""

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        try:
            await super().run(instruction, environment, context)
        finally:
            await self.exec_as_agent(
                environment,
                command=PATCH_COMMAND,
            )
