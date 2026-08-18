# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor agent that applies one saved patch without calling a model."""

import base64
import hashlib
import logging
import shlex
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import MCPServerConfig


class TestTimeScalingPatchAgent(BaseAgent):  # type: ignore[misc]
    """Apply a saved patch so Harbor can run the official verifier later."""

    def __init__(
        self,
        logs_dir: Path,
        patch_file: str,
        model_name: str | None = None,
        logger: logging.Logger | None = None,
        mcp_servers: list[MCPServerConfig] | None = None,
        skills_dir: str | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            logs_dir=logs_dir,
            model_name=model_name,
            logger=logger,
            mcp_servers=mcp_servers,
            skills_dir=skills_dir,
        )
        self.patch_file = Path(patch_file)
        del extra_env

    @staticmethod
    def name() -> str:
        """Return the name stored in the Harbor result."""
        return "test-time-scaling-patch"

    def version(self) -> str:
        """Return this small agent's record format version."""
        return "1"

    async def setup(self, environment: BaseEnvironment) -> None:
        """Use the task image without installing an agent."""

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Apply the patch in the clean task checkout."""
        del instruction
        patch = self.patch_file.read_bytes()
        context.metadata = {"patch_sha256": hashlib.sha256(patch).hexdigest()}
        if not patch:
            return
        encoded = base64.b64encode(patch).decode()
        result = await environment.exec(
            command=(f"printf %s {shlex.quote(encoded)} | base64 --decode | git apply --binary -"),
            cwd="/testbed",
        )
        if result.return_code != 0:
            raise RuntimeError(f"could not apply saved patch: {result.stderr}")
