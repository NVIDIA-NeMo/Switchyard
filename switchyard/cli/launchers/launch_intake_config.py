# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolved intake configuration for the ``launch claude`` / ``launch codex`` flows.

Both launchers compose two pieces when ``--intake-enabled`` is set:

* server-side — the in-process proxy gets an :class:`IntakeRequestProcessor`
  + :class:`IntakeResponseProcessor` attached to its chain so each completed
  turn is shipped to NMP intake.
* client-side — the spawned ``claude`` / ``codex`` process is given the
  per-request opt-in headers so every request the user issues actually
  triggers the sink (the intake processors short-circuit when the inbound
  request lacks the opt-in).

This module owns the resolved-config dataclass that flows from CLI args /
env vars through both seams. CLI parsing lives in
``switchyard.cli.switchyard_cli`` and the resolver in
``switchyard.cli.launch_command``.
"""

import re
import secrets
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from switchyard.lib.config import IntakeSinkConfig


_INTAKE_WARNING_LINES = (
    "",
    "  WARNING: Intake logging is enabled — your requests and responses will be",
    "           logged to a database. Ensure you are following all inference",
    "           rules and compliance requirements for LLM use.",
    "",
)


def print_intake_warning(stream: IO[str] | None = None) -> None:
    """Emit the compliance warning that runs whenever intake logging is on."""
    target = stream if stream is not None else sys.stderr
    target.write("\n".join(_INTAKE_WARNING_LINES) + "\n")
    target.flush()


@dataclass(frozen=True)
class LaunchIntakeConfig:
    """Everything a launcher needs to wire up the intake sink end-to-end.

    The auth-side fields (``base_url``, ``workspace``, ``api_key``) are
    optional because the path of least resistance is the docs' recommended
    one: leave them all unset, run
    ``uv run nmp auth login --base-url {INTAKE_BASE_URL}`` once, and the SDK
    reads everything (including a refresh-token-backed access token) from
    ``~/.config/nmp/config.yaml``. Set any of them to override that
    bootstrap (e.g. for CI/CD).

    ``app`` / ``task`` / ``session_id`` are always concrete because they
    drive the per-request opt-in headers regardless of which auth path the
    SDK takes.
    """

    base_url: str | None
    workspace: str | None
    api_key: str | None
    app: str
    task: str
    session_id: str
    user_id: str
    nvdataflow_project: str | None = None

    @classmethod
    def from_resolved(
        cls,
        *,
        base_url: str | None,
        workspace: str | None,
        api_key: str | None,
        app: str,
        task: str,
        session_id: str | None,
        user_id: str | None = None,
        nvdataflow_project: str | None = None,
        target: str,
    ) -> "LaunchIntakeConfig":
        """Resolve any blank fields and return the immutable config.

        ``session_id`` defaults to ``<target>-<unix-ms>-<uuid4-short>`` so each
        launch is uniquely identifiable in the intake store. Pass an explicit
        value to override (e.g. for re-runs that should join an existing
        thread).

        ``user_id`` defaults to the stable anonymous per-machine id so intake
        entries can be grouped by user without carrying any personal data.
        """
        return cls(
            base_url=base_url,
            workspace=workspace,
            api_key=api_key,
            app=app,
            task=task,
            session_id=session_id or _default_session_id(target),
            user_id=user_id or resolve_machine_user_id(),
            nvdataflow_project=nvdataflow_project,
        )

    def opt_in_headers(self) -> dict[str, str]:
        """Headers the spawned client must send on every request to opt in.

        ``IntakeRequestProcessor`` reads these via
        :meth:`RequestMetadata.from_headers` and stashes them on
        :class:`ProxyContext`; without them the sink short-circuits and no
        entry is shipped.
        """
        return {
            "x-switchyard-intake-enabled": "true",
            "x-switchyard-intake-app": self.app,
            "x-switchyard-intake-task": self.task,
            "proxy_x_session_id": self.session_id,
        }

    def to_sink_config(self) -> "IntakeSinkConfig":
        """Build the server-side intake sink config for this launch."""
        from switchyard.lib.config import IntakeSinkConfig

        return IntakeSinkConfig(
            intake_base_url=self.base_url,
            workspace=self.workspace,
            api_key=self.api_key,
            user_id=self.user_id,
            nvdataflow_project=self.nvdataflow_project,
        )


def build_intake_processors(
    intake: LaunchIntakeConfig | None,
) -> tuple[list[Any], list[Any]]:
    """Return intake processor lists for launchers, or empty lists when disabled."""
    if intake is None:
        return [], []

    from switchyard.lib.processors.intake_request_processor import IntakeRequestProcessor
    from switchyard.lib.processors.intake_response_processor import IntakeResponseProcessor

    config = intake.to_sink_config()
    return (
        [IntakeRequestProcessor()],
        [IntakeResponseProcessor(config)],
    )


def build_launch_capture_processors(
    intake: LaunchIntakeConfig | None,
    rl_log_dir: Path | None,
) -> tuple[list[Any], list[Any]]:
    """Combine intake + RL-logging request/response processor lists for launchers.

    The intake sink (``intake``) and the local RL trace logger (``rl_log_dir``)
    are independent: either, both, or neither may be active. Returns
    ``([], [])`` when neither is.
    """
    from switchyard.lib.processors.rl_logging_response_processor import (
        build_rl_logging_processors,
    )

    intake_request, intake_response = build_intake_processors(intake)
    rl_request, rl_response = build_rl_logging_processors(rl_log_dir)
    return [*intake_request, *rl_request], [*intake_response, *rl_response]


def _default_session_id(target: str) -> str:
    return f"{target}-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"


def _machine_user_id_path() -> Path:
    return Path.home() / ".switchyard" / "user_id"


_USER_ID_RE = re.compile(r"^[0-9a-f]{8}$")


def resolve_machine_user_id(path: Path | None = None) -> str:
    """Return a stable anonymous per-machine id, creating one on first use.

    The id is a random 8-char hex string, never derived from the username, so
    it is non-reversible. It is persisted at ``~/.switchyard/user_id`` and
    reused on later launches. Fail-open: if the file is missing, unreadable, or
    malformed, a fresh id is returned so intake never blocks a launch.
    """
    user_id = secrets.token_hex(4)
    try:
        target = path or _machine_user_id_path()
    except RuntimeError:
        # No resolvable home directory; use a fresh transient id.
        return user_id
    try:
        existing = target.read_text(encoding="utf-8").strip()
        if _USER_ID_RE.fullmatch(existing):
            return existing
    except (OSError, UnicodeError):
        pass
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(user_id, encoding="utf-8")
    except OSError:
        pass
    return user_id
