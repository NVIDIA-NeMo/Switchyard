# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for project-local skill-distillation session capture."""

import json
from pathlib import Path

import pytest

from switchyard.cli.config.user_config import (
    SkillDistillationConfig,
    UserConfig,
    save_user_config,
)
from switchyard.cli.launchers.launch_intake_config import build_launch_capture_processors
from switchyard.cli.launchers.skill_distillation import (
    build_launch_skill_distillation_session,
    launch_skill_distillation_session,
)
from switchyard.lib.processors.skill_distillation_session_processor import (
    CTX_SKILL_DISTILLATION_REQUEST,
    SkillDistillationRequestProcessor,
    SkillDistillationResponseProcessor,
)
from switchyard.lib.skill_distillation_store import (
    SkillDistillationSessionCapture,
    resolve_skill_distillation_store_path,
    summarize_skill_distillation_store,
)
from switchyard.lib.stats_accumulator import StatsAccumulator
from switchyard_rust.core import ChatRequest, ChatResponse, ProxyContext


def _request() -> ChatRequest:
    return ChatRequest.openai_chat({
        "model": "client-model",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [],
    })


def _completion() -> ChatResponse:
    return ChatResponse.openai_completion({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "served-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    })


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_store_layout_and_skipped_interrupted_session(tmp_path: Path) -> None:
    session = SkillDistillationSessionCapture(
        namespace="tooluniverse-trialqa",
        launch_target="claude",
        display_model="switchyard-default",
        strategy_summary="passthrough: switchyard-default",
        project_dir=tmp_path,
    )

    root = resolve_skill_distillation_store_path("tooluniverse-trialqa", tmp_path)
    assert session.store_path == root
    for child in ("active", "candidates", "history", "reports", "sessions"):
        assert (root / child).is_dir()

    summary = summarize_skill_distillation_store("tooluniverse-trialqa", tmp_path)
    assert summary.path == root
    assert summary.session_count == 1
    assert summary.active_skill_exists is False

    session.finish(exit_code=130, stats=StatsAccumulator())

    metadata = _read_json(session.session_path)
    assert metadata["status"] == "interrupted"
    assert metadata["exit_code"] == 130
    assert metadata["turn_count"] == 0
    assert metadata["distillation"]["status"] == "skipped"
    assert _read_json(session.stats_path)["total_requests"] == 0
    assert _read_jsonl(session.ledger_path)[0]["status"] == "skipped"


async def test_processors_write_completed_turn_and_finalize(tmp_path: Path) -> None:
    session = SkillDistillationSessionCapture(
        namespace="tooluniverse-trialqa",
        launch_target="codex",
        display_model="routing-model",
        project_dir=tmp_path,
    )
    ctx = ProxyContext()
    ctx.selected_model = "served-model"

    await SkillDistillationRequestProcessor().process(ctx, _request())
    assert isinstance(ctx.metadata[CTX_SKILL_DISTILLATION_REQUEST], dict)

    await SkillDistillationResponseProcessor(session).process(ctx, _completion())
    stats = StatsAccumulator()
    await stats.record_success(model="served-model")
    await stats.record_usage(
        model="served-model",
        prompt_tokens=5,
        completion_tokens=3,
    )
    session.finish(exit_code=0, stats=stats)

    turns = _read_jsonl(session.turns_path)
    assert len(turns) == 1
    assert turns[0]["served_model"] == "served-model"
    assert turns[0]["request"]["model"] == "client-model"
    assert turns[0]["messages"][-1] == {"role": "assistant", "content": "hi"}
    assert turns[0]["usage"] == {
        "prompt_tokens": 5,
        "completion_tokens": 3,
        "total_tokens": 8,
    }

    metadata = _read_json(session.session_path)
    assert metadata["status"] == "completed"
    assert metadata["turn_count"] == 1
    assert metadata["distillation"]["status"] == "pending"
    assert _read_json(session.stats_path)["total_requests"] == 1
    assert _read_jsonl(session.ledger_path)[0]["status"] == "pending"


async def test_request_snapshot_failure_keeps_request_going(mocker) -> None:
    processor = SkillDistillationRequestProcessor()
    mocker.patch.object(
        processor._translation,
        "request_to",
        side_effect=RuntimeError("translation failed"),
    )
    ctx = ProxyContext()
    request = _request()

    assert await processor.process(ctx, request) is request
    assert CTX_SKILL_DISTILLATION_REQUEST not in ctx.metadata


async def test_response_snapshot_failure_keeps_response_going(
    mocker,
    tmp_path: Path,
) -> None:
    session = SkillDistillationSessionCapture(
        namespace="tooluniverse-trialqa",
        launch_target="codex",
        display_model="routing-model",
        project_dir=tmp_path,
    )
    processor = SkillDistillationResponseProcessor(session)
    mocker.patch.object(
        processor._translation,
        "response_to",
        side_effect=RuntimeError("translation failed"),
    )
    ctx = ProxyContext()
    ctx.metadata[CTX_SKILL_DISTILLATION_REQUEST] = {"messages": []}
    response = _completion()

    assert await processor.process(ctx, response) is response
    assert not session.turns_path.exists()


def test_build_launch_capture_processors_includes_skill_capture(tmp_path: Path) -> None:
    session = SkillDistillationSessionCapture(
        namespace="tooluniverse-trialqa",
        launch_target="openclaw",
        display_model="model",
        project_dir=tmp_path,
    )

    request, response = build_launch_capture_processors(None, None, session)

    assert [type(p).__name__ for p in request] == [
        "SkillDistillationRequestProcessor",
    ]
    assert [type(p).__name__ for p in response] == [
        "SkillDistillationResponseProcessor",
    ]


def test_launcher_helper_uses_saved_namespace_and_cwd(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "config"
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    save_user_config(
        UserConfig(
            skill_distillation=SkillDistillationConfig(
                namespace="tooluniverse-trialqa",
            ),
        ),
        config_dir=config_dir,
    )
    monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(config_dir))
    monkeypatch.chdir(project_dir)

    session = build_launch_skill_distillation_session(
        target="claude",
        display_model="model",
        strategy_summary="passthrough: model",
    )

    assert session is not None
    assert session.store_path == (
        project_dir / ".switchyard" / "skill-distillation" / "tooluniverse-trialqa"
    )


def test_launch_session_context_finalizes_failed_launch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "config"
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    save_user_config(
        UserConfig(
            skill_distillation=SkillDistillationConfig(
                namespace="tooluniverse-trialqa",
            ),
        ),
        config_dir=config_dir,
    )
    monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(config_dir))
    monkeypatch.chdir(project_dir)

    with pytest.raises(RuntimeError, match="launcher failed"):
        with launch_skill_distillation_session(
            target="codex",
            display_model="model",
            strategy_summary="passthrough: model",
            stats=StatsAccumulator(),
        ) as session:
            assert session.capture is not None
            session_path = session.capture.session_path
            raise RuntimeError("launcher failed")

    metadata = _read_json(session_path)
    assert metadata["status"] == "failed"
    assert metadata["distillation"]["status"] == "skipped"
