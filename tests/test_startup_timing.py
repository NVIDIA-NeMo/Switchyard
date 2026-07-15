# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from switchyard.cli.launchers import startup_timing


def _reset() -> None:
    startup_timing._marks.clear()


def test_disabled_records_nothing_and_prints_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("SWITCHYARD_STARTUP_TIMING", raising=False)
    _reset()

    startup_timing.mark("launch invoked")
    startup_timing.mark("child agent spawned")
    assert startup_timing._marks == []

    startup_timing.dump()
    assert capsys.readouterr().err == ""


def test_falsey_value_is_disabled(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SWITCHYARD_STARTUP_TIMING", "0")
    _reset()

    startup_timing.mark("launch invoked")
    assert startup_timing._marks == []
    startup_timing.dump()
    assert capsys.readouterr().err == ""


def test_enabled_prints_per_stage_breakdown_and_resets(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SWITCHYARD_STARTUP_TIMING", "1")
    _reset()

    startup_timing.mark("launch invoked")
    startup_timing.mark("chain built (incl. backend-format probe)")
    startup_timing.mark("child agent spawned")
    startup_timing.dump()

    err = capsys.readouterr().err
    assert "switchyard startup timing" in err
    assert "chain built (incl. backend-format probe)" in err
    assert "child agent spawned" in err
    assert "total (launch invoked -> child spawn)" in err
    # dump() resets so a second launch in the same process starts clean.
    assert startup_timing._marks == []


def test_dump_needs_at_least_two_marks(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SWITCHYARD_STARTUP_TIMING", "1")
    _reset()

    startup_timing.mark("launch invoked")
    startup_timing.dump()
    assert capsys.readouterr().err == ""
    assert startup_timing._marks == []
