# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the structured per-attempt upstream-failure log."""

from __future__ import annotations

import json
import logging
from datetime import datetime

import pytest

from switchyard.lib.endpoints import upstream_error_log
from switchyard.lib.endpoints.upstream_error_log import (
    EVENT_NAME,
    log_upstream_attempt_failure,
)

_LOGGER_NAME = "switchyard.upstream_errors"


def _emit_and_parse(
    caplog: pytest.LogCaptureFixture,
    *,
    model: str,
    attempt: int,
    status_code: int | None,
    error: BaseException,
) -> dict[str, object]:
    """Call the logger and return the single emitted record parsed as JSON."""
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        log_upstream_attempt_failure(
            model=model, attempt=attempt, status_code=status_code, error=error
        )
    records = [r for r in caplog.records if r.name == _LOGGER_NAME]
    assert len(records) == 1, "expected exactly one structured record"
    assert records[0].levelno == logging.WARNING
    return json.loads(records[0].getMessage())


class TestRecordShape:
    def test_message_is_valid_json_with_expected_keys(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        rec = _emit_and_parse(
            caplog, model="m", attempt=1, status_code=500, error=RuntimeError("boom")
        )
        assert set(rec) == {
            "event",
            "timestamp",
            "model",
            "upstream_model",
            "attempt",
            "status_code",
            "code",
            "outcome",
            "error_source",
            "error_type",
            "error",
        }
        assert rec["event"] == EVENT_NAME == "upstream_attempt_failed"
        assert rec["model"] == "m"
        # No upstream_model supplied → logged as null; source is constant.
        assert rec["upstream_model"] is None
        assert rec["error_source"] == "provider"
        assert rec["attempt"] == 1
        assert rec["error_type"] == "RuntimeError"
        assert rec["error"] == "boom"

    def test_timestamp_is_iso8601_and_utc(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        rec = _emit_and_parse(
            caplog, model="m", attempt=1, status_code=429, error=ValueError("x")
        )
        # Round-trips through fromisoformat and carries a UTC offset.
        parsed = datetime.fromisoformat(str(rec["timestamp"]))
        assert parsed.utcoffset() is not None
        assert parsed.utcoffset().total_seconds() == 0  # type: ignore[union-attr]


class TestCodeAndOutcomeMirrorTheMetric:
    def test_none_is_network_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        rec = _emit_and_parse(
            caplog, model="m", attempt=2, status_code=None,
            error=ConnectionError("reset"),
        )
        assert rec["status_code"] is None
        assert rec["code"] == "none"
        assert rec["outcome"] == "retryable_error"

    @pytest.mark.parametrize("code", [429, 500, 504])
    def test_retryable_codes(
        self, caplog: pytest.LogCaptureFixture, code: int
    ) -> None:
        rec = _emit_and_parse(
            caplog, model="m", attempt=1, status_code=code, error=RuntimeError("e")
        )
        assert rec["status_code"] == code
        assert rec["code"] == str(code)
        assert rec["outcome"] == "retryable_error"

    def test_other_error_code(self, caplog: pytest.LogCaptureFixture) -> None:
        rec = _emit_and_parse(
            caplog, model="m", attempt=1, status_code=401, error=RuntimeError("e")
        )
        assert rec["status_code"] == 401
        assert rec["code"] == "401"
        assert rec["outcome"] == "other_error"

    def test_unknown_code_keeps_raw_status_but_clamps_label(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The raw code is preserved for audit; the joinable label is clamped."""
        rec = _emit_and_parse(
            caplog, model="m", attempt=1, status_code=418, error=RuntimeError("e")
        )
        assert rec["status_code"] == 418
        assert rec["code"] == "4xx"
        assert rec["outcome"] == "other_error"


class TestErrorTruncation:
    def test_long_error_is_capped(self, caplog: pytest.LogCaptureFixture) -> None:
        rec = _emit_and_parse(
            caplog, model="m", attempt=1, status_code=500,
            error=RuntimeError("x" * 5000),
        )
        assert isinstance(rec["error"], str)
        assert len(rec["error"]) == upstream_error_log._MAX_ERROR_CHARS
