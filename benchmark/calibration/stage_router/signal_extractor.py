# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python port of crates/switchyard-components/src/dimension_collector/tool_signals.rs.

Reads a claude-code session JSONL trajectory and reconstructs the per-turn
ToolResultSignal that the stage_router picker would have seen. Output is one record
per assistant turn.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SOFT, HARD, CRITICAL = 0.3, 0.7, 1.0

ERROR_PATTERNS: list[tuple[str, float, tuple[str, ...]]] = [
    ("oom", CRITICAL, ("out of memory", "memoryerror", "cannot allocate memory")),
    ("connection_refused", CRITICAL, ("connection refused", "connectionrefusederror", "econnrefused")),
    ("traceback", HARD, ("traceback (most recent call last)",)),
    ("import_error", HARD, ("modulenotfounderror:", "importerror:", "no module named ")),
    ("cmd_not_found", HARD, ("command not found", "not found\n", "/usr/bin/env: ")),
    ("assertion", HARD, ("assertionerror",)),
    ("value_error", HARD, ("valueerror:",)),
    ("syntax_error", HARD, ("syntaxerror:",)),
    ("timeout", HARD, ("timed out", "timeouterror", "timeout expired", "deadline exceeded")),
    ("no_such_file", HARD, ("filenotfounderror:", "no such file or directory")),
    ("exit_nonzero", SOFT, ("exit code 1", "exit code 2", "exit status 1", "returned non-zero", "exited with code")),
]

EDIT_TOOL_NAMES = {"edit", "multiedit", "notebookedit", "str_replace", "str_replace_based_edit_tool", "text_editor"}
WRITE_TOOL_NAMES = {"write", "create_file", "new_file"}
READ_TOOL_NAMES = {"read", "view"}
PLAN_TOOL_NAMES = {"todowrite", "todo_write", "todo", "update_plan"}
BASH_TOOL_NAMES = {"bash", "shell_command", "shell", "local_shell_call"}

BASH_WRITE_PATTERNS = ("cat >", "cat >>", "echo >", "echo >>", "tee ", "printf >", "printf >>", "> /", ">> /", "<< 'eof'", "<<eof", "<<'eof'", "<< eof")
BASH_EDIT_PATTERNS = ("sed -i", "sed --in-place", "awk -i inplace", "awk 'inplace=1'", "patch ", "patch -p", "perl -i", "perl -p -i", "perl -pi")
BASH_READ_PATTERNS = ("cat /", "cat ./", "cat ../", "grep ", "ls ", "ls -", "find ", "head ", "tail ", "wc ", "diff ", "which ", "ps ", "df ", "du ", "stat ", "file ", "less ", "more ")

TEST_PASS_PHRASES = (" passed", "passed in", "tests passed", "all tests passed", "test ok", "test result: ok", "passed.\n", "tests pass", "\nok ", "✓ ")
TEST_FAILURE_PHRASES = ("failed", "failure", "error:", " error", "assertionerror", "✗ ", "fatal:", "test failed")

RECENT_WINDOW = 3


@dataclass
class ToolResultSignal:
    severity: float = 0.0
    patterns: list[str] = field(default_factory=list)
    no_error_streak: int = 0
    edit_count: int = 0
    write_count: int = 0
    read_count: int = 0
    todowrite_count: int = 0
    recent_edit_count: int = 0
    recent_write_count: int = 0
    recent_read_count: int = 0
    recent_todowrite_count: int = 0
    pure_bash_streak: int = 0
    tests_passed: bool = False
    turn_depth: int = 0
    prompt_char_count: int = 0


def classify_text(text: str) -> tuple[float, list[str]]:
    lower = text.lower()
    patterns: list[str] = []
    severity = 0.0
    for name, sev, subs in ERROR_PATTERNS:
        if any(s in lower for s in subs):
            patterns.append(name)
            if sev > severity:
                severity = sev
    return severity, patterns


def classify_tool_call(name: str, command: str | None) -> str:
    lower = name.lower()
    if lower in WRITE_TOOL_NAMES:
        return "Write"
    if lower in EDIT_TOOL_NAMES:
        return "Edit"
    if lower in READ_TOOL_NAMES:
        return "Read"
    if lower in PLAN_TOOL_NAMES:
        return "Plan"
    if lower in BASH_TOOL_NAMES and command is not None:
        if any(p in command for p in BASH_WRITE_PATTERNS):
            return "Write"
        if any(p in command for p in BASH_EDIT_PATTERNS):
            return "Edit"
        if any(p in command for p in BASH_READ_PATTERNS):
            return "Read"
    return "Other"


def _detect_tests_passed(tool_texts: list[str]) -> bool:
    recent = tool_texts[-3:] if len(tool_texts) > 3 else tool_texts
    for text in recent:
        lower = text.lower()
        if any(p in lower for p in TEST_PASS_PHRASES) and not any(p in lower for p in TEST_FAILURE_PHRASES):
            return True
    return False


def _compute_no_error_streak(tool_texts: list[str]) -> int:
    streak = 0
    for text in reversed(tool_texts):
        sev, _ = classify_text(text)
        if sev > 0.0:
            break
        streak += 1
    return streak


def _build_signal(tool_texts: list[str], tool_calls: list[tuple[str, str | None]], turn_depth: int, prompt_char_count: int) -> ToolResultSignal:
    severity, patterns = classify_text(tool_texts[-1]) if tool_texts else (0.0, [])
    no_error_streak = _compute_no_error_streak(tool_texts)

    recent_start = max(0, len(tool_calls) - RECENT_WINDOW)
    write_count = edit_count = read_count = todowrite_count = 0
    recent_write_count = recent_edit_count = recent_read_count = recent_todowrite_count = 0
    pure_bash_streak = 0
    streak_open = True

    for i in range(len(tool_calls) - 1, -1, -1):
        name, cmd = tool_calls[i]
        cat = classify_tool_call(name, cmd)
        if streak_open:
            if cat == "Other":
                pure_bash_streak += 1
            else:
                streak_open = False
        in_recent = i >= recent_start
        if cat == "Write":
            write_count += 1
            if in_recent:
                recent_write_count += 1
        elif cat == "Edit":
            edit_count += 1
            if in_recent:
                recent_edit_count += 1
        elif cat == "Read":
            read_count += 1
            if in_recent:
                recent_read_count += 1
        elif cat == "Plan":
            todowrite_count += 1
            if in_recent:
                recent_todowrite_count += 1

    return ToolResultSignal(
        severity=severity,
        patterns=patterns,
        no_error_streak=no_error_streak,
        edit_count=edit_count,
        write_count=write_count,
        read_count=read_count,
        todowrite_count=todowrite_count,
        recent_edit_count=recent_edit_count,
        recent_write_count=recent_write_count,
        recent_read_count=recent_read_count,
        recent_todowrite_count=recent_todowrite_count,
        pure_bash_streak=pure_bash_streak,
        tests_passed=_detect_tests_passed(tool_texts),
        turn_depth=turn_depth,
        prompt_char_count=prompt_char_count,
    )


def _content_to_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str):
                parts.append(b["text"])
        return "\n".join(parts) if parts else None
    return None


def replay_trajectory(jsonl_path: Path) -> list[ToolResultSignal]:
    """Walk a claude-code session JSONL and emit per-assistant-turn signals.

    Mirrors `extract_from_messages_anthropic` in tool_signals.rs: each assistant
    turn produces a signal computed from all messages up to (but not including)
    the assistant turn itself — i.e. what the picker would have seen as input
    when deciding which model to call for that turn.
    """
    tool_texts: list[str] = []
    tool_calls: list[tuple[str, str | None]] = []
    prompt_char_count = 0
    msg_count = 0  # messages-so-far (proxy for turn_depth in the Rust impl)
    signals: list[ToolResultSignal] = []

    with open(jsonl_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            et = ev.get("type")
            if et not in ("user", "assistant"):
                continue

            if et == "assistant":
                # Snapshot signal before processing the assistant content — this
                # is what the picker would have seen when deciding which model
                # serves THIS assistant turn.
                signals.append(_build_signal(tool_texts.copy(), tool_calls.copy(), msg_count, prompt_char_count))

            msg = ev.get("message", {})
            content = msg.get("content")
            msg_count += 1

            if et == "user":
                # Claude-code stores user content as either a plain string
                # (initial prompt or text follow-up) or a list with
                # tool_result / text blocks.
                if isinstance(content, str):
                    prompt_char_count = len(content)
                elif isinstance(content, list):
                    for b in content:
                        if not isinstance(b, dict):
                            continue
                        bt = b.get("type")
                        if bt == "tool_result":
                            text = _content_to_text(b.get("content"))
                            if text is not None:
                                tool_texts.append(text)
                        elif bt == "text":
                            t = b.get("text")
                            if isinstance(t, str):
                                prompt_char_count = len(t)
            elif et == "assistant":
                if isinstance(content, list):
                    for b in content:
                        if not isinstance(b, dict):
                            continue
                        if b.get("type") == "tool_use":
                            name = b.get("name")
                            inp = b.get("input")
                            cmd = None
                            if isinstance(inp, dict):
                                c = inp.get("command")
                                if isinstance(c, str):
                                    cmd = c.lower()
                            if isinstance(name, str):
                                tool_calls.append((name, cmd))

    return signals


def task_summary(signals: list[ToolResultSignal]) -> dict[str, Any]:
    """Aggregate per-turn signals into one feature vector per task."""
    if not signals:
        return {
            "turns": 0,
            "max_severity": 0.0,
            "max_turn_depth": 0,
            "max_pure_bash": 0,
            "total_writes": 0,
            "total_edits": 0,
            "total_reads": 0,
            "total_todowrites": 0,
            "max_recent_reads": 0,
            "max_recent_writes": 0,
            "max_recent_todowrites": 0,
            "prompt_char_count": 0,
            "ever_tests_passed": False,
            "ever_severity_critical": False,
            "ever_severity_hard": False,
            "ever_severity_soft": False,
        }
    last = signals[-1]
    return {
        "turns": len(signals),
        "max_severity": max(s.severity for s in signals),
        "max_turn_depth": max(s.turn_depth for s in signals),
        "max_pure_bash": max(s.pure_bash_streak for s in signals),
        "total_writes": last.write_count,
        "total_edits": last.edit_count,
        "total_reads": last.read_count,
        "total_todowrites": last.todowrite_count,
        "max_recent_reads": max(s.recent_read_count for s in signals),
        "max_recent_writes": max(s.recent_write_count for s in signals),
        "max_recent_todowrites": max(s.recent_todowrite_count for s in signals),
        "prompt_char_count": signals[0].prompt_char_count if signals else 0,
        "ever_tests_passed": any(s.tests_passed for s in signals),
        "ever_severity_critical": any(s.severity >= CRITICAL for s in signals),
        "ever_severity_hard": any(s.severity >= HARD for s in signals),
        "ever_severity_soft": any(s.severity >= SOFT for s in signals),
    }
