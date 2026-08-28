#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Update alignment columns in a CSV from a pipeline state.json."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

try:
    from compare_eval_csv import _build_indexes, _load_state, _match_row
except ImportError:  # pragma: no cover - used when imported from tests as scripts.*
    from scripts.compare_eval_csv import _build_indexes, _load_state, _match_row


@dataclass(frozen=True)
class UpdateStats:
    rows: int
    matched: int

    @property
    def unmatched(self) -> int:
        return self.rows - self.matched


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as e:
            raise ValueError(f"{path}: CSV has no header row") from e
        return header, list(reader)


def _pad_rows(rows: list[list[str]], width: int) -> None:
    for row in rows:
        if len(row) < width:
            row.extend([""] * (width - len(row)))


def _column_indexes(header: list[str], name: str) -> list[int]:
    return [idx for idx, value in enumerate(header) if value == name]


def _default_insert_at(header: list[str]) -> int:
    for anchor in ("new_alignment_reason", "alignment_reason", "eval_reason", "new_eval_reason"):
        if anchor in header:
            return header.index(anchor) + 1
    return len(header)


def _insert_column(header: list[str], rows: list[list[str]], name: str, index: int) -> list[int]:
    old_width = len(header)
    _pad_rows(rows, old_width)
    header.insert(index, name)
    for row in rows:
        row.insert(index, "")
    return [index]


def _ensure_alignment_columns(
    header: list[str],
    rows: list[list[str]],
    *,
    verdict_column: str,
    reason_column: str,
) -> tuple[list[int], list[int]]:
    verdict_indexes = _column_indexes(header, verdict_column)
    reason_indexes = _column_indexes(header, reason_column)

    if not verdict_indexes and not reason_indexes:
        insert_at = _default_insert_at(header)
        verdict_indexes = _insert_column(header, rows, verdict_column, insert_at)
        reason_indexes = _insert_column(header, rows, reason_column, insert_at + 1)
    elif not verdict_indexes:
        verdict_indexes = _insert_column(header, rows, verdict_column, min(reason_indexes))
        reason_indexes = [idx + 1 if idx >= verdict_indexes[0] else idx for idx in reason_indexes]
    elif not reason_indexes:
        reason_indexes = _insert_column(header, rows, reason_column, max(verdict_indexes) + 1)
        verdict_indexes = [idx if idx < reason_indexes[0] else idx + 1 for idx in verdict_indexes]

    return verdict_indexes, reason_indexes


def _row_mapping(header: list[str], row: list[str]) -> dict[str, str]:
    _pad_rows([row], len(header))
    out: dict[str, str] = {}
    for idx, name in enumerate(header):
        if name not in out:
            out[name] = row[idx]
    return out


def _set_columns(row: list[str], indexes: list[int], value: object, width: int) -> None:
    if len(row) < width:
        row.extend([""] * (width - len(row)))
    text = "" if value is None else str(value)
    for idx in indexes:
        row[idx] = text


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    tmp_path.replace(path)


def update_alignment_csv(
    input_csv: Path,
    state_json: Path,
    output_csv: Path,
    *,
    verdict_column: str = "alignment_verdict",
    reason_column: str = "alignment_reason",
    max_text: int = 500,
) -> UpdateStats:
    state = _load_state(state_json)
    tasks = state.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError(f"{state_json}: expected top-level 'tasks' dict")

    by_instance_id, by_task_id, by_commit_sha = _build_indexes(tasks, max_text=max_text)
    header, rows = _read_csv(input_csv)
    verdict_indexes, reason_indexes = _ensure_alignment_columns(
        header,
        rows,
        verdict_column=verdict_column,
        reason_column=reason_column,
    )

    matched = 0
    width = len(header)
    for row in rows:
        payload, _match_key = _match_row(
            _row_mapping(header, row),
            by_instance_id,
            by_task_id,
            by_commit_sha,
        )
        if not payload:
            continue
        matched += 1
        _set_columns(row, verdict_indexes, payload.get("new_alignment_verdict", ""), width)
        _set_columns(row, reason_indexes, payload.get("new_alignment_reason", ""), width)

    _pad_rows(rows, len(header))
    _write_csv(output_csv, header, rows)
    return UpdateStats(rows=len(rows), matched=matched)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Update or add alignment_verdict/alignment_reason columns from a pipeline state.json."
    )
    parser.add_argument("input_csv", type=Path, help="CSV to update")
    parser.add_argument("state_json", type=Path, help="Pipeline state.json to join from")
    parser.add_argument("output_csv", type=Path, help="Destination CSV path; may be the same as input_csv")
    parser.add_argument("--verdict-column", default="alignment_verdict", help="Column name for the verdict")
    parser.add_argument("--reason-column", default="alignment_reason", help="Column name for the reason")
    parser.add_argument(
        "--max-text",
        type=int,
        default=500,
        help="Maximum length for copied alignment reasons",
    )
    args = parser.parse_args()

    stats = update_alignment_csv(
        args.input_csv,
        args.state_json,
        args.output_csv,
        verdict_column=args.verdict_column,
        reason_column=args.reason_column,
        max_text=args.max_text,
    )
    print(
        f"Wrote {stats.rows} rows to {args.output_csv} "
        f"({stats.matched} matched, {stats.unmatched} unmatched)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
