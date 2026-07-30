# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for importer CLI."""

from __future__ import annotations

from pathlib import Path

from craft_taskgen.importers import cli


def test_cli_dispatches_pr_refs_with_source_name(tmp_path, monkeypatch, capsys) -> None:
    called = {}

    def fake_run_import(**kwargs):
        called.update(kwargs)
        return {
            "rows_total": 4,
            "rows_unresolved": 0,
            "repos_written": 2,
            "prs_scanned": 2,
            "candidates_kept": 3,
            "skipped_unmerged_or_missing": 0,
        }

    monkeypatch.setattr("craft_taskgen.importers.pr_refs.run_import", fake_run_import)
    monkeypatch.setattr(
        "sys.argv",
        [
            "craft-taskgen-import",
            "--input",
            str(tmp_path / "map.json"),
            "--source-name",
            "curated-pr-map",
        ],
    )
    cli.main()

    assert called["input_path"] == Path(tmp_path / "map.json")
    assert called["out_dir"] == Path("candidates") / "pr-refs"
    assert called["top_per_repo"] == 0
    assert called["source_name"] == "curated-pr-map"
    out = capsys.readouterr().out
    assert "Summary:" in out
    assert "rows=4" in out


def test_cli_dispatches_swebench_pro(tmp_path, monkeypatch, capsys) -> None:
    called = {}

    def fake_run_import(**kwargs):
        called.update(kwargs)
        return {
            "rows_total": 2,
            "rows_unresolved": 0,
            "repos_written": 1,
            "prs_scanned": 2,
            "candidates_kept": 2,
            "skipped_unmerged_or_missing": 0,
        }

    monkeypatch.setattr("craft_taskgen.importers.swebench_pro.run_import", fake_run_import)
    monkeypatch.setattr(
        "sys.argv",
        [
            "craft-taskgen-import",
            "--format",
            "swebench-pro",
            "--input",
            str(tmp_path / "swebench_pro.jsonl"),
        ],
    )
    cli.main()

    assert called["input_path"] == Path(tmp_path / "swebench_pro.jsonl")
    assert called["out_dir"] == Path("candidates") / "swebench-pro"
    assert called["source_name"] == "swebench-pro"
    out = capsys.readouterr().out
    assert "Summary:" in out
    assert "rows=2" in out
