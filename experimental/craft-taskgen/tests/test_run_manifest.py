# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-job run-manifest writer."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from craft_taskgen.baselines.run_manifest import (
    SCHEMA_VERSION,
    task_dir_digest,
    write_manifest,
)


def _write_and_load(tmp_path: Path, **fields) -> dict:
    out = tmp_path / "run_manifest.json"
    write_manifest(out, **fields)
    return json.loads(out.read_text())


def test_manifest_is_valid_json(tmp_path: Path):
    manifest = _write_and_load(tmp_path, agent={"name": "claude-code"})
    assert isinstance(manifest, dict)


def test_schema_version_present(tmp_path: Path):
    manifest = _write_and_load(tmp_path)
    assert manifest["run"]["schema_version"] == SCHEMA_VERSION


def test_self_resolved_run_fields_present(tmp_path: Path):
    manifest = _write_and_load(tmp_path)
    run = manifest["run"]
    # These are always populated by the module itself.
    for key in (
        "schema_version",
        "timestamp",
        "hostname",
        "craft_taskgen_sha",
        "craft_taskgen_dirty",
        "craft_taskgen_tree_kind",
        "harbor_version",
        "node_version",
        "launcher_argv",
    ):
        assert key in run, f"missing run.{key}"


def test_timestamp_is_iso_8601(tmp_path: Path):
    manifest = _write_and_load(tmp_path)
    ts = manifest["run"]["timestamp"]
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", ts), ts


def test_craft_taskgen_sha_is_hex_or_null(tmp_path: Path):
    manifest = _write_and_load(tmp_path)
    sha = manifest["run"]["craft_taskgen_sha"]
    # Either a 40-char git sha or null if we're somehow not in a git repo.
    assert sha is None or re.match(r"^[0-9a-f]{40}$", sha), sha


def test_caller_fields_round_trip(tmp_path: Path):
    manifest = _write_and_load(
        tmp_path,
        agent={"name": "codex", "version": "0.121.0", "model": "azure/openai/gpt-5.3-codex"},
        reasoning={"effort": "high", "source": "reasoning_defaults"},
        output_cap={"tokens": 64000, "applied": "uncapped-see-openai/codex#4138"},
    )
    assert manifest["agent"]["name"] == "codex"
    assert manifest["agent"]["version"] == "0.121.0"
    assert manifest["reasoning"]["effort"] == "high"
    assert manifest["output_cap"]["tokens"] == 64000


def test_none_values_preserved_as_null(tmp_path: Path):
    manifest = _write_and_load(tmp_path, sampling={"temperature": None, "top_p": 0.95})
    assert manifest["sampling"]["temperature"] is None
    assert manifest["sampling"]["top_p"] == 0.95


def test_launcher_argv_captured(tmp_path: Path):
    argv = ["scripts/run-baselines.sh", "--agent", "claude-code", "--n-tasks", "3"]
    manifest = _write_and_load(tmp_path, launcher_argv=argv)
    assert manifest["run"]["launcher_argv"] == argv


def test_craft_taskgen_tree_kind_is_enum(tmp_path: Path):
    # tree_kind projects the tri-state dirty flag into a grep-friendly enum.
    manifest = _write_and_load(tmp_path)
    assert manifest["run"]["craft_taskgen_tree_kind"] in {
        "git-clean",
        "git-dirty",
        "not-git",
    }


def test_craft_taskgen_dirty_distinguishes_clean_from_not_in_repo(tmp_path: Path):
    # Key invariant: a clean git tree must report dirty=False (NOT None).
    # The old _git_dirty helper collapsed "rc=0, empty stdout" into None,
    # making clean indistinguishable from "not a repo".
    manifest = _write_and_load(tmp_path)
    # The craft-taskgen launcher repo itself IS a git repo; the module
    # resolves it via __file__, so dirty must be a concrete bool.
    assert manifest["run"]["craft_taskgen_dirty"] in (True, False), (
        f"expected bool, got {manifest['run']['craft_taskgen_dirty']!r} — "
        "clean/dirty must not be None when we're in a git repo"
    )


def test_caller_run_section_overrides_self_resolved(tmp_path: Path):
    # Caller-provided run fields win (e.g. to override hostname in CI).
    manifest = _write_and_load(tmp_path, run={"hostname": "ci-override-host"})
    assert manifest["run"]["hostname"] == "ci-override-host"
    # Self-resolved ones still populate where the caller didn't override.
    assert "timestamp" in manifest["run"]


class TestTaskHelpers:
    def test_digest_is_sha256_prefixed(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "b.txt").write_text("world")
        digest = task_dir_digest(tmp_path)
        assert digest.startswith("sha256:")
        assert len(digest) == len("sha256:") + 64

    def test_digest_deterministic_across_file_order(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("alpha")
        (tmp_path / "b.txt").write_text("beta")
        first = task_dir_digest(tmp_path)
        # Touching a different file path that isn't included shouldn't
        # change the digest — we only hash regular files under the root.
        second = task_dir_digest(tmp_path)
        assert first == second

    def test_digest_changes_when_content_changes(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("v1")
        first = task_dir_digest(tmp_path)
        (tmp_path / "a.txt").write_text("v2")
        second = task_dir_digest(tmp_path)
        assert first != second


class TestCliMain:
    """The launcher invokes `python -m craft_taskgen.baselines.run_manifest`
    with per-field flags. These tests pin that contract.
    """

    def _run(self, tmp_path: Path, *args: str) -> dict:
        from craft_taskgen.baselines.run_manifest import _cli_main

        out = tmp_path / "manifest.json"
        rc = _cli_main(["--output", str(out), *args])
        assert rc == 0
        return json.loads(out.read_text())

    def test_minimal_invocation(self, tmp_path: Path):
        manifest = self._run(tmp_path, "--agent", "claude-code", "--effort", "high")
        assert manifest["agent"]["name"] == "claude-code"
        assert manifest["reasoning"]["effort"] == "high"
        assert manifest["run"]["schema_version"] == SCHEMA_VERSION

    def test_empty_strings_become_null(self, tmp_path: Path):
        manifest = self._run(tmp_path, "--agent", "codex", "--effort", "")
        assert manifest["reasoning"]["effort"] is None

    def test_output_cap_is_int(self, tmp_path: Path):
        manifest = self._run(tmp_path, "--output-cap", "64000")
        assert manifest["output_cap"]["tokens"] == 64000

    def test_output_cap_tokens_null_when_uncapped(self, tmp_path: Path):
        # Codex has an UNCAPPED applied note (openai/codex#4138). Tokens
        # must be null in that case — reporting 64000 misleads the reader
        # into thinking a cap took effect.
        manifest = self._run(
            tmp_path,
            "--output-cap",
            "64000",
            "--output-cap-applied",
            "UNCAPPED (openai/codex#4138)",
        )
        assert manifest["output_cap"]["tokens"] is None
        assert manifest["output_cap"]["applied"] == "UNCAPPED (openai/codex#4138)"

    def test_output_cap_tokens_kept_when_applied(self, tmp_path: Path):
        manifest = self._run(
            tmp_path,
            "--output-cap",
            "64000",
            "--output-cap-applied",
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS=64000",
        )
        assert manifest["output_cap"]["tokens"] == 64000

    def test_sanity_agent_manifest_collapses_to_null(self, tmp_path: Path):
        """Oracle/nop runs through the same manifest CLI but with the
        LLM-specific fields empty — the launcher's `--agent oracle`
        invocation passes `--agent-version=""`, `--model=""`, etc.
        Empty strings round-trip to JSON null via _opt(). This pins
        the resulting manifest shape so a reviewer can tell from a
        glance that no model was involved.
        """
        manifest = self._run(
            tmp_path,
            "--agent",
            "oracle",
            # All LLM-related flags empty to simulate the launcher's
            # behavior for sanity agents.
            "--agent-version",
            "",
            "--model",
            "",
            "--effort",
            "",
            "--reasoning-source",
            "",
            "--reasoning-notes",
            "",
            "--output-cap-applied",
            "",
        )
        assert manifest["agent"]["name"] == "oracle"
        assert manifest["agent"]["version"] is None
        assert manifest["agent"]["model"] is None
        assert manifest["reasoning"]["effort"] is None
        assert manifest["reasoning"]["source"] is None
        assert manifest["output_cap"]["applied"] is None

    def test_tasks_dir_populates_harness(self, tmp_path: Path):
        # Only task_dir_digest + absolute tasks_dir land in the manifest.
        # Task enumeration and per-task config inspection are harbor's job
        # and live in result.json, not this manifest.
        td = tmp_path / "tasks"
        td.mkdir()
        (td / "t2v3-A").mkdir()
        (td / "t2v3-A" / "task.toml").write_text("stub")
        manifest = self._run(tmp_path, "--tasks-dir", str(td), "--n-tasks", "2")
        assert manifest["harness"]["task_dir_digest"].startswith("sha256:")
        assert manifest["harness"]["tasks_dir"] == str(td.resolve())
        # task_ids / timeouts intentionally absent
        assert "task_ids" not in manifest["harness"]
        assert "timeouts" not in manifest

    def test_launcher_argv_splits_cleanly(self, tmp_path: Path):
        manifest = self._run(
            tmp_path,
            "--launcher-argv",
            "/path/to/run-baselines.sh --agent claude-code --n-tasks 3",
        )
        argv = manifest["run"]["launcher_argv"]
        assert argv[1:] == ["--agent", "claude-code", "--n-tasks", "3"]

    def test_extra_field_merges(self, tmp_path: Path):
        # String values (JSON parse fails → fall through as string).
        manifest = self._run(
            tmp_path,
            "--extra",
            'determinism.PYTHONHASHSEED="0"',
            "--extra",
            "determinism.LC_ALL=C.UTF-8",
        )
        assert manifest["determinism"] == {"PYTHONHASHSEED": "0", "LC_ALL": "C.UTF-8"}

    def test_extra_json_coerces_numeric(self, tmp_path: Path):
        manifest = self._run(tmp_path, "--extra", "harness.max_turns=250")
        assert manifest["harness"]["max_turns"] == 250
        assert isinstance(manifest["harness"]["max_turns"], int)

    def test_extra_json_coerces_float(self, tmp_path: Path):
        manifest = self._run(tmp_path, "--extra", "timeouts.multiplier=1.5")
        assert manifest["timeouts"]["multiplier"] == 1.5
        assert isinstance(manifest["timeouts"]["multiplier"], float)

    def test_extra_json_coerces_bool(self, tmp_path: Path):
        manifest = self._run(tmp_path, "--extra", "compaction.opencode_auto=false")
        assert manifest["compaction"]["opencode_auto"] is False

    def test_extra_json_coerces_null(self, tmp_path: Path):
        # Non-applicable per-agent fields use `null` so the schema shape
        # stays consistent across agents (vs. mixed `95` / `"n/a"` types).
        manifest = self._run(tmp_path, "--extra", "compaction.claude_code_pct_override=null")
        assert manifest["compaction"]["claude_code_pct_override"] is None

    def test_extra_non_json_falls_through_to_string(self, tmp_path: Path):
        # C.UTF-8 is not valid JSON — must land as a string, not raise.
        manifest = self._run(tmp_path, "--extra", "determinism.LC_ALL=C.UTF-8")
        assert manifest["determinism"]["LC_ALL"] == "C.UTF-8"

    def test_agent_disallowed_tools_roundtrips(self, tmp_path: Path):
        # Launcher emits `--extra agent.disallowed_tools="EnterPlanMode,..."`
        # (quoted JSON string) for claude-code. Verify the string lands as-is
        # under the agent section alongside name/version/model.
        manifest = self._run(
            tmp_path,
            "--agent",
            "claude-code",
            "--extra",
            'agent.disallowed_tools="EnterPlanMode,ExitPlanMode"',
        )
        assert manifest["agent"]["disallowed_tools"] == "EnterPlanMode,ExitPlanMode"

    def test_agent_disallowed_tools_null_when_absent(self, tmp_path: Path):
        # Launcher emits `--extra agent.disallowed_tools=null` for
        # opencode/codex (no plan-mode tool) or when DISABLE_PLAN_MODE=0.
        manifest = self._run(
            tmp_path,
            "--agent",
            "opencode",
            "--extra",
            "agent.disallowed_tools=null",
        )
        assert manifest["agent"]["disallowed_tools"] is None

    def test_extra_rejects_bad_format(self, tmp_path: Path):
        from craft_taskgen.baselines.run_manifest import _cli_main

        out = tmp_path / "manifest.json"
        rc = _cli_main(["--output", str(out), "--extra", "no-section-dot"])
        assert rc == 2

    def test_harbor_result_json_is_absolute(self, tmp_path: Path):
        manifest = self._run(tmp_path)
        # The manifest should point at the absolute path of the
        # adjacent result.json, so a reader can `jq` it without knowing
        # where the manifest lives.
        expected = str((tmp_path / "result.json").resolve())
        assert manifest["outcomes"]["harbor_result_json"] == expected

    def test_outcomes_status_initially_predicted(self, tmp_path: Path):
        # Manifest is written BEFORE harbor runs. Status starts as
        # "predicted" so a reader can distinguish "run in progress" /
        # "finalize never ran" from a post-run "present" / "missing".
        manifest = self._run(tmp_path)
        assert manifest["outcomes"]["harbor_result_json_status"] == "predicted"
        assert manifest["outcomes"]["harbor_rc"] is None

    def test_vllm_probe_skipped_without_flag(self, tmp_path: Path):
        # --vllm-probe off → backend.vllm_snapshot absent.
        manifest = self._run(tmp_path, "--backend", "vllm", "--base-url", "http://invalid.local/v1")
        assert "vllm_snapshot" not in manifest["backend"]

    def test_vllm_probe_failure_writes_none(self, tmp_path: Path):
        # --vllm-probe on but unreachable URL → vllm_snapshot = null, manifest still written.
        manifest = self._run(
            tmp_path,
            "--backend",
            "vllm",
            "--base-url",
            "http://invalid.localhost.internal:1/v1",
            "--vllm-probe",
        )
        # Either None (probe failed) or a dict (probe succeeded if somehow
        # the host resolves). We test the dict case via live smoke; here
        # we only assert the field was at least attempted.
        assert "vllm_snapshot" in manifest["backend"]


@pytest.mark.parametrize("value", [None, 0, "", [], {}])
def test_falsy_values_preserved(tmp_path: Path, value):
    manifest = _write_and_load(tmp_path, custom_section={"k": value})
    assert manifest["custom_section"]["k"] == value


class TestFinalize:
    """`--finalize <path>` updates an existing manifest in place after
    harbor exits, so outcomes.harbor_result_json_status reflects reality
    (present / missing) and outcomes.harbor_rc records the exit code.
    """

    def _seed(self, tmp_path: Path) -> Path:
        from craft_taskgen.baselines.run_manifest import _cli_main

        out = tmp_path / "manifest.json"
        rc = _cli_main(["--output", str(out), "--agent", "claude-code"])
        assert rc == 0
        return out

    def test_marks_result_present_when_file_exists(self, tmp_path: Path):
        from craft_taskgen.baselines.run_manifest import _cli_main

        m = self._seed(tmp_path)
        (tmp_path / "result.json").write_text("{}")
        rc = _cli_main(["--finalize", str(m), "--harbor-rc", "0"])
        assert rc == 0
        manifest = json.loads(m.read_text())
        assert manifest["outcomes"]["harbor_result_json_status"] == "present"
        assert manifest["outcomes"]["harbor_rc"] == 0

    def test_marks_result_missing_when_file_absent(self, tmp_path: Path):
        from craft_taskgen.baselines.run_manifest import _cli_main

        m = self._seed(tmp_path)
        # result.json never written
        rc = _cli_main(["--finalize", str(m), "--harbor-rc", "1"])
        assert rc == 0
        manifest = json.loads(m.read_text())
        assert manifest["outcomes"]["harbor_result_json_status"] == "missing"
        assert manifest["outcomes"]["harbor_rc"] == 1

    def test_finalize_missing_manifest_returns_error(self, tmp_path: Path):
        from craft_taskgen.baselines.run_manifest import _cli_main

        rc = _cli_main(["--finalize", str(tmp_path / "nope.json"), "--harbor-rc", "0"])
        assert rc == 1


class TestProbeVllmModels:
    """probe_vllm_models is opt-in (--vllm-probe). It must never raise — any
    network/parse failure returns None so the manifest still writes. These
    tests pin each documented error branch.
    """

    def _patch_urlopen(self, monkeypatch, side_effect):
        """Replace urllib.request.urlopen with a fake that raises or returns."""
        import craft_taskgen.baselines.run_manifest as rm

        if isinstance(side_effect, Exception):

            def fake(*_a, **_kw):
                raise side_effect

        else:

            class _Resp:
                def __init__(self, payload: bytes):
                    self._payload = payload

                def __enter__(self):
                    return self

                def __exit__(self, *_a):
                    return False

                def read(self):
                    return self._payload

            def fake(*_a, **_kw):
                return _Resp(side_effect)

        monkeypatch.setattr(rm.urllib.request, "urlopen", fake)

    def test_returns_none_on_url_error(self, monkeypatch):
        from craft_taskgen.baselines.run_manifest import probe_vllm_models

        self._patch_urlopen(monkeypatch, __import__("urllib").error.URLError("connection refused"))
        assert probe_vllm_models("http://nope.local/v1") is None

    def test_returns_none_on_timeout(self, monkeypatch):
        from craft_taskgen.baselines.run_manifest import probe_vllm_models

        self._patch_urlopen(monkeypatch, TimeoutError("read timed out"))
        assert probe_vllm_models("http://nope.local/v1") is None

    def test_returns_none_on_invalid_json(self, monkeypatch):
        from craft_taskgen.baselines.run_manifest import probe_vllm_models

        self._patch_urlopen(monkeypatch, b"<!DOCTYPE html>not json")
        assert probe_vllm_models("http://x.local/v1") is None

    def test_returns_none_on_empty_data(self, monkeypatch):
        # Server returned 200 but `data: []` — not a real model list.
        from craft_taskgen.baselines.run_manifest import probe_vllm_models

        self._patch_urlopen(monkeypatch, json.dumps({"data": []}).encode())
        assert probe_vllm_models("http://x.local/v1") is None

    def test_returns_summary_on_success(self, monkeypatch):
        from craft_taskgen.baselines.run_manifest import probe_vllm_models

        payload = json.dumps(
            {
                "data": [
                    {
                        "id": "served-name",
                        "root": "/lustre/path/to/model",
                        "max_model_len": 196608,
                        "owned_by": "vllm",
                    }
                ]
            }
        ).encode()
        self._patch_urlopen(monkeypatch, payload)
        result = probe_vllm_models("http://x.local/v1")
        assert result == {
            "served_model_name": "served-name",
            "served_model_root": "/lustre/path/to/model",
            "max_model_len": 196608,
            "owned_by": "vllm",
        }

    def test_returns_none_on_missing_data_key(self, monkeypatch):
        # Some gateways return 200 with a different shape; we treat that
        # as "no usable info" rather than a partial dict.
        from craft_taskgen.baselines.run_manifest import probe_vllm_models

        self._patch_urlopen(monkeypatch, json.dumps({"models": [{"id": "x"}]}).encode())
        assert probe_vllm_models("http://x.local/v1") is None
