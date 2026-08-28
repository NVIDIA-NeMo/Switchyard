# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for task format validator."""

from __future__ import annotations

import os

from craft_taskgen.task_format import strip_instruction_boilerplate, validate_task_dir


def _make_valid_task(tmp_path):
    """Create a minimal valid task directory."""
    task_dir = str(tmp_path / "t2v3-TE1-test-feature")
    os.makedirs(os.path.join(task_dir, "environment"))
    os.makedirs(os.path.join(task_dir, "tests"))
    os.makedirs(os.path.join(task_dir, "solution"))

    with open(os.path.join(task_dir, "instruction.md"), "w") as f:
        f.write(
            "Implement the streaming response feature for the API endpoint. "
            "The server should support chunked transfer encoding and yield results "
            "incrementally as they become available. Ensure backwards compatibility "
            "with existing non-streaming clients. The StreamingResponse class must "
            "accept an async generator and properly set content-type headers. "
            "Test: pytest tests/. Environment: Python 3.12.\n"
        )
    with open(os.path.join(task_dir, "task.toml"), "w") as f:
        f.write('[task]\nname = "test"\n')
    with open(os.path.join(task_dir, "environment", "Dockerfile"), "w") as f:
        f.write("FROM python:3.12\n")
    with open(os.path.join(task_dir, "tests", "gold_reference_tests.py"), "w") as f:
        f.write("def test_streaming(): pass\ndef test_output(): pass\n")
    with open(os.path.join(task_dir, "tests", "verify_TE1.sh"), "w") as f:
        f.write("#!/bin/bash\necho PASS\n")
    with open(os.path.join(task_dir, "tests", "test_runner.py"), "w") as f:
        f.write("EXPECTED_REF_TESTS = 2\n")
    with open(os.path.join(task_dir, "tests", "test.sh"), "w") as f:
        f.write("#!/bin/bash\n")
    with open(os.path.join(task_dir, "solution", "solve.sh"), "w") as f:
        f.write("#!/bin/bash\necho 'apply solution'\n")
    return task_dir


def test_valid_task_passes(tmp_path):
    task_dir = _make_valid_task(tmp_path)
    errors = validate_task_dir(task_dir)
    assert errors == []


def test_missing_instruction(tmp_path):
    task_dir = _make_valid_task(tmp_path)
    os.remove(os.path.join(task_dir, "instruction.md"))
    errors = validate_task_dir(task_dir)
    assert any("instruction.md" in e for e in errors)


def test_instruction_too_short(tmp_path):
    task_dir = _make_valid_task(tmp_path)
    with open(os.path.join(task_dir, "instruction.md"), "w") as f:
        f.write("Do the thing.\n")
    errors = validate_task_dir(task_dir)
    assert any("too short" in e for e in errors)


def test_instruction_too_long(tmp_path):
    task_dir = _make_valid_task(tmp_path)
    with open(os.path.join(task_dir, "instruction.md"), "w") as f:
        f.write("word " * 250 + "\n")
    errors = validate_task_dir(task_dir)
    assert any("too long" in e for e in errors)


def test_no_gold_tests(tmp_path):
    task_dir = _make_valid_task(tmp_path)
    with open(os.path.join(task_dir, "tests", "gold_reference_tests.py"), "w") as f:
        f.write("# no tests\n")
    errors = validate_task_dir(task_dir)
    assert any("no test functions" in e for e in errors)


def test_missing_expected_ref_tests(tmp_path):
    task_dir = _make_valid_task(tmp_path)
    with open(os.path.join(task_dir, "tests", "test_runner.py"), "w") as f:
        f.write("# no constant\n")
    errors = validate_task_dir(task_dir)
    assert any("EXPECTED_REF_TESTS" in e for e in errors)


def test_template_task_has_build_agent_outputs():
    """The template task contains exactly what the build agent creates.

    The build agent creates instruction.md, task.toml, environment/Dockerfile.
    Everything else (tests/, solution/) is pipeline-generated, so the template
    is intentionally not a fully valid task. This test just checks the template
    directory exists and contains the expected starting files.
    """
    from craft_taskgen.config import PipelineContext

    ctx = PipelineContext()
    template_dir = ctx.template_task_dir
    if os.path.isdir(template_dir):
        for rel_path in ("instruction.md", "task.toml", "environment/Dockerfile"):
            full_path = os.path.join(template_dir, rel_path)
            assert os.path.isfile(full_path), f"Template missing {rel_path}"


def test_task_has_task_toml(tmp_path):
    """validate_task_dir flags missing task.toml."""
    task_dir = _make_valid_task(tmp_path)
    os.remove(os.path.join(task_dir, "task.toml"))
    errors = validate_task_dir(task_dir)
    assert any("task.toml" in e for e in errors)


def test_score_py_template_has_correct_formula():
    """SCORE_PY_TEMPLATE contains all required scoring variables."""
    from craft_taskgen.prompts import SCORE_PY_TEMPLATE

    for marker in ("resolved", "f2p_score", "p2p_score", "reward"):
        assert marker in SCORE_PY_TEMPLATE, f"SCORE_PY_TEMPLATE missing '{marker}'"


# ---------------------------------------------------------------------------
# New format (F2P/P2P) tests
# ---------------------------------------------------------------------------


def _make_valid_new_format_task(tmp_path):
    """Create a minimal valid task directory in the new F2P/P2P format."""
    task_dir = str(tmp_path / "t2v3-TE1-new-feature")
    os.makedirs(os.path.join(task_dir, "environment"))
    os.makedirs(os.path.join(task_dir, "tests"))
    os.makedirs(os.path.join(task_dir, "solution"))

    with open(os.path.join(task_dir, "instruction.md"), "w") as f:
        f.write(
            "Implement the streaming response feature for the API endpoint. "
            "The server should support chunked transfer encoding and yield results "
            "incrementally as they become available. Ensure backwards compatibility "
            "with existing non-streaming clients. The StreamingResponse class must "
            "accept an async generator and properly set content-type headers. "
            "Test: pytest tests/. Environment: Python 3.12.\n"
        )
    with open(os.path.join(task_dir, "task.toml"), "w") as f:
        f.write('[task]\nname = "test"\n')
    with open(os.path.join(task_dir, "environment", "Dockerfile"), "w") as f:
        f.write("FROM python:3.12\n")
    with open(os.path.join(task_dir, "tests", "fail_to_pass.txt"), "w") as f:
        f.write("tests/test_streaming.py::test_chunked_encoding\n")
    with open(os.path.join(task_dir, "tests", "pass_to_pass.txt"), "w") as f:
        f.write("tests/test_streaming.py::test_basic_response\n")
    with open(os.path.join(task_dir, "tests", "score.py"), "w") as f:
        f.write("# scoring script\n")
    with open(os.path.join(task_dir, "tests", "test.sh"), "w") as f:
        f.write("#!/bin/bash\npytest\n")
    with open(os.path.join(task_dir, "solution", "solve.sh"), "w") as f:
        f.write("#!/bin/bash\ngit apply changes.patch\n")
    return task_dir


def test_new_format_valid(tmp_path):
    """New F2P/P2P format passes validation when all files present."""
    task_dir = _make_valid_new_format_task(tmp_path)
    errors = validate_task_dir(task_dir)
    assert errors == []


def test_new_format_empty_f2p(tmp_path):
    """New format fails when fail_to_pass.txt is empty."""
    task_dir = _make_valid_new_format_task(tmp_path)
    with open(os.path.join(task_dir, "tests", "fail_to_pass.txt"), "w") as f:
        f.write("")
    errors = validate_task_dir(task_dir)
    assert any("fail_to_pass.txt is empty" in e for e in errors)


def test_new_format_missing_score_py(tmp_path):
    """New format fails when score.py is missing."""
    task_dir = _make_valid_new_format_task(tmp_path)
    os.remove(os.path.join(task_dir, "tests", "score.py"))
    errors = validate_task_dir(task_dir)
    assert any("score.py" in e for e in errors)


def test_new_format_does_not_require_gold_tests(tmp_path):
    """New format should NOT require gold_reference_tests.py or test_runner.py."""
    task_dir = _make_valid_new_format_task(tmp_path)
    errors = validate_task_dir(task_dir)
    assert not any("gold_reference_tests" in e for e in errors)
    assert not any("test_runner" in e for e in errors)
    assert not any("verify_" in e for e in errors)


def test_new_format_bad_exclusions(tmp_path):
    """f2p_skip.txt lines without :: are flagged."""
    task_dir = _make_valid_new_format_task(tmp_path)
    with open(os.path.join(task_dir, "tests", "f2p_skip.txt"), "w") as f:
        f.write("# comment line\ntests/test_foo.py::test_bar\nbad_line_no_separator\n")
    errors = validate_task_dir(task_dir)
    assert any("f2p_skip.txt line 3" in e for e in errors)
    assert len([e for e in errors if "f2p_skip" in e]) == 1


# ---------------------------------------------------------------------------
# instruction_preamble property tests
# ---------------------------------------------------------------------------


def test_instruction_preamble_returns_first_line(tmp_path):
    from craft_taskgen.config import PipelineContext

    ctx = PipelineContext(templates_dir=str(tmp_path))
    tmpl = tmp_path / "structural_template_instruction.md"
    tmpl.write_text("Solve the following task.\n\n(the task description)")
    assert ctx.instruction_preamble == "Solve the following task."


def test_instruction_preamble_strips_trailing_whitespace(tmp_path):
    from craft_taskgen.config import PipelineContext

    ctx = PipelineContext(templates_dir=str(tmp_path))
    tmpl = tmp_path / "structural_template_instruction.md"
    tmpl.write_text("Solve the following task.   \n\n(the task description)")
    assert ctx.instruction_preamble == "Solve the following task."


def test_instruction_preamble_missing_file_raises(tmp_path):
    import pytest

    from craft_taskgen.config import PipelineContext

    ctx = PipelineContext(templates_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="not found"):
        _ = ctx.instruction_preamble


def test_instruction_preamble_empty_file_raises(tmp_path):
    import pytest

    from craft_taskgen.config import PipelineContext

    ctx = PipelineContext(templates_dir=str(tmp_path))
    (tmp_path / "structural_template_instruction.md").write_text("")
    with pytest.raises(RuntimeError, match="blank"):
        _ = ctx.instruction_preamble


# ---------------------------------------------------------------------------
# Preamble-aware word count in validate_task_dir
# ---------------------------------------------------------------------------


def test_instruction_with_preamble_excludes_preamble_from_word_count(tmp_path):
    """Preamble line is excluded from word count so body words are counted correctly."""
    from craft_taskgen.config import PipelineContext

    task_dir = _make_valid_new_format_task(tmp_path)
    preamble = PipelineContext().instruction_preamble
    body = " ".join(["word"] * 60)
    with open(os.path.join(task_dir, "instruction.md"), "w") as f:
        f.write(f"{preamble}\n\n{body}\n")
    assert validate_task_dir(task_dir) == []


def test_instruction_with_preamble_too_short_body(tmp_path):
    """Word count check applies to body only when preamble is present."""
    from craft_taskgen.config import PipelineContext

    task_dir = _make_valid_new_format_task(tmp_path)
    preamble = PipelineContext().instruction_preamble
    body = " ".join(["word"] * 10)
    with open(os.path.join(task_dir, "instruction.md"), "w") as f:
        f.write(f"{preamble}\n\n{body}\n")
    errors = validate_task_dir(task_dir)
    assert any("too short" in e for e in errors)


# ---------------------------------------------------------------------------
# strip_instruction_boilerplate tests
# ---------------------------------------------------------------------------


def test_strip_boilerplate_no_boilerplate_unchanged(tmp_path):
    """File with no Environment heading is left byte-for-byte identical."""
    f = tmp_path / "instruction.md"
    original = "Solve the following task.\n\nAdd a retry mechanism to the HTTP client.\n"
    f.write_text(original)
    strip_instruction_boilerplate(str(f))
    assert f.read_text() == original


def test_strip_boilerplate_mid_body(tmp_path):
    """Content from the first Environment heading onward is stripped."""
    f = tmp_path / "instruction.md"
    f.write_text("Solve the following task.\n\nAdd a retry mechanism.\n\n## Environment\nFROM python:3.12\n")
    strip_instruction_boilerplate(str(f))
    assert f.read_text() == "Solve the following task.\n\nAdd a retry mechanism.\n"


def test_strip_boilerplate_missing_file_no_exception(tmp_path):
    """Missing file is silently ignored — covers the OSError guard."""
    strip_instruction_boilerplate(str(tmp_path / "nonexistent.md"))


def test_strip_boilerplate_preceded_by_blank_line(tmp_path):
    """Environment heading preceded by a blank line is stripped along with everything after."""
    f = tmp_path / "instruction.md"
    f.write_text("\n## Environment\nFROM python:3.12\n")
    strip_instruction_boilerplate(str(f))
    assert f.read_text() == "\n"


def test_strip_boilerplate_at_byte_zero(tmp_path):
    """Environment heading at byte offset 0 (no preceding newline) is stripped."""
    f = tmp_path / "instruction.md"
    f.write_text("## Environment\nFROM python:3.12\n")
    strip_instruction_boilerplate(str(f))
    assert f.read_text() == "\n"


def test_strip_boilerplate_case_insensitive_and_any_depth(tmp_path):
    """Heading depth and case are both ignored."""
    f = tmp_path / "instruction.md"
    f.write_text("Solve the following task.\n\nDo the thing.\n\n### environment\ndetails\n")
    strip_instruction_boilerplate(str(f))
    assert f.read_text() == "Solve the following task.\n\nDo the thing.\n"
