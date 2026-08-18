# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[1]
ROLLOUTS = REPO / "benchmark" / "test_time_scaling_rollouts.py"
GRADES = REPO / "benchmark" / "test_time_scaling_grades.py"
CONFIG = REPO / "benchmark" / "test_time_scaling_config.py"
HARBOR_AGENT = REPO / "benchmark" / "test_time_scaling_harbor_agent.py"


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    task = root / "task-1"
    task.mkdir(parents=True)
    (root / "switchyard_dataset_manifest.json").write_text("{}")
    (task / "instruction.md").write_text("Fix the task.\n")
    (task / "task.toml").write_text("[agent]\ntimeout_sec = 10\n")
    return root


def test_refined_dataset_prepends_one_shared_context(tmp_path: Path) -> None:
    module = _load(ROLLOUTS, "switchyard_test_time_scaling_rollouts")
    source = _source_dataset(tmp_path)
    requests = [
        {"iteration": 1, "rollout_index": index, "refinement_prompt": "Prior evidence."}
        for index in range(2)
    ]

    prepared = module._prepare_dataset(
        source,
        tmp_path / "method",
        {"id": "task-1", "prompt": "Fix the task."},
        requests,
        1,
    )

    instruction = (prepared / "task-1" / "instruction.md").read_text()
    assert instruction == "Prior evidence.\n\nORIGINAL TASK\nFix the task.\n"
    assert (source / "task-1" / "instruction.md").read_text() == "Fix the task.\n"


def test_public_rollout_excludes_private_grade_data(tmp_path: Path) -> None:
    rollouts = _load(ROLLOUTS, "switchyard_test_time_scaling_rollouts_public")
    grades = _load(GRADES, "switchyard_test_time_scaling_grades")
    trial = tmp_path / "trial-a"
    (trial / "agent").mkdir(parents=True)
    (trial / "artifacts").mkdir()
    (trial / "verifier").mkdir()
    (trial / "agent" / "mini-swe-agent.trajectory.json").write_text(
        json.dumps({"messages": [{"role": "assistant", "content": "changed code"}]})
    )
    (trial / "artifacts" / "patch.diff").write_text("diff --git a/a b/a\n")
    (trial / "verifier" / "reward.txt").write_text("1\n")

    public = rollouts._public_rollout(
        "task-1",
        "model-1",
        {"iteration": 0, "rollout_index": 0},
        trial,
    )

    encoded = json.dumps(public)
    assert "reward" not in encoded
    assert "verifier" not in encoded
    assert str(trial) not in encoded
    assert public["output_digest"].startswith("sha256:")
    assert grades._passed(trial) is True

    grade_root = tmp_path / "private-grades"
    rollouts._write_private_grade_map(grade_root, 0, [(public, trial)])
    mapping_text = (grade_root / "iteration-0.json").read_text()
    assert "trial_dir" not in mapping_text
    assert "reward" not in mapping_text
    mapping = json.loads(mapping_text)
    assert Path(mapping[0]["patch_file"]).read_text() == "diff --git a/a b/a\n"


def test_rollout_disables_verification(tmp_path: Path) -> None:
    module = _load(ROLLOUTS, "switchyard_test_time_scaling_rollouts_command")
    method_root = tmp_path / "method"
    config = {
        "repo_root": str(REPO),
        "method_root": str(method_root),
        "run_baseline": str(REPO / "benchmark" / "run-baseline.sh"),
        "model_id": "model-1",
        "agent_import_path": "module:Agent",
        "harbor_model": "openai/model-1",
        "n_concurrent": 1,
        "max_retries": 1,
        "agent_timeout_multiplier": 1.0,
        "upstream_base_url": "https://example.test/v1",
        "upstream_api_key_env": "TEST_KEY",
    }

    command = module._harbor_command(config, tmp_path / "dataset", "task-1", 2, 0)

    assert "--disable-verification" in command
    assert command[command.index("--harbor-extra") + 1] == "-k"


def test_config_writer_uses_paper_defaults(tmp_path: Path) -> None:
    module = _load(CONFIG, "switchyard_test_time_scaling_config")
    dataset = _source_dataset(tmp_path)
    output = tmp_path / "output"
    args = module._parser().parse_args(
        [
            "--dataset",
            str(dataset),
            "--task",
            "task-1",
            "--model",
            "azure/anthropic/claude-sonnet-4-5",
            "--output",
            str(output),
        ]
    )
    runner_path, harbor_path = module.write_configs(args, REPO)

    runner = json.loads(runner_path.read_text())
    harbor = json.loads(harbor_path.read_text())
    assert runner["scaling"] == {
        "rollout_count": 16,
        "refinement_count": 4,
        "group_size": 2,
        "votes_per_group": 8,
        "seed": 0,
        "pairing_order": "in_order",
        "display_order": "in_order",
        "tie_policy": "first_in_group",
        "invalid_vote_policy": {"mode": "abort"},
    }
    assert len(runner["manifest"]["fields"]) == 22
    assert runner["manifest"]["fields"]["ablation_fixed_settings"]["source"] == "paper"
    assert runner["evaluation_command"]["argv"][-2:] == ["--config", str(harbor_path)]
    assert harbor["harbor_model"] == "openai/azure/anthropic/claude-sonnet-4-5"
    assert harbor["run_record"] == str(output / "method" / "run.json")


def test_saved_patch_includes_new_files() -> None:
    module = _load(HARBOR_AGENT, "switchyard_test_time_scaling_harbor_agent")

    assert "git -C /testbed add -A" in module.PATCH_COMMAND
    assert "diff --cached --binary" in module.PATCH_COMMAND
