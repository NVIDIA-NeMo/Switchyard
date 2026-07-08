# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No-network tests for the container-free TrialQA orchestrator."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import signal
import stat
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from benchmark.trialqa_local_dataset import TRIALQA_SCHEMA, TrialQAMode

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "benchmark" / "trialqa_local_demo.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("switchyard_trialqa_local_demo", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _openai_transport(
    total_requests: int,
    *,
    retries: int = 0,
    charges: int = 0,
    unpriced: int = 0,
    prompt: int = 0,
    completion: int = 0,
) -> dict[str, object]:
    return {
        "physical_attempts": total_requests + retries,
        "null_eof_retries": retries,
        "retry_usage_charges": charges,
        "unpriced_null_eof_retries": unpriced,
        "retry_token_sensitivity": {
            "prompt": prompt,
            "completion": completion,
            "cached": 0,
            "cache_creation": 0,
            "reasoning": 0,
            "total": prompt + completion,
        },
    }


def _dataset(module: ModuleType, tmp_path: Path) -> object:
    parquet = tmp_path / "source" / "trialqa" / "train-00000-of-00001.parquet"
    parquet.parent.mkdir(parents=True)
    parquet.write_bytes(b"synthetic pinned gold artifact")
    digest = hashlib.sha256(parquet.read_bytes()).hexdigest()
    rows = []
    for index in range(120):
        rows.append(
            module.TrialQARow(
                dataset_row_index=index,
                id=f"trial-{index:03d}",
                tag="trialqa",
                version="1.0",
                question=f"What is the eligibility threshold for trial {index:03d}?",
                ideal=f"The eligibility threshold answer is exactly value {index:03d}.",
                files="",
                sources=(f"https://registry.invalid/trial/{index:03d}",),
                key_passage=f"Private gold passage for synthetic trial number {index:03d}.",
                canary="",
                is_opensource=True,
                ground_truth=True,
                prompt_suffix="Give the threshold and units.",
                trialqa_type="registry",
                mode=TrialQAMode(file=False, retrieve=True, inject=False),
                validator_params="{}",
                answer_regex="",
            )
        )
    return module.TrialQADataset(
        path=parquet,
        revision=module.TRIALQA_DATASET_REVISION,
        parquet_sha256=digest,
        rows=tuple(rows),
    )


def _prospective_dataset(module: ModuleType, tmp_path: Path, *, row_count: int = 8) -> object:
    parquet = tmp_path / "prospective" / "trialqa-ctgov-prospective-v1.parquet"
    parquet.parent.mkdir(parents=True)
    rows = []
    for index in range(row_count):
        rows.append(
            {
                "id": f"NCT9{index:07d}",
                "tag": "trialqa",
                "version": "1.0",
                "question": f"What is the primary outcome measure for trial {index:03d}?",
                "ideal": f"The primary outcome measure is prospective value {index:03d}.",
                "files": "",
                "sources": [f"https://clinicaltrials.gov/study/NCT9{index:07d}"],
                "key_passage": f"Prospective metadata passage for trial {index:03d}.",
                "canary": "",
                "is_opensource": True,
                "ground_truth": True,
                "prompt_suffix": "Answer with the exact field value.",
                "type": "registry",
                "mode": {"file": False, "retrieve": True, "inject": False},
                "validator_params": "{}",
                "answer_regex": "",
            }
        )
    table = pa.Table.from_pylist(rows, schema=TRIALQA_SCHEMA)
    pq.write_table(table, parquet)
    digest = hashlib.sha256(parquet.read_bytes()).hexdigest()
    return module.load_trialqa_compatible_parquet(
        parquet,
        expected_sha256=digest,
        expected_row_count=row_count,
        revision="clinicaltrials-gov-prospective-v1",
    )


def _prospective_population_report(dataset: object, path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "switchyard.trialqa_prospective_population.v1",
                "status": "passed",
                "population": {
                    "kind": "trialqa-compatible-clinicaltrials-gov-prospective",
                    "official_labbench2": False,
                    "sha256": dataset.parquet_sha256,
                    "row_count": len(dataset.rows),
                },
                "official_trialqa_exclusion": {
                    "selected_ncts_overlap_official_trialqa": [],
                },
                "use_constraints": {
                    "model_calls": 0,
                    "must_not_be_reported_as_official_labbench2_trialqa": True,
                    "performance_eligible_only_if_manifest_is_frozen_before_generation": True,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _candidate(module: ModuleType, root: Path) -> object:
    skill = (
        b"---\n"
        b"name: tooluniverse-trialqa\n"
        b"description: Answer TrialQA questions with targeted ToolUniverse evidence.\n"
        b"---\n\n"
        b"# TrialQA\n\nSearch trial registries before answering.\n"
    )
    skill_path = root / "tooluniverse-trialqa" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_bytes(skill)
    manifest = {
        "schema_version": 1,
        "candidate_id": "candidate-synthetic",
        "validation": {"status": "passed"},
        "skills": [
            {
                "path": "tooluniverse-trialqa/SKILL.md",
                "sha256": hashlib.sha256(skill).hexdigest(),
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    return module.validate_candidate_skill(root, module.NAMESPACE)


def test_trial_prompt_requires_compact_execute_contract(tmp_path: Path) -> None:
    module = _load()
    row = _dataset(module, tmp_path).rows[0]

    prompt = module.render_trial_prompt(row)

    assert "trialqa_load_active_skill" in prompt
    assert "compact meta-tools" in prompt
    assert "only through `execute_tool`" in prompt
    assert "never call a raw `ClinicalTrials_*` tool directly" in prompt
    assert "explicit stop conditions, call bounds, and `never` rules" in prompt
    assert "advisory and non-exhaustive" not in prompt
    assert "explicitly identifies the field" in prompt
    assert "another relevant read-only evidence slice" in prompt
    assert "never guess a specific value" in prompt
    assert "available `trialqa_*` MCP tools" not in prompt


def _profile(module: ModuleType, path: Path) -> Path:
    path.write_text(
        "defaults:\n"
        "  api_key: test\n"
        "  base_url: https://invalid.example/v1\n"
        "routes:\n"
        "  sd-executor:\n"
        "    type: model\n"
        "    target:\n"
        f"      model: {module.EXECUTOR_MODEL}\n"
        "      format: openai\n"
        "  sd-distiller:\n"
        "    type: model\n"
        "    target:\n"
        f"      model: {module.JUDGE_MODEL}\n"
        "      format: openai\n"
        "  sd-judge:\n"
        "    type: model\n"
        "    target:\n"
        f"      model: {module.JUDGE_MODEL}\n"
        "      format: openai\n",
        encoding="utf-8",
    )
    return path


def _tooluniverse(root: Path) -> Path:
    binary = _write_executable(root / "bin" / "tooluniverse-smcp-stdio")
    _write_executable(root / "bin" / "python")
    metadata = (
        root / "lib" / "python3.12" / "site-packages" / "tooluniverse-1.1.11.dist-info" / "METADATA"
    )
    metadata.parent.mkdir(parents=True)
    metadata.write_text("Name: tooluniverse\nVersion: 1.1.11\n", encoding="utf-8")
    return binary


def _namespace_translation_attestation(module: ModuleType) -> dict[str, object]:
    flattened = "__sy1n17_mcp__tooluniversetrialqa_load_active_skill"
    return {
        "schema_version": "switchyard.trialqa_namespace_translation.v1",
        "source_format": "openai_responses",
        "target_format": "openai_chat",
        "namespace": "mcp__tooluniverse",
        "child_name": "trialqa_load_active_skill",
        "flattened_name": flattened,
        "flattened_name_sha256": module._sha256_bytes(flattened.encode("ascii")),
        "request_flattened_tool_count": 1,
        "response_namespace": "mcp__tooluniverse",
        "response_child_name": "trialqa_load_active_skill",
        "response_call_id": "trialqa-doctor-call",
        "model_calls": 0,
    }


class _FakeNamespaceTranslationEngine:
    flattened_name = "__sy1n17_mcp__tooluniversetrialqa_load_active_skill"

    def __init__(self, *, response_namespace: object = "mcp__tooluniverse") -> None:
        self.response_namespace = response_namespace
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def translate_request(
        self,
        source: str,
        target: str,
        body: dict[str, object],
    ) -> dict[str, object]:
        self.calls.append((source, target, body))
        namespace = body["tools"][0]
        assert namespace["name"] == "mcp__tooluniverse"
        assert namespace["tools"][0]["name"] == "trialqa_load_active_skill"
        return {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": self.flattened_name,
                        "parameters": {"type": "object"},
                    },
                }
            ]
        }

    def translate_response(
        self,
        source: str,
        target: str,
        body: dict[str, object],
    ) -> dict[str, object]:
        self.calls.append((source, target, body))
        function = body["choices"][0]["message"]["tool_calls"][0]["function"]
        assert function["name"] == self.flattened_name
        return {
            "output": [
                {
                    "type": "function_call",
                    "namespace": self.response_namespace,
                    "name": "trialqa_load_active_skill",
                    "call_id": "trialqa-doctor-call",
                    "arguments": "{}",
                }
            ]
        }


def _runtime(module: ModuleType, tmp_path: Path) -> dict[str, Path]:
    runtime = {
        "switchyard": _write_executable(tmp_path / "bin" / "switchyard"),
        "codex": _write_executable(tmp_path / "bin" / "codex"),
        "tooluniverse": _tooluniverse(tmp_path / "tooluniverse-venv"),
        "profile": _profile(module, tmp_path / "profile.yaml"),
    }
    doctor_report = tmp_path / "doctor-report.json"
    doctor_report.write_text(
        json.dumps(
            {
                "schema_version": module.DOCTOR_SCHEMA_VERSION,
                "status": "passed",
                "model_calls": 0,
                "dataset": {
                    "id": module.TRIALQA_DATASET_ID,
                    "config": module.TRIALQA_DATASET_CONFIG,
                    "revision": module.TRIALQA_DATASET_REVISION,
                    "parquet_sha256": module.TRIALQA_PARQUET_SHA256,
                    "row_count": 120,
                    "split_counts": {"train": 24, "test": 96},
                },
                "routing": {
                    "first_route": module.EXECUTOR_ROUTE,
                    "executor_model": module.EXECUTOR_MODEL,
                    "judge_route": module.JUDGE_ROUTE,
                    "judge_model": module.JUDGE_MODEL,
                },
                "implementation": {
                    "source_sha256": module._execution_source_sha256(),
                },
                "mcp_adapter": module._adapter_schema_attestation(),
                "codex_safety": module._codex_safety_attestation(),
                "namespace_translation": _namespace_translation_attestation(module),
                "runtime_artifacts": {
                    "switchyard": module._runtime_binary_attestation(
                        runtime["switchyard"], "Switchyard binary"
                    ),
                    "codex": module._runtime_binary_attestation(runtime["codex"], "Codex binary"),
                    "switchyard_rust_native_extension": (module._native_extension_attestation()),
                    "tooluniverse": {
                        **module._runtime_binary_attestation(
                            runtime["tooluniverse"], "ToolUniverse binary"
                        ),
                        "version": module.TOOLUNIVERSE_VERSION,
                        "python": module._runtime_binary_attestation(
                            runtime["tooluniverse"].parent / "python",
                            "ToolUniverse venv Python",
                        ),
                    },
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    runtime["doctor_report"] = doctor_report
    return runtime


def test_manifests_enforce_candidate_free_donor_then_exact_paired_evaluation(
    tmp_path: Path,
) -> None:
    module = _load()
    dataset = _dataset(module, tmp_path)
    split = module.create_split_manifest(dataset)
    candidate = _candidate(module, tmp_path / "candidate")
    runtime = _runtime(module, tmp_path)
    profile = runtime["profile"]

    donor = module.build_experiment_manifest(
        dataset=dataset,
        split_manifest=split,
        kind="donor",
        candidate=None,
        routing_profile=profile,
        switchyard_bin=runtime["switchyard"],
        codex_bin=runtime["codex"],
        tooluniverse_bin=runtime["tooluniverse"],
        doctor_report=runtime["doctor_report"],
    )
    development = module.build_experiment_manifest(
        dataset=dataset,
        split_manifest=split,
        kind="development",
        candidate=candidate,
        routing_profile=profile,
        switchyard_bin=runtime["switchyard"],
        codex_bin=runtime["codex"],
        tooluniverse_bin=runtime["tooluniverse"],
        doctor_report=runtime["doctor_report"],
    )
    pilot = module.build_experiment_manifest(
        dataset=dataset,
        split_manifest=split,
        kind="pilot",
        candidate=candidate,
        routing_profile=profile,
        switchyard_bin=runtime["switchyard"],
        codex_bin=runtime["codex"],
        tooluniverse_bin=runtime["tooluniverse"],
        doctor_report=runtime["doctor_report"],
    )
    full = module.build_experiment_manifest(
        dataset=dataset,
        split_manifest=split,
        kind="full",
        candidate=candidate,
        routing_profile=profile,
        switchyard_bin=runtime["switchyard"],
        codex_bin=runtime["codex"],
        tooluniverse_bin=runtime["tooluniverse"],
        doctor_report=runtime["doctor_report"],
    )
    primary = module.build_experiment_manifest(
        dataset=dataset,
        split_manifest=split,
        kind="full",
        candidate=candidate,
        routing_profile=profile,
        switchyard_bin=runtime["switchyard"],
        codex_bin=runtime["codex"],
        tooluniverse_bin=runtime["tooluniverse"],
        doctor_report=runtime["doctor_report"],
        primary_question_start=module.PRIMARY_HELDOUT_QUESTION_START,
        primary_question_count=module.PRIMARY_HELDOUT_QUESTION_COUNT,
    )

    assert donor["candidate"] is None
    assert len(donor["tasks"]) == 24 * 5
    assert {task["condition"] for task in donor["tasks"]} == {"donor"}
    assert len(development["tasks"]) == 24 * 5
    assert {task["condition"] for task in development["tasks"]} == {"treatment"}
    assert {task["partition"] for task in development["tasks"]} == {"train"}
    assert {task["phase"] for task in development["tasks"]} == {"development"}
    assert development["protocol"]["performance_eligible"] is False
    assert development["protocol"]["control_design"] == "historical-cached-donor"
    for manifest in (donor, development, pilot, full, primary):
        assert manifest["protocol"]["max_generation_concurrency"] == 4
        assert module.manifest_max_generation_concurrency(manifest) == 4
    assert [task["pair_id"] for task in development["tasks"]] == [
        task["pair_id"] for task in donor["tasks"]
    ]
    assert len(pilot["tasks"]) == 2
    assert len(full["tasks"]) == 96 * 5 * 2
    assert full["protocol"]["performance_eligible"] is False
    assert full["protocol"]["primary_evaluation_scope"] is None
    assert full["protocol"]["heldout_quarantine"] is None
    assert {task["condition"] for task in full["tasks"]} == {
        "baseline",
        "treatment",
    }
    assignments = module.validate_split_manifest(dataset, split)
    heldout_rows = [row for row in dataset.rows if assignments[row.id] == "test"]
    assert len(primary["tasks"]) == 8 * 5 * 2
    assert primary["protocol"]["performance_eligible"] is True
    primary_scope = primary["protocol"]["primary_evaluation_scope"]
    assert primary_scope == {
        "question_start": 88,
        "question_count": 8,
        "repeat_count": 5,
        "task_count": 80,
        "question_group_keys_sha256": module._sha256_bytes(
            module._canonical_json([module.question_group_key(row) for row in heldout_rows[88:]])
        ),
    }
    quarantine = primary["protocol"]["heldout_quarantine"]
    assert quarantine["question_start"] == 0
    assert quarantine["question_count"] == 88
    assert quarantine["disposition"] == "excluded-exposed-heldout"
    assert primary["tasks"][0]["dataset_row_index"] == heldout_rows[88].dataset_row_index
    assert primary["tasks"][0]["row_id"] == heldout_rows[88].id
    assert primary["tasks"][-1]["row_id"] == heldout_rows[-1].id
    assert quarantine["question_group_keys_sha256"] == module._sha256_bytes(
        module._canonical_json([module.question_group_key(row) for row in heldout_rows[:88]])
    )
    heldout_ordering = primary["dataset"]["heldout_ordering"]
    assert heldout_ordering["question_group_keys"] == [
        module.question_group_key(row) for row in heldout_rows
    ]
    assert heldout_ordering["question_group_keys_sha256"] == module._sha256_bytes(
        module._canonical_json(heldout_ordering["question_group_keys"])
    )
    assert module.primary_evaluation_window(primary) == (88, 8)
    assert module.primary_evaluation_window(full) == (None, None)
    assert full["protocol"]["arm_order_policy"] == ("deterministic-balanced-crossover-v1")
    assert donor["protocol"]["arm_order_policy"] == "manifest-order-single-arm-v1"
    assert full["protocol"]["recovered_codex_error_events"] == ("telemetry-before-turn.completed")
    assert all("question" not in task and "ideal" not in task for task in full["tasks"])
    assert donor["manifest_id"] != full["manifest_id"]
    assert donor["manifest_id"] != development["manifest_id"]
    assert set(donor["implementation"]["source_sha256"]) == {
        "benchmark/trialqa_local_dataset.py",
        "benchmark/trialqa_tooluniverse_mcp.py",
        "benchmark/trialqa_local_batch.py",
        "benchmark/trialqa_local_gate.py",
        "benchmark/trialqa_local_regression.py",
        "benchmark/trialqa_local_search_gate.py",
        "benchmark/trialqa_local_runner.py",
        "benchmark/trialqa_local_demo.py",
        "crates/switchyard-components/src/lib.rs",
        "crates/switchyard-components/src/backends/openai.rs",
        "crates/switchyard-components/src/backends/stats.rs",
        "crates/switchyard-components/src/stats/accumulator.rs",
        "crates/switchyard-components/src/stats/mod.rs",
        "crates/switchyard-translation/src/codecs/responses/buffered.rs",
        "crates/switchyard-translation/src/codecs/responses/stream.rs",
        "crates/switchyard-translation/src/codecs/stream.rs",
        "crates/switchyard-translation/src/lib.rs",
        "crates/switchyard-translation/src/namespace_tools.rs",
        "switchyard/cli/launchers/codex_cli_launcher.py",
        "switchyard/cli/launchers/skill_distillation.py",
        "switchyard/lib/skill_distillation_native.py",
        "switchyard/lib/skill_distillation_store.py",
    }
    assert all(
        value.startswith("sha256:") and len(value) == 71
        for value in donor["implementation"]["source_sha256"].values()
    )
    assert donor["runtime"]["codex"]["resolved_path"] == str(runtime["codex"].resolve())
    assert donor["runtime"]["tooluniverse"]["version"] == "1.1.11"

    for invalid_value in (None, True, 3, 5):
        invalid = {**full, "protocol": dict(full["protocol"])}
        if invalid_value is None:
            del invalid["protocol"]["max_generation_concurrency"]
        else:
            invalid["protocol"]["max_generation_concurrency"] = invalid_value
        with pytest.raises(
            module.TrialQADemoError,
            match="max_generation_concurrency must be 4",
        ):
            module.validate_manifest_pairing(invalid)

    with pytest.raises(module.TrialQADemoError, match="must not reference"):
        module.build_experiment_manifest(
            dataset=dataset,
            split_manifest=split,
            kind="donor",
            candidate=candidate,
            routing_profile=profile,
            switchyard_bin=runtime["switchyard"],
            codex_bin=runtime["codex"],
            tooluniverse_bin=runtime["tooluniverse"],
            doctor_report=runtime["doctor_report"],
        )

    with pytest.raises(module.TrialQADemoError, match="start at or after held-out ordinal 88"):
        module.build_experiment_manifest(
            dataset=dataset,
            split_manifest=split,
            kind="full",
            candidate=candidate,
            routing_profile=profile,
            switchyard_bin=runtime["switchyard"],
            codex_bin=runtime["codex"],
            tooluniverse_bin=runtime["tooluniverse"],
            doctor_report=runtime["doctor_report"],
            primary_question_start=87,
            primary_question_count=9,
        )

    stale = json.loads(runtime["doctor_report"].read_text(encoding="utf-8"))
    stale["implementation"]["source_sha256"]["benchmark/trialqa_local_demo.py"] = (
        "sha256:" + "0" * 64
    )
    runtime["doctor_report"].write_text(json.dumps(stale) + "\n", encoding="utf-8")
    with pytest.raises(module.TrialQADemoError, match="stale"):
        module.build_experiment_manifest(
            dataset=dataset,
            split_manifest=split,
            kind="donor",
            candidate=None,
            routing_profile=profile,
            switchyard_bin=runtime["switchyard"],
            codex_bin=runtime["codex"],
            tooluniverse_bin=runtime["tooluniverse"],
            doctor_report=runtime["doctor_report"],
        )


def test_prospective_canary_manifest_is_non_official_and_reproducible(
    tmp_path: Path,
) -> None:
    module = _load()
    dataset = _prospective_dataset(module, tmp_path)
    candidate = _candidate(module, tmp_path / "candidate")
    runtime = _runtime(module, tmp_path)
    population_report = _prospective_population_report(
        dataset,
        tmp_path / "prospective-population-report.json",
    )

    manifest = module.build_prospective_experiment_manifest(
        dataset=dataset,
        population_report=population_report,
        candidate=candidate,
        routing_profile=runtime["profile"],
        switchyard_bin=runtime["switchyard"],
        codex_bin=runtime["codex"],
        tooluniverse_bin=runtime["tooluniverse"],
        doctor_report=runtime["doctor_report"],
    )

    assert manifest["kind"] == "full"
    assert manifest["dataset"]["official_labbench2"] is False
    assert manifest["dataset"]["id"] == "trialqa-compatible-prospective"
    assert manifest["dataset"]["test_count"] == 8
    assert manifest["dataset"]["train_count"] == 0
    assert len(manifest["tasks"]) == 8 * 5 * 2
    assert {task["partition"] for task in manifest["tasks"]} == {"test"}
    assert manifest["protocol"]["primary_evaluation_scope"] == {
        "question_start": 0,
        "question_count": 8,
        "repeat_count": 5,
        "task_count": 8 * 5 * 2,
        "question_group_keys_sha256": manifest["dataset"]["heldout_ordering"][
            "question_group_keys_sha256"
        ],
    }
    assert manifest["protocol"]["heldout_quarantine"] == {
        "question_start": 0,
        "question_count": 0,
        "disposition": "none-new-prospective-population",
        "question_group_keys_sha256": module._sha256_bytes(module._canonical_json([])),
    }

    loaded = module.load_manifest_dataset(dataset.path, manifest)
    split = module.create_manifest_split(loaded, manifest)
    assert set(module.validate_all_test_split_manifest(loaded, split).values()) == {"test"}
    expected = module.build_reproducible_manifest_from_supplied(
        supplied=manifest,
        dataset=loaded,
        split_manifest=split,
        candidate=candidate,
        routing_profile=runtime["profile"],
        switchyard_bin=runtime["switchyard"],
        codex_bin=runtime["codex"],
        tooluniverse_bin=runtime["tooluniverse"],
        doctor_report=runtime["doctor_report"],
        population_report=population_report,
    )

    assert expected == manifest

    with pytest.raises(module.TrialQADemoError, match="require --population-report"):
        module.build_reproducible_manifest_from_supplied(
            supplied=manifest,
            dataset=loaded,
            split_manifest=split,
            candidate=candidate,
            routing_profile=runtime["profile"],
            switchyard_bin=runtime["switchyard"],
            codex_bin=runtime["codex"],
            tooluniverse_bin=runtime["tooluniverse"],
            doctor_report=runtime["doctor_report"],
        )

    invalid = json.loads(json.dumps(manifest))
    invalid["protocol"]["heldout_quarantine"]["disposition"] = "excluded-exposed-heldout"
    with pytest.raises(module.TrialQADemoError, match="quarantine attestation"):
        module.validate_manifest_pairing(invalid)


def test_later_primary_scope_is_a_dynamically_validated_contiguous_suffix(
    tmp_path: Path,
) -> None:
    module = _load()
    dataset = _dataset(module, tmp_path)
    split = module.create_split_manifest(dataset)
    assignments = module.validate_split_manifest(dataset, split)
    candidate = _candidate(module, tmp_path / "candidate")
    runtime = _runtime(module, tmp_path)
    manifest = module.build_experiment_manifest(
        dataset=dataset,
        split_manifest=split,
        kind="full",
        candidate=candidate,
        routing_profile=runtime["profile"],
        switchyard_bin=runtime["switchyard"],
        codex_bin=runtime["codex"],
        tooluniverse_bin=runtime["tooluniverse"],
        doctor_report=runtime["doctor_report"],
        primary_question_start=89,
        primary_question_count=7,
    )
    heldout = [row for row in dataset.rows if assignments[row.id] == "test"]

    assert manifest["protocol"]["primary_evaluation_scope"] == {
        "question_start": 89,
        "question_count": 7,
        "repeat_count": 5,
        "task_count": 70,
        "question_group_keys_sha256": module._sha256_bytes(
            module._canonical_json([module.question_group_key(row) for row in heldout[89:]])
        ),
    }
    quarantine = manifest["protocol"]["heldout_quarantine"]
    assert quarantine["question_start"] == 0
    assert quarantine["question_count"] == 89
    assert quarantine["question_group_keys_sha256"] == module._sha256_bytes(
        module._canonical_json([module.question_group_key(row) for row in heldout[:89]])
    )
    assert len(manifest["tasks"]) == 70
    assert manifest["tasks"][0]["row_id"] == heldout[89].id
    assert manifest["tasks"][-1]["row_id"] == heldout[-1].id
    assert module.primary_evaluation_window(manifest) == (89, 7)
    module.validate_manifest_pairing(manifest)

    final_suffix = module.build_experiment_manifest(
        dataset=dataset,
        split_manifest=split,
        kind="full",
        candidate=candidate,
        routing_profile=runtime["profile"],
        switchyard_bin=runtime["switchyard"],
        codex_bin=runtime["codex"],
        tooluniverse_bin=runtime["tooluniverse"],
        doctor_report=runtime["doctor_report"],
        primary_question_start=95,
        primary_question_count=1,
    )
    assert len(final_suffix["tasks"]) == 10
    assert module.primary_evaluation_window(final_suffix) == (95, 1)
    assert {task["row_id"] for task in final_suffix["tasks"]} == {heldout[95].id}

    def clone() -> dict[str, object]:
        return json.loads(json.dumps(manifest))

    for start, count, task_count, match in (
        (87, 9, 90, "start at or after"),
        (96, 0, 0, "nonempty"),
        (89, 6, 60, "contiguous suffix"),
        (89, 8, 80, "contiguous suffix"),
    ):
        invalid = clone()
        invalid["protocol"]["primary_evaluation_scope"] = {
            "question_start": start,
            "question_count": count,
            "repeat_count": 5,
            "task_count": task_count,
            "question_group_keys_sha256": manifest["protocol"]["primary_evaluation_scope"][
                "question_group_keys_sha256"
            ],
        }
        with pytest.raises(module.TrialQADemoError, match=match):
            module.validate_manifest_pairing(invalid)

    for quarantine_count in (88, 90):
        invalid = clone()
        invalid["protocol"]["heldout_quarantine"]["question_count"] = quarantine_count
        with pytest.raises(module.TrialQADemoError, match="quarantine attestation"):
            module.validate_manifest_pairing(invalid)

    invalid = clone()
    invalid["protocol"]["primary_evaluation_scope"]["task_count"] = 69
    with pytest.raises(module.TrialQADemoError, match="primary evaluation scope"):
        module.validate_manifest_pairing(invalid)

    invalid = clone()
    invalid["tasks"] = invalid["tasks"][:-2]
    with pytest.raises(module.TrialQADemoError, match="repeat coverage"):
        module.validate_manifest_pairing(invalid)

    invalid = clone()
    invalid["protocol"]["heldout_quarantine"]["question_group_keys_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(module.TrialQADemoError, match="quarantine attestation"):
        module.validate_manifest_pairing(invalid)

    invalid = clone()
    invalid["dataset"]["heldout_ordering"]["question_group_keys_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(module.TrialQADemoError, match="ordering digest"):
        module.validate_manifest_pairing(invalid)

    invalid = clone()
    invalid["protocol"]["primary_evaluation_scope"]["question_group_keys_sha256"] = (
        "sha256:" + "0" * 64
    )
    with pytest.raises(module.TrialQADemoError, match="primary evaluation scope"):
        module.validate_manifest_pairing(invalid)

    invalid = clone()
    del invalid["tasks"][8:10]
    duplicate_pair = json.loads(json.dumps(invalid["tasks"][8:10]))
    for task in duplicate_pair:
        task["task_id"] += "-duplicate"
    invalid["tasks"].extend(duplicate_pair)
    with pytest.raises(module.TrialQADemoError, match="task identity is inconsistent"):
        module.validate_manifest_pairing(invalid)

    invalid = clone()
    invalid["tasks"][-10:] = [
        module._manifest_task(
            heldout[0],
            condition=condition,
            partition="test",
            repeat_index=repeat,
            n_repeats=module.FULL_REPEATS,
        )
        for repeat in range(1, module.FULL_REPEATS + 1)
        for condition in ("baseline", "treatment")
    ]
    with pytest.raises(module.TrialQADemoError, match="quarantine attestation"):
        module.validate_manifest_pairing(invalid)

    invalid = clone()
    invalid["manifest_id"] = "trialqa-full-" + "0" * 20
    with pytest.raises(module.TrialQADemoError, match="canonical contents"):
        module.validate_manifest_pairing(invalid)


@pytest.mark.parametrize("start", [8, 23, 87])
def test_primary_scope_rejects_every_exposed_prefix_start(start: int) -> None:
    module = _load()

    with pytest.raises(module.TrialQADemoError, match="start at or after held-out ordinal 88"):
        module._validate_primary_question_suffix(start, module.SERGEI_TEST_COUNT - start)


def test_live_run_rejects_manifest_tampering_before_capture_or_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()
    dataset = _dataset(module, tmp_path)
    split = module.create_split_manifest(dataset)
    candidate_root = tmp_path / "candidate"
    candidate = _candidate(module, candidate_root)
    runtime = _runtime(module, tmp_path)
    manifest = module.build_experiment_manifest(
        dataset=dataset,
        split_manifest=split,
        kind="full",
        candidate=candidate,
        routing_profile=runtime["profile"],
        switchyard_bin=runtime["switchyard"],
        codex_bin=runtime["codex"],
        tooluniverse_bin=runtime["tooluniverse"],
        doctor_report=runtime["doctor_report"],
        primary_question_start=88,
        primary_question_count=8,
    )
    manifest["protocol"]["heldout_quarantine"]["question_group_keys_sha256"] = "sha256:" + "0" * 64
    manifest_path = tmp_path / "tampered-manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    experiment_root = tmp_path / "experiment"
    monkeypatch.setattr(module, "load_pinned_trialqa_parquet", lambda _path: dataset)
    monkeypatch.setattr(
        module,
        "prepare_generation",
        lambda **_kwargs: pytest.fail("generation preparation must not run"),
    )
    monkeypatch.setattr(
        module,
        "execute_generation",
        lambda **_kwargs: pytest.fail("executor must not run"),
    )

    result = module.main(
        [
            "run-one",
            "--dataset",
            str(dataset.path),
            "--experiment-root",
            str(experiment_root),
            "--candidate-root",
            str(candidate_root),
            "--switchyard-bin",
            str(runtime["switchyard"]),
            "--codex-bin",
            str(runtime["codex"]),
            "--tooluniverse-bin",
            str(runtime["tooluniverse"]),
            "--routing-profile",
            str(runtime["profile"]),
            "--manifest",
            str(manifest_path),
            "--doctor-report",
            str(runtime["doctor_report"]),
            "--task-id",
            manifest["tasks"][0]["task_id"],
            "--yes-spend",
        ]
    )

    assert result == 2
    assert not experiment_root.exists()


def test_donor_preparation_has_no_candidate_or_project_skill(tmp_path: Path) -> None:
    module = _load()
    dataset = _dataset(module, tmp_path)
    split = module.create_split_manifest(dataset)
    runtime = _runtime(module, tmp_path)
    manifest = module.build_experiment_manifest(
        dataset=dataset,
        split_manifest=split,
        kind="donor",
        candidate=None,
        routing_profile=runtime["profile"],
        switchyard_bin=runtime["switchyard"],
        codex_bin=runtime["codex"],
        tooluniverse_bin=runtime["tooluniverse"],
        doctor_report=runtime["doctor_report"],
    )
    first = manifest["tasks"][0]

    planned = module.prepare_generation(
        manifest=manifest,
        task_id=first["task_id"],
        dataset=dataset,
        split_manifest=split,
        capture_cwd=tmp_path / "capture",
        candidate_root=None,
        switchyard_bin=runtime["switchyard"],
        codex_bin=runtime["codex"],
        routing_profile=runtime["profile"],
        tooluniverse_bin=runtime["tooluniverse"],
    )

    assert planned.spec.arm == "baseline"
    assert list((planned.pair.baseline.root / ".agents" / "skills").iterdir()) == []
    assert list((planned.pair.treatment.root / ".agents" / "skills").iterdir()) == []
    assert planned.pair.candidate.skill_path.name == "NO_SKILL"
    assert not list(planned.pair.runtime_root.rglob("SKILL.md"))
    bootstrap = json.loads((planned.pair.candidate.candidate_root / "attestation.json").read_text())
    assert bootstrap == {
        "schema_version": module.SCHEMA_VERSION,
        "loaded": False,
        "candidate_id": None,
    }
    assert planned.spec.cwd == (tmp_path / "capture").resolve()
    assert planned.spec.executor_model == module.EXECUTOR_MODEL


def test_doctor_validation_rejects_stale_native_extension_hash(tmp_path: Path) -> None:
    module = _load()
    runtime = _runtime(module, tmp_path)
    report = json.loads(runtime["doctor_report"].read_text(encoding="utf-8"))
    report["runtime_artifacts"]["switchyard_rust_native_extension"]["sha256"] = "sha256:" + "0" * 64
    runtime["doctor_report"].write_text(json.dumps(report) + "\n", encoding="utf-8")

    with pytest.raises(module.TrialQADemoError, match="selected runtime"):
        module._validate_doctor_report(
            runtime["doctor_report"],
            switchyard_bin=runtime["switchyard"],
            codex_bin=runtime["codex"],
            tooluniverse_bin=runtime["tooluniverse"],
        )


def test_doctor_validation_rejects_false_namespace_round_trip(tmp_path: Path) -> None:
    module = _load()
    runtime = _runtime(module, tmp_path)
    report = json.loads(runtime["doctor_report"].read_text(encoding="utf-8"))
    report["namespace_translation"]["response_namespace"] = "wrong_namespace"
    runtime["doctor_report"].write_text(json.dumps(report) + "\n", encoding="utf-8")

    with pytest.raises(module.TrialQADemoError, match="namespace translation probe"):
        module._validate_doctor_report(
            runtime["doctor_report"],
            switchyard_bin=runtime["switchyard"],
            codex_bin=runtime["codex"],
            tooluniverse_bin=runtime["tooluniverse"],
        )


def _fake_successful_executor(
    module: ModuleType,
    dataset: object,
    *,
    recovered_errors: int = 0,
):
    assert recovered_errors in {0, 1}

    def execute(spec: object, environment: dict[str, str]) -> int:
        assert stat.S_IMODE(dataset.path.stat().st_mode) == 0
        skill_load_event = ""
        if spec.arm == "treatment":
            skill_load_event = (
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "id": "item-skill-1",
                            "type": "mcp_tool_call",
                            "server": "tooluniverse",
                            "tool": "trialqa_load_active_skill",
                            "arguments": {},
                            "result": {"content": [{"type": "text", "text": "skill"}]},
                            "error": None,
                            "status": "completed",
                        },
                    }
                )
                + "\n"
            )
        spec.stdout_path.write_text(
            "Switchyard launching Codex...\n"
            + json.dumps({"type": "thread.started", "thread_id": "thread-1"})
            + "\n"
            + "banner {not-json}\n"
            + skill_load_event
            + json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "id": "item-tool-1",
                        "type": "mcp_tool_call",
                        "server": "tooluniverse",
                        "tool": "execute_tool",
                        "arguments": {"query": "threshold"},
                        "result": {"content": [{"type": "text", "text": "study"}]},
                        "error": None,
                        "status": "completed",
                    },
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 7, "output_tokens": 3},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        spec.stderr_path.write_text("local fake execution\n", encoding="utf-8")
        spec.final_output_path.write_text(
            json.dumps({"answer": "The threshold is 18 years."}) + "\n",
            encoding="utf-8",
        )
        context = json.loads(
            Path(environment["SWITCHYARD_SKILL_DISTILLATION_RUN_CONTEXT_PATH"]).read_text()
        )
        active = json.loads(
            Path(environment["SWITCHYARD_SKILL_DISTILLATION_ACTIVE_EVIDENCE_PATH"]).read_text()
        )
        assert not Path(
            environment["SWITCHYARD_SKILL_DISTILLATION_RUN_CONTEXT_PATH"]
        ).is_relative_to(spec.stdin_path.parent)
        assert context["executor_model"] == module.EXECUTOR_MODEL
        assert context["route"] == module.EXECUTOR_ROUTE
        assert context["skill_loaded"] is (spec.arm == "treatment")
        assert active["loaded"] is (spec.arm == "treatment")
        store = module.SkillDistillationStore(module.NAMESPACE, spec.cwd)
        session_id = f"codex-{context['task_id']}"
        session = store.sessions_path / session_id
        session.mkdir()
        turn = {
            "schema_version": 1,
            "session_id": session_id,
            "turn_index": 0,
            "served_model": module.EXECUTOR_MODEL,
            "request": {
                "model": module.EXECUTOR_ROUTE,
                "messages": [{"role": "user", "content": "threshold question"}],
            },
            "response": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The threshold is 18 years.",
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        }
        turns = (json.dumps(turn, sort_keys=True) + "\n").encode()
        (session / "turns.jsonl").write_bytes(turns)
        stats = {
            "total_requests": 1 + recovered_errors,
            "total_errors": recovered_errors,
            "total_tokens": {"prompt": 7, "completion": 3, "total": 10},
            "models": {module.EXECUTOR_MODEL: {"calls": 1, "errors": recovered_errors}},
            "classifier": {"total_requests": 0, "total_errors": 0},
            "planner": {"total_requests": 0, "total_errors": 0},
            "openai_transport": _openai_transport(1 + recovered_errors),
        }
        (session / "stats.json").write_text(json.dumps(stats) + "\n", encoding="utf-8")
        metadata = {
            "schema_version": 1,
            "session_id": session_id,
            "namespace": module.NAMESPACE,
            "launch_target": "codex",
            "display_model": module.EXECUTOR_ROUTE,
            "strategy_summary": module.EXECUTOR_ROUTE,
            "started_at": "2026-07-06T00:00:00Z",
            "ended_at": "2026-07-06T00:00:01Z",
            "status": "completed",
            "exit_code": 0,
            "turn_count": 1,
            "trajectory_sha256": f"sha256:{hashlib.sha256(turns).hexdigest()}",
            "run_context": context,
            "active_skill": active,
        }
        (session / "session.json").write_text(
            json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8"
        )
        return 0

    return execute


def test_streaming_subprocess_timeout_kills_process_group_and_is_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load()
    if module.os.name != "posix":
        pytest.skip("process-group timeout behavior is POSIX-specific")
    stdin_path = tmp_path / "prompt.md"
    stdin_path.write_text("prompt\n", encoding="utf-8")
    spec = SimpleNamespace(
        argv=("fake-switchyard",),
        cwd=tmp_path,
        env={},
        stdin_path=stdin_path,
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
    )

    class FakeProcess:
        pid = 4242

        def __init__(self) -> None:
            self.stdout = io.BytesIO(b"")
            self.stderr = io.BytesIO(b"")
            self.wait_timeouts: list[float] = []
            self.terminate_calls = 0
            self.kill_calls = 0

        def wait(self, *, timeout: float) -> int:
            self.wait_timeouts.append(timeout)
            if len(self.wait_timeouts) == 1:
                raise subprocess.TimeoutExpired(spec.argv, timeout)
            return -signal.SIGTERM

        def terminate(self) -> None:
            self.terminate_calls += 1

        def kill(self) -> None:
            self.kill_calls += 1

    process = FakeProcess()
    popen_kwargs: dict[str, object] = {}

    def fake_popen(*_args, **kwargs):
        popen_kwargs.update(kwargs)
        return process

    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        module.os,
        "killpg",
        lambda process_group, sent_signal: signals.append((process_group, sent_signal)),
    )

    with pytest.raises(module.GenerationTimeoutError) as raised:
        module.run_streaming_subprocess(spec, {}, timeout_seconds=60)

    assert raised.value.timeout_seconds == 60
    assert raised.value.process_group_terminated is True
    assert str(raised.value).endswith("timed out after 60s")
    assert signals == [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)]
    assert process.wait_timeouts == [60, 10, 10]
    assert process.terminate_calls == 0
    assert process.kill_calls == 0
    assert popen_kwargs["start_new_session"] is True
    assert spec.stdout_path.read_bytes() == b""
    assert spec.stderr_path.read_bytes() == b""


def test_streaming_subprocess_deadline_includes_capture_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load()
    if module.os.name != "posix":
        pytest.skip("process-group timeout behavior is POSIX-specific")
    stdin_path = tmp_path / "prompt.md"
    stdin_path.write_text("prompt\n", encoding="utf-8")
    spec = SimpleNamespace(
        argv=("fake-switchyard",),
        cwd=tmp_path,
        env={},
        stdin_path=stdin_path,
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
    )

    class FakeProcess:
        pid = 4343

        def __init__(self) -> None:
            self.stdout = io.BytesIO(b"")
            self.stderr = io.BytesIO(b"")
            self.wait_timeouts: list[float] = []

        def wait(self, *, timeout: float) -> int:
            self.wait_timeouts.append(timeout)
            return 0

        def terminate(self) -> None:  # pragma: no cover - POSIX test.
            raise AssertionError("unexpected Windows termination")

        def kill(self) -> None:  # pragma: no cover - POSIX test.
            raise AssertionError("unexpected Windows kill")

    class FakeThread:
        def __init__(self, **_kwargs) -> None:
            self.alive = True
            self.join_timeouts: list[float | None] = []

        def start(self) -> None:
            pass

        def join(self, timeout: float | None = None) -> None:
            self.join_timeouts.append(timeout)
            if timeout == 5:
                self.alive = False

        def is_alive(self) -> bool:
            return self.alive

    process = FakeProcess()
    threads: list[FakeThread] = []

    def fake_thread(**kwargs):
        thread = FakeThread(**kwargs)
        threads.append(thread)
        return thread

    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(module.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(module.threading, "Thread", fake_thread)
    monkeypatch.setattr(
        module.os,
        "killpg",
        lambda process_group, sent_signal: signals.append((process_group, sent_signal)),
    )

    with pytest.raises(module.GenerationTimeoutError):
        module.run_streaming_subprocess(spec, {}, timeout_seconds=60)

    assert process.wait_timeouts == [60, 10, 10]
    assert signals == [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)]
    assert all(thread.join_timeouts[-1] == 5 for thread in threads)


def test_timeout_cleanup_failure_remains_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()
    monkeypatch.setattr(
        module,
        "_terminate_timed_out_process",
        lambda *_args: (_ for _ in ()).throw(module.TrialQADemoError("cleanup failed")),
    )

    with pytest.raises(module.GenerationTimeoutError) as raised:
        module._raise_generation_timeout(
            SimpleNamespace(),
            (),
            timeout_seconds=600,
            cause=TimeoutError("deadline"),
        )

    assert raised.value.process_group_terminated is False
    assert isinstance(raised.value.__cause__, module.TrialQADemoError)


@pytest.mark.parametrize("recovered_errors", [0, 1])
def test_generation_locks_gold_validates_mixed_output_and_imports_native_session(
    tmp_path: Path,
    recovered_errors: int,
) -> None:
    module = _load()
    dataset = _dataset(module, tmp_path)
    original_mode = stat.S_IMODE(dataset.path.stat().st_mode)
    split = module.create_split_manifest(dataset)
    runtime = _runtime(module, tmp_path)
    manifest = module.build_experiment_manifest(
        dataset=dataset,
        split_manifest=split,
        kind="donor",
        candidate=None,
        routing_profile=runtime["profile"],
        switchyard_bin=runtime["switchyard"],
        codex_bin=runtime["codex"],
        tooluniverse_bin=runtime["tooluniverse"],
        doctor_report=runtime["doctor_report"],
    )
    task = manifest["tasks"][0]
    capture = tmp_path / "capture"
    planned = module.prepare_generation(
        manifest=manifest,
        task_id=task["task_id"],
        dataset=dataset,
        split_manifest=split,
        capture_cwd=capture,
        candidate_root=None,
        switchyard_bin=runtime["switchyard"],
        codex_bin=runtime["codex"],
        routing_profile=runtime["profile"],
        tooluniverse_bin=runtime["tooluniverse"],
    )

    result = module.execute_generation(
        manifest=manifest,
        planned=planned,
        dataset=dataset,
        executor=_fake_successful_executor(
            module,
            dataset,
            recovered_errors=recovered_errors,
        ),
    )

    assert stat.S_IMODE(dataset.path.stat().st_mode) == original_mode
    assert result.answer == "The threshold is 18 years."
    assert result.stats["models"] == {
        module.EXECUTOR_MODEL: {"calls": 1, "errors": recovered_errors}
    }
    module.validate_generation_for_import(result, project_dir=capture.resolve())
    with pytest.raises(module.TrialQADemoError, match="captured Codex output"):
        module.validate_generation_for_import(
            replace(result, answer="A tampered persisted answer."),
            project_dir=capture.resolve(),
        )

    scored = module.score_and_import_generation(
        generation=result,
        row=planned.row,
        judge=lambda _payload: json.dumps(
            {
                "judge_result": "incorrect",
                "score": 0,
                "rationale": "The submitted threshold differs.",
            }
        ),
        project_dir=capture.resolve(),
    )
    assert scored.outcome.score == 0.0
    assert scored.evidence.evidence_id.startswith("native-")
    assert scored.evidence.evidence_path.is_dir()
    retained_stats = json.loads((scored.evidence.evidence_path / "raw" / "stats.json").read_text())
    assert retained_stats["total_errors"] == recovered_errors


def test_dedicated_judge_rejects_fallback_and_audits_model_delta() -> None:
    module = _load()
    calls: list[tuple[str, str]] = []
    stats = [
        {"total_requests": 0, "total_errors": 0, "models": {}},
        {
            "total_requests": 1,
            "total_errors": 0,
            "models": {module.JUDGE_MODEL: {"calls": 1}},
        },
    ]

    def transport(method: str, url: str, payload: object, timeout: float):
        del timeout
        calls.append((method, url))
        if method == "GET":
            return 200, json.dumps(stats.pop(0)).encode()
        assert payload["model"] == module.JUDGE_ROUTE
        return 200, json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "judge_result": "correct",
                                    "score": 1,
                                    "rationale": "Equivalent.",
                                }
                            )
                        }
                    }
                ]
            }
        ).encode()

    client = module.DedicatedJudgeClient("http://127.0.0.1:4111", transport=transport)
    content = client({"model": module.JUDGE_ROUTE, "messages": []})
    assert json.loads(content)["judge_result"] == "correct"
    assert [method for method, _url in calls] == ["GET", "POST", "GET"]

    with pytest.raises(module.TrialQAJudgeError, match="must use route"):
        client({"model": "some-fallback", "messages": []})


def _write_codex_event_log(path: Path, items: list[dict[str, object]]) -> Path:
    events = [
        {"type": "thread.started", "thread_id": "thread-policy"},
        *({"type": "item.completed", "item": item} for item in items),
        {
            "type": "turn.completed",
            "usage": {"input_tokens": 7, "output_tokens": 3},
        },
    ]
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    return path


def test_codex_event_policy_requires_adapter_evidence_and_forbids_shell(
    tmp_path: Path,
) -> None:
    module = _load()
    evidence = {
        "id": "evidence",
        "type": "mcp_tool_call",
        "server": "tooluniverse",
        "tool": "execute_tool",
        "status": "completed",
        "error": None,
    }
    shell = {
        "id": "shell",
        "type": "command_execution",
        "command": "/bin/zsh -lc 'ls -la'",
        "status": "completed",
    }
    path = _write_codex_event_log(tmp_path / "events.jsonl", [evidence, shell])

    with pytest.raises(module.TrialQADemoError, match="forbidden or unknown item type"):
        module._parse_codex_events(path)


def test_codex_event_policy_requires_execute_tool_not_discovery_only(tmp_path: Path) -> None:
    module = _load()
    discovery = {
        "id": "discovery",
        "type": "mcp_tool_call",
        "server": "tooluniverse",
        "tool": "grep_tools",
        "status": "completed",
        "error": None,
    }
    path = _write_codex_event_log(tmp_path / "events.jsonl", [discovery])

    with pytest.raises(module.TrialQADemoError, match="without a TrialQA evidence-tool"):
        module._parse_codex_events(path)


def test_codex_event_policy_rejects_raw_clinicaltrials_tool(tmp_path: Path) -> None:
    module = _load()
    raw_tool = {
        "id": "raw",
        "type": "mcp_tool_call",
        "server": "tooluniverse",
        "tool": "ClinicalTrials_search_studies",
        "status": "completed",
        "error": None,
    }
    path = _write_codex_event_log(tmp_path / "events.jsonl", [raw_tool])

    with pytest.raises(module.TrialQADemoError, match="outside the TrialQA adapter"):
        module._parse_codex_events(path)


@pytest.mark.parametrize(
    "item_type",
    ["web_search", "image_view", "file_change", "future_tool"],
)
def test_codex_event_policy_fails_closed_for_every_non_trialqa_tool_item(
    tmp_path: Path,
    item_type: str,
) -> None:
    module = _load()
    evidence = {
        "id": "evidence",
        "type": "mcp_tool_call",
        "server": "tooluniverse",
        "tool": "execute_tool",
        "status": "completed",
        "error": None,
    }
    forbidden = {
        "id": "forbidden",
        "type": item_type,
        "status": "completed",
    }
    path = _write_codex_event_log(
        tmp_path / f"events-{item_type}.jsonl",
        [evidence, forbidden],
    )

    with pytest.raises(module.TrialQADemoError, match="forbidden or unknown item type"):
        module._parse_codex_events(path)


def test_codex_event_policy_allows_well_formed_passive_todo_list(tmp_path: Path) -> None:
    module = _load()
    evidence = {
        "id": "evidence",
        "type": "mcp_tool_call",
        "server": "tooluniverse",
        "tool": "execute_tool",
        "status": "completed",
        "error": None,
    }
    todo = {
        "id": "todo",
        "type": "todo_list",
        "items": [{"text": "Look up the trial", "completed": True}],
    }
    path = _write_codex_event_log(tmp_path / "events.jsonl", [evidence, todo])

    assert module._parse_codex_events(path) == {"input_tokens": 7, "output_tokens": 3}


def test_codex_event_policy_allows_recovered_trialqa_tool_error(tmp_path: Path) -> None:
    module = _load()
    failed = {
        "id": "failed",
        "type": "mcp_tool_call",
        "server": "tooluniverse",
        "tool": "execute_tool",
        "status": "failed",
        "error": {"message": "nct_id is required"},
    }
    recovered = {
        "id": "recovered",
        "type": "mcp_tool_call",
        "server": "tooluniverse",
        "tool": "execute_tool",
        "status": "completed",
        "error": None,
    }
    path = _write_codex_event_log(tmp_path / "events.jsonl", [failed, recovered])

    assert module._parse_codex_events(path) == {"input_tokens": 7, "output_tokens": 3}


def test_codex_tool_metrics_excludes_skill_load_and_deduplicates_events(
    tmp_path: Path,
) -> None:
    module = _load()
    items = [
        {
            "id": "load",
            "type": "mcp_tool_call",
            "server": "tooluniverse",
            "tool": "trialqa_load_active_skill",
            "status": "in_progress",
            "error": None,
        },
        {
            "id": "search",
            "type": "mcp_tool_call",
            "server": "tooluniverse",
            "tool": "execute_tool",
            "status": "in_progress",
            "error": None,
        },
        {
            "id": "failed-get",
            "type": "mcp_tool_call",
            "server": "tooluniverse",
            "tool": "execute_tool",
            "status": "in_progress",
            "error": None,
        },
    ]
    events = [
        {"type": "thread.started", "thread_id": "thread-metrics"},
        *({"type": "item.started", "item": item} for item in items),
        *({"type": "item.started", "item": item} for item in items),
        {
            "type": "item.completed",
            "item": {**items[0], "status": "completed"},
        },
        {
            "type": "item.completed",
            "item": {**items[1], "status": "completed"},
        },
        {
            "type": "item.completed",
            "item": {
                **items[2],
                "status": "failed",
                "error": {"message": "bad argument"},
            },
        },
        {
            "type": "turn.completed",
            "usage": {"input_tokens": 7, "output_tokens": 3},
        },
    ]
    path = tmp_path / "metrics.jsonl"
    path.write_text("".join(json.dumps(event) + "\n" for event in events))

    assert module.codex_tool_metrics(path) == {
        "operational_calls": 2,
        "successful_operational_calls": 1,
        "skill_load_calls": 1,
        "successful_skill_load_calls": 1,
    }


def test_codex_tool_metrics_rejects_completed_call_without_start(tmp_path: Path) -> None:
    module = _load()
    path = _write_codex_event_log(
        tmp_path / "events.jsonl",
        [
            {
                "id": "search",
                "type": "mcp_tool_call",
                "server": "tooluniverse",
                "tool": "execute_tool",
                "status": "completed",
                "error": None,
            }
        ],
    )

    with pytest.raises(module.TrialQADemoError, match="without starting"):
        module.codex_tool_metrics(path)


def test_codex_event_policy_allows_recovered_error_before_completion(
    tmp_path: Path,
) -> None:
    module = _load()
    events = [
        {"type": "thread.started", "thread_id": "thread-policy"},
        {"type": "error", "message": "Reconnecting... 1/5 (request timed out)"},
        {
            "type": "item.completed",
            "item": {
                "id": "evidence",
                "type": "mcp_tool_call",
                "server": "tooluniverse",
                "tool": "execute_tool",
                "status": "completed",
                "error": None,
            },
        },
        {
            "type": "turn.completed",
            "usage": {"input_tokens": 7, "output_tokens": 3},
        },
    ]
    path = tmp_path / "events.jsonl"
    path.write_text("".join(json.dumps(event) + "\n" for event in events))

    assert module._parse_codex_events(path) == {"input_tokens": 7, "output_tokens": 3}


@pytest.mark.parametrize("message", [None, "", "   "])
def test_codex_event_policy_rejects_malformed_recovered_error(
    tmp_path: Path,
    message: str | None,
) -> None:
    module = _load()
    path = _write_codex_event_log(
        tmp_path / "events.jsonl",
        [
            {
                "id": "evidence",
                "type": "mcp_tool_call",
                "server": "tooluniverse",
                "tool": "execute_tool",
                "status": "completed",
                "error": None,
            }
        ],
    )
    events = [json.loads(line) for line in path.read_text().splitlines()]
    events.insert(1, {"type": "error", "message": message})
    path.write_text("".join(json.dumps(event) + "\n" for event in events))

    with pytest.raises(module.TrialQADemoError, match="error telemetry"):
        module._parse_codex_events(path)


def test_codex_event_policy_rejects_error_after_completion(tmp_path: Path) -> None:
    module = _load()
    path = _write_codex_event_log(
        tmp_path / "events.jsonl",
        [
            {
                "id": "evidence",
                "type": "mcp_tool_call",
                "server": "tooluniverse",
                "tool": "execute_tool",
                "status": "completed",
                "error": None,
            }
        ],
    )
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"type": "error", "message": "late failure"}) + "\n")

    with pytest.raises(module.TrialQADemoError, match="precede turn.completed"):
        module._parse_codex_events(path)


@pytest.mark.parametrize(
    "event_type",
    ["turn.failed", "item.failed", "thread.failed", "fatal"],
)
def test_codex_event_policy_rejects_terminal_lifecycle_events(
    tmp_path: Path,
    event_type: str,
) -> None:
    module = _load()
    path = _write_codex_event_log(
        tmp_path / "events.jsonl",
        [
            {
                "id": "evidence",
                "type": "mcp_tool_call",
                "server": "tooluniverse",
                "tool": "execute_tool",
                "status": "completed",
                "error": None,
            }
        ],
    )
    events = [json.loads(line) for line in path.read_text().splitlines()]
    events.insert(1, {"type": event_type})
    path.write_text("".join(json.dumps(event) + "\n" for event in events))

    with pytest.raises(module.TrialQADemoError, match="reports failure"):
        module._parse_codex_events(path)


def test_codex_event_policy_rejects_malformed_passive_todo_list(tmp_path: Path) -> None:
    module = _load()
    evidence = {
        "id": "evidence",
        "type": "mcp_tool_call",
        "server": "tooluniverse",
        "tool": "execute_tool",
        "status": "completed",
        "error": None,
    }
    malformed = {
        "id": "todo",
        "type": "todo_list",
        "items": [{"text": "Look up the trial", "completed": "yes"}],
    }
    path = _write_codex_event_log(tmp_path / "events.jsonl", [evidence, malformed])

    with pytest.raises(module.TrialQADemoError, match="invalid passive todo"):
        module._parse_codex_events(path)


def test_final_answer_unwraps_exact_json_and_accepts_raw_text(tmp_path: Path) -> None:
    module = _load()
    exact = tmp_path / "exact.json"
    exact.write_text('{"answer":"  six months  "}')
    raw = tmp_path / "raw.txt"
    raw.write_text("Evidence summary.\n\nThe answer is six months.\n")

    assert module._parse_final_answer_with_source(exact) == (
        "six months",
        module.FINAL_ANSWER_JSON_SOURCE,
    )
    assert module._parse_final_answer_with_source(raw) == (
        "Evidence summary.\n\nThe answer is six months.",
        module.FINAL_ANSWER_TEXT_SOURCE,
    )


def test_execution_attestation_binds_batch_driver() -> None:
    module = _load()

    sources = module._execution_source_sha256()

    assert "benchmark/trialqa_local_batch.py" in sources
    assert sources["benchmark/trialqa_local_batch.py"] == module._sha256_file(
        module._PROJECT_ROOT / "benchmark/trialqa_local_batch.py"
    )


def test_codex_event_policy_requires_at_least_one_treatment_skill_load(
    tmp_path: Path,
) -> None:
    module = _load()
    evidence = {
        "id": "evidence",
        "type": "mcp_tool_call",
        "server": "tooluniverse",
        "tool": "execute_tool",
        "status": "completed",
        "error": None,
    }
    path = _write_codex_event_log(tmp_path / "events.jsonl", [evidence])

    with pytest.raises(module.TrialQADemoError, match="load its active"):
        module._parse_codex_events(path, require_skill_load=True)


def test_codex_event_policy_accepts_duplicate_successful_skill_loads(
    tmp_path: Path,
) -> None:
    module = _load()
    skill_load = {
        "id": "skill-load",
        "type": "mcp_tool_call",
        "server": "tooluniverse",
        "tool": "trialqa_load_active_skill",
        "status": "completed",
        "error": None,
    }
    evidence = {
        "id": "evidence",
        "type": "mcp_tool_call",
        "server": "tooluniverse",
        "tool": "execute_tool",
        "status": "completed",
        "error": None,
    }
    path = _write_codex_event_log(
        tmp_path / "events.jsonl",
        [skill_load, evidence, {**skill_load, "id": "skill-load-again"}],
    )

    module._parse_codex_events(path, require_skill_load=True)


def test_codex_event_policy_rejects_treatment_skill_load_after_evidence(
    tmp_path: Path,
) -> None:
    module = _load()
    evidence = {
        "id": "evidence",
        "type": "mcp_tool_call",
        "server": "tooluniverse",
        "tool": "execute_tool",
        "status": "completed",
        "error": None,
    }
    skill_load = {
        "id": "skill-load",
        "type": "mcp_tool_call",
        "server": "tooluniverse",
        "tool": "trialqa_load_active_skill",
        "status": "completed",
        "error": None,
    }
    path = _write_codex_event_log(
        tmp_path / "events.jsonl",
        [evidence, skill_load],
    )

    with pytest.raises(module.TrialQADemoError, match="before tool use"):
        module._parse_codex_events(path, require_skill_load=True)


def test_codex_event_policy_rejects_generic_mcp_resources(tmp_path: Path) -> None:
    module = _load()
    generic = {
        "id": "generic",
        "type": "mcp_tool_call",
        "server": "codex",
        "tool": "list_mcp_resources",
        "status": "completed",
        "error": None,
    }
    path = _write_codex_event_log(tmp_path / "events.jsonl", [generic])

    with pytest.raises(module.TrialQADemoError, match="outside the TrialQA adapter"):
        module._parse_codex_events(path)


def test_judge_fails_when_stats_attribute_call_to_another_model() -> None:
    module = _load()
    snapshots = iter(
        [
            {"total_requests": 0, "total_errors": 0, "models": {}},
            {
                "total_requests": 1,
                "total_errors": 0,
                "models": {"wrong/model": {"calls": 1}},
            },
        ]
    )

    def transport(method: str, _url: str, _payload: object, _timeout: float):
        if method == "GET":
            return 200, json.dumps(next(snapshots)).encode()
        return 200, json.dumps(
            {"choices": [{"message": {"content": '{"judge_result":"correct"}'}}]}
        ).encode()

    client = module.DedicatedJudgeClient("http://127.0.0.1:4111", transport=transport)
    with pytest.raises(module.TrialQAJudgeError, match="exclusively routed"):
        client({"model": module.JUDGE_ROUTE, "messages": []})


def test_executor_stats_allow_recovered_ultra_request_error() -> None:
    module = _load()
    stats = {
        "total_requests": 3,
        "total_errors": 1,
        "models": {module.EXECUTOR_MODEL: {"calls": 2, "errors": 1}},
        "classifier": {"total_requests": 0, "total_errors": 0},
        "planner": {"total_requests": 0, "total_errors": 0},
        "openai_transport": _openai_transport(3),
    }

    module._validate_executor_stats(stats)


def test_executor_stats_accept_priced_null_eof_retry() -> None:
    module = _load()
    stats = {
        "total_requests": 1,
        "total_errors": 0,
        "models": {module.EXECUTOR_MODEL: {"calls": 1, "errors": 0}},
        "classifier": {"total_requests": 0, "total_errors": 0},
        "planner": {"total_requests": 0, "total_errors": 0},
        "openai_transport": _openai_transport(
            1,
            retries=1,
            charges=1,
            prompt=12,
            completion=3,
        ),
    }

    module._validate_executor_stats(stats)


@pytest.mark.parametrize(
    "transport,match",
    [
        (_openai_transport(1, retries=1, unpriced=1), "unpriced"),
        ({**_openai_transport(1), "physical_attempts": 2}, "inconsistent"),
        ({}, "exact OpenAI transport"),
    ],
)
def test_executor_stats_reject_untrustworthy_transport_accounting(
    transport: dict[str, object], match: str
) -> None:
    module = _load()
    stats = {
        "total_requests": 1,
        "total_errors": 0,
        "models": {module.EXECUTOR_MODEL: {"calls": 1, "errors": 0}},
        "classifier": {"total_requests": 0, "total_errors": 0},
        "planner": {"total_requests": 0, "total_errors": 0},
        "openai_transport": transport,
    }

    with pytest.raises(module.TrialQADemoError, match=match):
        module._validate_executor_stats(stats)


def test_executor_stats_reject_error_attempt_attributed_to_another_model() -> None:
    module = _load()
    stats = {
        "total_requests": 2,
        "total_errors": 1,
        "models": {
            module.EXECUTOR_MODEL: {"calls": 1, "errors": 0},
            "wrong/model": {"calls": 0, "errors": 1},
        },
        "classifier": {"total_requests": 0, "total_errors": 0},
        "planner": {"total_requests": 0, "total_errors": 0},
        "openai_transport": _openai_transport(2),
    }

    with pytest.raises(module.TrialQADemoError, match="exclusively attributed"):
        module._validate_executor_stats(stats)


def test_resumable_ledger_is_hash_chained_and_does_not_repeat_completed_task(
    tmp_path: Path,
) -> None:
    module = _load()
    dataset = _dataset(module, tmp_path)
    split = module.create_split_manifest(dataset)
    runtime = _runtime(module, tmp_path)
    profile = runtime["profile"]
    manifest = module.build_experiment_manifest(
        dataset=dataset,
        split_manifest=split,
        kind="donor",
        candidate=None,
        routing_profile=profile,
        switchyard_bin=runtime["switchyard"],
        codex_bin=runtime["codex"],
        tooluniverse_bin=runtime["tooluniverse"],
        doctor_report=runtime["doctor_report"],
    )
    task_id = manifest["tasks"][0]["task_id"]
    ledger = module.ResumableLedger(tmp_path / "ledger.jsonl", manifest)

    for event in (
        "generation_started",
        "generation_completed",
        "scored",
        "evidence_imported",
        "completed",
    ):
        ledger.append(task_id, event)

    assert ledger.states()[task_id] == "completed"
    assert task_id not in ledger.pending_task_ids()
    lines = (tmp_path / "ledger.jsonl").read_text().splitlines()
    changed = json.loads(lines[0])
    changed["event"] = "completed"
    lines[0] = json.dumps(changed)
    (tmp_path / "ledger.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(module.TrialQADemoError, match="hash differs"):
        ledger.records()


def test_resumable_ledger_can_terminalize_failed_model_draw(tmp_path: Path) -> None:
    module = _load()
    dataset = _dataset(module, tmp_path)
    split = module.create_split_manifest(dataset)
    runtime = _runtime(module, tmp_path)
    manifest = module.build_experiment_manifest(
        dataset=dataset,
        split_manifest=split,
        kind="donor",
        candidate=None,
        routing_profile=runtime["profile"],
        switchyard_bin=runtime["switchyard"],
        codex_bin=runtime["codex"],
        tooluniverse_bin=runtime["tooluniverse"],
        doctor_report=runtime["doctor_report"],
    )
    task_id = manifest["tasks"][0]["task_id"]
    ledger = module.ResumableLedger(tmp_path / "terminal-ledger.jsonl", manifest)

    ledger.append(task_id, "generation_started")
    ledger.append(task_id, "failed", {"stage": "generation"})
    ledger.append(task_id, "completed", {"terminal_status": "error"})

    assert ledger.states()[task_id] == "completed"
    assert task_id not in ledger.pending_task_ids()


def test_resumable_ledger_retries_score_without_faking_generation(tmp_path: Path) -> None:
    module = _load()
    manifest = {
        "manifest_id": "trialqa-full-score-retry",
        "tasks": [{"task_id": "pair-baseline"}],
    }
    ledger = module.ResumableLedger(tmp_path / "score-retry-ledger.jsonl", manifest)

    ledger.append("pair-baseline", "generation_started")
    ledger.append("pair-baseline", "generation_completed")
    ledger.append("pair-baseline", "failed", {"stage": "score-import"})
    ledger.append("pair-baseline", "score_retry_started")
    ledger.append("pair-baseline", "scored")
    ledger.append("pair-baseline", "evidence_imported")
    ledger.append("pair-baseline", "completed")

    assert ledger.states()["pair-baseline"] == "completed"
    assert sum(record["event"] == "generation_completed" for record in ledger.records()) == 1


def test_failure_result_preserves_completed_draw_token_usage() -> None:
    module = _load()
    manifest = {"manifest_id": "trialqa-full-123"}
    task = {
        "task_id": "pair-baseline",
        "pair_id": "pair",
        "row_id": "row-1",
        "question_group_key": "question",
        "condition": "baseline",
        "repeat_index": 1,
        "n_repeats": 5,
    }

    record = module.failure_result_record(
        manifest=manifest,
        task=task,
        stage="generation",
        error=module.TrialQADemoError("no evidence call"),
        usage={"input_tokens": 120, "output_tokens": 30},
    )

    assert (record.prompt_tokens, record.completion_tokens, record.total_tokens) == (
        120,
        30,
        150,
    )


def test_result_collection_rejects_success_shadowing_terminal_generation_failure(
    tmp_path: Path,
) -> None:
    module = _load()
    task = {
        "task_id": "pair-baseline",
        "pair_id": "pair",
        "row_id": "row-1",
        "question_group_key": "question",
        "condition": "baseline",
        "repeat_index": 1,
        "n_repeats": 5,
    }
    manifest = {"manifest_id": "trialqa-full-123", "tasks": [task]}
    success = module.TrialResultRecord(
        manifest_id="trialqa-full-123",
        task_id="pair-baseline",
        pair_id="pair",
        row_id="row-1",
        question_group_key="question",
        condition="baseline",
        repeat_index=1,
        n_repeats=5,
        status="scored",
        score=1.0,
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        evidence_id="native-" + "0" * 32,
        error_stage=None,
        error_type=None,
    )
    module.write_trial_result(tmp_path / "pair-baseline.json", success)
    failure = module.failure_result_record(
        manifest=manifest,
        task=task,
        stage="generation",
        error=module.TrialQADemoError("no evidence"),
        usage={"input_tokens": 50, "output_tokens": 5},
    )
    module.write_failure_result(tmp_path, failure)

    with pytest.raises(module.TrialQADemoError, match="both a success"):
        module.collect_protocol_results(tmp_path, manifest)


def test_comparison_report_requires_pairs_and_aggregates_accuracy_and_tokens(
    tmp_path: Path,
) -> None:
    module = _load()
    base = module.GenerationResult(
        manifest_id="trialqa-full-123",
        task_id="pair-baseline",
        pair_id="pair",
        row_id="row-1",
        dataset_row_index=1,
        partition="test",
        condition="baseline",
        repeat_index=1,
        n_repeats=5,
        answer="a",
        answer_source="codex-output-last-message-json",
        session_dir=tmp_path,
        stats_path=tmp_path / "stats.json",
        trajectory_path=tmp_path / "turns.jsonl",
        codex_events_path=tmp_path / "events.log",
        final_output_path=tmp_path / "final.json",
        generation_path=tmp_path / "generation.json",
        stats={"total_tokens": {"prompt": 80, "completion": 20, "total": 100}},
        usage={},
        artifact_sha256={},
    )
    treatment = replace(
        base,
        task_id="pair-treatment",
        condition="treatment",
        stats={"total_tokens": {"prompt": 50, "completion": 20, "total": 70}},
    )
    evidence = module.NativeTrialQAEvidenceImportResult(
        evidence_id="native-" + "0" * 32,
        evidence_path=tmp_path,
        imported=True,
    )
    baseline_scored = module.ScoredGeneration(
        generation=base,
        outcome=module.JudgeOutcome(
            judge_result="incorrect",
            score=0.0,
            rationale="",
            judge_available=True,
            judge_model=module.JUDGE_ROUTE,
        ),
        reward={},
        evidence=evidence,
    )
    treatment_scored = replace(
        baseline_scored,
        generation=treatment,
        outcome=replace(baseline_scored.outcome, judge_result="correct", score=1.0),
    )

    report = module.build_comparison_report([baseline_scored, treatment_scored])
    assert report["pair_count"] == 1
    assert report["conditions"]["baseline"]["accuracy"] == 0.0
    assert report["conditions"]["treatment"]["accuracy"] == 1.0
    assert report["benefit"]["total_token_delta"] == -30
    assert report["benefit"]["token_reduction_fraction"] == pytest.approx(0.3)

    with pytest.raises(module.TrialQADemoError, match="incomplete pair"):
        module.build_comparison_report([baseline_scored])


def test_full_protocol_report_matches_reference_replicate_metrics_and_counts(
    tmp_path: Path,
) -> None:
    module = _load()
    dataset = _dataset(module, tmp_path)
    split = module.create_split_manifest(dataset)
    candidate = _candidate(module, tmp_path / "candidate")
    runtime = _runtime(module, tmp_path)
    profile = runtime["profile"]
    manifest = module.build_experiment_manifest(
        dataset=dataset,
        split_manifest=split,
        kind="full",
        candidate=candidate,
        routing_profile=profile,
        switchyard_bin=runtime["switchyard"],
        codex_bin=runtime["codex"],
        tooluniverse_bin=runtime["tooluniverse"],
        doctor_report=runtime["doctor_report"],
    )
    records = []
    for task in manifest["tasks"]:
        baseline = task["condition"] == "baseline"
        score = 1.0 if not baseline or task["repeat_index"] == 1 else 0.0
        records.append(
            module.TrialResultRecord(
                manifest_id=manifest["manifest_id"],
                task_id=task["task_id"],
                pair_id=task["pair_id"],
                row_id=task["row_id"],
                question_group_key=task["question_group_key"],
                condition=task["condition"],
                repeat_index=task["repeat_index"],
                n_repeats=task["n_repeats"],
                status="scored",
                score=score,
                prompt_tokens=8 if baseline else 5,
                completion_tokens=2,
                total_tokens=10 if baseline else 7,
                evidence_id="native-" + "0" * 32,
                error_stage=None,
                error_type=None,
            )
        )
    zero_baseline = next(
        index
        for index, record in enumerate(records)
        if record.condition == "baseline" and record.repeat_index == 2
    )
    records[zero_baseline] = replace(
        records[zero_baseline],
        status="error",
        evidence_id=None,
        error_stage="generation",
        error_type="TimeoutError",
    )

    report = module.build_protocol_report(manifest, records)

    assert report["count_gate"] == {
        "passed": True,
        "expected_questions_per_arm": 96,
        "expected_records_per_arm": 480,
    }
    baseline = report["conditions"]["baseline"]
    treatment = report["conditions"]["treatment"]
    assert baseline["trial_mean"] == pytest.approx(0.2)
    assert baseline["question_macro_mean"] == pytest.approx(0.2)
    assert baseline["worst_case"] == 0.0
    assert baseline["oracle"] == 1.0
    assert baseline["incomplete"] == 0
    assert baseline["error_records"] == 1
    assert treatment["trial_mean"] == 1.0
    assert treatment["questions"] == 96
    assert treatment["records"] == 480
    assert report["benefit"]["token_reduction_fraction"] == pytest.approx(0.3)

    with pytest.raises(module.TrialQADemoError, match="report is incomplete"):
        module.build_protocol_report(manifest, records[:-1])


def test_treatment_binds_candidate_hashes_and_rejects_mid_run_mutation(
    tmp_path: Path,
) -> None:
    module = _load()
    dataset = _dataset(module, tmp_path)
    split = module.create_split_manifest(dataset)
    candidate = _candidate(module, tmp_path / "candidate")
    runtime = _runtime(module, tmp_path)
    manifest = module.build_experiment_manifest(
        dataset=dataset,
        split_manifest=split,
        kind="pilot",
        candidate=candidate,
        routing_profile=runtime["profile"],
        switchyard_bin=runtime["switchyard"],
        codex_bin=runtime["codex"],
        tooluniverse_bin=runtime["tooluniverse"],
        doctor_report=runtime["doctor_report"],
    )
    task = next(task for task in manifest["tasks"] if task["condition"] == "treatment")
    planned = module.prepare_generation(
        manifest=manifest,
        task_id=task["task_id"],
        dataset=dataset,
        split_manifest=split,
        capture_cwd=tmp_path / "capture",
        candidate_root=candidate.candidate_root,
        switchyard_bin=runtime["switchyard"],
        codex_bin=runtime["codex"],
        routing_profile=runtime["profile"],
        tooluniverse_bin=runtime["tooluniverse"],
    )
    successful = _fake_successful_executor(module, dataset)

    def mutate(spec, environment):
        result = successful(spec, environment)
        context = json.loads(
            Path(environment["SWITCHYARD_SKILL_DISTILLATION_RUN_CONTEXT_PATH"]).read_text()
        )
        assert context["candidate_id"] == "candidate-synthetic"
        assert context["candidate_manifest_sha256"].startswith("sha256:")
        assert context["candidate_skill_sha256"] == candidate.sha256
        candidate.skill_path.write_text("tampered after launch\n", encoding="utf-8")
        return result

    with pytest.raises(module.TrialQADemoError, match="candidate skill changed"):
        module.execute_generation(
            manifest=manifest,
            planned=planned,
            dataset=dataset,
            executor=mutate,
        )


def test_doctor_runs_only_local_no_model_commands_and_attests_ab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load()
    dataset = _dataset(module, tmp_path)
    split = module.create_split_manifest(dataset)
    assert split["counts"] == {"test": 96, "train": 24}
    runtime = _runtime(module, tmp_path)
    candidate = _candidate(module, tmp_path / "candidate")
    monkeypatch.setattr(module, "load_pinned_trialqa_parquet", lambda _path: dataset)
    monkeypatch.setattr(module, "TRIALQA_PARQUET_SHA256", dataset.parquet_sha256)

    def attest(**kwargs):
        pair = kwargs["pair"]
        return SimpleNamespace(
            baseline_skills=(),
            treatment_skills=(
                SimpleNamespace(
                    name=module.NAMESPACE,
                    description=candidate.description,
                    path=pair.treatment.managed_skill_path / "SKILL.md",
                ),
            ),
        )

    monkeypatch.setattr(module, "attest_trial_workspace_pair", attest)
    namespace_proof = _namespace_translation_attestation(module)
    monkeypatch.setattr(
        module,
        "_probe_namespace_translation",
        lambda: namespace_proof,
    )
    commands: list[tuple[str, ...]] = []

    def run(command, **_kwargs):
        commands.append(tuple(command))
        if "--describe-tools" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(module.describe_tools_document()),
                "",
            )
        name = Path(command[0]).name
        if name == "switchyard":
            text = "switchyard 0.test"
        elif name == "codex" and "features" in command:
            text = "\n".join(
                f"{feature} stable false" for feature in module.CODEX_DISABLED_FEATURES
            )
        elif name == "codex":
            text = "codex-cli 0.test"
        else:
            text = "usage: tooluniverse-smcp-stdio --include-tools"
        return subprocess.CompletedProcess(command, 0, text, "")

    report = module.run_doctor(
        dataset_path=dataset.path,
        experiment_root=tmp_path,
        candidate_root=candidate.candidate_root,
        switchyard_bin=runtime["switchyard"],
        codex_bin=runtime["codex"],
        tooluniverse_bin=runtime["tooluniverse"],
        routing_profile=runtime["profile"],
        run=run,
    )

    assert report["status"] == "passed"
    assert report["model_calls"] == 0
    assert report["dataset"]["split_counts"] == {"train": 24, "test": 96}
    assert report["namespace_translation"] == namespace_proof
    assert report["runtime_artifacts"]["switchyard_rust_native_extension"] == (
        module._native_extension_attestation()
    )
    flattened = {argument for command in commands for argument in command}
    assert "launch" not in flattened
    assert "exec" not in flattened
    assert "serve" not in flattened


def test_namespace_translation_probe_is_a_pure_exact_round_trip() -> None:
    module = _load()
    engine = _FakeNamespaceTranslationEngine()

    proof = module._probe_namespace_translation(engine_factory=lambda: engine)

    assert proof == _namespace_translation_attestation(module)
    assert [(source, target) for source, target, _body in engine.calls] == [
        ("openai_responses", "openai_chat"),
        ("openai_chat", "openai_responses"),
    ]
    request = engine.calls[0][2]
    assert request["model"] == module.EXECUTOR_ROUTE
    assert request["tools"][0]["type"] == "namespace"
    completion = engine.calls[1][2]
    assert completion["choices"][0]["message"]["tool_calls"][0]["function"] == {
        "name": engine.flattened_name,
        "arguments": "{}",
    }
    assert proof["model_calls"] == 0


def test_namespace_translation_probe_passes_loaded_native_extension() -> None:
    module = _load()

    proof = module._probe_namespace_translation()

    assert proof == _namespace_translation_attestation(module)


def test_namespace_translation_probe_rejects_lost_namespace() -> None:
    module = _load()
    engine = _FakeNamespaceTranslationEngine(response_namespace=None)

    with pytest.raises(module.TrialQADemoError, match="exact namespace and child"):
        module._probe_namespace_translation(engine_factory=lambda: engine)


def test_native_extension_attestation_hashes_the_loaded_binary() -> None:
    module = _load()

    proof = module._native_extension_attestation()
    resolved = Path(proof["resolved_path"])

    assert proof["module"] == "switchyard_rust._switchyard_rust"
    assert resolved.is_file()
    assert resolved.name.startswith("_switchyard_rust")
    assert proof["sha256"] == module._sha256_file(resolved)


def test_native_extension_attestation_rejects_a_python_shadow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()
    shadow = tmp_path / "_switchyard_rust.py"
    shadow.write_text("# not a native extension\n", encoding="utf-8")
    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda _name: SimpleNamespace(__file__=str(shadow)),
    )

    with pytest.raises(module.TrialQADemoError, match="native extension file"):
        module._native_extension_attestation()


def test_gold_lockdown_restores_mode_after_exception(tmp_path: Path) -> None:
    module = _load()
    path = tmp_path / "gold.parquet"
    path.write_bytes(b"gold")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    original = stat.S_IMODE(path.stat().st_mode)

    with pytest.raises(RuntimeError, match="boom"):
        with module.GoldArtifactLockdown(path, expected_sha256=digest):
            assert stat.S_IMODE(path.stat().st_mode) == 0
            raise RuntimeError("boom")

    assert stat.S_IMODE(path.stat().st_mode) == original
    assert path.read_bytes() == b"gold"
