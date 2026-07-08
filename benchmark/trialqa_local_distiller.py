# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Container-free Trace2Skill distillation for native TrialQA evidence.

The module is deliberately fail closed.  It accepts only content-addressed
``native-*`` evidence from :class:`SkillDistillationStore`, proves that every
source was an unskilled train/donor run through the pinned Ultra executor, and
then performs a three-level Trace2Skill pipeline:

1. one success/error analyst patch per donor rollout;
2. one merge for all repeats of the same question; and
3. one cross-question merge rendered as ``tooluniverse-trialqa/SKILL.md``.

All model calls use the ``sd-distiller`` route on an already-running localhost
Switchyard server.  The HTTP implementation is replaceable so unit tests and
``plan``/``dry-run`` never make model or network calls.  Model output is not
trusted: every stage is schema checked, tool references are checked against the
source trajectory, task-specific literals are removed, and routing statistics
must account for every new call before a candidate can be saved or activated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import yaml  # type: ignore[import-untyped]

import switchyard.lib.skill_distillation_native as native_evidence_module
import switchyard.lib.skill_distillation_store as skill_store_module
from switchyard.lib.skill_distillation_native import (
    validate_native_trialqa_evidence_directory,
)
from switchyard.lib.skill_distillation_store import SkillDistillationStore

SCHEMA_VERSION = "switchyard.trialqa_native_distillation.v1"
RAW_RESPONSE_SCHEMA = "switchyard.trialqa_raw_model_result.v1"
PATCH_SCHEMA = "trace2skill_patch.v2"
QUESTION_MERGE_SCHEMA = "trace2skill_question_merge.v2"
FINAL_MERGE_SCHEMA = "trace2skill_merge.v2"
NAMESPACE = "tooluniverse-trialqa"
SKILL_NAME = "tooluniverse-trialqa"
SKILL_PATH = f"{SKILL_NAME}/SKILL.md"
COMPACT_SKILL_MAX_BYTES = 4096
COMPACT_SKILL_MAX_WORDS = 650
COMPACT_SKILL_MAX_RULES = 10
CACHED_CATALOG_TRANSPORT_MODE = "cached-final-catalog-transport-v1"
DEVELOPMENT_LAYER_MODE = "exposed-development-layer-v1"
MECHANISM_REPAIR_MODE = "exposed-mechanism-repair-v1"
SEARCH_DISCIPLINE_REPAIR_MODE = "exposed-search-discipline-repair-v1"
IDENTIFIER_TERMINAL_REPAIR_MODE = "exposed-identifier-terminal-repair-v1"
DISTILLER_ROUTE = "sd-distiller"
DISTILLER_MODEL = "aws/anthropic/bedrock-claude-opus-4-8"
EXECUTOR_ROUTE = "sd-executor"
EXECUTOR_MODEL = "nvidia/nvidia/nemotron-3-ultra"
EXECUTOR_MODEL_ALIASES = frozenset({EXECUTOR_MODEL, "nvidia/nemotron-3-ultra", "nemotron-3-ultra"})
JUDGE_VERIFIER = "trialqa-semantic-judge-v1"
FULL_QUESTION_COUNT = 24
FULL_REPEAT_COUNT = 5
FULL_EVIDENCE_COUNT = FULL_QUESTION_COUNT * FULL_REPEAT_COUNT
REFERENCE_REVISION = "0618068ccef126e2e5623cd44a379217dca449d8"
REFERENCE_ANALYST_RELATIVE_PATH = Path("skills/trajectory-analyst/SKILL.md")
REFERENCE_ANALYST_SHA256 = "baa1e77df5565736f73ae70f02c704e896e4e0277a70400fa250bb8bce73258a"

DEFAULT_CATEGORIES = frozenset(
    {
        "tool_discovery",
        "trial_identification",
        "evidence_retrieval",
        "answer_extraction",
        "exactness",
        "verification",
        "failure_avoidance",
        "other",
    }
)
RULE_TYPES = frozenset({"tool_rule", "failure_mode", "workflow_rule", "gotcha"})
RULE_TYPE_REQUIRED_FIELDS = {
    "tool_rule": frozenset({"tool_name", "rule", "when"}),
    "failure_mode": frozenset({"trigger", "symptom", "prevention"}),
    "workflow_rule": frozenset({"rule", "rationale"}),
    "gotcha": frozenset({"fact", "impact"}),
}
DEFAULT_CATEGORY_BY_RULE_TYPE = {
    "tool_rule": "evidence_retrieval",
    "failure_mode": "failure_avoidance",
    "workflow_rule": "other",
    "gotcha": "other",
}
PUBLIC_TRIALQA_TOOLS = frozenset(
    {
        "trialqa_load_active_skill",
        "trialqa_search",
        "trialqa_get_study",
        "trialqa_get_eligibility",
        "trialqa_get_descriptions",
        "trialqa_get_status_dates",
        "trialqa_get_outcome_measures",
        "trialqa_get_references",
        "trialqa_extract_outcomes",
        "trialqa_extract_adverse_events",
    }
)
UNSUPPORTED_TOOL_SENTINEL = "unsupported_tool_call"

_NATIVE_ID = re.compile(r"native-[0-9a-f]{32}\Z")
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_SHA256 = re.compile(r"(?:sha256:)?([0-9a-f]{64})\Z")
_NCT = re.compile(r"\bNCT\d{8,}\b", re.IGNORECASE)
_URL = re.compile(r"https?://[^\s<>()\]\[{}\"']+", re.IGNORECASE)
_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_NAMESPACE_TOOL = re.compile(r"__sy1n([1-9][0-9]{0,3})_(.+)\Z")
_NAMESPACE_TOOL_TOKEN = re.compile(r"__sy1n[1-9][0-9]{0,3}_[A-Za-z0-9_-]+")
_INTERNAL_TRIALQA_TOOL = re.compile(r"\bmcp__trialqa_+[A-Za-z0-9_-]+\b")
_TRIALQA_TOOL_TOKEN = re.compile(r"\btrialqa_+[A-Za-z0-9_-]+\b")
_KNOWN_UNSUPPORTED_TOOL_NAMES = frozenset(
    {
        "exec_command",
        "get_interventions",
        "get_protocol",
        "get_trial",
        "list_tools",
        "search_trials",
        "tools",
        "trialqa__search_trials",
        "trialqa_get_protocol",
        "trialqa_search_trials",
    }
)
_KNOWN_UNSUPPORTED_TOOL_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_-])(?:exec_command|get_interventions|get_protocol|get_trial|"
    r"list_tools|search_trials)(?![A-Za-z0-9_-])"
)
_STRUCTURAL_TEXT_FIELDS = frozenset(
    {
        "schema_version",
        "role",
        "source_task_name",
        "rule_type",
        "category",
        "tool_name",
        "target",
        "action",
        "question_group_key",
        "skill_name",
        "stage",
        "key",
        "route_model",
        "upstream_model",
        "request_id",
    }
)

JsonObject = dict[str, Any]
Stage = Literal["analyst", "question_merge", "final_merge"]
ToolContract = Literal["direct", "compact"]

TOOL_CONTRACTS: tuple[ToolContract, ...] = ("direct", "compact")
COMPACT_PUBLIC_TOOLS = (
    "trialqa_load_active_skill",
    "execute_tool",
    "grep_tools",
    "get_tool_info",
)
# Source aliases are evidence identities from the paid direct-tool run.  The
# compact candidate transports each alias to the exact ToolUniverse 1.1.11
# child name and argument shape; it does not pretend those aliases are callable
# on the compact MCP surface.
COMPACT_TRIALQA_TOOL_MAP = (
    (
        "trialqa_search",
        "ClinicalTrials_search_studies",
        (
            "query_cond?: string",
            "query_intr?: string",
            "query_term?: string",
            "filter_status?: string",
            "filter_phase?: string",
            "filter_study_type?: string",
            "page_size?: integer",
            "next_page_token?: string",
        ),
    ),
    ("trialqa_get_study", "ClinicalTrials_get_study", ("nct_id: string",)),
    (
        "trialqa_get_eligibility",
        "get_clinical_trial_eligibility_criteria",
        ("nct_ids: string[]",),
    ),
    (
        "trialqa_get_descriptions",
        "get_clinical_trial_descriptions",
        ("nct_ids: string[]", "description_type: brief|full"),
    ),
    (
        "trialqa_get_status_dates",
        "get_clinical_trial_status_and_dates",
        ("nct_ids: string[]",),
    ),
    (
        "trialqa_get_outcome_measures",
        "get_clinical_trial_outcome_measures",
        ("nct_ids: string[]", "outcome_measures?: primary|secondary|all"),
    ),
    (
        "trialqa_get_references",
        "get_clinical_trial_references",
        ("nct_ids: string[]",),
    ),
    (
        "trialqa_extract_outcomes",
        "extract_clinical_trial_outcomes",
        ("nct_ids: string[]", "outcome_measure?: primary|secondary|all|name"),
    ),
    (
        "trialqa_extract_adverse_events",
        "extract_clinical_trial_adverse_events",
        (
            "nct_ids: string[]",
            "organ_systems?: string[]",
            "adverse_event_type?: serious|other|all|name",
        ),
    ),
)


class TrialQADistillationError(RuntimeError):
    """Raised when native TrialQA distillation cannot be proven safe."""


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject redirects so donor prompts cannot leave the attested localhost origin."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


@dataclass(frozen=True)
class ModelCall:
    """One deterministic request to the local Switchyard distiller route."""

    stage: Stage
    key: str
    payload: JsonObject
    input_sha256: str


@dataclass(frozen=True)
class ModelCallResult:
    """Text plus routing evidence returned by an injected model caller."""

    content: str
    route_model: str
    upstream_model: str
    request_id: str
    usage: Mapping[str, Any] | None = None


class ModelCaller(Protocol):
    """Injectable boundary for all paid/model execution."""

    def __call__(self, call: ModelCall) -> ModelCallResult: ...


StatsReader = Callable[[], Mapping[str, Any]]


@dataclass(frozen=True)
class DonorEvidence:
    """A validated native donor bundle and its trusted grouping metadata."""

    evidence_id: str
    path: Path
    document: JsonObject
    manifest_sha256: str
    donor_run_id: str
    question_group_key: str
    repeat_index: int
    role: Literal["success", "error"]
    judge_result: Literal["correct", "incorrect", "unsure"]
    observed_tools: frozenset[str]
    sensitive_literals: tuple[str, ...]


@dataclass(frozen=True)
class DistillationPlan:
    """Content-addressed immutable plan for a resumable distillation run."""

    run_id: str
    run_path: Path
    namespace: str
    project_dir: Path
    routing_profile: Path
    routing_profile_sha256: str
    proxy_url: str
    reference_instruction: Path
    reference_instruction_sha256: str
    reference_instruction_text: str
    evidence: tuple[DonorEvidence, ...]
    expected_question_count: int
    expected_repeats: int
    mode: Literal["full", "pilot"]
    manifest: JsonObject


@dataclass(frozen=True)
class CompactDistillationPlan:
    """One-call compact merge over integrity-checked question aggregates."""

    run_id: str
    run_path: Path
    namespace: str
    project_dir: Path
    routing_profile: Path
    routing_profile_sha256: str
    proxy_url: str
    tool_contract: ToolContract
    source_run: Path
    source_run_id: str
    source_evidence_ids: tuple[str, ...]
    question_aggregates: tuple[JsonObject, ...]
    observed_tools: frozenset[str]
    source_bindings: tuple[JsonObject, ...]
    transport_source_final_catalog: bool
    source_final_catalog: JsonObject | None
    source_final_binding: JsonObject | None
    source_final_attestation: JsonObject | None
    paid_raw_run: Path | None
    paid_raw_binding: JsonObject | None
    manifest: JsonObject


@dataclass(frozen=True)
class DevelopmentEvidence:
    """One integrity-checked skilled evaluation bundle used as exposed development."""

    evidence_id: str
    path: Path
    document: JsonObject
    manifest_sha256: str
    question_group_key: str
    repeat_index: int
    role: Literal["failure", "support"]
    direct_support_observed: bool


@dataclass(frozen=True)
class DevelopmentLayerPlan:
    """Deterministic layer over one train-derived compact parent candidate."""

    run_id: str
    run_path: Path
    namespace: str
    project_dir: Path
    parent_candidate_id: str
    parent_candidate_path: Path
    parent_manifest_sha256: str
    parent_skill_sha256: str
    parent_catalog: JsonObject
    parent_catalog_binding: JsonObject
    train_evidence_ids: tuple[str, ...]
    failure_evidence: DevelopmentEvidence
    support_evidence: DevelopmentEvidence
    verdict_path: Path
    verdict: JsonObject
    verdict_binding: JsonObject
    descriptive_manifest_path: Path
    descriptive_manifest_binding: JsonObject
    primary_manifest_path: Path
    primary_manifest_binding: JsonObject
    manifest: JsonObject


@dataclass(frozen=True)
class DistillationResult:
    """Saved candidate and optional activation produced by execution."""

    run_id: str
    candidate_id: str
    candidate_path: Path
    skill_path: Path
    validation_report_path: Path
    activated: bool
    model_call_count: int


class LocalSwitchyardCaller:
    """Minimal OpenAI-compatible client restricted to a localhost proxy."""

    def __init__(self, proxy_url: str, *, timeout: float = 1800.0) -> None:
        self.proxy_url = _validate_local_proxy_url(proxy_url)
        if not math.isfinite(timeout) or timeout <= 0:
            raise TrialQADistillationError("model timeout must be positive and finite")
        self.timeout = timeout
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _NoRedirectHandler(),
        )

    def __call__(self, call: ModelCall) -> ModelCallResult:
        endpoint = f"{self.proxy_url.rstrip('/')}/chat/completions"
        request = urllib.request.Request(
            endpoint,
            data=_canonical_bytes(call.payload),
            headers={
                "Authorization": "Bearer switchyard-local",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                if response.geturl().rstrip("/") != endpoint.rstrip("/"):
                    raise TrialQADistillationError(
                        "local Switchyard completion changed origin or endpoint"
                    )
                response_bytes = response.read()
                header_request_id = response.headers.get("x-request-id")
        except (OSError, urllib.error.URLError) as exc:
            raise TrialQADistillationError(
                f"local Switchyard model call failed for {call.stage}/{call.key}"
            ) from exc
        body = _decode_json_object(response_bytes, "Switchyard chat completion")
        choices = body.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise TrialQADistillationError("Switchyard completion must contain one choice")
        message = choices[0].get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise TrialQADistillationError("Switchyard completion has no text content")
        observed_model = body.get("model")
        request_id = body.get("id") or header_request_id
        if not isinstance(observed_model, str) or not observed_model.strip():
            raise TrialQADistillationError("Switchyard completion omitted its upstream model")
        if not isinstance(request_id, str) or not request_id.strip():
            raise TrialQADistillationError("Switchyard completion omitted its request id")
        usage = body.get("usage")
        return ModelCallResult(
            content=cast(str, message["content"]),
            route_model=DISTILLER_ROUTE,
            upstream_model=observed_model,
            request_id=request_id,
            usage=usage if isinstance(usage, dict) else None,
        )

    def read_stats(self) -> Mapping[str, Any]:
        endpoint = f"{self.proxy_url.rstrip('/')}/routing/stats"
        request = urllib.request.Request(
            endpoint,
            headers={"Authorization": "Bearer switchyard-local"},
        )
        try:
            with self._opener.open(request, timeout=10) as response:
                if response.geturl().rstrip("/") != endpoint.rstrip("/"):
                    raise TrialQADistillationError(
                        "local Switchyard stats changed origin or endpoint"
                    )
                return _decode_json_object(response.read(), "Switchyard routing stats")
        except (OSError, urllib.error.URLError) as exc:
            raise TrialQADistillationError("could not read local Switchyard routing stats") from exc


class _PaidRawNoCallCaller:
    """Fail closed if paid-raw recovery reaches the model boundary."""

    def __call__(self, call: ModelCall) -> ModelCallResult:
        del call
        raise TrialQADistillationError("paid raw transport adaptation attempted a model call")


def _canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TrialQADistillationError("value is not finite JSON") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_json_object(payload: bytes, label: str) -> JsonObject:
    try:
        value: object = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrialQADistillationError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise TrialQADistillationError(f"{label} must be a JSON object")
    return cast(JsonObject, value)


def _read_json_file(path: Path, label: str) -> JsonObject:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise TrialQADistillationError(f"missing {label}: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise TrialQADistillationError(f"{label} must be a single-link regular file: {path}")
    return _decode_json_object(path.read_bytes(), label)


def _read_jsonl_file(path: Path, label: str) -> list[JsonObject]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise TrialQADistillationError(f"missing {label}: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise TrialQADistillationError(f"{label} must be a single-link regular file: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise TrialQADistillationError(f"{label} is not UTF-8") from exc
    values: list[JsonObject] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            value: object = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrialQADistillationError(f"{label} line {line_number} is not valid JSON") from exc
        if not isinstance(value, dict):
            raise TrialQADistillationError(f"{label} line {line_number} must be an object")
        values.append(cast(JsonObject, value))
    return values


def _required_text(value: object, field: str, *, minimum: int = 1) -> str:
    if not isinstance(value, str) or len(value.strip()) < minimum:
        raise TrialQADistillationError(f"{field} must contain at least {minimum} characters")
    return value.strip()


def _required_int(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TrialQADistillationError(f"{field} must be an integer >= {minimum}")
    return value


def _number(value: object, field: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TrialQADistillationError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise TrialQADistillationError(
            f"{field} must be finite and between {minimum} and {maximum}"
        )
    return result


def _safe_component(value: str, field: str) -> str:
    if value in {"", ".", ".."} or _SAFE_COMPONENT.fullmatch(value) is None:
        raise TrialQADistillationError(f"unsafe {field}: {value!r}")
    return value


def _real_directory(path: Path, label: str) -> Path:
    absolute = path.expanduser().absolute()
    if absolute.is_symlink() or not absolute.is_dir():
        raise TrialQADistillationError(f"{label} must be a real directory: {absolute}")
    return absolute.resolve(strict=True)


def _real_file(path: Path, label: str) -> Path:
    absolute = path.expanduser().absolute()
    if absolute.is_symlink() or not absolute.is_file():
        raise TrialQADistillationError(f"{label} must be a real file: {absolute}")
    return absolute.resolve(strict=True)


def _validate_local_proxy_url(value: str) -> str:
    parsed = urllib.parse.urlparse(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise TrialQADistillationError("distillation proxy must be an explicit localhost HTTP URL")
    path = parsed.path.rstrip("/")
    if path not in {"", "/v1"}:
        raise TrialQADistillationError("distillation proxy path must be /v1")
    host = f"[{parsed.hostname}]" if parsed.hostname == "::1" else parsed.hostname
    return f"http://{host}:{parsed.port}/v1"


def _validate_routing_profile(path: Path) -> tuple[Path, str]:
    profile = _real_file(path, "routing profile")
    try:
        document: object = yaml.safe_load(profile.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise TrialQADistillationError(f"invalid routing profile: {profile}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("routes"), dict):
        raise TrialQADistillationError("routing profile must contain a routes mapping")
    route = document["routes"].get(DISTILLER_ROUTE)
    if not isinstance(route, dict) or route.get("type") != "model":
        raise TrialQADistillationError(f"routing profile has no model route {DISTILLER_ROUTE}")
    target = route.get("target")
    if not isinstance(target, dict) or target.get("model") != DISTILLER_MODEL:
        raise TrialQADistillationError(f"{DISTILLER_ROUTE} must target exactly {DISTILLER_MODEL}")
    return profile, _file_sha256(profile)


def _load_reference_instruction(reference_repo: Path) -> tuple[Path, str]:
    repo = _real_directory(reference_repo, "pinned reference repository")
    try:
        revision = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TrialQADistillationError(
            f"could not attest pinned reference revision: {repo}"
        ) from exc
    if revision != REFERENCE_REVISION:
        raise TrialQADistillationError(
            f"reference repository must be pinned to {REFERENCE_REVISION}, got {revision}"
        )
    instruction = _real_file(
        repo / REFERENCE_ANALYST_RELATIVE_PATH,
        "pinned trajectory-analyst instruction",
    )
    digest = _file_sha256(instruction)
    if digest != REFERENCE_ANALYST_SHA256:
        raise TrialQADistillationError(
            "trajectory-analyst instructions do not match pinned reference "
            f"{REFERENCE_REVISION}: {digest}"
        )
    try:
        text = instruction.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise TrialQADistillationError("trajectory-analyst instructions are not UTF-8") from exc
    return instruction, text


def _mapping(value: object, field: str) -> JsonObject:
    if not isinstance(value, dict):
        raise TrialQADistillationError(f"{field} must be a JSON object")
    return cast(JsonObject, value)


def _list(value: object, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise TrialQADistillationError(f"{field} must be a JSON list")
    return value


def _public_tool_name(name: str, *, structural: bool = False) -> str:
    """Decode one tool name, failing closed for structural trajectory fields."""

    if name.startswith("__sy1n"):
        match = _NAMESPACE_TOOL.fullmatch(name)
        if match is None:
            raise TrialQADistillationError(f"malformed namespace tool name: {name!r}")
        namespace_length = int(match.group(1))
        payload = match.group(2)
        if namespace_length >= len(payload):
            raise TrialQADistillationError(f"malformed namespace tool length: {name!r}")
        namespace = payload[:namespace_length]
        child = payload[namespace_length:]
        if namespace != "mcp__tooluniverse" or child not in PUBLIC_TRIALQA_TOOLS:
            return UNSUPPORTED_TOOL_SENTINEL
        if _SAFE_COMPONENT.fullmatch(child) is None:
            raise TrialQADistillationError(f"unsafe public TrialQA tool name: {child!r}")
        return child
    if name in PUBLIC_TRIALQA_TOOLS:
        return name
    if (
        name in _KNOWN_UNSUPPORTED_TOOL_NAMES
        or name.startswith("mcp__trialqa")
        or (name.startswith("trialqa") and name not in PUBLIC_TRIALQA_TOOLS)
        or structural
    ):
        return UNSUPPORTED_TOOL_SENTINEL
    return name


def _public_tool_view(value: object) -> object:
    """Recursively remove transport/internal tool spellings from model-visible data."""

    if isinstance(value, str):
        if value in _KNOWN_UNSUPPORTED_TOOL_NAMES:
            return UNSUPPORTED_TOOL_SENTINEL
        normalized = _NAMESPACE_TOOL_TOKEN.sub(
            lambda match: _public_tool_name(match.group(0), structural=True), value
        )
        normalized = _INTERNAL_TRIALQA_TOOL.sub(UNSUPPORTED_TOOL_SENTINEL, normalized)
        normalized = _TRIALQA_TOOL_TOKEN.sub(
            lambda match: _public_tool_name(match.group(0), structural=True), normalized
        )
        normalized = _KNOWN_UNSUPPORTED_TOOL_TOKEN.sub(UNSUPPORTED_TOOL_SENTINEL, normalized)
        if "__sy1n" in normalized or "mcp__trialqa" in normalized:
            raise TrialQADistillationError(
                f"malformed or residual internal tool name in model-visible text: {value!r}"
            )
        return normalized
    if isinstance(value, list):
        return [_public_tool_view(item) for item in value]
    if isinstance(value, dict):
        is_tool_function = "arguments" in value and isinstance(value.get("name"), str)
        normalized_mapping: dict[str, object] = {}
        for key, item in value.items():
            if isinstance(item, str) and (
                key == "tool_name" or (key == "name" and is_tool_function)
            ):
                normalized_mapping[str(key)] = _public_tool_name(item, structural=True)
            else:
                normalized_mapping[str(key)] = _public_tool_view(item)
        return normalized_mapping
    return value


def _observed_tools(events: Sequence[Any]) -> frozenset[str]:
    names: set[str] = set()
    for raw_event in events:
        event = _mapping(raw_event, "evidence event")
        if event.get("kind") != "tool_call":
            continue
        payload = _mapping(event.get("payload"), "tool-call payload")
        function = payload.get("function")
        raw_names: list[str] = []
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            raw_names.append(cast(str, function["name"]))
        if isinstance(payload.get("name"), str):
            raw_names.append(cast(str, payload["name"]))
        unique_raw_names = set(raw_names)
        if len(unique_raw_names) != 1:
            continue
        name = _public_tool_name(unique_raw_names.pop(), structural=True)
        if name in PUBLIC_TRIALQA_TOOLS:
            names.add(name)
    return frozenset(names)


def _assert_no_internal_tool_names(value: object, label: str) -> None:
    """Reject transport spellings, placeholders, and unsupported TrialQA aliases."""

    for text in _strings(value):
        if (
            "__sy1n" in text
            or "mcp__trialqa" in text
            or UNSUPPORTED_TOOL_SENTINEL in text
            or _KNOWN_UNSUPPORTED_TOOL_TOKEN.search(text)
        ):
            raise TrialQADistillationError(f"{label} contains an internal tool name")
        for match in _TRIALQA_TOOL_TOKEN.finditer(text):
            if match.group(0) not in PUBLIC_TRIALQA_TOOLS:
                raise TrialQADistillationError(
                    f"{label} contains unsupported TrialQA tool {match.group(0)!r}"
                )


def _strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _sensitive_literals(document: Mapping[str, Any]) -> tuple[str, ...]:
    task = _mapping(document.get("task"), "evidence.task")
    outcome = _mapping(document.get("outcome"), "evidence.outcome")
    candidates: set[str] = set()
    for field in ("id", "row_id", "question_id", "question_group_key", "task_name"):
        value = task.get(field)
        if isinstance(value, str) and len(value.strip()) >= 5:
            candidates.add(value.strip())
    for field in ("row_id", "question_id", "question_group_key", "task_name"):
        value = outcome.get(field)
        if isinstance(value, str) and len(value.strip()) >= 5:
            candidates.add(value.strip())
    for field in ("question",):
        value = task.get(field)
        if isinstance(value, str) and len(value.strip()) >= 20:
            candidates.add(value.strip())
    for field in (
        "ideal",
        "ideal_answer",
        "expected_answer",
        "answer",
        "submitted_answer",
        "judge_rationale",
    ):
        value = outcome.get(field)
        if isinstance(value, str) and value.strip():
            candidates.add(value.strip())
    for value in _strings({"task": task, "outcome": outcome}):
        candidates.update(match.group(0) for match in _NCT.finditer(value))
        candidates.update(match.group(0) for match in _URL.finditer(value))
    return tuple(sorted(candidates, key=lambda item: (-len(item), item)))


def _stats_model_calls(stats: Mapping[str, Any]) -> dict[str, int]:
    models = stats.get("models")
    if not isinstance(models, dict):
        raise TrialQADistillationError("session stats must contain a models object")
    result: dict[str, int] = {}
    for name, raw in models.items():
        if not isinstance(name, str) or not isinstance(raw, dict):
            raise TrialQADistillationError("session stats models are malformed")
        result[name] = _required_int(raw.get("calls"), f"stats.models.{name}.calls")
    return result


def _validate_executor_source(evidence_dir: Path, document: JsonObject) -> None:
    """Cross-check caller-supplied evidence metadata against raw capture artifacts."""

    raw = evidence_dir / "raw"
    session = _read_json_file(raw / "session.json", "raw native session")
    task_artifact = _read_json_file(raw / "task.json", "raw donor task")
    outcome_artifact = _read_json_file(raw / "outcome.json", "raw donor outcome")
    run_artifact = _read_json_file(raw / "run.json", "raw donor run")
    stats = _read_json_file(raw / "stats.json", "raw donor stats")
    raw_turns = _read_jsonl_file(raw / "turns.jsonl", "raw donor trajectory")
    task = _mapping(document.get("task"), "evidence.task")
    outcome = _mapping(document.get("outcome"), "evidence.outcome")
    if outcome.get("verifier") != JUDGE_VERIFIER:
        raise TrialQADistillationError(
            f"TrialQA donor outcome must use pinned verifier {JUDGE_VERIFIER!r}"
        )
    execution = _mapping(document.get("execution"), "evidence.execution")

    if task_artifact != task or outcome_artifact != outcome:
        raise TrialQADistillationError("native evidence task/outcome differ from raw artifacts")
    for key, value in run_artifact.items():
        if execution.get(key) != value:
            raise TrialQADistillationError(
                f"native evidence execution.{key} differs from raw run metadata"
            )
    if (
        session.get("status") != "completed"
        or session.get("exit_code") != 0
        or session.get("display_model") != EXECUTOR_ROUTE
    ):
        raise TrialQADistillationError(
            "donor session must be completed with exit 0 through sd-executor"
        )
    if task.get("partition") != "train" or run_artifact.get("phase") != "donor":
        raise TrialQADistillationError("only train-partition donor evidence may be distilled")
    if outcome.get("partition") != "train":
        raise TrialQADistillationError("donor outcome partition does not match train evidence")
    condition = _required_text(task.get("condition"), "task.condition")
    if condition != "donor":
        raise TrialQADistillationError("donor evidence condition must be exactly 'donor'")
    if run_artifact.get("route") != EXECUTOR_ROUTE:
        raise TrialQADistillationError("donor run route must be sd-executor")
    if run_artifact.get("model") not in EXECUTOR_MODEL_ALIASES:
        raise TrialQADistillationError("donor run model must be Nemotron-3-Ultra")
    candidate_fields = (
        "candidate_id",
        "candidate_manifest_sha256",
        "candidate_skill_sha256",
    )
    if (
        "skill_loaded" not in run_artifact
        or any(field not in run_artifact for field in candidate_fields)
        or run_artifact.get("skill_loaded") is not False
        or any(run_artifact.get(field) is not None for field in candidate_fields)
    ):
        raise TrialQADistillationError("donor run metadata must attest an unskilled execution")

    active = _mapping(session.get("active_skill"), "session.active_skill")
    active_candidate_fields = (
        "candidate_id",
        "manifest_sha256",
        "skill_sha256",
        "path",
    )
    if (
        any(field not in active for field in ("loaded", *active_candidate_fields))
        or active.get("loaded") is not False
        or any(active.get(field) is not None for field in active_candidate_fields)
    ):
        raise TrialQADistillationError("donor session must attest that no skill was loaded")
    context = _mapping(session.get("run_context"), "session.run_context")
    comparisons = {
        "task_id": task.get("id"),
        "row_id": task.get("row_id", task.get("question_id")),
        "partition": task.get("partition"),
        "condition": task.get("condition"),
        "repeat_index": task.get("repeat_index"),
        "n_repeats": task.get("n_repeats"),
        "manifest_id": run_artifact.get("run_id"),
        "phase": run_artifact.get("phase"),
        "route": run_artifact.get("route"),
        "executor_model": run_artifact.get("model"),
        "skill_loaded": run_artifact.get("skill_loaded"),
        "candidate_id": run_artifact.get("candidate_id"),
        "candidate_manifest_sha256": run_artifact.get("candidate_manifest_sha256"),
        "candidate_skill_sha256": run_artifact.get("candidate_skill_sha256"),
    }
    if task.get("question_group_key") is not None:
        comparisons["question_group_key"] = task["question_group_key"]
    for field, expected in comparisons.items():
        if field not in context or context.get(field) != expected:
            raise TrialQADistillationError(
                f"session.run_context.{field} does not match imported donor metadata"
            )

    outcome_bindings = {
        "row_id": task.get("row_id", task.get("question_id")),
        "question": task.get("question"),
        "question_group_key": task.get("question_group_key"),
        "repeat_index": task.get("repeat_index"),
        "n_repeats": task.get("n_repeats"),
        "task_name": task.get("id"),
        "condition": task.get("condition"),
    }
    for field, expected in outcome_bindings.items():
        if field not in outcome or outcome.get(field) != expected:
            raise TrialQADistillationError(
                f"donor outcome.{field} does not match the trusted task metadata"
            )
    _required_text(outcome.get("ideal_answer"), "donor outcome.ideal_answer")
    _required_text(outcome.get("submitted_answer"), "donor outcome.submitted_answer")
    _required_text(outcome.get("judge_rationale"), "donor outcome.judge_rationale")

    total_requests = _required_int(stats.get("total_requests"), "stats.total_requests", minimum=1)
    total_errors = _required_int(stats.get("total_errors"), "stats.total_errors")
    if total_errors != 0:
        raise TrialQADistillationError("donor session stats contain executor errors")
    model_calls = _stats_model_calls(stats)
    if not model_calls or set(model_calls).difference(EXECUTOR_MODEL_ALIASES):
        raise TrialQADistillationError("donor session stats contain a non-Ultra model")
    if sum(model_calls.values()) != total_requests:
        raise TrialQADistillationError("donor model-call stats do not match total requests")
    if len(raw_turns) != total_requests:
        raise TrialQADistillationError("raw donor turn count does not match executor stats")
    for index, turn in enumerate(raw_turns):
        request = _mapping(turn.get("request"), f"raw donor turn {index} request")
        if request.get("model") != EXECUTOR_ROUTE:
            raise TrialQADistillationError("raw donor request did not use sd-executor")
        if turn.get("served_model") not in EXECUTOR_MODEL_ALIASES:
            raise TrialQADistillationError("raw donor turn was not served by Ultra")
        turn_skill_fields = (
            "active_skill_version",
            "active_skill_candidate_id",
            "active_skill_manifest_sha256",
        )
        if any(field not in turn for field in turn_skill_fields) or any(
            turn.get(field) is not None for field in turn_skill_fields
        ):
            raise TrialQADistillationError("raw donor turn records an active skill")

    served_models = execution.get("served_models")
    if (
        not isinstance(served_models, list)
        or not served_models
        or any(model not in EXECUTOR_MODEL_ALIASES for model in served_models)
    ):
        raise TrialQADistillationError("native donor turns were not all served by Ultra")
    events = _list(document.get("events"), "evidence.events")
    response_turns: set[int] = set()
    for raw_event in events:
        event = _mapping(raw_event, "evidence event")
        metadata = _mapping(event.get("metadata"), "evidence event metadata")
        if metadata.get("source") == "response":
            response_turns.add(
                _required_int(metadata.get("turn_index"), "response event turn_index")
            )
            if metadata.get("served_model") not in EXECUTOR_MODEL_ALIASES:
                raise TrialQADistillationError("donor event has a non-Ultra served model")
    if response_turns != set(range(total_requests)):
        raise TrialQADistillationError("donor event count does not match executor stats")
    final_events = [
        _mapping(event, "final output event")
        for event in events
        if isinstance(event, dict) and event.get("kind") == "final_output"
    ]
    if len(final_events) != 1:
        raise TrialQADistillationError("donor evidence must contain exactly one final output")
    final_payload = _mapping(final_events[0].get("payload"), "final output payload")
    raw_final_answer = final_payload.get("content")
    if not isinstance(raw_final_answer, str) or not raw_final_answer.strip():
        raise TrialQADistillationError("donor final output must contain a nonempty answer")
    final_answer = raw_final_answer.strip()
    if final_answer.startswith("{"):
        try:
            parsed_final: object = json.loads(final_answer)
        except json.JSONDecodeError as exc:
            raise TrialQADistillationError(
                "donor structured final output is not valid JSON"
            ) from exc
        if (
            not isinstance(parsed_final, dict)
            or set(parsed_final) != {"answer"}
            or not isinstance(parsed_final.get("answer"), str)
            or not cast(str, parsed_final["answer"]).strip()
        ):
            raise TrialQADistillationError(
                "donor structured final output must be exactly {answer: string}"
            )
        final_answer = cast(str, parsed_final["answer"]).strip()
    recorded_answer = outcome.get("submitted_answer", outcome.get("answer"))
    if recorded_answer is not None and (
        not isinstance(recorded_answer, str) or recorded_answer.strip() != final_answer
    ):
        raise TrialQADistillationError(
            "donor outcome answer differs from the trajectory final output"
        )


def _load_donor_evidence(store: SkillDistillationStore, evidence_id: str) -> DonorEvidence:
    if _NATIVE_ID.fullmatch(evidence_id) is None:
        raise TrialQADistillationError(f"unsupported donor evidence id: {evidence_id!r}")
    path = store.evidence_path / evidence_id
    validate_native_trialqa_evidence_directory(path, expected_evidence_id=evidence_id)
    document = _read_json_file(path / "evidence.json", "native evidence document")
    manifest_path = path / "manifest.json"
    _read_json_file(manifest_path, "native evidence manifest")
    _validate_executor_source(path, document)
    task = _mapping(document.get("task"), "evidence.task")
    outcome = _mapping(document.get("outcome"), "evidence.outcome")
    execution = _mapping(document.get("execution"), "evidence.execution")
    donor_run_id = _required_text(execution.get("run_id"), "evidence.execution.run_id")
    group = _required_text(
        task.get("question_group_key") or task.get("question_id") or task.get("row_id"),
        "task.question_group_key",
    )
    _safe_component(group, "question group key")
    repeat_index = _required_int(task.get("repeat_index"), "task.repeat_index", minimum=1)
    n_repeats = _required_int(task.get("n_repeats"), "task.n_repeats", minimum=1)
    if repeat_index > n_repeats:
        raise TrialQADistillationError("task.repeat_index exceeds task.n_repeats")
    score = _number(outcome.get("score"), "outcome.score", minimum=0, maximum=1)
    if score not in {0.0, 1.0}:
        raise TrialQADistillationError("TrialQA donor outcome score must be exactly 0 or 1")
    judge_result = outcome.get("judge_result", outcome.get("label"))
    if judge_result not in {"correct", "incorrect", "unsure"}:
        raise TrialQADistillationError("TrialQA donor outcome has an invalid judge_result")
    if (judge_result == "correct") != (score == 1.0):
        raise TrialQADistillationError("TrialQA judge_result and score are inconsistent")
    role: Literal["success", "error"] = "success" if score == 1.0 else "error"
    events = _list(document.get("events"), "evidence.events")
    return DonorEvidence(
        evidence_id=evidence_id,
        path=path.resolve(strict=True),
        document=document,
        manifest_sha256=_file_sha256(manifest_path),
        donor_run_id=donor_run_id,
        question_group_key=group,
        repeat_index=repeat_index,
        role=role,
        judge_result=cast(Literal["correct", "incorrect", "unsure"], judge_result),
        observed_tools=_observed_tools(events),
        sensitive_literals=tuple(
            sorted(
                {*_sensitive_literals(document), evidence_id},
                key=lambda value: (-len(value), value),
            )
        ),
    )


def _validate_evidence_set(
    evidence: Sequence[DonorEvidence],
    *,
    expected_question_count: int,
    expected_repeats: int,
    require_complete: bool,
) -> None:
    if not evidence:
        raise TrialQADistillationError("at least one native donor evidence id is required")
    if expected_question_count < 1 or expected_repeats < 1:
        raise TrialQADistillationError("expected question/repeat counts must be positive")
    pair_keys: set[tuple[str, int]] = set()
    questions: dict[str, str] = {}
    for item in evidence:
        pair = (item.question_group_key, item.repeat_index)
        if pair in pair_keys:
            raise TrialQADistillationError(f"duplicate donor pair key: {pair!r}")
        pair_keys.add(pair)
        task = _mapping(item.document.get("task"), "evidence.task")
        if task.get("n_repeats") != expected_repeats:
            raise TrialQADistillationError("donor evidence n_repeats differs from the plan")
        question = _required_text(task.get("question"), "task.question")
        previous = questions.setdefault(item.question_group_key, question)
        if previous != question:
            raise TrialQADistillationError("one question group contains different question text")
    groups = {item.question_group_key for item in evidence}
    if require_complete:
        donor_run_ids = {item.donor_run_id for item in evidence}
        if len(donor_run_ids) != 1:
            raise TrialQADistillationError(
                "full donor evidence must come from exactly one experiment manifest"
            )
        expected_pairs = {
            (group, repeat) for group in groups for repeat in range(1, expected_repeats + 1)
        }
        if len(groups) != expected_question_count or pair_keys != expected_pairs:
            raise TrialQADistillationError(
                "donor evidence is not the complete expected question x repeat matrix"
            )


def build_distillation_plan(
    *,
    project_dir: Path,
    namespace: str,
    evidence_ids: Sequence[str],
    work_dir: Path,
    reference_repo: Path,
    routing_profile: Path,
    proxy_url: str,
    expected_question_count: int = 24,
    expected_repeats: int = 5,
    mode: Literal["full", "pilot"] = "full",
) -> DistillationPlan:
    """Validate all local inputs and return a deterministic no-call plan."""

    namespace = _safe_component(namespace, "namespace")
    if namespace != NAMESPACE:
        raise TrialQADistillationError(f"TrialQA namespace must be exactly {NAMESPACE!r}")
    project = _real_directory(project_dir, "project directory")
    work = work_dir.expanduser().absolute()
    if work.is_symlink():
        raise TrialQADistillationError(f"distillation work directory cannot be a symlink: {work}")
    if work.exists() and not work.is_dir():
        raise TrialQADistillationError(f"distillation work path is not a directory: {work}")
    profile, profile_sha = _validate_routing_profile(routing_profile)
    instruction, instruction_text = _load_reference_instruction(reference_repo)
    local_proxy = _validate_local_proxy_url(proxy_url)
    normalized_ids = tuple(sorted(set(evidence_ids)))
    if len(normalized_ids) != len(evidence_ids):
        raise TrialQADistillationError("donor evidence ids must be unique")
    store = SkillDistillationStore(namespace, project)
    evidence = tuple(_load_donor_evidence(store, item) for item in normalized_ids)
    evidence = tuple(
        sorted(evidence, key=lambda item: (item.question_group_key, item.repeat_index))
    )
    if mode not in {"full", "pilot"}:
        raise TrialQADistillationError(f"invalid distillation mode: {mode!r}")
    if mode == "full" and (
        expected_question_count != FULL_QUESTION_COUNT
        or expected_repeats != FULL_REPEAT_COUNT
        or len(evidence) != FULL_EVIDENCE_COUNT
    ):
        raise TrialQADistillationError(
            "full mode requires exactly 24 questions x 5 repeats = 120 evidence bundles"
        )
    if mode == "pilot" and len(evidence) != 1:
        raise TrialQADistillationError(
            "pilot mode requires exactly one donor evidence bundle and is non-performance"
        )
    _validate_evidence_set(
        evidence,
        expected_question_count=expected_question_count,
        expected_repeats=expected_repeats,
        require_complete=mode == "full",
    )
    manifest_seed: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "namespace": namespace,
        "pipeline": "analyst-per-evidence/question-repeat-merge/cross-question-merge",
        "implementation": {
            "trialqa_local_distiller_sha256": _file_sha256(Path(__file__).resolve()),
            "skill_distillation_native_sha256": _file_sha256(
                Path(native_evidence_module.__file__).resolve()
            ),
            "skill_distillation_store_sha256": _file_sha256(
                Path(skill_store_module.__file__).resolve()
            ),
        },
        "reference": {
            "revision": REFERENCE_REVISION,
            "instruction_path": REFERENCE_ANALYST_RELATIVE_PATH.as_posix(),
            "instruction_sha256": REFERENCE_ANALYST_SHA256,
        },
        "routing": {
            "route": DISTILLER_ROUTE,
            "upstream_model": DISTILLER_MODEL,
            "profile_sha256": profile_sha,
            "proxy_url": local_proxy,
        },
        "executor_attestation": {
            "route": EXECUTOR_ROUTE,
            "upstream_model": EXECUTOR_MODEL,
            "required_partition": "train",
            "required_phase": "donor",
            "required_skill_loaded": False,
            "required_total_errors": 0,
        },
        "matrix": {
            "expected_question_count": expected_question_count,
            "expected_repeats": expected_repeats,
            "mode": mode,
            "require_complete": mode == "full",
            "performance_eligible": mode == "full",
        },
        "source_evidence": [
            {
                "evidence_id": item.evidence_id,
                "manifest_sha256": item.manifest_sha256,
                "donor_run_id": item.donor_run_id,
                "question_group_key": item.question_group_key,
                "repeat_index": item.repeat_index,
                "role": item.role,
            }
            for item in evidence
        ],
    }
    run_id = f"trialqa-distill-{_digest(manifest_seed)[:32]}"
    manifest = {"run_id": run_id, **manifest_seed}
    return DistillationPlan(
        run_id=run_id,
        run_path=work / run_id,
        namespace=namespace,
        project_dir=project,
        routing_profile=profile,
        routing_profile_sha256=profile_sha,
        proxy_url=local_proxy,
        reference_instruction=instruction,
        reference_instruction_sha256=REFERENCE_ANALYST_SHA256,
        reference_instruction_text=instruction_text,
        evidence=evidence,
        expected_question_count=expected_question_count,
        expected_repeats=expected_repeats,
        mode=mode,
        manifest=manifest,
    )


def _bound_source_artifact(source: Path, entry: Mapping[str, Any]) -> Path:
    relative_text = _required_text(entry.get("path"), "source artifact path")
    relative = Path(relative_text)
    if relative.is_absolute() or ".." in relative.parts:
        raise TrialQADistillationError("source artifact path escapes the source run")
    path = source / relative
    if path.is_symlink() or not path.is_file():
        raise TrialQADistillationError(f"source artifact is missing or unsafe: {path}")
    expected_sha = entry.get("sha256")
    expected_size = entry.get("size_bytes")
    if expected_sha != f"sha256:{_file_sha256(path)}" or expected_size != path.stat().st_size:
        raise TrialQADistillationError(f"source artifact binding mismatch: {path}")
    return path


def build_compact_distillation_plan(
    *,
    project_dir: Path,
    namespace: str,
    work_dir: Path,
    source_run: Path,
    routing_profile: Path,
    proxy_url: str,
    paid_raw_run: Path | None = None,
    tool_contract: ToolContract = "direct",
    transport_source_final_catalog: bool = False,
) -> CompactDistillationPlan:
    """Plan one compact final merge without repeating paid analyst stages."""

    namespace = _safe_component(namespace, "namespace")
    if namespace != NAMESPACE:
        raise TrialQADistillationError(f"TrialQA namespace must be exactly {NAMESPACE!r}")
    if tool_contract not in TOOL_CONTRACTS:
        raise TrialQADistillationError(f"tool contract must be one of {', '.join(TOOL_CONTRACTS)}")
    if transport_source_final_catalog and tool_contract != "compact":
        raise TrialQADistillationError(
            "source-final catalog transport requires the compact tool contract"
        )
    if transport_source_final_catalog and paid_raw_run is not None:
        raise TrialQADistillationError(
            "source-final catalog transport cannot also recover a paid raw response"
        )
    project = _real_directory(project_dir, "project directory")
    work = work_dir.expanduser().absolute()
    if work.is_symlink():
        raise TrialQADistillationError(f"distillation work directory cannot be a symlink: {work}")
    if work.exists() and not work.is_dir():
        raise TrialQADistillationError(f"distillation work path is not a directory: {work}")
    source = _real_directory(source_run, "source distillation run")
    profile, profile_sha = _validate_routing_profile(routing_profile)
    local_proxy = _validate_local_proxy_url(proxy_url)

    source_manifest_path = source / "run_manifest.json"
    source_manifest = _read_json_file(source_manifest_path, "source run manifest")
    source_run_id = _required_text(source_manifest.get("run_id"), "source run id")
    if (
        source_manifest.get("schema_version") != SCHEMA_VERSION
        or source_manifest.get("namespace") != namespace
    ):
        raise TrialQADistillationError("source run manifest identity is invalid")
    matrix = _mapping(source_manifest.get("matrix"), "source run matrix")
    if (
        matrix.get("mode") != "full"
        or matrix.get("performance_eligible") is not True
        or matrix.get("expected_question_count") != FULL_QUESTION_COUNT
        or matrix.get("expected_repeats") != FULL_REPEAT_COUNT
    ):
        raise TrialQADistillationError("compact merge requires one complete full donor run")
    source_evidence = [
        _mapping(item, "source run evidence")
        for item in _list(source_manifest.get("source_evidence"), "source run evidence")
    ]
    if len(source_evidence) != FULL_EVIDENCE_COUNT:
        raise TrialQADistillationError("source run does not contain exactly 120 evidence rows")
    evidence_ids: list[str] = []
    evidence_matrix: dict[str, dict[int, str]] = {}
    for entry in source_evidence:
        evidence_id = _required_text(entry.get("evidence_id"), "source evidence id")
        group = _safe_component(
            _required_text(entry.get("question_group_key"), "source evidence question group"),
            "source evidence question group",
        )
        repeat = _required_int(entry.get("repeat_index"), "source evidence repeat index", minimum=1)
        role = entry.get("role")
        if repeat > FULL_REPEAT_COUNT or role not in {"success", "error"}:
            raise TrialQADistillationError("source evidence matrix metadata is invalid")
        group_repeats = evidence_matrix.setdefault(group, {})
        if repeat in group_repeats:
            raise TrialQADistillationError("source evidence matrix contains a duplicate repeat")
        group_repeats[repeat] = cast(str, role)
        evidence_ids.append(evidence_id)
    if (
        len(set(evidence_ids)) != FULL_EVIDENCE_COUNT
        or len(evidence_matrix) != FULL_QUESTION_COUNT
        or any(
            set(repeats) != set(range(1, FULL_REPEAT_COUNT + 1))
            for repeats in evidence_matrix.values()
        )
    ):
        raise TrialQADistillationError("source evidence is not an exact 24 x 5 matrix")

    validation_path = source / "candidate_validation.json"
    validation = _read_json_file(validation_path, "source candidate validation")
    source_ids_raw = validation.get("source_evidence_ids")
    validation_checks = _mapping(validation.get("checks"), "source validation checks")
    required_source_checks = {
        "all_evidence_native_and_content_validated",
        "all_evidence_train_donor",
        "all_executor_sessions_unskilled",
        "one_analyst_patch_per_evidence",
        "repeats_grouped_by_question",
        "distiller_route_only",
        "distiller_model_only",
        "routing_stats_accounted",
    }
    if (
        validation.get("schema_version") != SCHEMA_VERSION
        or validation.get("status") != "passed"
        or validation.get("performance_eligible") is not True
        or validation.get("run_id") != source_run_id
        or not isinstance(source_ids_raw, list)
        or len(source_ids_raw) != FULL_EVIDENCE_COUNT
        or not all(isinstance(item, str) for item in source_ids_raw)
    ):
        raise TrialQADistillationError("source candidate validation is not full-run eligible")
    source_ids = tuple(cast(list[str], source_ids_raw))
    if len(set(source_ids)) != len(source_ids):
        raise TrialQADistillationError("source candidate evidence ids are not unique")
    if source_ids != tuple(evidence_ids):
        raise TrialQADistillationError("source candidate evidence ids differ from the run manifest")
    if not required_source_checks.issubset(validation_checks) or any(
        value is not True for value in validation_checks.values()
    ):
        raise TrialQADistillationError("source candidate validation contains a failed check")

    completion_path = source / "completion_manifest.json"
    completion = _read_json_file(completion_path, "source completion manifest")
    if (
        completion.get("schema_version") != SCHEMA_VERSION
        or completion.get("run_id") != source_run_id
    ):
        raise TrialQADistillationError("source completion manifest identity is invalid")
    artifacts = _mapping(validation.get("artifacts"), "source validation artifacts")
    if artifacts.get("completion_manifest_sha256") != f"sha256:{_file_sha256(completion_path)}":
        raise TrialQADistillationError("source completion manifest hash mismatch")
    stage_entries = _list(completion.get("stage_artifacts"), "source stage artifacts")
    stage_paths = [
        _required_text(
            _mapping(entry, "source stage artifact").get("path"),
            "source stage artifact path",
        )
        for entry in stage_entries
    ]
    if len(stage_paths) != len(set(stage_paths)):
        raise TrialQADistillationError("source completion manifest repeats a stage artifact")
    analyst_paths = [
        path for path in stage_paths if path.startswith("analyst/") and path.endswith(".json")
    ]
    final_entries = [
        _mapping(entry, "source final catalog artifact")
        for entry in stage_entries
        if isinstance(entry, dict) and entry.get("path") == "final_catalog.json"
    ]
    if (
        len(stage_paths) != FULL_EVIDENCE_COUNT + FULL_QUESTION_COUNT + 1
        or len(analyst_paths) != FULL_EVIDENCE_COUNT
        or len(final_entries) != 1
    ):
        raise TrialQADistillationError("source completion manifest has an incomplete stage matrix")
    question_entries = [
        _mapping(entry, "source question artifact")
        for entry in stage_entries
        if isinstance(entry, dict)
        and isinstance(entry.get("path"), str)
        and cast(str, entry["path"]).startswith("questions/")
        and cast(str, entry["path"]).endswith(".json")
    ]
    if len(question_entries) != FULL_QUESTION_COUNT:
        raise TrialQADistillationError("source run does not contain exactly 24 question merges")

    question_aggregates: list[JsonObject] = []
    question_bindings: list[JsonObject] = []
    observed_tools: set[str] = set()
    seen_groups: set[str] = set()
    for entry in sorted(question_entries, key=lambda item: cast(str, item["path"])):
        path = _bound_source_artifact(source, entry)
        _validate_stage_integrity(path)
        artifact = _read_json_file(path, "source question stage")
        key = _required_text(artifact.get("key"), "source question key")
        input_sha = _required_text(artifact.get("input_sha256"), "source question input hash")
        if artifact.get("stage") != "question_merge" or key in seen_groups:
            raise TrialQADistillationError("source question stage identity is invalid")
        seen_groups.add(key)
        aggregate, _attestation = _load_stage_artifact(
            path,
            stage="question_merge",
            key=key,
            input_sha256=input_sha,
        )
        if (
            aggregate.get("schema_version") != QUESTION_MERGE_SCHEMA
            or aggregate.get("question_group_key") != key
            or aggregate.get("source_patch_count") != FULL_REPEAT_COUNT
            or aggregate.get("repeat_count") != FULL_REPEAT_COUNT
            or key not in evidence_matrix
        ):
            raise TrialQADistillationError("source question aggregate matrix is invalid")
        expected_roles = dict(sorted(Counter(evidence_matrix[key].values()).items()))
        if _mapping(aggregate.get("role_counts"), "source question role counts") != expected_roles:
            raise TrialQADistillationError("source question role counts differ from evidence")
        judge_counts = _mapping(
            aggregate.get("judge_result_counts"), "source question judge counts"
        )
        if (
            not judge_counts
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 1
                for value in judge_counts.values()
            )
            or sum(cast(int, value) for value in judge_counts.values()) != FULL_REPEAT_COUNT
        ):
            raise TrialQADistillationError("source question judge counts are invalid")
        aggregate_tools: set[str] = set()
        for group in _list(aggregate.get("tool_rules"), "source question tool rules"):
            tool_name = _required_text(
                _mapping(group, "source tool rule group").get("tool_name"),
                "source tool name",
            )
            if tool_name not in PUBLIC_TRIALQA_TOOLS:
                raise TrialQADistillationError(
                    f"source question aggregate references unsupported tool {tool_name!r}"
                )
            aggregate_tools.add(tool_name)
            observed_tools.add(tool_name)
        typed_rules = _typed_rules(aggregate)
        if not typed_rules:
            raise TrialQADistillationError("source question aggregate contains no rules")
        for index, (_rule_type, item) in enumerate(typed_rules):
            _validate_rule_item(
                item,
                field=f"source question rule[{index}]",
                observed_tools=frozenset(aggregate_tools),
            )
            source_count = _required_int(
                item.get("source_patch_count"),
                f"source question rule[{index}].source_patch_count",
                minimum=1,
            )
            if source_count > FULL_REPEAT_COUNT:
                raise TrialQADistillationError(
                    "source question rule exceeds the repeat evidence count"
                )
        question_aggregates.append(aggregate)
        question_bindings.append(
            {
                "path": path.relative_to(source).as_posix(),
                "sha256": f"sha256:{_file_sha256(path)}",
                "size_bytes": path.stat().st_size,
                "integrity_sha256": f"sha256:{_file_sha256(_integrity_path(path))}",
            }
        )
    if "trialqa_search" not in observed_tools:
        raise TrialQADistillationError("source question aggregates lack trialqa_search evidence")
    if seen_groups != set(evidence_matrix):
        raise TrialQADistillationError("source question merges differ from the evidence matrix")

    source_final_catalog: JsonObject | None = None
    source_final_binding: JsonObject | None = None
    source_final_attestation: JsonObject | None = None
    if transport_source_final_catalog:
        source_final_path = _bound_source_artifact(source, final_entries[0])
        source_final_stage = _read_json_file(source_final_path, "source final catalog stage")
        source_final_input_sha = _required_text(
            source_final_stage.get("input_sha256"), "source final catalog input hash"
        )
        source_final_catalog, source_final_attestation = _load_stage_artifact(
            source_final_path,
            stage="final_merge",
            key=source_run_id,
            input_sha256=source_final_input_sha,
        )
        source_final_catalog = _validate_final_catalog(
            source_final_catalog,
            observed_tools=frozenset(observed_tools),
            sensitive_literals=(),
            max_source_patch_count=FULL_EVIDENCE_COUNT,
        )
        source_final_binding = {
            "path": source_final_path.relative_to(source).as_posix(),
            "sha256": f"sha256:{_file_sha256(source_final_path)}",
            "size_bytes": source_final_path.stat().st_size,
            "integrity_sha256": f"sha256:{_file_sha256(_integrity_path(source_final_path))}",
            "input_sha256": source_final_input_sha,
        }

    paid_run: Path | None = None
    paid_binding: JsonObject | None = None
    if paid_raw_run is not None:
        paid_run = _real_directory(paid_raw_run, "paid compact raw-response run")
        paid_manifest_path = paid_run / "run_manifest.json"
        paid_manifest = _read_json_file(paid_manifest_path, "paid compact raw-response manifest")
        paid_raw_path = paid_run / "final_catalog.raw-response.json"
        _validate_stage_integrity(paid_raw_path)
        paid_raw = _read_json_file(paid_raw_path, "paid compact raw response")
        paid_call = _mapping(paid_raw.get("call"), "paid compact raw call")
        paid_result = _mapping(paid_raw.get("result"), "paid compact raw result")
        paid_source = _mapping(paid_manifest.get("source"), "paid compact source")
        if (
            paid_manifest.get("run_id") != paid_call.get("key")
            or paid_manifest.get("namespace") != namespace
            or paid_manifest.get("pipeline") != "cached-question-merges/compact-final-merge"
            or paid_source.get("run_id") != source_run_id
        ):
            raise TrialQADistillationError("paid compact raw run identity is invalid")
        if (
            paid_call.get("stage") != "final_merge"
            or not isinstance(paid_call.get("input_sha256"), str)
            or paid_result.get("route_model") != DISTILLER_ROUTE
            or paid_result.get("upstream_model") != DISTILLER_MODEL
            or not isinstance(paid_result.get("request_id"), str)
            or not isinstance(paid_result.get("content"), str)
        ):
            raise TrialQADistillationError("paid compact raw response attestation is invalid")
        paid_binding = {
            "run_id": paid_manifest["run_id"],
            "run_path": str(paid_run),
            "run_manifest_sha256": f"sha256:{_file_sha256(paid_manifest_path)}",
            "raw_response_sha256": f"sha256:{_file_sha256(paid_raw_path)}",
            "raw_response_integrity_sha256": (
                f"sha256:{_file_sha256(_integrity_path(paid_raw_path))}"
            ),
            "input_sha256": paid_call["input_sha256"],
            "request_id": paid_result["request_id"],
        }

    source_bindings: tuple[JsonObject, ...] = (
        {
            "path": source_manifest_path.name,
            "sha256": f"sha256:{_file_sha256(source_manifest_path)}",
            "size_bytes": source_manifest_path.stat().st_size,
        },
        {
            "path": completion_path.name,
            "sha256": f"sha256:{_file_sha256(completion_path)}",
            "size_bytes": completion_path.stat().st_size,
        },
        {
            "path": validation_path.name,
            "sha256": f"sha256:{_file_sha256(validation_path)}",
            "size_bytes": validation_path.stat().st_size,
        },
        *question_bindings,
        *([source_final_binding] if source_final_binding is not None else []),
    )
    pipeline = (
        "cached-final-catalog/deterministic-compact-transport"
        if transport_source_final_catalog
        else "cached-question-merges/compact-final-merge"
    )
    manifest_seed: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "namespace": namespace,
        "pipeline": pipeline,
        "tool_contract": tool_contract,
        "implementation": {
            "trialqa_local_distiller_sha256": _file_sha256(Path(__file__).resolve()),
            "skill_distillation_store_sha256": _file_sha256(
                Path(skill_store_module.__file__).resolve()
            ),
        },
        "routing": {
            "route": DISTILLER_ROUTE,
            "upstream_model": DISTILLER_MODEL,
            "profile_sha256": profile_sha,
            "proxy_url": local_proxy,
        },
        "source": {
            "run_id": source_run_id,
            "run_path": str(source),
            "source_evidence_count": len(source_ids),
            "question_merge_count": len(question_aggregates),
            "artifacts": list(source_bindings),
            "final_catalog": source_final_binding,
        },
        "compact_policy": {
            "max_bytes": COMPACT_SKILL_MAX_BYTES,
            "max_words": COMPACT_SKILL_MAX_WORDS,
            "max_rules": COMPACT_SKILL_MAX_RULES,
            "search_budget": 3,
            "deterministic_pruning": (
                CACHED_CATALOG_TRANSPORT_MODE
                if transport_source_final_catalog
                else "top-supported-rules-plus-bounded-agent-contract-v2"
            ),
            "transport_adaptation": (
                "source-alias-to-tooluniverse-compact-meta-tools-v1"
                if tool_contract == "compact"
                else "none"
            ),
        },
    }
    if paid_binding is not None:
        manifest_seed["paid_raw_recovery"] = paid_binding
    if transport_source_final_catalog:
        manifest_seed["model_call_budget"] = 0
    run_prefix = "trialqa-transport" if transport_source_final_catalog else "trialqa-compact"
    run_id = f"{run_prefix}-{_digest(manifest_seed)[:32]}"
    manifest = {"run_id": run_id, **manifest_seed}
    return CompactDistillationPlan(
        run_id=run_id,
        run_path=work / run_id,
        namespace=namespace,
        project_dir=project,
        routing_profile=profile,
        routing_profile_sha256=profile_sha,
        proxy_url=local_proxy,
        tool_contract=tool_contract,
        source_run=source,
        source_run_id=source_run_id,
        source_evidence_ids=source_ids,
        question_aggregates=tuple(question_aggregates),
        observed_tools=frozenset(observed_tools),
        source_bindings=source_bindings,
        transport_source_final_catalog=transport_source_final_catalog,
        source_final_catalog=source_final_catalog,
        source_final_binding=source_final_binding,
        source_final_attestation=source_final_attestation,
        paid_raw_run=paid_run,
        paid_raw_binding=paid_binding,
        manifest=manifest,
    )


def _sha256_binding(path: Path, *, name: str) -> JsonObject:
    return {
        "name": name,
        "sha256": f"sha256:{_file_sha256(path)}",
        "size_bytes": path.stat().st_size,
    }


def _candidate_binding(candidate_id: str, manifest_sha256: str, skill_sha256: str) -> JsonObject:
    return {
        "candidate_id": candidate_id,
        "manifest_sha256": f"sha256:{manifest_sha256}",
        "skill_sha256": f"sha256:{skill_sha256}",
    }


def _development_operations(events: Sequence[Any]) -> frozenset[str]:
    operations: set[str] = set()
    target_to_alias = {target: alias for alias, target, _arguments in COMPACT_TRIALQA_TOOL_MAP}
    for raw_event in events:
        if not isinstance(raw_event, dict) or raw_event.get("kind") != "tool_call":
            continue
        payload = raw_event.get("payload")
        if not isinstance(payload, dict):
            continue
        function = payload.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str):
            continue
        public_name = _public_tool_name(name, structural=True)
        if public_name in PUBLIC_TRIALQA_TOOLS:
            operations.add(public_name)
        if not name.endswith("execute_tool"):
            continue
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            continue
        try:
            outer: object = json.loads(arguments)
        except json.JSONDecodeError:
            continue
        if not isinstance(outer, dict):
            continue
        target = outer.get("tool_name")
        if isinstance(target, str) and target in target_to_alias:
            operations.add(target_to_alias[target])
    return frozenset(operations)


def _load_development_evidence(
    path: Path,
    *,
    role: Literal["failure", "support"],
    parent_binding: Mapping[str, Any],
) -> DevelopmentEvidence:
    evidence_path = _real_directory(path, f"{role} development evidence")
    evidence_id = evidence_path.name
    if _NATIVE_ID.fullmatch(evidence_id) is None:
        raise TrialQADistillationError(
            f"unsupported {role} development evidence id: {evidence_id!r}"
        )
    validate_native_trialqa_evidence_directory(evidence_path, expected_evidence_id=evidence_id)
    document = _read_json_file(
        evidence_path / "evidence.json", f"{role} development evidence document"
    )
    manifest_path = evidence_path / "manifest.json"
    _read_json_file(manifest_path, f"{role} development evidence manifest")
    task = _mapping(document.get("task"), f"{role} evidence.task")
    outcome = _mapping(document.get("outcome"), f"{role} evidence.outcome")
    execution = _mapping(document.get("execution"), f"{role} evidence.execution")
    raw = evidence_path / "raw"
    raw_task = _read_json_file(raw / "task.json", f"raw {role} development task")
    raw_outcome = _read_json_file(raw / "outcome.json", f"raw {role} development outcome")
    raw_run = _read_json_file(raw / "run.json", f"raw {role} development run")
    session = _read_json_file(raw / "session.json", f"raw {role} development session")
    if raw_task != task or raw_outcome != outcome:
        raise TrialQADistillationError(
            f"{role} development evidence differs from its raw task/outcome"
        )
    for key, value in raw_run.items():
        if execution.get(key) != value:
            raise TrialQADistillationError(
                f"{role} development execution.{key} differs from raw run metadata"
            )
    if (
        task.get("partition") != "test"
        or task.get("condition") != "treatment"
        or outcome.get("partition") != "test"
        or outcome.get("condition") != "treatment"
        or outcome.get("verifier") != JUDGE_VERIFIER
    ):
        raise TrialQADistillationError(
            "exposed development must be judged test-partition treatment evidence"
        )
    if (
        execution.get("phase") != "evaluation"
        or execution.get("route") != EXECUTOR_ROUTE
        or execution.get("model") not in EXECUTOR_MODEL_ALIASES
        or execution.get("skill_loaded") is not True
        or execution.get("candidate_id") != parent_binding.get("candidate_id")
        or execution.get("candidate_manifest_sha256") != parent_binding.get("manifest_sha256")
        or execution.get("candidate_skill_sha256") != parent_binding.get("skill_sha256")
    ):
        raise TrialQADistillationError(
            "exposed development must attest the exact skilled Ultra parent execution"
        )
    active = _mapping(session.get("active_skill"), f"raw {role} active skill")
    context = _mapping(session.get("run_context"), f"raw {role} run context")
    if (
        session.get("status") != "completed"
        or session.get("exit_code") != 0
        or session.get("display_model") != EXECUTOR_ROUTE
        or active.get("loaded") is not True
        or active.get("candidate_id") != parent_binding.get("candidate_id")
        or active.get("manifest_sha256") != parent_binding.get("manifest_sha256")
        or active.get("skill_sha256") != parent_binding.get("skill_sha256")
        or context.get("partition") != "test"
        or context.get("condition") != "treatment"
        or context.get("phase") != "evaluation"
        or context.get("candidate_id") != parent_binding.get("candidate_id")
    ):
        raise TrialQADistillationError(
            "exposed development session does not attest the exact skilled parent"
        )
    repeat_index = _required_int(task.get("repeat_index"), f"{role} task.repeat_index", minimum=1)
    group = _safe_component(
        _required_text(task.get("question_group_key"), f"{role} question group"),
        f"{role} question group",
    )
    score = _number(outcome.get("score"), f"{role} outcome.score", minimum=0, maximum=1)
    expected_score = 0.0 if role == "failure" else 1.0
    expected_label = "incorrect" if role == "failure" else "correct"
    if score != expected_score or outcome.get("label") != expected_label:
        raise TrialQADistillationError(f"{role} development evidence has the wrong judge outcome")
    operations = _development_operations(
        _list(document.get("events"), f"{role} development events")
    )
    direct_support = "trialqa_extract_adverse_events" in operations
    if direct_support != (role == "support"):
        raise TrialQADistillationError(
            f"{role} development evidence has the wrong direct-support observation"
        )
    return DevelopmentEvidence(
        evidence_id=evidence_id,
        path=evidence_path,
        document=document,
        manifest_sha256=_file_sha256(manifest_path),
        question_group_key=group,
        repeat_index=repeat_index,
        role=role,
        direct_support_observed=direct_support,
    )


def _manifest_group_order(manifest: Mapping[str, Any], *, label: str) -> tuple[str, ...]:
    tasks = [
        _mapping(item, f"{label} task") for item in _list(manifest.get("tasks"), f"{label} tasks")
    ]
    order: list[str] = []
    seen_groups: set[str] = set()
    matrix: Counter[tuple[str, int, str]] = Counter()
    for task in tasks:
        group = _safe_component(
            _required_text(task.get("question_group_key"), f"{label} question group"),
            f"{label} question group",
        )
        repeat = _required_int(task.get("repeat_index"), f"{label} repeat", minimum=1)
        condition = task.get("condition")
        if (
            task.get("partition") != "test"
            or task.get("phase") != "evaluation"
            or task.get("n_repeats") != FULL_REPEAT_COUNT
            or repeat > FULL_REPEAT_COUNT
            or condition not in {"baseline", "treatment"}
        ):
            raise TrialQADistillationError(f"{label} contains an invalid evaluation task")
        if group not in seen_groups:
            seen_groups.add(group)
            order.append(group)
        matrix[(group, repeat, cast(str, condition))] += 1
    expected = {
        (group, repeat, condition)
        for group in order
        for repeat in range(1, FULL_REPEAT_COUNT + 1)
        for condition in ("baseline", "treatment")
    }
    if set(matrix) != expected or any(count != 1 for count in matrix.values()):
        raise TrialQADistillationError(f"{label} is not an exact paired repeat matrix")
    return tuple(order)


def _validate_development_manifests(
    descriptive: JsonObject,
    primary: JsonObject,
    *,
    parent_binding: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    for label, document in (("descriptive manifest", descriptive), ("primary manifest", primary)):
        if (
            document.get("schema_version") != "switchyard.trialqa_experiment_manifest.v1"
            or document.get("kind") != "full"
            or _mapping(document.get("candidate"), f"{label} candidate") != parent_binding
        ):
            raise TrialQADistillationError(f"{label} identity or parent candidate is invalid")
    descriptive_protocol = _mapping(descriptive.get("protocol"), "descriptive manifest protocol")
    primary_protocol = _mapping(primary.get("protocol"), "primary manifest protocol")
    descriptive_groups = _manifest_group_order(descriptive, label="descriptive manifest")
    primary_groups = _manifest_group_order(primary, label="primary manifest")
    if (
        len(descriptive_groups) != 96
        or descriptive_protocol.get("performance_eligible") is not False
        or descriptive_protocol.get("primary_evaluation_scope") is not None
        or descriptive_protocol.get("heldout_quarantine") is not None
    ):
        raise TrialQADistillationError(
            "descriptive manifest must cover all 96 held-out questions as nonperformance"
        )
    expected_primary_scope = {
        "question_start": 8,
        "question_count": 88,
        "repeat_count": 5,
        "task_count": 880,
    }
    quarantine = _mapping(primary_protocol.get("heldout_quarantine"), "primary heldout quarantine")
    quarantined_groups = list(descriptive_groups[:8])
    quarantine_sha = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                quarantined_groups,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )
    if (
        primary_protocol.get("performance_eligible") is not True
        or primary_protocol.get("primary_evaluation_scope") != expected_primary_scope
        or quarantine
        != {
            "question_start": 0,
            "question_count": 8,
            "disposition": "excluded-exposed-heldout",
            "question_group_keys_sha256": quarantine_sha,
        }
        or primary_groups != descriptive_groups[8:]
    ):
        raise TrialQADistillationError(
            "primary manifest must exclude exactly the exposed first-eight quarantine"
        )
    return descriptive_groups, primary_groups


def _validate_regression_verdict(
    verdict: JsonObject,
    *,
    parent_binding: Mapping[str, Any],
    failure: DevelopmentEvidence,
    support: DevelopmentEvidence,
    descriptive_manifest: JsonObject,
    descriptive_sha256: str,
    primary_manifest: JsonObject,
    primary_sha256: str,
    descriptive_groups: Sequence[str],
    primary_groups: Sequence[str],
) -> None:
    if (
        verdict.get("schema_version") != "switchyard.trialqa_exposed_regression_verdict.v1"
        or verdict.get("decision") != "kill"
        or verdict.get("performance_eligible") is not False
        or verdict.get("source_attestation_current") is not True
        or _mapping(verdict.get("candidate"), "regression verdict candidate") != parent_binding
    ):
        raise TrialQADistillationError("regression verdict identity or parent binding is invalid")
    manifest_binding = _mapping(verdict.get("manifest"), "regression verdict manifest")
    primary_binding = _mapping(
        verdict.get("primary_evaluation"), "regression verdict primary evaluation"
    )
    if (
        manifest_binding.get("manifest_id") != descriptive_manifest.get("manifest_id")
        or manifest_binding.get("manifest_sha256") != f"sha256:{descriptive_sha256}"
        or primary_binding.get("manifest_id") != primary_manifest.get("manifest_id")
        or primary_binding.get("manifest_sha256") != f"sha256:{primary_sha256}"
        or primary_binding.get("capture_started") is not False
    ):
        raise TrialQADistillationError(
            "regression verdict does not bind the untouched primary evaluation scope"
        )
    scope = _mapping(verdict.get("scope"), "regression verdict scope")
    if (
        scope.get("question_start") != 7
        or scope.get("question_limit") != 1
        or scope.get("repeat_limit") != 2
        or scope.get("heldout_classification") != "exposed-heldout-quarantine"
        or _SHA256.fullmatch(str(scope.get("scope_attestation_sha256"))) is None
        or failure.question_group_key != support.question_group_key
        or failure.question_group_key != descriptive_groups[7]
        or failure.question_group_key in primary_groups
        or support.repeat_index != 1
        or failure.repeat_index != 2
    ):
        raise TrialQADistillationError(
            "regression evidence is not exactly q7 repeats 1-2 in the excluded quarantine"
        )
    results = [
        _mapping(item, "regression verdict result")
        for item in _list(verdict.get("results"), "regression verdict results")
    ]
    result_by_evidence = {
        _required_text(item.get("evidence_id"), "regression result evidence id"): item
        for item in results
    }
    if len(results) != 2 or len(result_by_evidence) != 2:
        raise TrialQADistillationError("regression verdict must bind exactly two evidence rows")
    for evidence, expected_score in ((support, 1.0), (failure, 0.0)):
        result = result_by_evidence.get(evidence.evidence_id)
        task = _mapping(evidence.document.get("task"), "development evidence task")
        if (
            result is None
            or result.get("task_id") != task.get("id")
            or result.get("score") != expected_score
            or result.get("direct_support_operation") != "extract_clinical_trial_adverse_events"
            or result.get("direct_support_observed") is not evidence.direct_support_observed
            or any(
                _SHA256.fullmatch(str(result.get(field))) is None
                for field in (
                    "generation_sha256",
                    "codex_events_sha256",
                    "result_sha256",
                )
            )
        ):
            raise TrialQADistillationError(
                "regression verdict result differs from validated development evidence"
            )
    policy = _mapping(verdict.get("policy"), "regression verdict policy")
    summary = _mapping(verdict.get("summary"), "regression verdict summary")
    if (
        policy.get("name") != "exposed-mechanism-regression-v1"
        or policy.get("required_treatment_correct_repeats") != 2
        or policy.get("required_treatment_direct_support_repeats") != 2
        or summary.get("treatment_correct_repeats") != 1
        or summary.get("treatment_direct_support_repeats") != 1
        or summary.get("required_repeats") != 2
    ):
        raise TrialQADistillationError("regression verdict does not attest the failed mechanism")


def build_development_layer_plan(
    *,
    project_dir: Path,
    namespace: str,
    work_dir: Path,
    parent_candidate_id: str,
    development_evidence_dir: Path,
    supporting_development_evidence_dir: Path,
    regression_verdict: Path,
    descriptive_manifest: Path,
    primary_manifest: Path,
) -> DevelopmentLayerPlan:
    """Build a zero-call plan from quarantined, explicitly exposed failure evidence."""

    namespace = _safe_component(namespace, "namespace")
    if namespace != NAMESPACE:
        raise TrialQADistillationError(f"TrialQA namespace must be exactly {NAMESPACE!r}")
    project = _real_directory(project_dir, "project directory")
    work = work_dir.expanduser().absolute()
    if work.is_symlink() or (work.exists() and not work.is_dir()):
        raise TrialQADistillationError(
            f"development-layer work path must be a real directory or absent: {work}"
        )
    candidate_id = _safe_component(parent_candidate_id, "parent candidate id")
    store = SkillDistillationStore(namespace, project)
    candidate_path = _real_directory(
        store.candidates_path / candidate_id, "parent candidate directory"
    )
    parent_manifest_path = candidate_path / "manifest.json"
    parent_manifest = _read_json_file(parent_manifest_path, "parent candidate manifest")
    parent_manifest_sha = _file_sha256(parent_manifest_path)
    parent_validation = _mapping(parent_manifest.get("validation"), "parent validation")
    parent_checks = _mapping(parent_validation.get("checks"), "parent validation checks")
    provenance = _mapping(parent_manifest.get("provenance"), "parent provenance")
    train_ids_raw = _list(provenance.get("source_evidence_ids"), "parent source evidence ids")
    train_ids = tuple(cast(list[str], train_ids_raw))
    if (
        parent_manifest.get("schema_version") != 1
        or parent_manifest.get("namespace") != namespace
        or parent_manifest.get("candidate_id") != candidate_id
        or parent_validation.get("status") != "passed"
        or parent_validation.get("candidate_id") != candidate_id
        or parent_validation.get("performance_eligible") is not True
        or parent_validation.get("tool_contract") != "compact"
        or parent_validation.get("distillation_mode") != CACHED_CATALOG_TRANSPORT_MODE
        or not parent_checks
        or any(value is not True for value in parent_checks.values())
        or len(train_ids) != FULL_EVIDENCE_COUNT
        or len(set(train_ids)) != FULL_EVIDENCE_COUNT
        or any(_NATIVE_ID.fullmatch(item) is None for item in train_ids)
    ):
        raise TrialQADistillationError(
            "parent candidate is not one validated train-only compact transport"
        )
    skills = [
        _mapping(item, "parent candidate skill")
        for item in _list(parent_manifest.get("skills"), "parent candidate skills")
    ]
    skill_entry = next((item for item in skills if item.get("path") == SKILL_PATH), None)
    skill_path = _real_file(candidate_path / SKILL_PATH, "parent executable skill")
    parent_skill_sha = _file_sha256(skill_path)
    if (
        skill_entry is None
        or skill_entry.get("sha256") != parent_skill_sha
        or _mapping(parent_validation.get("artifacts"), "parent validation artifacts").get(
            "skill_sha256"
        )
        != f"sha256:{parent_skill_sha}"
    ):
        raise TrialQADistillationError("parent candidate skill hash binding is invalid")
    parent_binding = _candidate_binding(candidate_id, parent_manifest_sha, parent_skill_sha)
    parent_run_id = _safe_component(
        _required_text(parent_validation.get("run_id"), "parent run id"), "parent run id"
    )
    parent_run = _real_directory(work / parent_run_id, "parent distillation run")
    report_path = parent_run / "candidate_validation.json"
    if _read_json_file(report_path, "parent validation report") != parent_validation:
        raise TrialQADistillationError(
            "parent candidate validation differs from its immutable run report"
        )
    completion_path = parent_run / "completion_manifest.json"
    completion = _read_json_file(completion_path, "parent completion manifest")
    parent_artifacts = _mapping(parent_validation.get("artifacts"), "parent artifacts")
    if (
        parent_artifacts.get("completion_manifest_sha256")
        != f"sha256:{_file_sha256(completion_path)}"
        or completion.get("run_id") != parent_run_id
        or completion.get("new_model_call_count") != 0
        or completion.get("transport_mode") != CACHED_CATALOG_TRANSPORT_MODE
    ):
        raise TrialQADistillationError("parent completion provenance is invalid")
    final_entries = [
        _mapping(item, "parent final catalog entry")
        for item in _list(completion.get("stage_artifacts"), "parent stage artifacts")
        if isinstance(item, dict) and item.get("path") == "final_catalog.json"
    ]
    if len(final_entries) != 1:
        raise TrialQADistillationError("parent completion must bind one final catalog")
    final_path = _real_file(parent_run / "final_catalog.json", "parent final catalog")
    final_entry = final_entries[0]
    if (
        final_entry.get("sha256") != f"sha256:{_file_sha256(final_path)}"
        or final_entry.get("size_bytes") != final_path.stat().st_size
    ):
        raise TrialQADistillationError("parent final catalog binding changed")
    _validate_stage_integrity(final_path)
    final_document = _read_json_file(final_path, "parent final catalog")
    parent_catalog = _mapping(final_document.get("output"), "parent catalog output")
    if (
        final_document.get("stage") != "catalog_transport"
        or final_document.get("key") != parent_run_id
        or parent_catalog.get("compaction_mode") != CACHED_CATALOG_TRANSPORT_MODE
        or parent_catalog.get("tool_contract") != "compact"
    ):
        raise TrialQADistillationError("parent catalog is not a compact train transport")
    parent_skill = skill_path.read_text(encoding="utf-8")
    if render_skill_markdown(parent_catalog, tool_contract="compact") != parent_skill:
        raise TrialQADistillationError("parent catalog does not render the parent skill")
    validate_compact_skill(parent_catalog, parent_skill, tool_contract="compact")
    parent_catalog_binding = {
        "run_id": parent_run_id,
        "path": "final_catalog.json",
        "sha256": f"sha256:{_file_sha256(final_path)}",
        "integrity_sha256": f"sha256:{_file_sha256(_integrity_path(final_path))}",
        "size_bytes": final_path.stat().st_size,
    }

    failure = _load_development_evidence(
        development_evidence_dir, role="failure", parent_binding=parent_binding
    )
    support = _load_development_evidence(
        supporting_development_evidence_dir,
        role="support",
        parent_binding=parent_binding,
    )
    if failure.evidence_id == support.evidence_id:
        raise TrialQADistillationError("development evidence ids must be distinct")

    verdict_path = _real_file(regression_verdict, "regression verdict")
    verdict = _read_json_file(verdict_path, "regression verdict")
    descriptive_path = _real_file(descriptive_manifest, "descriptive manifest")
    descriptive_document = _read_json_file(descriptive_path, "descriptive manifest")
    primary_path = _real_file(primary_manifest, "primary manifest")
    primary_document = _read_json_file(primary_path, "primary manifest")
    descriptive_groups, primary_groups = _validate_development_manifests(
        descriptive_document, primary_document, parent_binding=parent_binding
    )
    descriptive_sha = _file_sha256(descriptive_path)
    primary_sha = _file_sha256(primary_path)
    _validate_regression_verdict(
        verdict,
        parent_binding=parent_binding,
        failure=failure,
        support=support,
        descriptive_manifest=descriptive_document,
        descriptive_sha256=descriptive_sha,
        primary_manifest=primary_document,
        primary_sha256=primary_sha,
        descriptive_groups=descriptive_groups,
        primary_groups=primary_groups,
    )
    verdict_binding = _sha256_binding(verdict_path, name=verdict_path.name)
    descriptive_binding = {
        **_sha256_binding(descriptive_path, name=descriptive_path.name),
        "manifest_id": descriptive_document.get("manifest_id"),
    }
    primary_binding = {
        **_sha256_binding(primary_path, name=primary_path.name),
        "manifest_id": primary_document.get("manifest_id"),
    }
    manifest_seed: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "namespace": namespace,
        "pipeline": "train-parent/exposed-development/deterministic-layer",
        "mode": DEVELOPMENT_LAYER_MODE,
        "model_call_budget": 0,
        "implementation": {
            "trialqa_local_distiller_sha256": _file_sha256(Path(__file__).resolve())
        },
        "parent_candidate": parent_binding,
        "parent_catalog": parent_catalog_binding,
        "provenance_strata": {
            "train_base": {
                "source_evidence_ids": list(train_ids),
                "evidence_count": len(train_ids),
            },
            "exposed_development": {
                "performance_eligible": False,
                "failure": {
                    "evidence_id": failure.evidence_id,
                    "manifest_sha256": f"sha256:{failure.manifest_sha256}",
                },
                "support": {
                    "evidence_id": support.evidence_id,
                    "manifest_sha256": f"sha256:{support.manifest_sha256}",
                },
                "regression_verdict": verdict_binding,
            },
            "evaluation_scope": {
                "descriptive_full96": descriptive_binding,
                "primary_untouched88": primary_binding,
                "excluded_question_start": 0,
                "excluded_question_count": 8,
                "primary_question_start": 8,
                "primary_question_count": 88,
                "capture_started": False,
            },
        },
    }
    run_id = f"trialqa-development-{_digest(manifest_seed)[:32]}"
    manifest = {"run_id": run_id, **manifest_seed}
    return DevelopmentLayerPlan(
        run_id=run_id,
        run_path=work / run_id,
        namespace=namespace,
        project_dir=project,
        parent_candidate_id=candidate_id,
        parent_candidate_path=candidate_path,
        parent_manifest_sha256=parent_manifest_sha,
        parent_skill_sha256=parent_skill_sha,
        parent_catalog=dict(parent_catalog),
        parent_catalog_binding=parent_catalog_binding,
        train_evidence_ids=train_ids,
        failure_evidence=failure,
        support_evidence=support,
        verdict_path=verdict_path,
        verdict=verdict,
        verdict_binding=verdict_binding,
        descriptive_manifest_path=descriptive_path,
        descriptive_manifest_binding=descriptive_binding,
        primary_manifest_path=primary_path,
        primary_manifest_binding=primary_binding,
        manifest=manifest,
    )


def _replace_outside_json_strings(text: str, old: str, new: str) -> tuple[str, int]:
    """Replace an exact syntax token without touching JSON string contents."""

    output: list[str] = []
    index = 0
    in_string = False
    escaped = False
    replacements = 0
    while index < len(text):
        if not in_string and text.startswith(old, index):
            output.append(new)
            index += len(old)
            replacements += 1
            continue
        char = text[index]
        output.append(char)
        index += 1
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
    return "".join(output), replacements


def _repair_final_merge_array_closures(candidate: str) -> JsonObject:
    """Repair one observed Opus error that closes flat final arrays per item."""

    repaired = candidate
    counts: dict[str, int] = {}
    for old, new in (
        (']},{"rule"', ',{"rule"'),
        (']}],"failure_modes"', '],"failure_modes"'),
        (']},{"trigger"', ',{"trigger"'),
        (']}],"gotchas"', '],"gotchas"'),
        (']},{"fact"', ',{"fact"'),
    ):
        repaired, counts[old] = _replace_outside_json_strings(repaired, old, new)
    if counts[']}],"failure_modes"'] != 1 or counts[']}],"gotchas"'] != 1:
        raise TrialQADistillationError("final merge JSON does not match the repairable shape")
    if not repaired.endswith("]}]}"):
        raise TrialQADistillationError("final merge JSON has an unexpected closing shape")
    repaired = repaired[:-4] + "]}"
    try:
        value: object = json.loads(repaired)
    except json.JSONDecodeError as exc:
        raise TrialQADistillationError("final merge JSON repair did not produce JSON") from exc
    if not isinstance(value, dict):
        raise TrialQADistillationError("repaired final merge JSON must be an object")
    return cast(JsonObject, value)


def _extract_json(text: str, label: str) -> JsonObject:
    if not isinstance(text, str) or not text.strip():
        raise TrialQADistillationError(f"{label} returned empty output")
    stripped = text.strip()
    match = _JSON_BLOCK.search(stripped)
    candidate = match.group(1) if match else stripped
    if not candidate.startswith("{"):
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise TrialQADistillationError(f"{label} returned no JSON object")
        candidate = candidate[start : end + 1]
    try:
        value: object = json.loads(candidate)
    except json.JSONDecodeError as exc:
        if label.startswith("final_merge/"):
            try:
                return _repair_final_merge_array_closures(candidate)
            except TrialQADistillationError:
                pass
        raise TrialQADistillationError(f"{label} returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise TrialQADistillationError(f"{label} JSON must be an object")
    return cast(JsonObject, value)


def _replace_sensitive_text(text: str, literals: Sequence[str]) -> str:
    result = _URL.sub("<REDACTED_URL>", _NCT.sub("<REDACTED_TRIAL_ID>", text))
    for literal in literals:
        if literal:
            result = _literal_pattern(literal).sub("<REDACTED_TASK_LITERAL>", result)
    return result


def _literal_pattern(literal: str) -> re.Pattern[str]:
    escaped = re.escape(literal)
    if len(literal) <= 4:
        escaped = rf"(?<!\w){escaped}(?!\w)"
    return re.compile(escaped, re.IGNORECASE)


def _sanitize(value: object, literals: Sequence[str]) -> object:
    if isinstance(value, str):
        return _replace_sensitive_text(value, literals)
    if isinstance(value, list):
        return [_sanitize(item, literals) for item in value]
    if isinstance(value, dict):
        return {
            str(key): item if key in _STRUCTURAL_TEXT_FIELDS else _sanitize(item, literals)
            for key, item in value.items()
        }
    return value


def _free_text_strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _free_text_strings(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key not in _STRUCTURAL_TEXT_FIELDS:
                yield from _free_text_strings(item)


def _assert_no_sensitive(value: object, literals: Sequence[str], label: str) -> None:
    for text in _free_text_strings(value):
        if _URL.search(text) or _NCT.search(text):
            raise TrialQADistillationError(f"{label} contains a task-specific URL or trial id")
        for literal in literals:
            if _literal_pattern(literal).search(text):
                raise TrialQADistillationError(f"{label} contains a source task literal")


def _validate_rule_item(
    raw: object,
    *,
    field: str,
    observed_tools: frozenset[str],
    require_rule_type: bool = True,
) -> JsonObject:
    item = _mapping(raw, field)
    rule_type = item.get("rule_type")
    if not require_rule_type:
        rule_type = field.rsplit(".", 1)[-1].removesuffix("s")
    if rule_type not in RULE_TYPES:
        raise TrialQADistillationError(f"{field}.rule_type is invalid")
    if require_rule_type:
        category = item.get("category")
        if category not in DEFAULT_CATEGORIES:
            raise TrialQADistillationError(f"{field}.category is invalid")
    _number(item.get("confidence"), f"{field}.confidence", minimum=0, maximum=1)
    if rule_type == "tool_rule":
        tool_name = _required_text(item.get("tool_name"), f"{field}.tool_name")
        if tool_name not in observed_tools:
            raise TrialQADistillationError(
                f"{field} references tool {tool_name!r} absent from source trajectories"
            )
        _required_text(item.get("rule"), f"{field}.rule", minimum=20)
        _required_text(item.get("when"), f"{field}.when", minimum=10)
    elif rule_type == "failure_mode":
        _required_text(item.get("trigger"), f"{field}.trigger", minimum=15)
        _required_text(item.get("symptom"), f"{field}.symptom", minimum=15)
        _required_text(item.get("prevention"), f"{field}.prevention", minimum=20)
    elif rule_type == "workflow_rule":
        _required_text(item.get("rule"), f"{field}.rule", minimum=20)
        _required_text(item.get("rationale"), f"{field}.rationale", minimum=15)
    else:
        _required_text(item.get("fact"), f"{field}.fact", minimum=20)
        _required_text(item.get("impact"), f"{field}.impact", minimum=15)
    return item


def _validate_diagnosis(value: object, field: str) -> JsonObject:
    diagnosis = _mapping(value, field)
    for name in ("failure_surface", "expected_vs_actual", "root_cause", "corrected_strategy"):
        _required_text(diagnosis.get(name), f"{field}.{name}", minimum=8)
    steps = _list(diagnosis.get("causal_trace_steps"), f"{field}.causal_trace_steps")
    if not steps:
        raise TrialQADistillationError(f"{field}.causal_trace_steps must not be empty")
    for index, step in enumerate(steps):
        _required_text(step, f"{field}.causal_trace_steps[{index}]", minimum=8)
    return diagnosis


def _validate_analyst_patch(raw: JsonObject, evidence: DonorEvidence) -> JsonObject:
    if raw.get("source_task_name") != evidence.evidence_id:
        raise TrialQADistillationError("analyst patch source_task_name is not the evidence id")
    content = dict(raw)
    content.pop("source_task_name", None)
    sanitized = _sanitize(content, evidence.sensitive_literals)
    patch = _mapping(sanitized, "analyst patch")
    patch["source_task_name"] = evidence.evidence_id
    if patch.get("schema_version") != PATCH_SCHEMA:
        raise TrialQADistillationError("analyst patch has the wrong schema_version")
    if patch.get("role") != evidence.role:
        raise TrialQADistillationError("analyst patch role does not match verifier outcome")
    items = _list(patch.get("memory_items"), "analyst patch memory_items")
    if not items:
        raise TrialQADistillationError("analyst patch must contain at least one memory item")
    validated_items = [
        _validate_rule_item(
            item,
            field=f"analyst patch memory_items[{index}]",
            observed_tools=evidence.observed_tools,
        )
        for index, item in enumerate(items)
    ]
    skill_patch = _mapping(patch.get("skill_patch"), "analyst patch skill_patch")
    if skill_patch.get("target") != SKILL_PATH:
        raise TrialQADistillationError("analyst patch target must be tooluniverse-trialqa/SKILL.md")
    sections = _list(skill_patch.get("sections"), "analyst patch skill_patch.sections")
    for index, section_raw in enumerate(sections):
        section = _mapping(section_raw, f"analyst patch section[{index}]")
        _required_text(section.get("heading"), f"analyst patch section[{index}].heading")
        _required_text(
            section.get("content"), f"analyst patch section[{index}].content", minimum=20
        )
        if section.get("action") != "append":
            raise TrialQADistillationError("analyst skill-patch actions must be append")
    if evidence.role == "error":
        _validate_diagnosis(patch.get("diagnosis"), "analyst patch diagnosis")
    elif "diagnosis" in patch:
        raise TrialQADistillationError("success analyst patch must not contain diagnosis")
    patch["memory_items"] = validated_items
    leakage_view = dict(patch)
    leakage_view.pop("source_task_name", None)
    _assert_no_sensitive(leakage_view, evidence.sensitive_literals, "analyst patch")
    return patch


def _analyst_prompt(evidence: DonorEvidence, instruction: str) -> tuple[str, str]:
    task = _mapping(evidence.document.get("task"), "evidence.task")
    outcome = _mapping(evidence.document.get("outcome"), "evidence.outcome")
    events = cast(
        list[Any],
        _public_tool_view(_list(evidence.document.get("events"), "evidence.events")),
    )
    trusted_input = {
        "task_context": {
            "task_name": evidence.evidence_id,
            "role": evidence.role,
            "domain": "trialqa",
            "skill_target": SKILL_PATH,
            "categories": sorted(DEFAULT_CATEGORIES),
            "available_mcp_servers": [],
        },
        "task": task,
        "reward": outcome,
        "trajectory": events,
    }
    system = (
        "You are the pinned Trace2Skill trajectory analyst. Follow the complete "
        "reference instructions below. This host-side adaptation supplies the named "
        "artifacts inline and requires the patch.json object as your entire response. "
        "No MCP tools are available, so ground tool rules only in exact trajectory events.\n\n"
        f'<pinned-reference revision="{REFERENCE_REVISION}" '
        f'sha256="{REFERENCE_ANALYST_SHA256}">\n{instruction}\n</pinned-reference>'
    )
    user = (
        "Analyze exactly this finalized train/donor evidence. The ideal/expected answer "
        "and judge rationale are donor-only training signals. Use them for diagnosis, but "
        "do not copy any task answer, question, URL, trial identifier, or row identifier "
        "into a rule. Tool names have been decoded to the exact public trialqa_* names "
        "visible to Codex; preserve those names exactly. `unsupported_tool_call` is a "
        "failure marker, never a callable tool or a valid tool_rule target. Every memory "
        "item must use exactly one rule_type from "
        f"{sorted(RULE_TYPES)} and include "
        f"one category from {sorted(DEFAULT_CATEGORIES)}. Return one "
        "trace2skill_patch.v2 JSON object only.\n\n"
        + json.dumps(trusted_input, ensure_ascii=False, sort_keys=True)
    )
    return system, user


def _openai_payload(system: str, user: str, *, max_tokens: int) -> JsonObject:
    return {
        "model": DISTILLER_ROUTE,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }


def _build_call(stage: Stage, key: str, system: str, user: str, max_tokens: int) -> ModelCall:
    payload = _openai_payload(system, user, max_tokens=max_tokens)
    return ModelCall(stage=stage, key=key, payload=payload, input_sha256=_digest(payload))


def _question_prompt(
    group: str,
    patches: Sequence[JsonObject],
    repeat_metadata: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    merge_patches = [
        {
            "role": patch.get("role"),
            "memory_items": patch.get("memory_items"),
            "diagnosis": patch.get("diagnosis"),
            "skill_patch_sections": _mapping(
                patch.get("skill_patch"), "analyst patch skill_patch"
            ).get("sections"),
        }
        for patch in patches
    ]
    system = (
        "You are the Trace2Skill question-level merge analyst. Merge repeated rollout "
        "patches into reusable rules. Output one valid JSON object only. Do not wrap it "
        "under a schema-name key and do not use a Markdown fence."
    )
    role_counts = dict(sorted(Counter(cast(str, item["role"]) for item in repeat_metadata).items()))
    judge_counts = dict(
        sorted(Counter(cast(str, item["judge_result"]) for item in repeat_metadata).items())
    )
    repeat_count = len(patches)
    user = f"""Merge analyst patches from independent repeats of one TrialQA training question.

Rules:
- Preserve concrete tool names, fields, response shapes, decision points, and prevention steps.
- Tool rules may target only these public names: {json.dumps(sorted(PUBLIC_TRIALQA_TOOLS))}.
- `unsupported_tool_call` is diagnostic input only; do not reproduce it in any output field.
- Treat repeats as independent evidence and set source_patch_count/repeat_count accurately.
- Deduplicate equivalent facts into exactly one natural rule type.
- Set the top-level structural question_group_key exactly to the trusted group key below.
  That required structural value is allowed; the group-ID prohibition applies only to free text.
- Never reproduce task questions, answers, URLs, NCT identifiers, row IDs, or group IDs in
  summary, rule text, rationales, failure descriptions, or other free-text fields.
- Return exactly the top-level object shape below. Do not emit a
  `trace2skill_question_merge.v2` wrapper key. Every rule leaf must include
  confidence (number from 0 to 1) and source_patch_count (integer from 1 to
  {repeat_count}). Empty rule arrays are allowed.

{{
  "schema_version": "{QUESTION_MERGE_SCHEMA}",
  "question_group_key": {json.dumps(group)},
  "summary": "reusable summary without task identifiers or answers",
  "source_patch_count": {repeat_count},
  "repeat_count": {repeat_count},
  "role_counts": {json.dumps(role_counts, sort_keys=True)},
  "judge_result_counts": {json.dumps(judge_counts, sort_keys=True)},
  "tool_rules": [{{"tool_name": "exact observed tool name", "rule": "at least 20 characters", "when": "at least 10 characters", "confidence": 0.9, "source_patch_count": 1}}],
  "workflow_rules": [{{"rule": "at least 20 characters", "rationale": "at least 15 characters", "confidence": 0.9, "source_patch_count": 1}}],
  "failure_modes": [{{"trigger": "at least 15 characters", "symptom": "at least 15 characters", "prevention": "at least 20 characters", "confidence": 0.9, "source_patch_count": 1}}],
  "gotchas": [{{"fact": "at least 20 characters", "impact": "at least 15 characters", "confidence": 0.9, "source_patch_count": 1}}]
}}

Trusted group key: {json.dumps(group)}
Trusted repeat metadata (contains no question or answer text):
{json.dumps(list(repeat_metadata), ensure_ascii=False, sort_keys=True)}
Sanitized patches:
{json.dumps(merge_patches, ensure_ascii=False, sort_keys=True)}
"""
    return system, user


def _typed_rules(aggregate: Mapping[str, Any]) -> list[tuple[str, JsonObject]]:
    values: list[tuple[str, JsonObject]] = []
    for plural, rule_type, category in (
        ("tool_rules", "tool_rule", "evidence_retrieval"),
        ("workflow_rules", "workflow_rule", "other"),
        ("failure_modes", "failure_mode", "failure_avoidance"),
        ("gotchas", "gotcha", "other"),
    ):
        raw_items = _list(aggregate.get(plural), f"question merge {plural}")
        for raw in raw_items:
            item = _mapping(raw, f"question merge {plural} item")
            item = {"rule_type": rule_type, "category": category, **item}
            values.append((rule_type, item))
    return values


def _validate_question_merge(
    raw: JsonObject,
    *,
    group: str,
    patches: Sequence[JsonObject],
    judge_results: Sequence[str],
    observed_tools: frozenset[str],
    sensitive_literals: Sequence[str],
) -> JsonObject:
    if raw.get("question_group_key") != group:
        raise TrialQADistillationError("question merge has the wrong trusted group key")
    content = dict(raw)
    content.pop("question_group_key", None)
    for text in _strings(content):
        if _literal_pattern(group).search(text):
            raise TrialQADistillationError(
                "question merge contains its structural group key in free text"
            )
    sanitized = _sanitize(content, sensitive_literals)
    aggregate = _mapping(sanitized, "question merge")
    aggregate["question_group_key"] = group
    if aggregate.get("schema_version") != QUESTION_MERGE_SCHEMA:
        raise TrialQADistillationError("question merge has the wrong schema_version")
    if aggregate.get("source_patch_count") != len(patches):
        raise TrialQADistillationError("question merge source_patch_count is incorrect")
    if aggregate.get("repeat_count") != len(patches):
        raise TrialQADistillationError("question merge repeat_count is incorrect")
    expected_roles = Counter(cast(str, patch["role"]) for patch in patches)
    role_counts = _mapping(aggregate.get("role_counts"), "question merge role_counts")
    if role_counts != dict(sorted(expected_roles.items())):
        raise TrialQADistillationError("question merge role_counts is incorrect")
    judge_counts = _mapping(
        aggregate.get("judge_result_counts"), "question merge judge_result_counts"
    )
    expected_judges = dict(sorted(Counter(judge_results).items()))
    if judge_counts != expected_judges:
        raise TrialQADistillationError("question merge judge_result_counts is incorrect")
    _required_text(aggregate.get("summary"), "question merge summary")
    typed_rules = _typed_rules(aggregate)
    if not typed_rules:
        raise TrialQADistillationError("question merge contains no reusable rules")
    for index, (_rule_type, item) in enumerate(typed_rules):
        _validate_rule_item(
            item,
            field=f"question merge rule[{index}]",
            observed_tools=observed_tools,
        )
        source_count = _required_int(
            item.get("source_patch_count"),
            f"question merge rule[{index}].source_patch_count",
            minimum=1,
        )
        if source_count > len(patches):
            raise TrialQADistillationError(
                "question merge rule source_patch_count exceeds source patches"
            )
    leakage_view = dict(aggregate)
    leakage_view.pop("question_group_key", None)
    _assert_no_sensitive(leakage_view, sensitive_literals, "question merge")
    return aggregate


def _final_prompt(question_aggregates: Sequence[JsonObject]) -> tuple[str, str]:
    merge_inputs = [
        {
            "summary": aggregate.get("summary"),
            "source_patch_count": aggregate.get("source_patch_count"),
            "repeat_count": aggregate.get("repeat_count"),
            "tool_rules": aggregate.get("tool_rules"),
            "workflow_rules": aggregate.get("workflow_rules"),
            "failure_modes": aggregate.get("failure_modes"),
            "gotchas": aggregate.get("gotchas"),
        }
        for aggregate in question_aggregates
    ]
    system = (
        "You are the Trace2Skill cross-question merge analyst. Produce one concise, "
        "benchmark-safe structured skill catalog. Output one valid JSON object only. "
        "Do not wrap it under a schema-name key and do not use a Markdown fence."
    )
    max_source_patch_count = sum(
        cast(int, aggregate.get("source_patch_count", 0)) for aggregate in question_aggregates
    )
    user = f"""Merge sanitized aggregates from independent TrialQA training questions.

Rules:
- Preserve specific, generalizable ToolUniverse tool behavior and efficient workflows.
- Tool rules may target only these public names: {json.dumps(sorted(PUBLIC_TRIALQA_TOOLS))}.
- Omit the diagnostic `unsupported_tool_call` marker and every internal/transport tool name.
- Deduplicate equivalent facts; keep each fact in exactly one natural rule type.
- Prefer evidence repeated across questions, while retaining high-confidence unique tool quirks.
- Never include a task question, task answer, URL, NCT identifier, source/group/row ID,
  benchmark split information, or an answer recipe for one question.
- Return exactly the top-level object shape below. Do not emit a
  `trace2skill_merge.v2` wrapper key. Every rule leaf must include confidence
  (number from 0 to 1) and source_patch_count (integer from 1 to
  {max_source_patch_count}). Empty rule arrays are allowed.

{{
  "schema_version": "{FINAL_MERGE_SCHEMA}",
  "skill_name": "{SKILL_NAME}",
  "summary": "concise reusable TrialQA workflow summary",
  "tool_rules": [{{"tool_name": "exact observed tool name", "rules": [{{"rule": "at least 20 characters", "when": "at least 10 characters", "confidence": 0.9, "source_patch_count": 1}}]}}],
  "workflow_rules": [{{"rule": "at least 20 characters", "rationale": "at least 15 characters", "confidence": 0.9, "source_patch_count": 1}}],
  "failure_modes": [{{"trigger": "at least 15 characters", "symptom": "at least 15 characters", "prevention": "at least 20 characters", "confidence": 0.9, "source_patch_count": 1}}],
  "gotchas": [{{"fact": "at least 20 characters", "impact": "at least 15 characters", "confidence": 0.9, "source_patch_count": 1}}]
}}

Question aggregates:
{json.dumps(merge_inputs, ensure_ascii=False, sort_keys=True)}
"""
    return system, user


def _compact_final_prompt(question_aggregates: Sequence[JsonObject]) -> tuple[str, str]:
    """Build the one-call final merge used for rapid candidate iteration."""

    merge_inputs = [
        {
            "summary": aggregate.get("summary"),
            "source_patch_count": aggregate.get("source_patch_count"),
            "repeat_count": aggregate.get("repeat_count"),
            "tool_rules": aggregate.get("tool_rules"),
            "workflow_rules": aggregate.get("workflow_rules"),
            "failure_modes": aggregate.get("failure_modes"),
            "gotchas": aggregate.get("gotchas"),
        }
        for aggregate in question_aggregates
    ]
    max_source_patch_count = sum(
        cast(int, aggregate.get("source_patch_count", 0)) for aggregate in question_aggregates
    )
    system = (
        "You are the Trace2Skill compact cross-question merge analyst. Produce one "
        "minimal, benchmark-safe skill catalog that reduces agent search and tool-call "
        "loops without sacrificing answer accuracy. Output one JSON object only."
    )
    user = f"""Compress sanitized aggregates from independent TrialQA training questions.

Hard efficiency contract:
- Return at most {COMPACT_SKILL_MAX_RULES} total rule leaves. Prefer rules independently
  supported across questions; omit narrow one-question recipes and tool schema restatements.
- The rendered skill must fit within {COMPACT_SKILL_MAX_BYTES:,} UTF-8 bytes and
  {COMPACT_SKILL_MAX_WORDS} words, so use short imperative sentences.
- Require at most 3 semantically distinct searches. Never repeat the same query or
  arguments, never invent an acronym expansion, and accept an exact title match unless
  a question constraint contradicts it.
- After resolving the NCT identifier, require one question-specific getter. Permit a
  second getter only when the required field is absent, then answer immediately.
- Do not tell the agent that an empty result requires more searching, that every possible
  constraint must be checked, or that it must avoid a not-found answer at any cost.
- Tool rules may target only these public names: {json.dumps(sorted(PUBLIC_TRIALQA_TOOLS))}.
- Never include a task question, answer, URL, NCT identifier, row/group/source ID,
  benchmark split metadata, placeholder, or redaction marker.
- Every leaf must include confidence and source_patch_count from 1 to
  {max_source_patch_count}. Empty arrays are allowed.

Return exactly:
{{
  "schema_version": "{FINAL_MERGE_SCHEMA}",
  "skill_name": "{SKILL_NAME}",
  "summary": "one-sentence bounded TrialQA workflow",
  "tool_rules": [{{"tool_name": "observed public name", "rules": [{{"rule": "imperative rule", "when": "short trigger", "confidence": 0.9, "source_patch_count": 1}}]}}],
  "workflow_rules": [{{"rule": "imperative rule", "rationale": "short evidence rationale", "confidence": 0.9, "source_patch_count": 1}}],
  "failure_modes": [{{"trigger": "short trigger", "symptom": "short symptom", "prevention": "imperative prevention", "confidence": 0.9, "source_patch_count": 1}}],
  "gotchas": [{{"fact": "concise fact", "impact": "concise impact", "confidence": 0.9, "source_patch_count": 1}}]
}}

Question aggregates:
{json.dumps(merge_inputs, ensure_ascii=False, sort_keys=True)}
"""
    return system, user


def _validate_final_catalog(
    raw: JsonObject,
    *,
    observed_tools: frozenset[str],
    sensitive_literals: Sequence[str],
    max_source_patch_count: int,
) -> JsonObject:
    sanitized = _sanitize(raw, sensitive_literals)
    catalog = _mapping(sanitized, "final catalog")
    if catalog.get("schema_version") != FINAL_MERGE_SCHEMA:
        raise TrialQADistillationError("final catalog has the wrong schema_version")
    if catalog.get("skill_name") != SKILL_NAME:
        raise TrialQADistillationError("final catalog has the wrong skill_name")
    _required_text(catalog.get("summary"), "final catalog summary")
    rule_count = 0
    tool_groups = _list(catalog.get("tool_rules"), "final catalog tool_rules")
    for group_index, raw_group in enumerate(tool_groups):
        group = _mapping(raw_group, f"final catalog tool group[{group_index}]")
        tool_name = _required_text(group.get("tool_name"), "final catalog tool_name")
        if tool_name not in observed_tools:
            raise TrialQADistillationError(
                f"final catalog references unobserved tool {tool_name!r}"
            )
        rules = _list(group.get("rules"), "final catalog grouped rules")
        if not rules:
            raise TrialQADistillationError("final catalog tool group is empty")
        for index, raw_rule in enumerate(rules):
            item = {
                "rule_type": "tool_rule",
                "category": "other",
                "tool_name": tool_name,
                **_mapping(raw_rule, "final tool rule"),
            }
            _validate_rule_item(
                item,
                field=f"final catalog tool rule[{index}]",
                observed_tools=observed_tools,
            )
            source_count = _required_int(
                item.get("source_patch_count"), "final rule source_patch_count", minimum=1
            )
            if source_count > max_source_patch_count:
                raise TrialQADistillationError(
                    "final rule source_patch_count exceeds donor evidence count"
                )
            rule_count += 1
    for plural, rule_type in (
        ("workflow_rules", "workflow_rule"),
        ("failure_modes", "failure_mode"),
        ("gotchas", "gotcha"),
    ):
        for index, raw_item in enumerate(_list(catalog.get(plural), f"final catalog {plural}")):
            item = {
                "rule_type": rule_type,
                "category": "other",
                **_mapping(raw_item, f"final catalog {plural} item"),
            }
            _validate_rule_item(
                item,
                field=f"final catalog {plural}[{index}]",
                observed_tools=observed_tools,
            )
            source_count = _required_int(
                item.get("source_patch_count"), "final rule source_patch_count", minimum=1
            )
            if source_count > max_source_patch_count:
                raise TrialQADistillationError(
                    "final rule source_patch_count exceeds donor evidence count"
                )
            rule_count += 1
    if rule_count == 0:
        raise TrialQADistillationError("final catalog contains no rules")
    _assert_no_sensitive(catalog, sensitive_literals, "final catalog")
    _assert_no_internal_tool_names(catalog, "final catalog")
    return catalog


def _compact_transport_metadata() -> JsonObject:
    return {
        "public_tools": list(COMPACT_PUBLIC_TOOLS),
        "source_alias_mapping": [
            {
                "source_alias": source_alias,
                "public_tool": "execute_tool",
                "tool_name": tool_name,
                "arguments_parameter": "arguments_json",
                "arguments_json_shape": list(arguments),
            }
            for source_alias, tool_name, arguments in COMPACT_TRIALQA_TOOL_MAP
        ],
    }


def adapt_compact_tool_contract(
    catalog: Mapping[str, Any], *, tool_contract: ToolContract
) -> JsonObject:
    """Bind deterministic transport metadata without changing learned rule text."""

    if tool_contract not in TOOL_CONTRACTS:
        raise TrialQADistillationError(f"tool contract must be one of {', '.join(TOOL_CONTRACTS)}")
    adapted = json.loads(_canonical_bytes(catalog))
    if not isinstance(adapted, dict):  # pragma: no cover - Mapping guarantees this
        raise TrialQADistillationError("compact catalog must be a JSON object")
    result = cast(JsonObject, adapted)
    result["tool_contract"] = tool_contract
    result.pop("transport", None)
    if tool_contract == "compact":
        result["transport"] = _compact_transport_metadata()
    return result


def _catalog_tool_contract(
    catalog: Mapping[str, Any], requested: ToolContract | None
) -> ToolContract:
    embedded = catalog.get("tool_contract")
    if embedded is not None and embedded not in TOOL_CONTRACTS:
        raise TrialQADistillationError("skill catalog has an invalid tool contract")
    if requested is not None and embedded is not None and requested != embedded:
        raise TrialQADistillationError(
            "requested tool contract differs from the skill catalog binding"
        )
    contract = requested if requested is not None else embedded
    return cast(ToolContract, contract if contract is not None else "direct")


def _compact_transport_lines() -> list[str]:
    lines = [
        "## Compact ToolUniverse contract",
        "",
        (
            "- Call `trialqa_load_active_skill` exactly once. For every known operation "
            "below, call `execute_tool` directly with `tool_name` and an `arguments_json` "
            "string encoding one JSON object; never call an underlying clinical-trial "
            "tool directly."
        ),
        (
            "- Discovery is fallback-only for a genuinely unlisted operation: use "
            "`grep_tools` with `{pattern: string}`, then `get_tool_info` with "
            "`{tool_names: string[], detail_level: full}`. Do not discover a "
            "known tool from this map."
        ),
        "",
        "### Exact operation map",
        "",
    ]
    for source_alias, tool_name, arguments in COMPACT_TRIALQA_TOOL_MAP:
        shape = ", ".join(arguments)
        lines.append(
            f"- `{source_alias}` evidence -> `execute_tool` / `{tool_name}`; "
            f"`arguments_json` encodes `{{{shape}}}`."
        )
    lines.append("")
    return lines


def render_skill_markdown(
    catalog: Mapping[str, Any], *, tool_contract: ToolContract | None = None
) -> str:
    """Render a validated catalog deterministically with Codex skill frontmatter."""

    _assert_no_internal_tool_names(catalog, "skill catalog")
    resolved_contract = _catalog_tool_contract(catalog, tool_contract)
    if resolved_contract == "compact" and catalog.get("transport") != _compact_transport_metadata():
        raise TrialQADistillationError("compact catalog transport mapping is not exact")

    lines = [
        "---",
        f"name: {SKILL_NAME}",
        "description: Answer LABBench2 TrialQA questions with ToolUniverse evidence.",
        "---",
        "",
        "# TrialQA with ToolUniverse",
        "",
    ]
    if resolved_contract == "compact":
        lines.extend(_compact_transport_lines())
    tool_rules = cast(list[JsonObject], catalog.get("tool_rules", []))
    if tool_rules:
        lines.extend(["## Tool reference", ""])
        for group in tool_rules:
            tool_name = _required_text(group.get("tool_name"), "skill catalog tool_name")
            if tool_name not in PUBLIC_TRIALQA_TOOLS:
                raise TrialQADistillationError(
                    f"skill catalog references unsupported tool {tool_name!r}"
                )
            if resolved_contract == "compact" and tool_name != "trialqa_load_active_skill":
                target_by_alias = {
                    source_alias: target
                    for source_alias, target, _arguments in COMPACT_TRIALQA_TOOL_MAP
                }
                target = target_by_alias.get(tool_name)
                if target is None:
                    raise TrialQADistillationError(
                        f"compact skill cannot transport source tool {tool_name!r}"
                    )
                heading = f"### `execute_tool` -> `{target}` (from `{tool_name}` evidence)"
            else:
                heading = f"### `{tool_name}`"
            lines.extend([heading, ""])
            for rule in cast(list[JsonObject], group["rules"]):
                lines.append(f"- **When:** {rule['when']}")
                lines.append(f"  {rule['rule']}")
            lines.append("")
    workflows = cast(list[JsonObject], catalog.get("workflow_rules", []))
    if workflows:
        lines.extend(["## Workflow rules", ""])
        for item in workflows:
            lines.extend([f"- {item['rule']}\n  *Rationale:* {item['rationale']}", ""])
    failures = cast(list[JsonObject], catalog.get("failure_modes", []))
    if failures:
        lines.extend(["## Failure modes", ""])
        for item in failures:
            lines.extend(
                [
                    f"- **Trigger:** {item['trigger']}\n"
                    f"  **Symptom:** {item['symptom']}\n"
                    f"  **Prevention:** {item['prevention']}",
                    "",
                ]
            )
    gotchas = cast(list[JsonObject], catalog.get("gotchas", []))
    if gotchas:
        lines.extend(["## Gotchas", ""])
        for item in gotchas:
            lines.extend([f"- {item['fact']}\n  *Impact if missed:* {item['impact']}", ""])
    rendered = "\n".join(lines).rstrip() + "\n"
    _assert_no_internal_tool_names(rendered, "rendered skill")
    return rendered


def validate_compact_skill(
    catalog: Mapping[str, Any],
    skill: str,
    *,
    tool_contract: ToolContract | None = None,
) -> dict[str, int]:
    """Enforce the paid-canary size and bounded-control-flow contract."""

    resolved_contract = _catalog_tool_contract(catalog, tool_contract)
    if "host-enforced" in skill.lower():
        raise TrialQADistillationError("compact skill falsely claims host enforcement")
    if resolved_contract == "compact":
        if catalog.get("transport") != _compact_transport_metadata():
            raise TrialQADistillationError("compact catalog transport mapping is not exact")
        required_transport_text = (
            "trialqa_load_active_skill",
            "execute_tool",
            "grep_tools",
            "get_tool_info",
            "fallback-only",
            "call `execute_tool` directly",
            "arguments_json",
            "never call an underlying clinical-trial tool directly",
            "{pattern: string}",
            "{tool_names: string[], detail_level: full}",
            *(tool_name for _alias, tool_name, _arguments in COMPACT_TRIALQA_TOOL_MAP),
            *(
                argument.split(":", maxsplit=1)[0].rstrip("?")
                for _alias, _tool_name, arguments in COMPACT_TRIALQA_TOOL_MAP
                for argument in arguments
            ),
        )
        missing = [text for text in required_transport_text if text not in skill]
        if missing:
            raise TrialQADistillationError(
                f"compact skill is missing exact transport text: {missing[0]}"
            )

    placeholder = re.search(
        r"(?:<|\[|\{\{)(?:REDACTED(?:_[A-Z0-9]+)*|TASK_LITERAL|NCT_ID|URL)"
        r"(?:>|\]|\}\})|\bREDACTED(?:_[A-Z0-9]+)+\b|\bTASK_LITERAL\b",
        skill,
        re.IGNORECASE,
    )
    if placeholder is not None:
        raise TrialQADistillationError("compact skill contains a redaction placeholder")
    size_bytes = len(skill.encode("utf-8"))
    if size_bytes > COMPACT_SKILL_MAX_BYTES:
        raise TrialQADistillationError(
            f"compact skill exceeds {COMPACT_SKILL_MAX_BYTES} UTF-8 bytes"
        )
    word_count = len(re.findall(r"\S+", skill))
    if word_count > COMPACT_SKILL_MAX_WORDS:
        raise TrialQADistillationError(f"compact skill exceeds {COMPACT_SKILL_MAX_WORDS} words")
    tool_groups = cast(list[Mapping[str, Any]], catalog.get("tool_rules", []))
    tool_names = [str(group.get("tool_name")) for group in tool_groups]
    if len(tool_names) != len(set(tool_names)):
        raise TrialQADistillationError("compact skill repeats a tool rule group")
    rule_count = sum(
        len(cast(list[object], group.get("rules", []))) for group in tool_groups
    ) + sum(
        len(cast(list[object], catalog.get(field, [])))
        for field in ("workflow_rules", "failure_modes", "gotchas")
    )
    if rule_count > COMPACT_SKILL_MAX_RULES:
        raise TrialQADistillationError(
            f"compact skill exceeds {COMPACT_SKILL_MAX_RULES} rule leaves"
        )
    normalized = " ".join(skill.lower().split())
    forbidden = {
        "unbounded search instruction": (
            r"\b(?:continue|keep) searching (?:indefinitely|without (?:a )?limit)\b"
        ),
        "exhaustive getter instruction": r"\bcall (?:all|every) (?:available )?getters?\b",
        "search-after-match instruction": r"\beven after an exact title match\b",
        "mandatory second getter": r"\b(?:always|must) (?:call|use) (?:a )?second getter\b",
    }
    for label, pattern in forbidden.items():
        if re.search(pattern, normalized) is not None:
            raise TrialQADistillationError(f"compact skill contains a contradictory {label}")
    required = {
        "search budget": r"\bat most 3 semantically distinct searches\b",
        "repeat-call prohibition": (
            r"\b(?:never|do not) repeat\b.{0,100}\b(?:search|query|arguments?)\b"
        ),
        "exact-title stop rule": (
            r"\b(?:stop|cease|end) searching\b.{0,100}\bexact title match\b"
            r"|\bexact title match\b.{0,100}\b(?:stop|cease|end) searching\b"
        ),
        "acronym invention prohibition": r"\bnever invent an acronym expansion\b",
    }
    if catalog.get("compaction_mode") == CACHED_CATALOG_TRANSPORT_MODE:
        transport_forbidden = {
            "one-getter shortcut": r"\bone question-specific getter\b",
            "hard operational-call cap": r"\bnever exceed \d+ operational\b",
        }
        for label, pattern in transport_forbidden.items():
            if re.search(pattern, normalized) is not None:
                raise TrialQADistillationError(
                    f"cached catalog transport contains a contradictory {label}"
                )
        required.update(
            {
                "field-specific getter routing": r"\bfield-specific getter\b",
                "non-exhaustive route declaration": r"\bnon-exhaustive\b",
                "conditional cross-getter fallback": (
                    r"\bif the selected slice lacks the requested field\b.{0,120}"
                    r"\banother relevant getter\b"
                ),
                "evidence sufficiency": (
                    r"\banswer only when retrieved evidence directly supports every requested field\b"
                ),
                "evidence-complete finalization": r"\bonce it does, finalize\b",
            }
        )
    else:
        required.update(
            {
                "bounded getter step": r"\bone question-specific getter\b",
                "conditional second getter": (
                    r"\b(?:use|call) (?:a )?second getter only (?:when|if)\b.{0,100}"
                    r"\b(?:absent|missing|not present)\b"
                ),
                "immediate answer step": r"\b(?:then answer|answer immediately)\b",
            }
        )
    for label, pattern in required.items():
        if re.search(pattern, normalized) is None:
            raise TrialQADistillationError(f"compact skill is missing its {label}")
    return {
        "size_bytes": size_bytes,
        "word_count": word_count,
        "rule_count": rule_count,
    }


def compact_final_catalog(
    catalog: Mapping[str, Any], *, tool_contract: ToolContract = "direct"
) -> JsonObject:
    """Deterministically prune an overlong paid merge without another model call."""

    if tool_contract not in TOOL_CONTRACTS:
        raise TrialQADistillationError(f"tool contract must be one of {', '.join(TOOL_CONTRACTS)}")

    def support(item: Mapping[str, Any]) -> tuple[int, float]:
        source_count = item.get("source_patch_count")
        confidence = item.get("confidence")
        return (
            source_count
            if isinstance(source_count, int) and not isinstance(source_count, bool)
            else 0,
            float(confidence)
            if isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
            else 0.0,
        )

    groups = [
        _mapping(item, "compact source tool group")
        for item in _list(catalog.get("tool_rules"), "compact source tool groups")
    ]
    selected_by_name: dict[str, JsonObject] = {}
    for group in groups:
        tool_name = _required_text(group.get("tool_name"), "compact source tool name")
        rules = [
            _mapping(item, "compact source tool rule")
            for item in _list(group.get("rules"), "compact source tool rules")
        ]
        best = max(rules, key=lambda item: (*support(item), _digest(item)))
        if tool_name == "trialqa_search":
            best = {
                "rule": (
                    "Make at most 3 total search calls: first use the exact title or "
                    "acronym from the question, then use intervention plus condition, then "
                    "one short alternate query. Never make a fourth search. Never repeat "
                    "a query or arguments, and never invent an acronym expansion. Stop "
                    "searching after an exact title match."
                ),
                "when": "resolving the trial identifier before any evidence getter",
                "confidence": 1.0,
                "source_patch_count": 1,
            }
        selected_by_name[tool_name] = {"tool_name": tool_name, "rules": [dict(best)]}
    preferred = ["trialqa_load_active_skill", "trialqa_search"]
    selected_groups = [selected_by_name[name] for name in preferred if name in selected_by_name]
    if {cast(str, group["tool_name"]) for group in selected_groups} != set(preferred):
        raise TrialQADistillationError("compact catalog lacks loader/search evidence")

    agent_contract: JsonObject = {
        "rule": (
            "Load the active skill exactly once and never call the loader again. Use at "
            "most 3 semantically distinct searches; never repeat the same search query "
            "or arguments, never invent an acronym expansion, and stop searching after "
            "an exact title match. Resolve the NCT identifier, call one question-specific "
            "getter, use a second getter only when the required field is absent, then "
            "answer immediately. Never exceed 5 operational TrialQA calls total."
        ),
        "rationale": (
            "This explicit agent bound prevents the search and getter loops observed in "
            "the rejected first candidate without claiming runtime enforcement."
        ),
        "confidence": 1.0,
        "source_patch_count": 1,
    }
    model_workflows: list[JsonObject] = []
    selected_gotchas: list[JsonObject] = []

    while True:
        compacted: JsonObject = {
            "schema_version": FINAL_MERGE_SCHEMA,
            "skill_name": SKILL_NAME,
            "summary": (
                "Resolve one trial with bounded search, retrieve one targeted evidence "
                "slice, and answer immediately."
            ),
            "tool_rules": selected_groups,
            "workflow_rules": [agent_contract, *model_workflows],
            "failure_modes": [],
            "gotchas": selected_gotchas,
        }
        adapted = adapt_compact_tool_contract(compacted, tool_contract=tool_contract)
        try:
            skill = render_skill_markdown(adapted, tool_contract=tool_contract)
            validate_compact_skill(adapted, skill, tool_contract=tool_contract)
            return adapted
        except TrialQADistillationError as exc:
            if not any(
                marker in str(exc) for marker in ("exceeds", "contradictory", "placeholder")
            ):
                raise
            if selected_gotchas:
                selected_gotchas.pop()
            elif model_workflows:
                model_workflows.pop()
            elif len(selected_groups) > 3:
                selected_groups.pop()
            else:
                raise TrialQADistillationError(
                    "deterministic catalog pruning cannot satisfy the compact contract"
                ) from exc


def compact_cached_final_catalog(
    catalog: Mapping[str, Any], *, tool_contract: ToolContract = "compact"
) -> JsonObject:
    """Transport a validated train-only final catalog without another model call."""

    if tool_contract != "compact":
        raise TrialQADistillationError(
            "cached final catalog transport requires the compact tool contract"
        )

    tool_groups = [
        _mapping(item, "cached source tool group")
        for item in _list(catalog.get("tool_rules"), "cached source tool groups")
    ]
    search_rules = [
        _mapping(rule, "cached source search rule")
        for group in tool_groups
        if group.get("tool_name") == "trialqa_search"
        for rule in _list(group.get("rules"), "cached source search rules")
    ]
    if not search_rules:
        raise TrialQADistillationError("cached final catalog lacks train-derived search rules")

    workflows = [
        _mapping(item, "cached source workflow")
        for item in _list(catalog.get("workflow_rules"), "cached source workflows")
    ]
    field_route = next(
        (
            item
            for item in workflows
            if "field-specific getter" in str(item.get("rule", "")).lower()
            and "get_eligibility" in str(item.get("rule", ""))
            and "get_outcome_measures" in str(item.get("rule", ""))
            and "get_descriptions" in str(item.get("rule", ""))
            and "get_study" in str(item.get("rule", ""))
        ),
        None,
    )
    evidence_complete = next(
        (
            item
            for item in workflows
            if "already contains every field needed to answer" in str(item.get("rule", "")).lower()
        ),
        None,
    )
    gotchas = [
        _mapping(item, "cached source gotcha")
        for item in _list(catalog.get("gotchas"), "cached source gotchas")
    ]
    cross_slice = next(
        (
            item
            for item in gotchas
            if "one wrapper" in str(item.get("fact", "")).lower()
            and "other trialqa_* tools expose" in str(item.get("fact", "")).lower()
        ),
        None,
    )
    if field_route is None or evidence_complete is None or cross_slice is None:
        raise TrialQADistillationError(
            "cached final catalog lacks generic routing or evidence-sufficiency support"
        )

    search_support = max(
        _required_int(item.get("source_patch_count"), "cached search support", minimum=1)
        for item in search_rules
    )
    field_support = _required_int(
        field_route.get("source_patch_count"), "cached field-route support", minimum=1
    )
    evidence_support = _required_int(
        evidence_complete.get("source_patch_count"),
        "cached evidence-complete support",
        minimum=1,
    )
    cross_slice_support = _required_int(
        cross_slice.get("source_patch_count"), "cached cross-slice support", minimum=1
    )
    compacted: JsonObject = {
        "schema_version": FINAL_MERGE_SCHEMA,
        "skill_name": SKILL_NAME,
        "summary": "Resolve the trial, route by requested field, and require direct evidence.",
        "compaction_mode": CACHED_CATALOG_TRANSPORT_MODE,
        "tool_rules": [
            {
                "tool_name": "trialqa_search",
                "rules": [
                    {
                        "rule": (
                            "Use at most 3 semantically distinct searches; never repeat the "
                            "same query or arguments, never invent an acronym expansion, and "
                            "stop searching after an exact title match."
                        ),
                        "when": "resolving the trial identifier before evidence retrieval",
                        "confidence": max(
                            _number(
                                item.get("confidence"),
                                "cached search confidence",
                                minimum=0,
                                maximum=1,
                            )
                            for item in search_rules
                        ),
                        "source_patch_count": search_support,
                    }
                ],
            }
        ],
        "workflow_rules": [
            {
                "rule": (
                    "After trialqa_search resolves the NCT id, select the field-specific "
                    "getter: trialqa_get_eligibility for criteria or thresholds, "
                    "trialqa_get_outcome_measures for outcome counts or timeFrames, "
                    "trialqa_get_descriptions for narrative or per-arm detail, and "
                    "trialqa_get_study for enrollment or structured summary fields."
                ),
                "rationale": str(field_route["rationale"]),
                "confidence": _number(
                    field_route.get("confidence"),
                    "cached field-route confidence",
                    minimum=0,
                    maximum=1,
                ),
                "source_patch_count": field_support,
            },
            {
                "rule": (
                    "Treat the listed field routes as non-exhaustive. If the selected slice "
                    "lacks the requested field, call another relevant getter whose documented "
                    "output can contain it; one empty wrapper does not prove the datum absent."
                ),
                "rationale": str(cross_slice["impact"]),
                "confidence": _number(
                    cross_slice.get("confidence"),
                    "cached cross-slice confidence",
                    minimum=0,
                    maximum=1,
                ),
                "source_patch_count": cross_slice_support,
            },
            {
                "rule": (
                    "Answer only when retrieved evidence directly supports every requested "
                    "field. Once it does, finalize without repeating calls; otherwise change "
                    "evidence slices instead of guessing from a related field."
                ),
                "rationale": str(evidence_complete["rationale"]),
                "confidence": _number(
                    evidence_complete.get("confidence"),
                    "cached evidence-complete confidence",
                    minimum=0,
                    maximum=1,
                ),
                "source_patch_count": evidence_support,
            },
        ],
        "failure_modes": [],
        "gotchas": [],
    }
    adapted = adapt_compact_tool_contract(compacted, tool_contract=tool_contract)
    skill = render_skill_markdown(adapted, tool_contract=tool_contract)
    validate_compact_skill(adapted, skill, tool_contract=tool_contract)
    return adapted


def layer_exposed_development_catalog(parent_catalog: Mapping[str, Any]) -> JsonObject:
    """Apply the attested q7 mechanism correction without embedding q7 literals."""

    catalog = cast(JsonObject, json.loads(json.dumps(parent_catalog)))
    workflows = [
        _mapping(item, "parent development workflow")
        for item in _list(catalog.get("workflow_rules"), "parent development workflows")
    ]
    matches = [
        index
        for index, item in enumerate(workflows)
        if "answer only when retrieved evidence directly supports every requested field"
        in str(item.get("rule", "")).lower()
    ]
    if len(matches) != 1:
        raise TrialQADistillationError(
            "parent catalog must contain exactly one evidence-sufficiency workflow"
        )
    workflows[matches[0]] = {
        "rule": (
            "For intervention starting-dose, regimen, or ordered arm/group questions, if "
            "the selected slice lacks direct support, use trialqa_extract_adverse_events "
            "as a fallback evidence slice; inspect group titles/descriptions. Do not infer "
            "starting, lowest, or highest values from outcome timeFrames or cohort labels. "
            "Answer only when retrieved "
            "evidence directly supports every requested field. Once it does, finalize; "
            "otherwise use another relevant getter."
        ),
        "rationale": (
            "The exposed failure inferred a value without direct intervention-group evidence."
        ),
        "confidence": 1.0,
        "source_patch_count": 1,
        "provenance_stratum": "exposed-development",
    }
    catalog["workflow_rules"] = workflows
    catalog["development_layer_mode"] = DEVELOPMENT_LAYER_MODE
    skill = render_skill_markdown(catalog, tool_contract="compact")
    validate_compact_skill(catalog, skill, tool_contract="compact")
    return catalog


def layer_exposed_mechanism_repair_catalog(
    parent_catalog: Mapping[str, Any],
) -> JsonObject:
    """Strengthen generic search and eligibility routing while preserving fallbacks."""

    catalog = cast(JsonObject, json.loads(json.dumps(parent_catalog)))
    tool_groups = [
        _mapping(item, "repair parent tool group")
        for item in _list(catalog.get("tool_rules"), "repair parent tool groups")
    ]
    search_groups = [group for group in tool_groups if group.get("tool_name") == "trialqa_search"]
    if len(search_groups) != 1:
        raise TrialQADistillationError(
            "repair parent must contain exactly one trialqa_search rule group"
        )
    search_rules = _list(search_groups[0].get("rules"), "repair parent search rules")
    if len(search_rules) != 1:
        raise TrialQADistillationError("repair parent must contain exactly one search rule")
    search_groups[0]["rules"] = [
        {
            "when": "resolving the trial identifier before evidence retrieval",
            "rule": (
                "Use at most 3 semantically distinct searches; never repeat a query or arguments; "
                "never invent an acronym expansion. Stop searching at an exact title match or "
                "once a result identifies the matching trial and NCT id; never search after "
                "resolution."
            ),
            "confidence": 1.0,
            "source_patch_count": 1,
            "provenance_stratum": "exposed-mechanism-repair",
        }
    ]

    workflows = [
        _mapping(item, "repair parent workflow")
        for item in _list(catalog.get("workflow_rules"), "repair parent workflows")
    ]
    field_routes = [
        index
        for index, item in enumerate(workflows)
        if "field-specific getter" in str(item.get("rule", ""))
    ]
    adverse_event_fallbacks = [
        index
        for index, item in enumerate(workflows)
        if "trialqa_extract_adverse_events" in str(item.get("rule", ""))
    ]
    if len(field_routes) != 1 or len(adverse_event_fallbacks) != 1:
        raise TrialQADistillationError(
            "repair parent must contain one field route and the adverse-event fallback"
        )
    workflows[field_routes[0]] = {
        "rule": (
            "After resolving the NCT id, select the field-specific getter: "
            "trialqa_get_eligibility first for eligibility, criteria, or thresholds (never "
            "trialqa_get_study); "
            "trialqa_get_outcome_measures for outcome counts or timeFrames; "
            "trialqa_get_descriptions for narrative or per-arm detail; and trialqa_get_study "
            "for enrollment or structured summary fields."
        ),
        "rationale": "Broad study retrieval can mask the requested eligibility slice.",
        "confidence": 1.0,
        "source_patch_count": 1,
        "provenance_stratum": "exposed-mechanism-repair",
    }
    catalog["tool_rules"] = tool_groups
    catalog["workflow_rules"] = workflows
    catalog["mechanism_repair_mode"] = MECHANISM_REPAIR_MODE
    skill = render_skill_markdown(catalog, tool_contract="compact")
    validate_compact_skill(catalog, skill, tool_contract="compact")
    return catalog


def layer_exposed_search_discipline_repair_catalog(
    parent_catalog: Mapping[str, Any],
) -> JsonObject:
    """Make a resolved trial identifier terminal without changing field routing."""

    catalog = cast(JsonObject, json.loads(json.dumps(parent_catalog)))
    if catalog.get("mechanism_repair_mode") != MECHANISM_REPAIR_MODE:
        raise TrialQADistillationError(
            "search-discipline parent must be the reviewed mechanism repair"
        )
    tool_groups = [
        _mapping(item, "search-discipline parent tool group")
        for item in _list(catalog.get("tool_rules"), "search-discipline parent tool groups")
    ]
    search_groups = [group for group in tool_groups if group.get("tool_name") == "trialqa_search"]
    if len(search_groups) != 1:
        raise TrialQADistillationError(
            "search-discipline parent must contain exactly one trialqa_search rule group"
        )
    search_rules = _list(search_groups[0].get("rules"), "search-discipline parent search rules")
    if len(search_rules) != 1:
        raise TrialQADistillationError(
            "search-discipline parent must contain exactly one search rule"
        )
    search_groups[0]["rules"] = [
        {
            "when": "resolving the trial identifier before evidence retrieval",
            "rule": (
                "Search once. At an exact title match with one NCT result, stop searching even "
                "if the topic is absent; call the field getter. Otherwise use at most 3 "
                "semantically distinct searches. Never repeat a query or arguments; never "
                "invent an acronym expansion."
            ),
            "confidence": 1.0,
            "source_patch_count": 1,
            "provenance_stratum": "exposed-search-discipline-repair",
        }
    ]
    catalog["tool_rules"] = tool_groups
    catalog["search_discipline_repair_mode"] = SEARCH_DISCIPLINE_REPAIR_MODE
    skill = render_skill_markdown(catalog, tool_contract="compact")
    validate_compact_skill(catalog, skill, tool_contract="compact")
    return catalog


def layer_exposed_identifier_terminal_repair_catalog(
    parent_catalog: Mapping[str, Any],
) -> JsonObject:
    """Define an identifier-containing singleton title as terminal resolution."""

    catalog = cast(JsonObject, json.loads(json.dumps(parent_catalog)))
    if catalog.get("search_discipline_repair_mode") != SEARCH_DISCIPLINE_REPAIR_MODE:
        raise TrialQADistillationError(
            "identifier-terminal parent must be the reviewed search-discipline repair"
        )
    tool_groups = [
        _mapping(item, "identifier-terminal parent tool group")
        for item in _list(catalog.get("tool_rules"), "identifier-terminal parent tool groups")
    ]
    search_groups = [group for group in tool_groups if group.get("tool_name") == "trialqa_search"]
    if len(search_groups) != 1:
        raise TrialQADistillationError(
            "identifier-terminal parent must contain exactly one trialqa_search rule group"
        )
    search_rules = _list(search_groups[0].get("rules"), "identifier-terminal parent search rules")
    if len(search_rules) != 1:
        raise TrialQADistillationError(
            "identifier-terminal parent must contain exactly one search rule"
        )
    search_groups[0]["rules"] = [
        {
            "when": "resolving the trial identifier before evidence retrieval",
            "rule": (
                "Search trial id once. One NCT result with id in its title is an exact title "
                "match: stop searching; call field getter now. Else use at most 3 semantically "
                "distinct searches. No answer-term searches. Never repeat a query or arguments; "
                "never invent an acronym expansion."
            ),
            "confidence": 1.0,
            "source_patch_count": 1,
            "provenance_stratum": "exposed-identifier-terminal-repair",
        }
    ]
    catalog["tool_rules"] = tool_groups
    catalog["identifier_terminal_repair_mode"] = IDENTIFIER_TERMINAL_REPAIR_MODE
    skill = render_skill_markdown(catalog, tool_contract="compact")
    validate_compact_skill(catalog, skill, tool_contract="compact")
    return catalog


def _write_json_atomic(path: Path, value: object, *, identical_ok: bool = True) -> None:
    payload = _canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise TrialQADistillationError(f"refusing symlinked distillation artifact: {path}")
    if path.exists():
        if identical_ok and path.is_file() and path.read_bytes() == payload:
            return
        raise TrialQADistillationError(f"immutable distillation artifact conflict: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _write_text_atomic(path: Path, value: str) -> None:
    payload = value.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise TrialQADistillationError(f"refusing symlinked distillation artifact: {path}")
    if path.exists():
        if path.is_file() and path.read_bytes() == payload:
            return
        raise TrialQADistillationError(f"immutable distillation artifact conflict: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _integrity_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.integrity.json")


def _raw_response_path(stage_path: Path) -> Path:
    return stage_path.with_name(f"{stage_path.stem}.raw-response.json")


def _artifact_binding(path: Path) -> JsonObject:
    payload = path.read_bytes()
    return {
        "artifact": path.name,
        "sha256": f"sha256:{hashlib.sha256(payload).hexdigest()}",
        "size_bytes": len(payload),
    }


def _write_stage_artifact(path: Path, value: JsonObject) -> None:
    _write_json_atomic(path, value)
    payload = path.read_bytes()
    _write_json_atomic(
        _integrity_path(path),
        {
            "schema_version": SCHEMA_VERSION,
            "artifact": path.name,
            "sha256": f"sha256:{hashlib.sha256(payload).hexdigest()}",
            "size_bytes": len(payload),
        },
    )


def _validate_stage_integrity(path: Path) -> None:
    marker = _read_json_file(_integrity_path(path), "stage artifact integrity marker")
    payload = path.read_bytes()
    if (
        marker.get("schema_version") != SCHEMA_VERSION
        or marker.get("artifact") != path.name
        or marker.get("sha256") != f"sha256:{hashlib.sha256(payload).hexdigest()}"
        or marker.get("size_bytes") != len(payload)
    ):
        raise TrialQADistillationError(f"stage artifact integrity mismatch: {path}")


def _quarantine_incomplete_stage(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise TrialQADistillationError(f"unsafe incomplete stage artifact: {path}")
    digest = _file_sha256(path)[:16]
    destination = path.with_name(f"{path.name}.orphan-{digest}")
    if destination.exists() or destination.is_symlink():
        raise TrialQADistillationError(f"incomplete stage quarantine collision: {destination}")
    path.replace(destination)
    return destination


def _load_stage_artifact(
    path: Path,
    *,
    stage: Stage,
    key: str,
    input_sha256: str,
) -> tuple[JsonObject, JsonObject]:
    _validate_stage_integrity(path)
    artifact = _read_json_file(path, f"resumable {stage} artifact")
    if (
        artifact.get("schema_version") != SCHEMA_VERSION
        or artifact.get("stage") != stage
        or artifact.get("key") != key
        or artifact.get("input_sha256") != input_sha256
    ):
        raise TrialQADistillationError(f"resumable {stage} artifact identity mismatch: {path}")
    output = _mapping(artifact.get("output"), f"resumable {stage} output")
    attestation = _mapping(artifact.get("attestation"), f"resumable {stage} attestation")
    _validate_attestation(attestation, stage=stage, key=key)
    raw_path = _raw_response_path(path)
    raw_marker_path = _integrity_path(raw_path)
    raw_binding = artifact.get("raw_response")
    if raw_binding is None:
        # Complete artifacts created before raw capture was introduced remain
        # resumable; they already contain validated output and route evidence.
        if (
            raw_path.exists()
            or raw_path.is_symlink()
            or raw_marker_path.exists()
            or raw_marker_path.is_symlink()
        ):
            raise TrialQADistillationError(
                f"resumable {stage} artifact has an unbound raw response: {path}"
            )
    else:
        expected_binding = _mapping(raw_binding, f"resumable {stage} raw response binding")
        _content, raw_attestation, actual_binding = _load_raw_response_artifact(
            raw_path,
            stage=stage,
            key=key,
            input_sha256=input_sha256,
        )
        if expected_binding != actual_binding:
            raise TrialQADistillationError(
                f"resumable {stage} raw response binding mismatch: {raw_path}"
            )
        if attestation != raw_attestation:
            raise TrialQADistillationError(
                f"resumable {stage} attestation differs from its raw response"
            )
    return output, attestation


def _validate_attestation(attestation: Mapping[str, Any], *, stage: Stage, key: str) -> None:
    if (
        attestation.get("stage") != stage
        or attestation.get("key") != key
        or attestation.get("route_model") != DISTILLER_ROUTE
        or attestation.get("upstream_model") != DISTILLER_MODEL
    ):
        raise TrialQADistillationError(
            f"invalid distiller route/model attestation for {stage}/{key}"
        )
    _required_text(attestation.get("request_id"), "distiller request id")
    usage = attestation.get("usage")
    if usage is not None and not isinstance(usage, dict):
        raise TrialQADistillationError("distiller usage must be a JSON object or null")


def _raw_response_document(call: ModelCall, result: ModelCallResult) -> JsonObject:
    """Return the unvalidated model result envelope written before parsing."""

    return {
        "schema_version": RAW_RESPONSE_SCHEMA,
        "call": {
            "stage": call.stage,
            "key": call.key,
            "input_sha256": call.input_sha256,
        },
        "result": {
            "content": result.content,
            "route_model": result.route_model,
            "upstream_model": result.upstream_model,
            "request_id": result.request_id,
            "usage": dict(result.usage) if result.usage is not None else None,
        },
    }


def _canonicalize_model_output(call: ModelCall, output: JsonObject) -> JsonObject:
    """Replace model-owned stage identity with the content-addressed call identity."""

    stage_schema = {
        "analyst": PATCH_SCHEMA,
        "question_merge": QUESTION_MERGE_SCHEMA,
        "final_merge": FINAL_MERGE_SCHEMA,
    }[call.stage]
    if set(output) == {stage_schema}:
        wrapped = output[stage_schema]
        if not isinstance(wrapped, dict):
            raise TrialQADistillationError(
                f"{call.stage}/{call.key} schema wrapper must contain an object"
            )
        canonical = dict(cast(JsonObject, wrapped))
        canonical.setdefault("schema_version", stage_schema)
    else:
        canonical = dict(output)
    canonical = cast(JsonObject, _public_tool_view(canonical))
    canonical.setdefault("schema_version", stage_schema)
    if call.stage == "analyst":
        # The request key, not model-authored prose, is the trusted evidence identity.
        canonical["source_task_name"] = call.key
        # An omitted empty patch shell carries no learned content; preserve all
        # model-authored memory items while supplying the fixed target structure.
        canonical.setdefault("skill_patch", {"target": SKILL_PATH, "sections": []})
        skill_patch = canonical.get("skill_patch")
        if isinstance(skill_patch, dict) and isinstance(skill_patch.get("sections"), list):
            skill_patch.setdefault("target", SKILL_PATH)
            normalized_sections: list[object] = []
            for raw_section in cast(list[object], skill_patch["sections"]):
                if not isinstance(raw_section, dict):
                    normalized_sections.append(raw_section)
                    continue
                section = dict(cast(JsonObject, raw_section))
                section.setdefault("action", "append")
                normalized_sections.append(section)
            skill_patch["sections"] = normalized_sections
        elif isinstance(skill_patch, dict):
            skill_patch.setdefault("target", SKILL_PATH)
            skill_patch.setdefault("sections", [])
        if isinstance(canonical.get("memory_items"), list):
            normalized_items: list[object] = []
            for raw_item in cast(list[object], canonical["memory_items"]):
                if not isinstance(raw_item, dict):
                    normalized_items.append(raw_item)
                    continue
                item = dict(cast(JsonObject, raw_item))
                rule_type = item.get("rule_type")
                if rule_type in DEFAULT_CATEGORIES:
                    matching_types = [
                        candidate
                        for candidate, required_fields in RULE_TYPE_REQUIRED_FIELDS.items()
                        if required_fields <= item.keys()
                    ]
                    if len(matching_types) == 1:
                        rule_type = matching_types[0]
                        item["rule_type"] = rule_type
                if "category" not in item and rule_type in DEFAULT_CATEGORY_BY_RULE_TYPE:
                    item["category"] = DEFAULT_CATEGORY_BY_RULE_TYPE[cast(str, rule_type)]
                normalized_items.append(item)
            canonical["memory_items"] = normalized_items
    if call.stage == "question_merge":
        # The raw-response artifact retains the exact model text. The parsed
        # stage value comes only from the trusted call key/input hash.
        canonical["question_group_key"] = call.key
    if call.stage == "final_merge":
        canonical.setdefault("skill_name", SKILL_NAME)
    return canonical


def _load_raw_response_artifact(
    path: Path,
    *,
    stage: Stage,
    key: str,
    input_sha256: str,
) -> tuple[str, JsonObject, JsonObject]:
    """Load one integrity-checked raw result without repeating its paid call."""

    _validate_stage_integrity(path)
    artifact = _read_json_file(path, f"raw {stage} model result")
    if artifact.get("schema_version") != RAW_RESPONSE_SCHEMA:
        raise TrialQADistillationError(f"raw {stage} model result has the wrong schema")
    identity = _mapping(artifact.get("call"), f"raw {stage} call identity")
    if identity != {
        "stage": stage,
        "key": key,
        "input_sha256": input_sha256,
    }:
        raise TrialQADistillationError(f"raw {stage} model result identity mismatch: {path}")
    result = _mapping(artifact.get("result"), f"raw {stage} result")
    content = result.get("content")
    if not isinstance(content, str):
        raise TrialQADistillationError(f"raw {stage} model result content must be text")
    usage = result.get("usage")
    if usage is not None and not isinstance(usage, dict):
        raise TrialQADistillationError(f"raw {stage} model result usage is malformed")
    attestation: JsonObject = {
        "stage": stage,
        "key": key,
        "request_id": result.get("request_id"),
        "route_model": result.get("route_model"),
        "upstream_model": result.get("upstream_model"),
        "usage": usage,
    }
    _validate_attestation(attestation, stage=stage, key=key)
    return content, attestation, _artifact_binding(path)


def _call_or_resume(
    *,
    call: ModelCall,
    artifact_path: Path,
    caller: ModelCaller,
    resume: bool,
    validator: Callable[[JsonObject], JsonObject],
) -> tuple[JsonObject, JsonObject, bool]:
    marker_path = _integrity_path(artifact_path)
    raw_path = _raw_response_path(artifact_path)
    raw_marker_path = _integrity_path(raw_path)
    artifact_exists = artifact_path.exists() or artifact_path.is_symlink()
    marker_exists = marker_path.exists() or marker_path.is_symlink()
    if artifact_exists != marker_exists:
        if not resume:
            raise TrialQADistillationError(
                "incomplete stage artifact exists; use resume for quarantined recovery"
            )
        if artifact_exists:
            _quarantine_incomplete_stage(artifact_path)
        if marker_exists:
            _quarantine_incomplete_stage(marker_path)
    if artifact_path.exists() or artifact_path.is_symlink():
        if not resume:
            raise TrialQADistillationError(
                f"stage artifact exists; use resume explicitly: {artifact_path}"
            )
        output, attestation = _load_stage_artifact(
            artifact_path,
            stage=call.stage,
            key=call.key,
            input_sha256=call.input_sha256,
        )
        return validator(output), attestation, False
    raw_exists = raw_path.exists() or raw_path.is_symlink()
    raw_marker_exists = raw_marker_path.exists() or raw_marker_path.is_symlink()
    if raw_exists != raw_marker_exists:
        raise TrialQADistillationError(
            f"incomplete raw model result must not be repeated: {raw_path}"
        )
    if raw_exists:
        if not resume:
            raise TrialQADistillationError(
                f"raw model result exists; use resume explicitly: {raw_path}"
            )
        content, attestation, raw_binding = _load_raw_response_artifact(
            raw_path,
            stage=call.stage,
            key=call.key,
            input_sha256=call.input_sha256,
        )
        parsed = _extract_json(content, f"{call.stage}/{call.key}")
        output = validator(_canonicalize_model_output(call, parsed))
        _write_stage_artifact(
            artifact_path,
            {
                "schema_version": SCHEMA_VERSION,
                "stage": call.stage,
                "key": call.key,
                "input_sha256": call.input_sha256,
                "output": output,
                "attestation": attestation,
                "raw_response": raw_binding,
            },
        )
        return output, attestation, False
    result = caller(call)
    # Persist the exact result before route attestation, parsing, sanitization,
    # or semantic validation. A failed validator can therefore be resumed
    # locally without paying for the same call again.
    _write_stage_artifact(raw_path, _raw_response_document(call, result))
    content, new_attestation, raw_binding = _load_raw_response_artifact(
        raw_path,
        stage=call.stage,
        key=call.key,
        input_sha256=call.input_sha256,
    )
    parsed = _extract_json(content, f"{call.stage}/{call.key}")
    output = validator(_canonicalize_model_output(call, parsed))
    _write_stage_artifact(
        artifact_path,
        {
            "schema_version": SCHEMA_VERSION,
            "stage": call.stage,
            "key": call.key,
            "input_sha256": call.input_sha256,
            "output": output,
            "attestation": new_attestation,
            "raw_response": raw_binding,
        },
    )
    return output, new_attestation, True


def _stats_total(stats: Mapping[str, Any], field: str) -> int:
    return _required_int(stats.get(field), f"routing stats {field}")


def _stats_calls(stats: Mapping[str, Any]) -> dict[str, int]:
    return _stats_model_calls(stats)


def _validate_stats_delta(
    before: Mapping[str, Any], after: Mapping[str, Any], *, expected_calls: int
) -> JsonObject:
    request_delta = _stats_total(after, "total_requests") - _stats_total(before, "total_requests")
    error_delta = _stats_total(after, "total_errors") - _stats_total(before, "total_errors")
    before_models, after_models = _stats_calls(before), _stats_calls(after)
    names = set(before_models) | set(after_models)
    deltas = {
        name: after_models.get(name, 0) - before_models.get(name, 0)
        for name in sorted(names)
        if after_models.get(name, 0) - before_models.get(name, 0)
    }
    if request_delta != expected_calls or error_delta != 0:
        raise TrialQADistillationError(
            "Switchyard routing stats do not account for all successful distiller calls"
        )
    if deltas != ({DISTILLER_MODEL: expected_calls} if expected_calls else {}):
        raise TrialQADistillationError(
            f"distillation routing stats contain unexpected model deltas: {deltas}"
        )
    return {
        "request_delta": request_delta,
        "error_delta": error_delta,
        "model_call_deltas": deltas,
    }


def _manifest_or_initialize(
    plan: DistillationPlan | CompactDistillationPlan, *, resume: bool
) -> None:
    if plan.run_path.is_symlink():
        raise TrialQADistillationError(
            f"distillation run path cannot be a symlink: {plan.run_path}"
        )
    manifest_path = plan.run_path / "run_manifest.json"
    if plan.run_path.exists():
        if not resume:
            raise TrialQADistillationError(
                f"content-addressed run already exists; use resume: {plan.run_path}"
            )
        existing = _read_json_file(manifest_path, "distillation run manifest")
        if existing != plan.manifest:
            raise TrialQADistillationError("existing run manifest differs from the content plan")
        return
    plan.run_path.mkdir(parents=True)
    _write_json_atomic(manifest_path, plan.manifest)


def _artifact_name(value: str) -> str:
    return _safe_component(value, "artifact key")


def execute_distillation(
    plan: DistillationPlan,
    *,
    caller: ModelCaller,
    stats_reader: StatsReader,
    resume: bool = False,
    activate: bool = False,
) -> DistillationResult:
    """Execute or explicitly resume the plan, validating before store mutation."""

    _manifest_or_initialize(plan, resume=resume)
    before = dict(stats_reader())
    new_call_count = 0
    attestations: list[JsonObject] = []
    stage_paths: list[Path] = []
    analyst_patches: dict[str, JsonObject] = {}
    for item in plan.evidence:
        system, user = _analyst_prompt(item, plan.reference_instruction_text)
        call = _build_call("analyst", item.evidence_id, system, user, 5000)
        artifact_path = plan.run_path / "analyst" / f"{_artifact_name(item.evidence_id)}.json"
        patch, attestation, called = _call_or_resume(
            call=call,
            artifact_path=artifact_path,
            caller=caller,
            resume=resume,
            validator=partial(_validate_analyst_patch, evidence=item),
        )
        analyst_patches[item.evidence_id] = patch
        stage_paths.append(artifact_path)
        attestations.append(attestation)
        new_call_count += int(called)

    grouped: dict[str, list[DonorEvidence]] = defaultdict(list)
    for item in plan.evidence:
        grouped[item.question_group_key].append(item)
    question_aggregates: list[JsonObject] = []
    for group in sorted(grouped):
        group_evidence = sorted(grouped[group], key=lambda item: item.repeat_index)
        patches = [analyst_patches[item.evidence_id] for item in group_evidence]
        repeat_metadata = [
            {
                "repeat_index": item.repeat_index,
                "role": item.role,
                "judge_result": item.judge_result,
            }
            for item in group_evidence
        ]
        observed_tools = frozenset().union(*(item.observed_tools for item in group_evidence))
        literals = tuple(
            sorted(
                {literal for item in group_evidence for literal in item.sensitive_literals},
                key=lambda value: (-len(value), value),
            )
        )
        system, user = _question_prompt(group, patches, repeat_metadata)
        call = _build_call("question_merge", group, system, user, 5000)
        artifact_path = plan.run_path / "questions" / f"{_artifact_name(group)}.json"
        aggregate, attestation, called = _call_or_resume(
            call=call,
            artifact_path=artifact_path,
            caller=caller,
            resume=resume,
            validator=partial(
                _validate_question_merge,
                group=group,
                patches=patches,
                judge_results=[item.judge_result for item in group_evidence],
                observed_tools=observed_tools,
                sensitive_literals=literals,
            ),
        )
        question_aggregates.append(aggregate)
        stage_paths.append(artifact_path)
        attestations.append(attestation)
        new_call_count += int(called)

    all_tools = frozenset().union(*(item.observed_tools for item in plan.evidence))
    all_literals = tuple(
        sorted(
            {literal for item in plan.evidence for literal in item.sensitive_literals},
            key=lambda value: (-len(value), value),
        )
    )
    system, user = _final_prompt(question_aggregates)
    final_call = _build_call("final_merge", plan.run_id, system, user, 8000)
    final_artifact_path = plan.run_path / "final_catalog.json"
    catalog, final_attestation, called = _call_or_resume(
        call=final_call,
        artifact_path=final_artifact_path,
        caller=caller,
        resume=resume,
        validator=partial(
            _validate_final_catalog,
            observed_tools=all_tools,
            sensitive_literals=all_literals,
            max_source_patch_count=len(plan.evidence),
        ),
    )
    stage_paths.append(final_artifact_path)
    attestations.append(final_attestation)
    new_call_count += int(called)
    after = dict(stats_reader())
    _validate_stats_delta(before, after, expected_calls=new_call_count)
    expected_attestations = len(plan.evidence) + len(grouped) + 1
    request_ids = [cast(str, item["request_id"]) for item in attestations]
    if len(attestations) != expected_attestations or len(set(request_ids)) != len(request_ids):
        raise TrialQADistillationError(
            "distiller request attestations are incomplete or contain duplicate request ids"
        )
    raw_response_paths = sorted(
        raw_path
        for path in stage_paths
        if (raw_path := _raw_response_path(path)).exists() or raw_path.is_symlink()
    )
    for raw_path in raw_response_paths:
        _validate_stage_integrity(raw_path)
    recovery_paths = sorted(plan.run_path.rglob("*.orphan-?*"))
    completion_manifest: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "run_id": plan.run_id,
        "stage_artifacts": [
            {
                "path": path.relative_to(plan.run_path).as_posix(),
                "sha256": f"sha256:{_file_sha256(path)}",
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(stage_paths)
        ],
        "raw_response_artifacts": [
            {
                "path": path.relative_to(plan.run_path).as_posix(),
                "sha256": f"sha256:{_file_sha256(path)}",
                "size_bytes": path.stat().st_size,
            }
            for path in raw_response_paths
        ],
        "recovery_artifacts": [
            {
                "path": path.relative_to(plan.run_path).as_posix(),
                "sha256": f"sha256:{_file_sha256(path)}",
                "size_bytes": path.stat().st_size,
            }
            for path in recovery_paths
        ],
    }
    completion_path = plan.run_path / "completion_manifest.json"
    _write_json_atomic(completion_path, completion_manifest)
    completion_sha256 = _file_sha256(completion_path)

    skill = render_skill_markdown(catalog)
    _assert_no_sensitive(skill, all_literals, "rendered skill")
    if len(skill.encode("utf-8")) > 128 * 1024:
        raise TrialQADistillationError("rendered skill exceeds 128 KiB")
    skill_sha = hashlib.sha256(skill.encode("utf-8")).hexdigest()
    source_ids = [item.evidence_id for item in plan.evidence]
    candidate_seed = {
        "run_id": plan.run_id,
        "skill_path": SKILL_PATH,
        "skill_sha256": skill_sha,
        "source_evidence_ids": source_ids,
    }
    candidate_id = f"trialqa-{_digest(candidate_seed)[:32]}"
    validation: JsonObject = {
        "status": "passed",
        "schema_version": SCHEMA_VERSION,
        "scope": "native-train-evidence-and-static-candidate-validation",
        "performance_validated": False,
        "distillation_mode": plan.mode,
        "performance_eligible": plan.mode == "full",
        "run_id": plan.run_id,
        "candidate_id": candidate_id,
        "source_evidence_ids": source_ids,
        "checks": {
            "all_evidence_native_and_content_validated": True,
            "all_evidence_train_donor": True,
            "all_executor_sessions_unskilled": True,
            "all_executor_sessions_ultra_zero_errors": True,
            "one_analyst_patch_per_evidence": len(analyst_patches) == len(plan.evidence),
            "repeats_grouped_by_question": len(question_aggregates) == len(grouped),
            "pinned_reference_instruction": True,
            "distiller_route_only": True,
            "distiller_model_only": True,
            "routing_stats_accounted": True,
            "request_ids_unique": True,
            "no_task_literal_leakage": True,
            "skill_frontmatter_present": skill.startswith(f"---\nname: {SKILL_NAME}\n"),
        },
        "routing": {
            "route": DISTILLER_ROUTE,
            "upstream_model": DISTILLER_MODEL,
            "profile_sha256": plan.routing_profile_sha256,
            "attested_call_count": len(attestations),
            "attestations": attestations,
        },
        "artifacts": {
            "reference_instruction_sha256": plan.reference_instruction_sha256,
            "completion_manifest_sha256": f"sha256:{completion_sha256}",
            "recovery_artifact_count": len(recovery_paths),
            "skill_sha256": f"sha256:{skill_sha}",
            "analyst_patch_count": len(analyst_patches),
            "question_merge_count": len(question_aggregates),
            "raw_response_count": len(raw_response_paths),
        },
    }
    if not all(validation["checks"].values()):
        raise TrialQADistillationError("candidate validation report contains a failed check")
    report_path = plan.run_path / "candidate_validation.json"
    _write_json_atomic(report_path, validation)
    generated_skill_path = plan.run_path / "candidate" / SKILL_PATH
    _write_text_atomic(generated_skill_path, skill)

    # SkillDistillationStore requires a top-level index in addition to the one
    # actual Codex skill directory.  Both are immutable and content hashed.
    index = (
        "# TrialQA distilled skill bundle\n\n"
        f"The executable skill is [`{SKILL_PATH}`]({SKILL_PATH}).\n"
    )
    store = SkillDistillationStore(plan.namespace, plan.project_dir)
    candidate_path = store.save_candidate(
        candidate_id=candidate_id,
        skills={"SKILL.md": index, SKILL_PATH: skill},
        generator=f"{DISTILLER_ROUTE} ({DISTILLER_MODEL})",
        evidence_ids=source_ids,
        validation=validation,
    )
    activated = False
    if activate:
        store.activate(candidate_id)
        activated = True
    return DistillationResult(
        run_id=plan.run_id,
        candidate_id=candidate_id,
        candidate_path=candidate_path,
        skill_path=candidate_path / SKILL_PATH,
        validation_report_path=report_path,
        activated=activated,
        model_call_count=new_call_count,
    )


def _transport_cached_catalog_artifact(
    plan: CompactDistillationPlan, artifact_path: Path, *, resume: bool
) -> tuple[JsonObject, JsonObject, bool]:
    if (
        plan.source_final_catalog is None
        or plan.source_final_binding is None
        or plan.source_final_attestation is None
    ):
        raise TrialQADistillationError("source-final catalog transport lacks source provenance")
    catalog = compact_cached_final_catalog(
        plan.source_final_catalog, tool_contract=plan.tool_contract
    )
    input_sha = _digest(
        {
            "mode": CACHED_CATALOG_TRANSPORT_MODE,
            "source_run_id": plan.source_run_id,
            "source_final_catalog": plan.source_final_binding,
        }
    )
    document: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "stage": "catalog_transport",
        "key": plan.run_id,
        "input_sha256": input_sha,
        "output": catalog,
        "provenance": {
            "mode": CACHED_CATALOG_TRANSPORT_MODE,
            "source_run_id": plan.source_run_id,
            "source_final_catalog": dict(plan.source_final_binding),
            "source_final_attestation": dict(plan.source_final_attestation),
            "new_model_call_count": 0,
        },
    }
    marker_path = _integrity_path(artifact_path)
    artifact_exists = artifact_path.exists() or artifact_path.is_symlink()
    marker_exists = marker_path.exists() or marker_path.is_symlink()
    if artifact_exists != marker_exists:
        raise TrialQADistillationError("incomplete cached catalog transport artifact")
    if artifact_exists:
        if not resume:
            raise TrialQADistillationError(
                f"transport artifact exists; use resume explicitly: {artifact_path}"
            )
        _validate_stage_integrity(artifact_path)
        if _read_json_file(artifact_path, "cached catalog transport artifact") != document:
            raise TrialQADistillationError("cached catalog transport artifact changed")
    else:
        _write_stage_artifact(artifact_path, document)
    return catalog, dict(plan.source_final_attestation), False


def execute_compact_distillation(
    plan: CompactDistillationPlan,
    *,
    caller: ModelCaller,
    stats_reader: StatsReader,
    resume: bool = False,
    activate: bool = False,
) -> DistillationResult:
    """Run exactly one compact final merge over cached question aggregates."""

    rebuilt = build_compact_distillation_plan(
        project_dir=plan.project_dir,
        namespace=plan.namespace,
        work_dir=plan.run_path.parent,
        source_run=plan.source_run,
        routing_profile=plan.routing_profile,
        proxy_url=plan.proxy_url,
        paid_raw_run=plan.paid_raw_run,
        tool_contract=plan.tool_contract,
        transport_source_final_catalog=plan.transport_source_final_catalog,
    )
    if rebuilt != plan:
        raise TrialQADistillationError("compact plan differs from its re-attested source artifacts")
    _manifest_or_initialize(plan, resume=resume)
    before = dict(stats_reader())
    artifact_path = plan.run_path / "final_catalog.json"
    if plan.transport_source_final_catalog:
        catalog, attestation, called = _transport_cached_catalog_artifact(
            plan, artifact_path, resume=resume
        )
    else:
        system, user = _compact_final_prompt(plan.question_aggregates)
        call = _build_call("final_merge", plan.run_id, system, user, 2500)
        recovered_paid_raw = False
        if plan.paid_raw_run is not None:
            if plan.paid_raw_binding is None:
                raise TrialQADistillationError("compact paid-raw recovery lacks its binding")
            source_raw = plan.paid_raw_run / "final_catalog.raw-response.json"
            source_manifest = plan.paid_raw_run / "run_manifest.json"
            if (
                plan.paid_raw_binding.get("run_manifest_sha256")
                != f"sha256:{_file_sha256(source_manifest)}"
                or plan.paid_raw_binding.get("raw_response_sha256")
                != f"sha256:{_file_sha256(source_raw)}"
                or plan.paid_raw_binding.get("raw_response_integrity_sha256")
                != f"sha256:{_file_sha256(_integrity_path(source_raw))}"
            ):
                raise TrialQADistillationError("paid compact raw-response binding changed")
            _validate_stage_integrity(source_raw)
            source_document = _read_json_file(source_raw, "paid compact raw response")
            source_call = _mapping(source_document.get("call"), "paid compact raw call")
            source_result = _mapping(source_document.get("result"), "paid compact raw result")
            if (
                source_call.get("stage") != "final_merge"
                or source_call.get("input_sha256") != call.input_sha256
                or source_result.get("request_id") != plan.paid_raw_binding.get("request_id")
                or source_result.get("route_model") != DISTILLER_ROUTE
                or source_result.get("upstream_model") != DISTILLER_MODEL
            ):
                raise TrialQADistillationError(
                    "paid compact raw response does not match the current final-merge input"
                )
            destination_raw = _raw_response_path(artifact_path)
            _write_stage_artifact(
                destination_raw,
                {
                    "schema_version": RAW_RESPONSE_SCHEMA,
                    "call": {
                        "stage": "final_merge",
                        "key": plan.run_id,
                        "input_sha256": call.input_sha256,
                    },
                    "result": dict(source_result),
                    "recovered_from": dict(plan.paid_raw_binding),
                },
            )
            recovered_paid_raw = True

        def validate_catalog(raw: JsonObject) -> JsonObject:
            source_catalog = _validate_final_catalog(
                raw,
                observed_tools=plan.observed_tools,
                sensitive_literals=(),
                max_source_patch_count=FULL_EVIDENCE_COUNT,
            )
            compacted = compact_final_catalog(source_catalog, tool_contract=plan.tool_contract)
            skill = render_skill_markdown(compacted, tool_contract=plan.tool_contract)
            validate_compact_skill(compacted, skill, tool_contract=plan.tool_contract)
            return compacted

        catalog, attestation, called = _call_or_resume(
            call=call,
            artifact_path=artifact_path,
            caller=caller,
            resume=resume or recovered_paid_raw,
            validator=validate_catalog,
        )
    after = dict(stats_reader())
    new_call_count = int(called)
    _validate_stats_delta(before, after, expected_calls=new_call_count)

    raw_path = _raw_response_path(artifact_path)
    _validate_stage_integrity(artifact_path)
    if raw_path.exists() or raw_path.is_symlink():
        _validate_stage_integrity(raw_path)
    completion_manifest: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "run_id": plan.run_id,
        "tool_contract": plan.tool_contract,
        "new_model_call_count": new_call_count,
        "source_run_id": plan.source_run_id,
        "source_artifacts": list(plan.source_bindings),
        "transport_mode": (
            CACHED_CATALOG_TRANSPORT_MODE if plan.transport_source_final_catalog else None
        ),
        "source_final_catalog": (
            dict(plan.source_final_binding) if plan.source_final_binding is not None else None
        ),
        "paid_raw_recovery": (
            dict(plan.paid_raw_binding) if plan.paid_raw_binding is not None else None
        ),
        "stage_artifacts": [
            {
                "path": artifact_path.name,
                "sha256": f"sha256:{_file_sha256(artifact_path)}",
                "size_bytes": artifact_path.stat().st_size,
            }
        ],
        "raw_response_artifacts": (
            [
                {
                    "path": raw_path.name,
                    "sha256": f"sha256:{_file_sha256(raw_path)}",
                    "size_bytes": raw_path.stat().st_size,
                }
            ]
            if raw_path.exists()
            else []
        ),
    }
    completion_path = plan.run_path / "completion_manifest.json"
    _write_json_atomic(completion_path, completion_manifest)

    skill = render_skill_markdown(catalog, tool_contract=plan.tool_contract)
    compact_metrics = validate_compact_skill(catalog, skill, tool_contract=plan.tool_contract)
    _assert_no_sensitive(skill, (), "compact rendered skill")
    skill_sha = hashlib.sha256(skill.encode("utf-8")).hexdigest()
    candidate_seed = {
        "run_id": plan.run_id,
        "source_run_id": plan.source_run_id,
        "tool_contract": plan.tool_contract,
        "skill_path": SKILL_PATH,
        "skill_sha256": skill_sha,
        "source_evidence_ids": list(plan.source_evidence_ids),
        "compaction_mode": catalog.get("compaction_mode"),
        "source_final_catalog": plan.source_final_binding,
    }
    candidate_id = f"trialqa-{_digest(candidate_seed)[:32]}"
    validation: JsonObject = {
        "status": "passed",
        "schema_version": SCHEMA_VERSION,
        "scope": (
            "cached-train-final-catalog-deterministic-transport"
            if plan.transport_source_final_catalog
            else "cached-question-aggregates-and-compact-final-merge"
        ),
        "performance_validated": False,
        "distillation_mode": (
            CACHED_CATALOG_TRANSPORT_MODE
            if plan.transport_source_final_catalog
            else "compact-final-only"
        ),
        "performance_eligible": True,
        "run_id": plan.run_id,
        "source_run_id": plan.source_run_id,
        "tool_contract": plan.tool_contract,
        "new_model_call_count": new_call_count,
        "candidate_id": candidate_id,
        "source_evidence_ids": list(plan.source_evidence_ids),
        "checks": {
            "source_full_run_validated": True,
            "source_artifacts_hash_bound": True,
            "exact_question_merge_count": len(plan.question_aggregates) == FULL_QUESTION_COUNT,
            "one_new_distiller_call_or_resumed": (
                new_call_count == 0
                if plan.transport_source_final_catalog
                else new_call_count in {0, 1}
            ),
            "distiller_route_only": attestation.get("route_model") == DISTILLER_ROUTE,
            "distiller_model_only": attestation.get("upstream_model") == DISTILLER_MODEL,
            "source_final_catalog_hash_bound": (
                not plan.transport_source_final_catalog or plan.source_final_binding is not None
            ),
            "train_only_source_provenance": True,
            "zero_new_model_calls": (
                not plan.transport_source_final_catalog or new_call_count == 0
            ),
            "compact_size": compact_metrics["size_bytes"] <= COMPACT_SKILL_MAX_BYTES,
            "compact_words": compact_metrics["word_count"] <= COMPACT_SKILL_MAX_WORDS,
            "compact_rules": compact_metrics["rule_count"] <= COMPACT_SKILL_MAX_RULES,
            "deterministic_compaction_applied": True,
            "tool_contract_bound": (
                plan.manifest.get("tool_contract") == plan.tool_contract
                and completion_manifest.get("tool_contract") == plan.tool_contract
                and catalog.get("tool_contract") == plan.tool_contract
            ),
            "tool_contract_rendered": (
                (plan.tool_contract == "compact") == ("## Compact ToolUniverse contract" in skill)
            ),
            "compact_transport_map_exact": (
                plan.tool_contract != "compact"
                or catalog.get("transport") == _compact_transport_metadata()
            ),
            "paid_raw_response_reused_without_recall": (
                plan.paid_raw_run is None or new_call_count == 0
            ),
            "bounded_search_contract": "at most 3 semantically distinct searches" in skill,
            "generic_field_specific_getter_routing": (
                not plan.transport_source_final_catalog or "field-specific getter" in skill
            ),
            "evidence_sufficiency_fallback": (
                not plan.transport_source_final_catalog
                or (
                    "If the selected slice lacks the requested field" in skill
                    and "another relevant getter" in skill
                )
            ),
            "non_exhaustive_routes_preserved": (
                not plan.transport_source_final_catalog or "non-exhaustive" in skill
            ),
            "no_hard_one_getter_or_five_call_shortcut": (
                not plan.transport_source_final_catalog
                or (
                    "one question-specific getter" not in skill
                    and "Never exceed 5 operational" not in skill
                )
            ),
            "no_task_literal_leakage": True,
            "skill_frontmatter_present": skill.startswith(f"---\nname: {SKILL_NAME}\n"),
        },
        "routing": {
            "route": DISTILLER_ROUTE,
            "upstream_model": DISTILLER_MODEL,
            "profile_sha256": plan.routing_profile_sha256,
            "attested_call_count": 0 if plan.transport_source_final_catalog else 1,
            "attestations": [] if plan.transport_source_final_catalog else [attestation],
            "source_final_attestation": (
                attestation if plan.transport_source_final_catalog else None
            ),
        },
        "artifacts": {
            "completion_manifest_sha256": f"sha256:{_file_sha256(completion_path)}",
            "skill_sha256": f"sha256:{skill_sha}",
            "source_question_merge_count": len(plan.question_aggregates),
            "source_final_catalog": (
                dict(plan.source_final_binding) if plan.source_final_binding is not None else None
            ),
            "paid_raw_recovery": (
                dict(plan.paid_raw_binding) if plan.paid_raw_binding is not None else None
            ),
            **compact_metrics,
        },
    }
    if not all(cast(dict[str, bool], validation["checks"]).values()):
        raise TrialQADistillationError("compact candidate validation contains a failed check")
    report_path = plan.run_path / "candidate_validation.json"
    _write_json_atomic(report_path, validation)
    generated_skill_path = plan.run_path / "candidate" / SKILL_PATH
    _write_text_atomic(generated_skill_path, skill)

    index = (
        "# TrialQA compact distilled skill bundle\n\n"
        f"The executable skill is [`{SKILL_PATH}`]({SKILL_PATH}).\n"
    )
    store = SkillDistillationStore(plan.namespace, plan.project_dir)
    candidate_path = store.save_candidate(
        candidate_id=candidate_id,
        skills={"SKILL.md": index, SKILL_PATH: skill},
        generator=(
            (f"deterministic {CACHED_CATALOG_TRANSPORT_MODE} source={plan.source_run_id}")
            if plan.transport_source_final_catalog
            else (
                f"{DISTILLER_ROUTE} ({DISTILLER_MODEL}) compact-final-only "
                f"tool-contract={plan.tool_contract}"
            )
        ),
        evidence_ids=list(plan.source_evidence_ids),
        validation=validation,
    )
    activated = False
    if activate:
        store.activate(candidate_id)
        activated = True
    return DistillationResult(
        run_id=plan.run_id,
        candidate_id=candidate_id,
        candidate_path=candidate_path,
        skill_path=candidate_path / SKILL_PATH,
        validation_report_path=report_path,
        activated=activated,
        model_call_count=new_call_count,
    )


def _copy_development_evidence(
    store: SkillDistillationStore, evidence: DevelopmentEvidence
) -> None:
    """Atomically make validated external evidence local to the layered candidate."""

    destination = store.evidence_path / evidence.evidence_id
    with store.exclusive_lock():
        if destination.exists() or destination.is_symlink():
            validate_native_trialqa_evidence_directory(
                destination, expected_evidence_id=evidence.evidence_id
            )
            return
        staging_root = Path(
            tempfile.mkdtemp(
                prefix=f".{evidence.evidence_id}.development-", dir=store.evidence_path
            )
        )
        staging = staging_root / evidence.evidence_id
        try:
            shutil.copytree(evidence.path, staging)
            validate_native_trialqa_evidence_directory(
                staging, expected_evidence_id=evidence.evidence_id
            )
            staging.replace(destination)
        finally:
            if staging_root.exists():
                shutil.rmtree(staging_root)


def execute_development_layer(
    plan: DevelopmentLayerPlan, *, resume: bool = False
) -> DistillationResult:
    """Materialize one provenance-separated, zero-call development candidate."""

    rebuilt = build_development_layer_plan(
        project_dir=plan.project_dir,
        namespace=plan.namespace,
        work_dir=plan.run_path.parent,
        parent_candidate_id=plan.parent_candidate_id,
        development_evidence_dir=plan.failure_evidence.path,
        supporting_development_evidence_dir=plan.support_evidence.path,
        regression_verdict=plan.verdict_path,
        descriptive_manifest=plan.descriptive_manifest_path,
        primary_manifest=plan.primary_manifest_path,
    )
    if rebuilt != plan:
        raise TrialQADistillationError(
            "development-layer plan differs from its re-attested source artifacts"
        )
    if plan.run_path.is_symlink():
        raise TrialQADistillationError(
            f"development-layer run path cannot be a symlink: {plan.run_path}"
        )
    run_manifest_path = plan.run_path / "run_manifest.json"
    if plan.run_path.exists():
        if not resume:
            raise TrialQADistillationError(
                f"content-addressed run already exists; use resume: {plan.run_path}"
            )
        if _read_json_file(run_manifest_path, "development-layer run manifest") != plan.manifest:
            raise TrialQADistillationError(
                "existing development-layer manifest differs from the content plan"
            )
    else:
        plan.run_path.mkdir(parents=True)
        _write_json_atomic(run_manifest_path, plan.manifest)

    catalog = layer_exposed_development_catalog(plan.parent_catalog)
    catalog_path = plan.run_path / "final_catalog.json"
    catalog_document: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "stage": "development_layer",
        "key": plan.run_id,
        "input_sha256": _digest(
            {
                "mode": DEVELOPMENT_LAYER_MODE,
                "parent_catalog": plan.parent_catalog_binding,
                "verdict": plan.verdict_binding,
            }
        ),
        "output": catalog,
        "provenance": {
            "mode": DEVELOPMENT_LAYER_MODE,
            "parent_candidate_id": plan.parent_candidate_id,
            "parent_catalog": plan.parent_catalog_binding,
            "failure_evidence_id": plan.failure_evidence.evidence_id,
            "support_evidence_id": plan.support_evidence.evidence_id,
            "regression_verdict": plan.verdict_binding,
            "new_model_call_count": 0,
        },
    }
    artifact_exists = catalog_path.exists() or catalog_path.is_symlink()
    marker_exists = (
        _integrity_path(catalog_path).exists() or _integrity_path(catalog_path).is_symlink()
    )
    if artifact_exists != marker_exists:
        raise TrialQADistillationError("incomplete development-layer catalog artifact")
    if artifact_exists:
        if not resume:
            raise TrialQADistillationError(
                f"development-layer artifact exists; use resume: {catalog_path}"
            )
        _validate_stage_integrity(catalog_path)
        if _read_json_file(catalog_path, "development-layer catalog") != catalog_document:
            raise TrialQADistillationError("development-layer catalog artifact changed")
    else:
        _write_stage_artifact(catalog_path, catalog_document)

    completion: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "run_id": plan.run_id,
        "mode": DEVELOPMENT_LAYER_MODE,
        "new_model_call_count": 0,
        "parent_candidate_id": plan.parent_candidate_id,
        "parent_catalog": plan.parent_catalog_binding,
        "provenance_strata": plan.manifest["provenance_strata"],
        "stage_artifacts": [
            {
                "path": catalog_path.name,
                "sha256": f"sha256:{_file_sha256(catalog_path)}",
                "size_bytes": catalog_path.stat().st_size,
            }
        ],
    }
    completion_path = plan.run_path / "completion_manifest.json"
    _write_json_atomic(completion_path, completion)
    skill = render_skill_markdown(catalog, tool_contract="compact")
    metrics = validate_compact_skill(catalog, skill, tool_contract="compact")
    _assert_no_sensitive(
        skill,
        tuple(
            sorted(
                {
                    *_sensitive_literals(plan.failure_evidence.document),
                    *_sensitive_literals(plan.support_evidence.document),
                    plan.failure_evidence.evidence_id,
                    plan.support_evidence.evidence_id,
                },
                key=lambda value: (-len(value), value),
            )
        ),
        "development-layer rendered skill",
    )
    skill_sha = hashlib.sha256(skill.encode("utf-8")).hexdigest()
    development_ids = tuple(
        sorted(
            (
                plan.failure_evidence.evidence_id,
                plan.support_evidence.evidence_id,
            )
        )
    )
    all_evidence_ids = (*plan.train_evidence_ids, *development_ids)
    candidate_seed = {
        "run_id": plan.run_id,
        "parent_candidate_id": plan.parent_candidate_id,
        "parent_manifest_sha256": plan.parent_manifest_sha256,
        "skill_path": SKILL_PATH,
        "skill_sha256": skill_sha,
        "source_evidence_ids": list(all_evidence_ids),
        "mode": DEVELOPMENT_LAYER_MODE,
    }
    candidate_id = f"trialqa-{_digest(candidate_seed)[:32]}"
    validation: JsonObject = {
        "status": "passed",
        "schema_version": SCHEMA_VERSION,
        "scope": "train-base-plus-exposed-development-primary88-only",
        "distillation_mode": DEVELOPMENT_LAYER_MODE,
        "performance_validated": False,
        "performance_eligible": True,
        "full_96_performance_eligible": False,
        "run_id": plan.run_id,
        "candidate_id": candidate_id,
        "parent_candidate_id": plan.parent_candidate_id,
        "tool_contract": "compact",
        "new_model_call_count": 0,
        "source_evidence_ids": list(all_evidence_ids),
        "provenance_strata": plan.manifest["provenance_strata"],
        "checks": {
            "parent_candidate_hash_bound": True,
            "parent_train_provenance_inherited": len(plan.train_evidence_ids)
            == FULL_EVIDENCE_COUNT,
            "development_evidence_native_validated": True,
            "development_evidence_parent_bound": True,
            "regression_verdict_hash_bound": True,
            "failure_and_support_mechanism_attested": True,
            "first_eight_explicitly_quarantined": True,
            "primary_88_capture_not_started": True,
            "full_96_ineligible": True,
            "zero_new_model_calls": True,
            "compact_size": metrics["size_bytes"] <= COMPACT_SKILL_MAX_BYTES,
            "compact_words": metrics["word_count"] <= COMPACT_SKILL_MAX_WORDS,
            "compact_rules": metrics["rule_count"] <= COMPACT_SKILL_MAX_RULES,
            "generic_mechanism_only": all(
                literal not in skill for literal in ("PF-06463922", "NCT01970865", "10 mg", "25 mg")
            ),
            "conditional_adverse_event_fallback": (
                "if the selected slice lacks direct support" in skill
                and "as a fallback evidence slice" in skill
                and "all dose questions" not in skill.lower()
            ),
            "outcome_inference_prohibited": (
                "Do not infer starting, lowest, or highest values from outcome timeFrames" in skill
            ),
            "skill_frontmatter_present": skill.startswith(f"---\nname: {SKILL_NAME}\n"),
        },
        "routing": {
            "attested_call_count": 0,
            "attestations": [],
        },
        "artifacts": {
            "completion_manifest_sha256": f"sha256:{_file_sha256(completion_path)}",
            "catalog_sha256": f"sha256:{_file_sha256(catalog_path)}",
            "skill_sha256": f"sha256:{skill_sha}",
            **metrics,
        },
    }
    if not all(cast(dict[str, bool], validation["checks"]).values()):
        raise TrialQADistillationError(
            "development-layer candidate validation contains a failed check"
        )
    report_path = plan.run_path / "candidate_validation.json"
    _write_json_atomic(report_path, validation)
    generated_skill_path = plan.run_path / "candidate" / SKILL_PATH
    _write_text_atomic(generated_skill_path, skill)
    store = SkillDistillationStore(plan.namespace, plan.project_dir)
    _copy_development_evidence(store, plan.failure_evidence)
    _copy_development_evidence(store, plan.support_evidence)
    index = (
        "# TrialQA exposed-development skill bundle\n\n"
        f"The executable skill is [`{SKILL_PATH}`]({SKILL_PATH}).\n"
    )
    candidate_path = store.save_candidate(
        candidate_id=candidate_id,
        skills={"SKILL.md": index, SKILL_PATH: skill},
        generator=(f"deterministic {DEVELOPMENT_LAYER_MODE} parent={plan.parent_candidate_id}"),
        evidence_ids=list(all_evidence_ids),
        validation=validation,
    )
    return DistillationResult(
        run_id=plan.run_id,
        candidate_id=candidate_id,
        candidate_path=candidate_path,
        skill_path=candidate_path / SKILL_PATH,
        validation_report_path=report_path,
        activated=False,
        model_call_count=0,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "plan",
            "dry-run",
            "execute",
            "resume",
            "compact-plan",
            "compact-execute",
            "compact-resume",
            "development-layer-plan",
            "development-layer-execute",
            "development-layer-resume",
        ),
    )
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--reference-repo", type=Path)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--parent-candidate-id")
    parser.add_argument("--development-evidence-dir", type=Path)
    parser.add_argument("--supporting-development-evidence-dir", type=Path)
    parser.add_argument("--regression-verdict", type=Path)
    parser.add_argument("--descriptive-manifest", type=Path)
    parser.add_argument("--primary-manifest", type=Path)
    parser.add_argument(
        "--paid-raw-run",
        type=Path,
        help="reuse one integrity-bound paid compact raw response without another call",
    )
    parser.add_argument(
        "--transport-source-final-catalog",
        action="store_true",
        help=(
            "deterministically compact the integrity-bound train-only source final catalog "
            "without a new model call"
        ),
    )
    parser.add_argument(
        "--tool-contract",
        choices=TOOL_CONTRACTS,
        default="direct",
        help=(
            "candidate tool surface for compact commands; compact deterministically "
            "maps learned aliases through ToolUniverse meta-tools"
        ),
    )
    parser.add_argument("--routing-profile", type=Path)
    parser.add_argument("--proxy-url")
    parser.add_argument("--namespace", default=NAMESPACE)
    parser.add_argument("--evidence-id", action="append")
    parser.add_argument("--expected-question-count", type=int, default=24)
    parser.add_argument("--expected-repeats", type=int, default=5)
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="require exactly one donor evidence bundle; output is non-performance only",
    )
    parser.add_argument(
        "--execute-model-calls",
        action="store_true",
        help=(
            "required confirmation when execution can make a model call; unnecessary "
            "when --paid-raw-run makes compact execution deterministic and local"
        ),
    )
    parser.add_argument("--activate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI with explicit no-call planning and paid execution boundaries."""

    args = _parser().parse_args(argv)
    try:
        if args.command.startswith("development-layer-"):
            required = {
                "--parent-candidate-id": args.parent_candidate_id,
                "--development-evidence-dir": args.development_evidence_dir,
                "--supporting-development-evidence-dir": (args.supporting_development_evidence_dir),
                "--regression-verdict": args.regression_verdict,
                "--descriptive-manifest": args.descriptive_manifest,
                "--primary-manifest": args.primary_manifest,
            }
            missing = [flag for flag, value in required.items() if value is None]
            if missing:
                raise TrialQADistillationError(
                    "development-layer commands require " + ", ".join(missing)
                )
            if args.activate:
                raise TrialQADistillationError(
                    "development-layer commands never activate candidates"
                )
            development_plan = build_development_layer_plan(
                project_dir=args.project_dir,
                namespace=args.namespace,
                work_dir=args.work_dir,
                parent_candidate_id=cast(str, args.parent_candidate_id),
                development_evidence_dir=cast(Path, args.development_evidence_dir),
                supporting_development_evidence_dir=cast(
                    Path, args.supporting_development_evidence_dir
                ),
                regression_verdict=cast(Path, args.regression_verdict),
                descriptive_manifest=cast(Path, args.descriptive_manifest),
                primary_manifest=cast(Path, args.primary_manifest),
            )
            if args.command == "development-layer-plan":
                print(json.dumps(development_plan.manifest, indent=2, sort_keys=True))
                return 0
            development_result = execute_development_layer(
                development_plan,
                resume=args.command == "development-layer-resume",
            )
            print(
                json.dumps(
                    {
                        "run_id": development_result.run_id,
                        "candidate_id": development_result.candidate_id,
                        "candidate_path": str(development_result.candidate_path),
                        "skill_path": str(development_result.skill_path),
                        "validation_report": str(development_result.validation_report_path),
                        "activated": development_result.activated,
                        "model_call_count": development_result.model_call_count,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command.startswith("compact-"):
            if args.source_run is None:
                raise TrialQADistillationError("compact commands require --source-run")
            if args.routing_profile is None or args.proxy_url is None:
                raise TrialQADistillationError(
                    "compact commands require --routing-profile and --proxy-url"
                )
            compact_plan = build_compact_distillation_plan(
                project_dir=args.project_dir,
                namespace=args.namespace,
                work_dir=args.work_dir,
                source_run=args.source_run,
                routing_profile=args.routing_profile,
                proxy_url=args.proxy_url,
                paid_raw_run=args.paid_raw_run,
                tool_contract=args.tool_contract,
                transport_source_final_catalog=args.transport_source_final_catalog,
            )
            if args.command == "compact-plan":
                print(json.dumps(compact_plan.manifest, indent=2, sort_keys=True))
                return 0
            if (
                not args.execute_model_calls
                and args.paid_raw_run is None
                and not args.transport_source_final_catalog
            ):
                raise TrialQADistillationError(
                    "compact execute/resume requires --execute-model-calls confirmation"
                )
            if args.paid_raw_run is not None or args.transport_source_final_catalog:

                def paid_raw_zero_stats() -> Mapping[str, Any]:
                    return {
                        "total_requests": 0,
                        "total_errors": 0,
                        "models": {},
                    }

                compact_caller: ModelCaller = _PaidRawNoCallCaller()
                compact_stats_reader: StatsReader = paid_raw_zero_stats
            else:
                compact_client = LocalSwitchyardCaller(compact_plan.proxy_url)
                compact_caller = compact_client
                compact_stats_reader = compact_client.read_stats
            compact_result = execute_compact_distillation(
                compact_plan,
                caller=compact_caller,
                stats_reader=compact_stats_reader,
                resume=args.command == "compact-resume",
                activate=args.activate,
            )
            print(
                json.dumps(
                    {
                        "run_id": compact_result.run_id,
                        "candidate_id": compact_result.candidate_id,
                        "candidate_path": str(compact_result.candidate_path),
                        "skill_path": str(compact_result.skill_path),
                        "validation_report": str(compact_result.validation_report_path),
                        "activated": compact_result.activated,
                        "model_call_count": compact_result.model_call_count,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.tool_contract != "direct":
            raise TrialQADistillationError(
                "--tool-contract compact is valid only for compact commands"
            )
        if args.reference_repo is None or not args.evidence_id:
            raise TrialQADistillationError(
                "full/pilot commands require --reference-repo and --evidence-id"
            )
        if args.routing_profile is None or args.proxy_url is None:
            raise TrialQADistillationError(
                "full/pilot commands require --routing-profile and --proxy-url"
            )
        plan = build_distillation_plan(
            project_dir=args.project_dir,
            namespace=args.namespace,
            evidence_ids=args.evidence_id,
            work_dir=args.work_dir,
            reference_repo=args.reference_repo,
            routing_profile=args.routing_profile,
            proxy_url=args.proxy_url,
            expected_question_count=args.expected_question_count,
            expected_repeats=args.expected_repeats,
            mode="pilot" if args.pilot else "full",
        )
        if args.command in {"plan", "dry-run"}:
            print(json.dumps(plan.manifest, indent=2, sort_keys=True))
            return 0
        if not args.execute_model_calls:
            raise TrialQADistillationError(
                "execute/resume requires explicit --execute-model-calls confirmation"
            )
        client = LocalSwitchyardCaller(plan.proxy_url)
        result = execute_distillation(
            plan,
            caller=client,
            stats_reader=client.read_stats,
            resume=args.command == "resume",
            activate=args.activate,
        )
        print(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "candidate_id": result.candidate_id,
                    "candidate_path": str(result.candidate_path),
                    "skill_path": str(result.skill_path),
                    "validation_report": str(result.validation_report_path),
                    "activated": result.activated,
                    "model_call_count": result.model_call_count,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except TrialQADistillationError as exc:
        print(f"trialqa_local_distiller: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
