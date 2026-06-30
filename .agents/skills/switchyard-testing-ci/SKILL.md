---
name: switchyard-testing-ci
description: Use when validating a Switchyard change, preparing or reviewing a PR, debugging CI failures (ruff, mypy, SPDX, pytest, slim-install), choosing which local tests to run, or diagnosing dependency, optional-extra, stale-name, CLI, server, translation, routing, stats, or live e2e failures. Triggers on phrases like "is this ready to merge", "CI is failing", "which tests should I run", or "how do I reproduce this locally".
---

# Switchyard Testing and CI

## Overview

**Pick the smallest local gate that is genuinely equivalent to what CI will run, then map failures
to targeted fixes.** The two goals — move fast locally, never report a green run weaker than the CI
gate you are claiming — usually conflict; this skill resolves that conflict from the current diff
rather than from memory.

If your change also touches code you have not yet read, run [`switchyard-codebase-exploration`](../switchyard-codebase-exploration/SKILL.md) first so the validation set covers the real impact, not just the diff surface.

Rust crates under `crates/` (`switchyard-core`, `switchyard-translation`,
`switchyard-components`, `switchyard-server`, `switchyard-py`) are active on the Rust core branch.
For Rust-touching diffs, include `cargo fmt --all --check`, `cargo clippy --workspace
--all-targets -- -D warnings`, and the smallest trustworthy `cargo test` scope;
use `cargo test --workspace` before calling a broad Rust MR ready.

## Quick Reference

| Situation | Command |
|---|---|
| Pick gates from the current diff | `python .agents/skills/switchyard-testing-ci/scripts/select_validation.py --changed` |
| Pre-PR hermetic gate (default) | `uv run ruff check . && uv run mypy switchyard && env -u OPENROUTER_API_KEY -u NVIDIA_API_KEY -u OPENAI_API_KEY -u ANTHROPIC_API_KEY uv run pytest tests/ -v -m "not integration"` |
| Mirror CI pytest with no live creds | `env -u OPENROUTER_API_KEY -u NVIDIA_API_KEY -u OPENAI_API_KEY -u ANTHROPIC_API_KEY uv run pytest tests/ -v -m "not integration"` |
| Rust component crate change | `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test -p switchyard-components` |
| Rust server crate change | `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test -p switchyard-server` |
| Broad Rust crate change | `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace` |
| Slim-install regression guard | see [Slim-install smoke gate](#slim-install-smoke-gate) |
| Live e2e (only on explicit user request) | `NVIDIA_API_KEY=… uv run pytest tests/e2e/ -v -m integration -o addopts= --maxfail=10` |
| Skill/docs change only | YAML frontmatter check + `git diff --check` (see [Skill/docs-only gate](#skilldocs-only-gate)) |

## Dynamic Selection First

From the repo root, inspect the diff and let the selector propose focused gates:

```bash
cd "$(git rev-parse --show-toplevel)"
git status -sb
git diff --stat
python .agents/skills/switchyard-testing-ci/scripts/select_validation.py --changed
```

For a not-yet-edited area, pass likely owners explicitly:

```bash
python .agents/skills/switchyard-testing-ci/scripts/select_validation.py \
  --path switchyard/lib/translation/request_engine.py \
  --path tests/test_request_translation_engine.py
```

Use the output as the starting plan, then add any tests revealed by code search or by the failure.

## CI Gates and Local Equivalence

The hard GitHub Actions gates live in `.github/workflows/ci.yml`:

- `uv run ruff check .`
- SPDX header check for every Python file found by CI, including `.agents/skills/**/scripts/*.py`
- `uv run pytest tests/ -v -m "not integration"` on Python 3.12 through 3.14
- Rust workspace gate: `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`,
  and `cargo test --workspace`
- slim-install smoke: isolated `uv run --with` default install/import checks, heavy-package
  absence, and CLI help checks with `nemo-switchyard[cli,server] @ file://...`
- `uv run mypy switchyard` is a signal job (`continue-on-error`) but should still be run for typed
  package code, public APIs, profiles, route bundles, backends, request/response models, and translation changes.

The local pre-commit config also runs the Rust fmt/clippy gate when Rust files,
`Cargo.toml`, or `Cargo.lock` change. Treat those hooks as hard failures; do not
bypass them with `--no-verify`.

A CI-equivalent `ruff` claim means a clean checkout or generated artifacts removed. Do not treat
`uv run ruff check . --exclude docs/.venv-docs --exclude docs/_build` as canonical PR validation;
that is only a dirty-workspace workaround.

If generated docs artifacts exist locally, prefer:

```bash
rm -rf docs/.venv-docs docs/_build site/_build
git status -sb
uv run ruff check .
```

## Standard Command Sets

### Fast local loop

```bash
uv run pytest tests/test_<area>.py -v
uv run pytest tests/test_<area>.py::test_name -v
uv run ruff check .
```

Pytest uses `addopts = "-x"`; use `-o addopts= --maxfail=10` when finding all failures:

```bash
uv run pytest tests/ -v -o addopts= --maxfail=10 -m "not integration"
```

### Pre-PR package-code gate

Use the hermetic test gate by default so local credentials cannot trigger live provider calls:

```bash
uv run ruff check .
uv run mypy switchyard
env -u OPENROUTER_API_KEY -u NVIDIA_API_KEY -u OPENAI_API_KEY -u ANTHROPIC_API_KEY uv run pytest tests/ -v -m "not integration"
```

If you need to mirror the CI pytest command locally, run from a clean checkout or explicitly remove
provider credentials first:

```bash
env -u OPENROUTER_API_KEY -u NVIDIA_API_KEY -u OPENAI_API_KEY -u ANTHROPIC_API_KEY uv run pytest tests/ -v -m "not integration"
```

### Live integration/e2e gate

Run this only when the user asked for live validation and accepts upstream API calls:

```bash
NVIDIA_API_KEY=... uv run pytest tests/e2e/ -v -m integration -o addopts= --maxfail=10
```

`tests/e2e/conftest.py` and per-file markers/skips own credential behavior. Do not run live e2e as
a default PR gate.

### Slim-install smoke gate

Run after touching `pyproject.toml`, lockfiles, top-level imports, optional extras,
`switchyard/__init__.py`, CLI/server exports, or anything that could pull heavy packages into the
default install:

```bash
SWITCHYARD_ROOT="$(git rev-parse --show-toplevel)"
SWITCHYARD_DEFAULT_PACKAGE="nemo-switchyard @ file://${SWITCHYARD_ROOT}"
SWITCHYARD_EXTRAS_PACKAGE="nemo-switchyard[cli,server] @ file://${SWITCHYARD_ROOT}"
cd /tmp
uv run --isolated --no-project --python 3.12 \
  --with "${SWITCHYARD_DEFAULT_PACKAGE}" \
  python - <<'PY'
import importlib.util
import switchyard
print('import OK, version:', switchyard.__version__)
from switchyard import RandomRoutingPresets
print('preset import OK:', sorted(RandomRoutingPresets.PRESETS))
forbidden = [
    'torch', 'transformers', 'huggingface_hub', 'routellm', 'litellm',
    'datasets', 'tokenizers', 'safetensors', 'pyarrow', 'agents', 'mcp', 'docx',
]
extras = [m for m in forbidden if importlib.util.find_spec(m) is not None]
if extras:
    raise SystemExit(f'FAIL: heavy packages pulled into slim install: {extras}')
print('OK: heavy packages absent')
PY
uv run --isolated --no-project \
  --python 3.12 \
  --with "${SWITCHYARD_EXTRAS_PACKAGE}" \
  switchyard launch claude --help >/dev/null
uv run --isolated --no-project \
  --python 3.12 \
  --with "${SWITCHYARD_EXTRAS_PACKAGE}" \
  switchyard launch claude --help 2>&1 \
  | grep -q -- '--routing-profiles' || { echo 'FAIL: --routing-profiles flag missing from help'; exit 1; }
uv run --isolated --no-project \
  --python 3.12 \
  --with "${SWITCHYARD_EXTRAS_PACKAGE}" \
  switchyard serve --help >/dev/null
```

### Skill/docs-only gate

For `.agents/skills/` changes, validate YAML frontmatter, whitespace, and any Python helper scripts:

```bash
python - <<'PY'
from pathlib import Path
import yaml
for p in Path('.agents/skills').glob('*/SKILL.md'):
    text = p.read_text(encoding='utf-8')
    if not text.startswith('---\n'):
        raise SystemExit(f'{p}: missing frontmatter start')
    end = text.find('\n---\n', 4)
    if end == -1:
        raise SystemExit(f'{p}: missing frontmatter end')
    data = yaml.safe_load(text[4:end])
    if not isinstance(data, dict) or not data.get('name') or not data.get('description'):
        raise SystemExit(f'{p}: missing name/description')
print('OK: skill frontmatter parses')
PY
git diff --check
find .agents/skills -path '*/scripts/*.py' -print
```

For any listed skill Python helper, run targeted lint and SPDX checks:

```bash
.venv/bin/ruff check --isolated .agents/skills/<skill>/scripts/<helper>.py
python - <<'PY'
from importlib.machinery import SourceFileLoader
from pathlib import Path
spdx = SourceFileLoader('spdx', '.hooks/add_spdx_headers.py').load_module()
paths = [Path('.agents/skills/<skill>/scripts/<helper>.py')]
bad = [str(p) for p in paths if not spdx.has_required_header(p)]
if bad:
    raise SystemExit('missing SPDX header: ' + ', '.join(bad))
print('OK: SPDX headers present')
PY
```

`.agents` is gitignored, so stage new or modified skills explicitly:

```bash
git status --ignored -sb .agents/skills
git add -f .agents/skills/<skill>
```

## Failure Map

### Ruff

- Generated docs/venv paths (`docs/.venv-docs/`, `docs/_build/`, `site/_build/`) are local debris;
  remove them rather than editing vendored/generated files.
- Use autofix narrowly, then inspect the diff:

  ```bash
  uv run ruff check --fix .
  uv run ruff check .
  ```

### SPDX

CI requires this exact header in every Python file, after an optional shebang:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

Fix with:

```bash
python .hooks/add_spdx_headers.py path/to/file.py
uv run ruff check .
```

### Pytest

Classify before editing:

- Optional package import (`routellm`, `torch`, `fastapi`, `prompt_toolkit`, `nemo_platform`) → keep
  optional deps out of top-level/default install paths; use lazy imports and clear runtime errors.
- Unit test makes network call → mock with `respx`, `httpx.ASGITransport`, or a fake backend.
- Translation mismatch → run translation-focused tests before touching broad plumbing.
- Streaming failure → inspect streaming response accumulator, SSE helpers, and translation-stream tests.
- CLI/config regression → run launcher/config/model/verify tests.
- Routing/stats regression → run random-routing and RouteLLM suites separately.

Useful focused groups:

```bash
uv run pytest tests/test_anthropic_openai_translation.py tests/test_responses_openai_translation.py tests/test_translation_engine_chaos.py -v -o addopts=
uv run pytest tests/test_launch_claude.py tests/test_launch_codex.py tests/test_launch_route_builder.py tests/test_user_config.py tests/test_model_discovery.py tests/test_verify.py -v -o addopts=
uv run pytest tests/test_random_routing_config.py tests/test_random_routing_llm_backend.py tests/test_random_routing_presets.py tests/test_random_routing_profile.py -v -o addopts=
uv run pytest tests/test_routellm_config.py tests/test_routellm_llm_backend.py tests/test_routellm_request_processor.py -v -o addopts=
```

### Mypy

Mypy is strict on `switchyard/` and ignores tests. Prefer concrete request/response subclasses after
`isinstance` narrowing, `TYPE_CHECKING` imports for circular or optional types, and aliases from
`switchyard/lib/roles.py` instead of broad `Any`. Use targeted ignores with error codes only when an
external incompatibility is localized.

### Slim install / dependencies

Default install must not pull heavyweight packages:

- `torch`, `transformers`, `huggingface_hub`, `routellm`, `litellm`, `datasets`, `tokenizers`,
  `safetensors`, `pyarrow`, `agents`, `mcp`, `docx`

Keep them in optional extras or dev groups, not `[project.dependencies]`. Dependency changes require
explicit user intent; after changing dependencies run `uv lock`, `uv sync`, and the selected tests.

### Stale package names

The distribution package is `nemo-switchyard`; imports and the CLI are `switchyard`. Do not
introduce `nemo_switchyard` imports or publish docs that imply `pip install switchyard`. Check:

```bash
uv run pytest tests/test_cli_stale_names.py tests/test_no_stale_module_paths.py tests/test_version_package_name.py -v -o addopts=
```

## Common Mistakes

| Mistake | Reality |
|---|---|
| Running `uv run ruff check . --exclude docs/.venv-docs --exclude docs/_build` and calling it CI-equivalent | That is a dirty-workspace workaround, not the CI command. Either remove the generated artifacts (`rm -rf docs/.venv-docs docs/_build site/_build`) or run from a clean checkout. |
| Running `uv run pytest tests/ -v` with `OPENROUTER_API_KEY`/`NVIDIA_API_KEY`/`OPENAI_API_KEY`/`ANTHROPIC_API_KEY` set in your shell | Tests that mock providers can accidentally hit live endpoints. Default to `-m "not integration"`, or strip credentials with `env -u`. |
| Treating mypy as optional because CI marks it `continue-on-error` | Mypy still catches real bugs in `switchyard/` typed code. Run it for any change to profiles, route bundles, backends, request/response models, or translation. |
| Skipping the slim-install smoke gate after a dependency or top-level import change | This is the *actual* hard CI gate that catches `torch`/`transformers`/`routellm` accidentally landing in the default install. |
| Claiming validation from `scripts/select_validation.py` without rerunning it after committing changes | The script diffs against `HEAD` plus untracked files; if your changes are already committed, `--changed` returns "no diff". Pass `--path` explicitly, or diff against the branch base. |
| Adding `# noqa` or per-line ignores to make ruff green | Ruff is a hard CI gate. Fix the code or, if the rule is wrong here, lift the ignore to the file or project level with a one-line justification. |
| Editing a generated artifact (e.g., `docs/.venv-docs/...`) because ruff complained about it | Delete the artifact instead. CI does not have it; you should not either. |

## Report Format

Report validation with exact commands and live-call status:

```markdown
Validation:
- `uv run ruff check .` — pass
- `uv run mypy switchyard` — pass / not run (<reason>)
- `uv run pytest tests/ -v -m "not integration"` — pass / not run (<reason>)
- Slim install smoke — pass / not run (<reason>)
- Live provider tests — not run / pass (<explicit user request>)
```

## Related Skills

- [`switchyard-codebase-exploration`](../switchyard-codebase-exploration/SKILL.md) — run this **before** the testing-ci skill on any non-trivial change, so the validation set you pick covers every file the change actually touches (importers, profile builders, tests), not just the diff surface.
