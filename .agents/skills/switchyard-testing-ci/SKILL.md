---
name: switchyard-testing-ci
description: Select or debug Switchyard validation and GitHub CI. Use when asked which tests to run, whether a change is ready, why a CI job failed, how to reproduce CI, or whether live provider tests are appropriate.
---

# Switchyard Testing And CI

Read `.github/workflows/ci.yml` before claiming local CI equivalence. Select tests from the current
diff and symbol references rather than a static subsystem inventory.

## Validation Matrix

| Change | Minimum trustworthy validation |
|---|---|
| Markdown or skill Markdown only | Parse skill frontmatter when applicable; `git diff --check` |
| Python implementation | Focused pytest, `uv run ruff check .`, `uv run mypy switchyard` |
| Broad Python or public API | Hermetic full pytest plus lint and mypy |
| One Rust crate | `cargo fmt --all --check`, workspace clippy, focused crate tests |
| Shared Rust/API/FFI | Rust workspace tests plus affected Python tests |
| Packaging, extras, top-level imports | The slim-install smoke commands from the current CI workflow |
| Live provider behavior | Focused integration test only after explicit authorization |

## Standard Gates

Python pre-PR gate:

```bash
uv run ruff check .
uv run mypy switchyard
env -u OPENROUTER_API_KEY -u NVIDIA_API_KEY -u OPENAI_API_KEY \
  -u ANTHROPIC_API_KEY uv run pytest tests/ -v -m "not integration"
```

Rust workspace gate:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Use `-o addopts=` when the repository's pytest defaults would stop failure collection too early.
Run focused tests first, then broaden only when the changed contract or shared ownership warrants it.

## Live Tests

Integration tests can spend money, require credentials, and contact real services. Never run them
as an automatic pre-PR step. Provision `NVIDIA_API_KEY` through the environment or a secret manager
before running the user-authorized path, and state which provider was called:

```bash
uv run pytest tests/e2e/<focused_test>.py \
  -v -m integration -o addopts=
```

## Skill Validation

Each retained skill must have YAML frontmatter with `name` and `description`:

```bash
uv run python - <<'PY'
from pathlib import Path

import yaml

for path in Path(".agents/skills").glob("*/SKILL.md"):
    parts = path.read_text().split("---", 2)
    data = yaml.safe_load(parts[1]) if len(parts) == 3 else None
    if not isinstance(data, dict) or not data.get("name") or not data.get("description"):
        raise SystemExit(f"{path}: missing name or description")
print("skill frontmatter is valid")
PY
git diff --check
```

## Reporting

Report exact commands, pass/fail status, and whether any live network call occurred. Do not call a
focused test run "full CI", and do not treat mypy or clippy warnings as unrelated without checking
whether the diff introduced them.
