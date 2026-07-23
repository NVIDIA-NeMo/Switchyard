# Contributing to Switchyard

Thank you for your interest in contributing! This document outlines the development workflow, testing practices, and code standards.

## External Contributions

We welcome contributions of all sizes, from typo fixes to new features. The short version:

1. [Fork the repository](https://github.com/NVIDIA-NeMo/Switchyard/fork) and clone your fork:

   ```bash
   git clone https://github.com/YOUR-USERNAME/Switchyard.git
   cd Switchyard
   git remote add upstream https://github.com/NVIDIA-NeMo/Switchyard.git
   ```

2. Pick the right process for the size of your change:
   - **Small changes** (typos, docs, focused bug fixes under ~100 lines): open a PR directly, no issue needed.
   - **Larger changes** (new features, refactors, anything 100+ lines): [open an issue](https://github.com/NVIDIA-NeMo/Switchyard/issues/new/choose) first so maintainers can confirm the direction before you invest time.

3. Create a branch, make your change, and run the checks in [Code standards](#2-code-standards).

4. Commit with a DCO sign-off (see [Signing Your Work](#signing-your-work)):

   ```bash
   git commit -s -m "fix: description of the change"
   ```

5. Push to your fork and open a PR against `main`, linking any related issues (e.g. "Closes #42").

Review is requested automatically from the core team via [CODEOWNERS](.github/CODEOWNERS), so there is no need to pick reviewers. Keep each PR focused on one concern, include tests for behavior changes, and respond to feedback with follow-up commits rather than force-pushes.

Using AI tools to write code is fine, but you must understand and be able to explain every change in your PR.

## Setup

See [Development](DEVELOPMENT.md) for full setup instructions.

Quick start:

```bash
git clone https://github.com/NVIDIA-NeMo/Switchyard.git
cd Switchyard
uv sync
uvx pre-commit install --install-hooks --hook-type pre-commit --hook-type commit-msg
source .venv/bin/activate
```

## Development Workflow

### 1. Branch naming

Use one of these prefixes for clarity:

- `feature/...` — new functionality
- `fix/...` — bug fixes
- `docs/...` — documentation updates
- `refactor/...` — non-functional code changes
- `test/...` — test additions or improvements

Example: `feature/add-new-router-backend` or `fix/async-context-leak`

### 2. Code standards

All code must pass these gates before a PR is merged:

```bash
# Linting — zero errors required
uv run ruff check .

# Type checking — strict mode
uv run mypy switchyard

# Tests — no failures
uv run pytest tests/ -v
```

These commands run in CI on every push. **Fix linting errors locally before pushing:**

```bash
uv run ruff check --fix .
```

### 3. Commit messages

Commit messages must follow
[Conventional Commits v1.0.0](https://www.conventionalcommits.org/en/v1.0.0/).
This is enforced locally by the `commit-msg` hook and in GitHub Actions.

Every commit must also carry a DCO sign-off (`git commit -s`), enforced by the
required DCO check on every PR. See [Signing Your Work](#signing-your-work).

- ✓ `fix: handle async context cleanup in ProxyContext`
- ✓ `feat: add stage-router routing backend`
- ✗ `Fixed stuff` / `Updated code`

Use one of:

- `build:` — build system or packaging changes
- `feat:` — new feature
- `fix:` — bug fix
- `ci:` — CI configuration
- `docs:` — documentation
- `refactor:` — code structure (no behavior change)
- `test:` — test additions
- `perf:` — performance improvement
- `chore:` — tooling, dependencies, CI
- `revert:` — revert a previous commit
- `style:` — formatting-only changes

Scopes are optional: `fix(cli): preserve launcher args`.
Breaking changes use `!` or a `BREAKING CHANGE:` footer:

```text
feat(api)!: remove legacy route option
```

### 4. Pull request process

1. **Create a feature branch** off `main`:
   ```bash
   git checkout -b feature/your-feature
   ```

2. **Write tests** for new functionality or bug fixes. Tests live in `tests/`:
   - Unit tests: `tests/test_*.py`

3. **Run the full test suite locally** before pushing:
   ```bash
   uv run ruff check .
   uv run mypy switchyard
   uv run pytest tests/ -v
   ```

4. **Push and open a PR** on GitHub. Include:
   - Clear description of the change and why
   - Link to any related issues (e.g., "Closes #42")
   - Test coverage notes

5. **Address review feedback** — push additional commits (don't force-push unless explicitly asked).

6. **Squash and merge** when approved. One commit per feature keeps history clean.
   Keep the PR title conventional too, because GitHub can use the PR title for
   the squash-merge commit.

The `CI Success` and `DCO` status checks are required on `main`; the other
workflows (commitlint, PR title) run on every PR but are advisory.

## Testing

### Unit tests (fast, no API keys)

```bash
uv run pytest tests/ -v
```

### Integration tests (requires API keys)

Integration tests hit live LLM APIs. Set credentials before running:

```bash
export OPENAI_API_KEY="sk-..."
# or NVIDIA_API_KEY / ANTHROPIC_API_KEY depending on the backend you target
```

The default unit test suite runs with no network access and no API keys. Live, end-to-end tests are not currently part of the public CI pipeline.

## Architecture

See [Agents](AGENTS.md) for the full architecture guide. Key points:

- **Typed requests/responses** — use `ChatRequest` and `ChatResponse` subtypes
- **Composable chain** — `RequestProcessor` → `LLMBackend` → `ResponseProcessor` → `TranslationEngine`
- **Profiles** — pre-built chains in `switchyard/lib/profiles/`

When adding a new component:

1. Decide the role: `RequestProcessor`, `LLMBackend`, `ResponseProcessor`, or translation engine work.
2. Subclass the ABC from `switchyard/lib/roles.py`.
3. Put Python middleware in the matching subpackage (`switchyard/lib/processors/`, `switchyard/lib/backends/`) and provider translation logic in `crates/switchyard-translation`.
4. Add tests in `tests/`.
5. Export from the relevant `__init__.py` and from `switchyard/__init__.py`'s `__all__`.

## Documentation

- **Code comments** — explain the *why*, not the *what*. One line max unless complex.
- **Docstrings** — include for public APIs. Format: one-line summary, blank line, details, examples.
- **README** — keep in sync with CLI surface (commands, flags, new subcommands).
- **docs/** — architecture, design decisions, advanced usage.

## Questions?

- **Setup issues?** See [Development](DEVELOPMENT.md)
- **Architecture questions?** See [Agents](AGENTS.md)
- **Design docs?** See [docs/](docs/)
- **Report a bug?** [Open an issue](https://github.com/NVIDIA-NeMo/Switchyard/issues)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. See [Code of Conduct](CODE_OF_CONDUCT.md).

## Signing Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:

  ```bash
  $ git commit -s -m "feat: add cool feature"
  ```

  This will append the following to your commit message:

  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO (https://developercertificate.org/):

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

    Everyone is permitted to copy and distribute verbatim copies of this
    license document, but changing it is not allowed.


    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I
        have the right to submit it under the open source license
        indicated in the file; or

    (b) The contribution is based upon previous work that, to the best
        of my knowledge, is covered under an appropriate open source
        license and I have the right under that license to submit that
        work with modifications, whether created in whole or in part
        by me, under the same open source license (unless I am
        permitted to submit under a different license), as indicated
        in the file; or

    (c) The contribution was provided directly to me by some other
        person who certified (a), (b) or (c) and I have not modified
        it.

    (d) I understand and agree that this project and the contribution
        are public and that a record of the contribution (including all
        personal information I submit with it, including my sign-off) is
        maintained indefinitely and may be redistributed consistent with
        this project or the open source license(s) involved.
  ```
