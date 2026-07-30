# Adapter reproducibility pattern

For owners of craft-taskgen track adapters (`planning`, `search-native`,
`tools`, future tracks). Explains how to migrate your adapter to the shared
reproducibility-first Dockerfile builder so every shipped task has pinned,
hermetic dependencies.

Status: `planning` is the reference implementation. `search-native` and
`tools` still use their own Dockerfile logic. Migration is straightforward
and low-risk.

## What this gets you

- Base image pinned by sha256 digest (`python:{ver}-slim-bookworm@sha256:...`).
- `uv==0.7.12` pinned as the installer.
- Full PyPI transitive deps pinned via `uv pip install --system --no-deps -r requirements.lock`.
- Every task has `environment/requirements.lock` alongside its `Dockerfile`.
- Hard gate: candidates without `pinned_requirements` raise at adapter time. Unpinned tasks do not ship.
- Optional: pinned Claude Code + codex baked into the image for in-container agent execution (`bake_agents=True`).

## Migration steps

### 1. Replace your Dockerfile construction

Before:

```python
def _build_dockerfile(candidate):
    docker = candidate["docker"]
    lines = [
        f"FROM python:{docker.get('python', '3.11')}-slim",
        "RUN apt-get update && apt-get install -y git curl ...",
        f"RUN git clone https://github.com/{candidate['repo']}.git /repo ...",
        f"RUN {docker['install']}",
    ]
    return "\n".join(lines) + "\n"

(env_dir / "Dockerfile").write_text(_build_dockerfile(candidate))
```

After:

```python
from craft_taskgen.adapters._docker import (
    build_dockerfile,
    spec_from_candidate,
    write_environment,
)

docker_spec = spec_from_candidate(candidate)
write_environment(env_dir, build_dockerfile(docker_spec), docker_spec.pinned_requirements)
```

### 2. Adjust your candidate schema

Required fields on each candidate JSON:

| Field | Notes |
|---|---|
| `repo` | `"owner/name"` |
| `parent_sha` | The commit to check out |
| `docker.python` | `"3.9"` / `"3.10"` / `"3.11"` / `"3.12"`. Defaults to `"3.11"`. |
| `docker.install` | Shell command run after the repo is cloned. e.g. `uv pip install --system -e .` |
| `pinned_requirements` | Raw `requirements.lock` content (text). **No opt-out.** |

Optional:

| Field | Notes |
|---|---|
| `docker.pre_install` | List of shell commands run before `install` (patches, etc.) |
| `docker.test_deps` | Space-separated extra deps. Ignored once `pinned_requirements` is present — the lock supersedes. |

### 3. Produce the pins

Three-step loop per task (manual for now, automatable later):

1. Build the task unpinned (use a scratch Dockerfile outside the adapter, or a one-off `bake_lock.sh`).
2. `docker run <image> uv pip freeze` → paste output into the candidate's `pinned_requirements`.
3. Rebuild via the adapter. Oracle must still hit 1.0. If not, the pin set is incomplete.

A `craft-taskgen-lock-deps` tool that automates this loop is a follow-up.

### 4. Update your adapter tests

- Add `pinned_requirements` to every `_make_candidate` / fixture.
- Assert the lock file lands in the generated task's `environment/` dir.
- Assert the Dockerfile contains `uv pip install --system --no-deps -r /tmp/requirements.lock`.
- Add a test that pinless candidates raise.

See `tests/test_adapters_planning.py::test_convert_single_rejects_candidate_without_pinned_requirements` for the pattern.

## Baked agent runtimes

Tracks where harbor runs the agent *inside* the task container (like
`search-native`) should flip `bake_agents=True`:

```python
docker_spec = spec_from_candidate(candidate)
docker_spec.bake_agents = True
```

That adds a layer installing pinned Claude Code, codex, and opencode via
npm. Versions are module constants in `_docker.py`
(`CLAUDE_CODE_VERSION`, `CODEX_VERSION`, `OPENCODE_VERSION`) — bump in
one place, every track picks it up. Default `False` (matches tracks
where harbor runs the agent on the host).

## Not goals (for now)

- Version-indexed repo_specs (hoisting per-repo install / test_command out of per-PR LLM discovery). Reproducibility-adjacent, separate concern.
- Automated lock-deps tool. Manual workflow suffices for the 2-week release.
