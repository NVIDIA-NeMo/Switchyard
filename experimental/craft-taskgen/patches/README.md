# Harbor Agent Patches

Patches for Harbor agents to work with the NVIDIA inference gateway.

**Harbor pin:** `harbor-framework/harbor@a56546f` (post-v0.13.1 `main`; includes #1826 native pi log-trim and #1840 wildcard network allowlist). The patches were re-derived against this pin on 2026-06-08 (bump from `331dcba3`/v0.6.4). The agent contract is the post-PR #1255 `await self.exec_as_agent(...)` shape. If you bump the pin, patches will need re-deriving again.

**Apply:** `cd .venv/lib/python3.12/site-packages && patch -p1 < ../../../../patches/harbor-agent-patches.diff`

## Patches

**base.py** — Raise the `_truncate_output` default cap from 1000 to 8000 chars so installed-agent setup/exec stdout+stderr captured into the trial log stays diagnosable.

> **models/trial/config.py (dropped).** Previously rewrote the `__` trial-name separator to `-`. Verified unnecessary at `a56546f`: harbor's `_sanitize_docker_image_name` / `_sanitize_docker_compose_project_name` already preserve `_`, Docker 29.x accepts `__` in image/repo names and tags, and a smoke run with `__` trial names scored clean. Removed from the patch.

> **Host-gateway compose override (no longer a site-packages patch).** `extra_hosts: "host.docker.internal:host-gateway"` (needed for `--backend vllm` against a host-local vLLM) is no longer patched into harbor's compose base — that file no longer exists at `a56546f`. It now lives at `patches/compose-overrides/host-gateway.yaml` and is passed via harbor's native `--extra-docker-compose` flag (wired into the vLLM path of `scripts/run-baselines.sh`).

**claude_code.py** — Install hardening only: skip the system-package and npm install steps when `claude` is already on PATH **and its version matches the pinned version** (else reinstall; if unpinned, skip if present), and tolerate dead/unsigned apt repos on EOL base images (`(apt-get update 2>/dev/null || true)`). The version-aware guard (claude/codex/opencode) extracts the installed semver and exact-compares it to the pin, mirroring upstream harbor #1848.

> **Env-var settings moved to the launcher.** `CLAUDE_CODE_ATTRIBUTION_HEADER=0`, `CLAUDE_CODE_ENABLE_TELEMETRY=0`, and `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1` are now passed via `scripts/run-baselines.sh` `--agent-env` instead of patched into the `run()` env block. Verified: a claude-code gateway smoke shows the vars reach the container (trial `config.json`), `context_management` is null in every response turn, and no API errors. (The gateway now also tolerates the `context_management` field directly, so the betas disable is defensive.)

**codex.py** — Three changes (the old fourth — switching MCP-config write to `>>` — is dropped: harbor `a56546f` already appends):
- Preserve full model path instead of stripping to last `/`-segment (breaks gateway routing).
- When `OPENAI_BASE_URL` is set inside the container, write a full `[model_providers.nvidia_gateway]` block (`wire_api = "responses"`) to `$CODEX_HOME/config.toml` so codex routes there instead of `api.openai.com`, plus `[tools] view_image=false` / `[features] unified_exec=false` and an optional `web_search = "disabled"` closed-book toggle (`CODEX_DISABLE_WEB_SEARCH`).
- Load a custom model catalog from the host path in `CODEX_MODEL_CATALOG_JSON` and write it into the container as `$CODEX_HOME/custom-model-catalog.json`; pass `-c model_catalog_json=...` and `--disable unified_exec` to `codex exec`. This makes codex recognize gateway-routed model slugs as reasoning-capable; without it, codex does not pass `reasoning.effort` through to the Responses API (short trajectories, early "want me to continue?" stops). See `codex-model-catalog.json` in this directory.

Also includes install hardening: skip-if-codex-present-at-pinned-version, a 3-try NVM bootstrap retry with a Node 16 fallback, and a ripgrep release-tarball fallback for restricted-network agent-bake environments.

**opencode.py** — Gateway and container-runtime fixes:
- Custom `nvidia`/`vllm` providers route through `@ai-sdk/openai-compatible` (chat-completions API) instead of `/v1/responses`, which vLLM-style gateways don't expose; `models.<id>.temperature=true` unlocks the capability gate; `reasoningEffort` from `OPENCODE_REASONING_EFFORT`.
- The config is written via an unquoted heredoc so `${OPENAI_BASE_URL}`/`${OPENAI_API_KEY}` expand at container write-time.
- Extends the `nvidia`/`vllm` provider required-key list with `OPENAI_API_KEY` + `OPENAI_BASE_URL`.
- Injects per-mode sampling params from env vars (`OPENCODE_BUILD_*`, `OPENCODE_PLAN_*` — temperature, top_p, top_k, max_tokens, presence_penalty).
- `OPENCODE_SMALL_MODEL` override so title generation doesn't fall back to `gpt-5-nano` on single-model vLLM endpoints.
- `config["permission"] = "allow"` (or `webfetch`/`websearch` = `deny` under `OPENCODE_DISABLE_WEBFETCH`) so non-interactive container runs don't silently reject tool prompts; `OPENCODE_DISABLE_AUTO_COMPACT` for baseline reproducibility.
- `OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX` passthrough into the container env for vLLM endpoints that reject oversized requests.
- Install Node 16 directly from nodejs.org (skip-if-installed), bypassing NVM's github clone whose IPs rotate past the firewall snapshot.

**pi.py** — Three changes (the old fourth — a `message_update` event filter to curb multi-GB `pi.txt` — is dropped: harbor `a56546f` filters `message_update` natively, see #1826):
- Rename the npm package `@mariozechner/pi-coding-agent` → `@earendil-works/pi-coding-agent` (the current upstream).
- Add a custom `nvidia` provider: plant `~/.pi/agent/models.json` declaring `nvidia` as an `openai-completions` provider pointing at `OPENAI_BASE_URL` (pi has no built-in `nvidia` provider and no `--base-url` flag), and strip exactly one leading provider segment for the CLI model name.

**openhands_sdk.py** — Bootstrap a uv-managed Python 3.12 when the container's system `python3` is older. `openhands-sdk` requires Python ≥3.12 but a non-trivial slice of task containers pin Python 3.10 or 3.11 for upstream-codebase compatibility. The patch probes `python3 --version` and, if <3.12, downloads uv via `curl | sh` and uses `uv venv --python 3.12 --seed` to create the SDK venv from a parallel `python-build-standalone` interpreter. System `python3` (which the task code depends on) is never modified. One-time ~80MB download per container; no-op when `python3 >= 3.12`.

## OpenHands

The `openhands-sdk` agent reads `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` directly from env, with no provider-name sniffing. To route through the NVIDIA gateway, run with:

```bash
uv run harbor run -y \
  --agent openhands-sdk \
  --model openai/<gateway-slug> \
  --agent-kwarg version=1.17.0 \
  --agent-kwarg max_iterations=200 \
  --agent-env LLM_API_KEY="$OPENAI_API_KEY" \
  --agent-env LLM_BASE_URL="$OPENAI_BASE_URL"
```

The `openai/` prefix tells LiteLLM to use the OpenAI chat-completions wire format against `LLM_BASE_URL`; everything after it (e.g. `nvidia/nvidia/nemotron-3-ultra-rl-050826`) is treated as an opaque model id and is passed to the gateway verbatim.

The legacy `openhands` adapter (using `openhands-ai` 0.x) is also present but not currently exercised on this pin.

## Applying

After installing or reinstalling Harbor:

```bash
uv sync --reinstall-package harbor
cd .venv/lib/python3.12/site-packages && patch -p1 < ../../../../patches/harbor-agent-patches.diff
```

(Adjust the `python3.12` segment to your interpreter version.)

To enable codex reasoning, point codex at the model catalog file before running:

```bash
export CODEX_MODEL_CATALOG_JSON="$(pwd)/patches/codex-model-catalog.json"
```

Verify reasoning is active by grepping `reasoning_output_tokens` in a codex trial's rollout file (under `<trial>/agent/sessions/.../rollout-*.jsonl`). Non-zero = working. Zero on every turn = `CODEX_MODEL_CATALOG_JSON` not set, patch not applied, or the slug in the catalog doesn't match the `--model` you're passing.

If the patch fails due to context mismatch, derive afresh: the diff was produced by editing pristine `a56546f` site-packages files and `diff -u`-ing against a pristine copy. The authoritative per-change content is the diff hunks themselves; the per-patch summaries above describe intent. Anchor on surrounding code, not line numbers (a future pin bump will shift them again). Key load-bearing items: codex `nvidia_gateway` provider block uses a distinct heredoc tag (`EOF2`) to avoid colliding with the outer `EOF`; the 3 claude_code gateway env vars go directly in the `run()` env block; opencode's config write must use an *unquoted* heredoc so `${VAR}` expands at container write-time. Do **not** re-add the codex MCP `>>` change or the pi `message_update` filter — both are native at `a56546f`.

The host-gateway compose override is not part of this diff — it is `patches/compose-overrides/host-gateway.yaml`, passed via `--extra-docker-compose`.
