# Baseline smoke reproducibility

`scripts/run-baselines.sh` automatically picks a `reasoning_effort` value per
(agent, model) pair and passes everything the agent needs to actually honor it.
Without this wiring, reasoning is silently disabled across all three agents in
ways that are hard to notice — see [Production baseline evidence](#evidence).

## Claude Code version pin

Pinned to **2.1.118** in `src/craft_taskgen/adapters/_docker.py::CLAUDE_CODE_VERSION`
and `src/craft_taskgen/config.py::CC_VERSION`. Rationale:

The [Anthropic 2026-04-23 postmortem](https://www.anthropic.com/engineering/april-23-postmortem)
describes three regressions in Claude Code between 2026-03-04 and 2026-04-20:

1. **Default effort flipped medium** (2026-03-04 → 2026-04-07). Reverted in
   v2.1.94 (Pro/Max users in v2.1.101).
2. **`clear_thinking_20251015` caching bug** (2026-03-26 → 2026-04-10).
   Shipped in v2.1.85; fixed in v2.1.101. Caused "forgetfulness, repetition,
   and odd tool choices" in multi-turn trials.
3. **≤25 words verbosity prompt** (2026-04-16 → 2026-04-20). Added in
   v2.1.111 (with Opus 4.7 support); reverted in v2.1.116.
   Anthropic's own eval: "3% drop for both Opus 4.6 and 4.7".

Per-candidate version cleanliness (dates from GitHub tag commits):

| Version | Date | Issue 1 | Issue 2 | Issue 3 | Verdict |
|---|---|---|---|---|---|
| 2.1.87 | 2026-03-29 | medium default (we override) | **active** | predates | ❌ |
| 2.1.110 | 2026-04-15 | clean | clean | predates | ✅ |
| 2.1.111–2.1.115 | 2026-04-16 to 2026-04-18 | clean | clean | **active** | ❌ |
| 2.1.116 | 2026-04-20 | clean | clean | reverted | ✅ |
| **2.1.118** | 2026-04-23 | clean | clean | reverted | ✅ **pinned** |

Between 2.1.110 (last pre-Opus-4.7) and 2.1.118 (latest clean), 2.1.118
wins on pragmatic grounds: it's what `npm install @anthropic-ai/claude-code@latest`
will install for a reviewer, and Anthropic explicitly attests "all resolved
as of v2.1.116". Opus 4.7 code paths are present but we pin to Opus 4.6 models
for baselines, so the 4.7-specific branches don't fire on our runs.

An 18-trial 2.1.87 vs 2.1.110 version-compare sweep confirmed 2.1.87 is
contaminated by Issue 2 (caching bug) — the model's behavior on long
multi-turn trials matches the postmortem's description of the bug's effects.
Sweep data is in `baselines/sweep-cc-version-20260423-112647/`; small-N
scores are not used to justify the pin, the postmortem evidence is.

## Table

Source of truth: `src/craft_taskgen/baselines/reasoning_defaults.py`.

### codex

| Model | Effort | Source |
|---|---|---|
| `azure/openai/gpt-5.3-codex` | `high` | Harbor default; OpenAI suggests `medium` for day-to-day, we go higher for harder smoke tasks |
| `openai/openai/gpt-5.3-codex` | `high` | same |

Companion: `CODEX_MODEL_CATALOG_JSON=patches/codex-model-catalog.json` (auto-set by launcher).

### claude-code (Anthropic-route via `/v1/messages`)

Per [Anthropic's effort docs](https://platform.claude.com/docs/en/build-with-claude/effort), effort is supported on Opus 4.7, Opus 4.6, Sonnet 4.6, and Opus 4.5. Haiku is not effort-capable.

| Model | Effort | Source |
|---|---|---|
| `aws/anthropic/bedrock-claude-opus-4-7` | `high` | Anthropic recommends `xhigh` for coding, but harbor's CliFlag validator accepts only {low, medium, high}; capped here until we patch harbor |
| `aws/anthropic/bedrock-claude-opus-4-6` | `high` | API default |
| `azure/anthropic/claude-opus-4-6` | `high` | API default |
| `aws/anthropic/claude-opus-4-5` | `high` | API default (no specific recommendation) |
| `azure/anthropic/claude-opus-4-5` | `high` | API default |
| `aws/anthropic/bedrock-claude-sonnet-4-6` | `medium` | "Medium effort (recommended default)" for agentic coding |
| `azure/anthropic/claude-sonnet-4-6` | `medium` | same |

Companion: `ANTHROPIC_DEFAULT_{OPUS,SONNET}_MODEL_SUPPORTED_CAPABILITIES=effort,thinking,adaptive_thinking,interleaved_thinking` (auto-set by launcher — without this, effort is silently dropped on gateway-routed models).

### opencode (OpenAI-compat route via `/v1/chat/completions`)

Same effort intent as claude-code but routes through `@ai-sdk/openai-compatible` → NVIDIA gateway's `/v1/chat/completions`, which rejects `xhigh` per [CLIProxyAPI#2185](https://github.com/router-for-me/CLIProxyAPI/issues/2185). Opus-4-7 capped at `high` until that changes.

Opencode slugs are the **canonical gateway slug** — the same string claude-code and codex use. The opencode-internal `nvidia/` provider dispatch prefix is added by the launcher (see "opencode provider dispatch" below), not typed by the user.

| Model | Effort | Source |
|---|---|---|
| `aws/anthropic/bedrock-claude-opus-4-7` | `high` | Capped below `xhigh` (gateway limitation) |
| `aws/anthropic/bedrock-claude-opus-4-6` | `high` | API default |
| `azure/anthropic/claude-opus-4-6` | `high` | API default |
| `aws/anthropic/claude-opus-4-5` | `high` | API default |
| `azure/anthropic/claude-opus-4-5` | `high` | API default |
| `aws/anthropic/bedrock-claude-sonnet-4-6` | `medium` | Anthropic's sonnet-4-6 agentic recommendation |
| `azure/anthropic/claude-sonnet-4-6` | `medium` | same |
| `aws/anthropic/claude-haiku-4-5-v1` | `medium` | Haiku via chat-completions (gateway accepts effort; not applicable via claude-code's `/v1/messages`) |
| `azure/anthropic/claude-haiku-4-5` | `medium` | same |

Companion: `OPENCODE_REASONING_EFFORT` env → harbor opencode patch writes `reasoningEffort` on the nvidia provider's model entry.

#### opencode provider dispatch (`nvidia/` prefix)

Opencode's config format uses the first `/`-delimited segment of the model name as a **provider key** — it selects which JS client opencode's AI SDK instantiates (`@ai-sdk/openai-compatible` vs. stock `/v1/messages`). The remainder is the **model ID** that goes on the wire to the endpoint.

For our gateway-routed opencode runs we want provider=`nvidia` (the custom provider our harbor patch defines, which forces `@ai-sdk/openai-compatible`). So harbor's `--model` argument takes the form `nvidia/<canonical-slug>`, but the gateway only ever sees `<canonical-slug>` on the wire.

Consequences:

- **User-facing `--model`** is the canonical slug (e.g. `aws/anthropic/claude-haiku-4-5-v1`). Same string claude-code and codex use.
- **Launcher** prepends `nvidia/` automatically when `--agent opencode --backend gateway`. For legacy convenience, `--model nvidia/...` is also accepted (prefix stripped, then re-added).
- **Preflight** uses the canonical slug (what the gateway's policy check actually looks up).
- **`reasoning_defaults.py`** keys on the canonical slug.
- **Manifest** records `agent.model` (canonical) and `agent.harbor_model_arg` (dispatched form, opencode-only).

### pi (earendil-works/pi-coding-agent, OpenAI-compat via planted models.json)

Pi is a coding CLI from [earendil-works/pi](https://github.com/earendil-works/pi) (formerly `@mariozechner/pi-coding-agent`). Harbor 0.6.4 ships a `Pi` agent class, but it has no built-in path for arbitrary OpenAI-compatible endpoints — `--provider` only accepts hard-coded names (`openai`, `anthropic`, `google`, ...) and there's no `--base-url` flag. Our harbor patch (in `patches/harbor-agent-patches.diff`) adds a `nvidia` provider branch that, at run time, plants `~/.pi/agent/models.json` declaring `nvidia` as an `openai-completions` provider whose `baseUrl` is the host's `OPENAI_BASE_URL`. The launcher prepends `nvidia/` to `--model` as a dispatch token; pi.py strips it before sending the canonical slug on the wire.

The patch also pipes pi's JSON output through `grep -v '"type":"message_update"'` to drop pi's quadratic-bloat thinking-delta events (each emits the full accumulated thinking buffer on every token; without the filter, `pi.txt` grows ~1 GB/min on a reasoning trace). The parser only reads `message_end` events for token totals, so dropping `message_update` is lossless for our usage.

Pinned to **0.75.5** in `src/craft_taskgen/adapters/_docker.py::PI_VERSION`. Installed at container start via `npm install -g @earendil-works/pi-coding-agent@0.75.5`.

Reasoning effort: pi exposes `--thinking {off,minimal,low,medium,high,xhigh}`. Harbor surfaces it as the `thinking` agent-kwarg; the launcher renames `reasoning_effort` → `thinking` in its `pi` branch and forwards the value resolved from `reasoning_defaults.py`. Models with `compat.supportsReasoningEffort=true` in our planted `models.json` (set unconditionally for the `nvidia` provider) translate `thinking=high` to `reasoning_effort=high` in the chat-completions request body.

| Model | --thinking | Source |
|---|---|---|
| `nvidia/qwen/qwen3.6-35b-a3b` | `high` | Qwen3.x reasoning model |
| `nvidia/zai-org/glm-5.1` | `high` | GLM reasoning model |
| `nvidia/nvidia/nemotron-3-ultra-preview` | `high` | Nemotron-Ultra reasoning model |
| haiku-4-5 | — | No row → no `--thinking` (haiku is not reasoning-capable) |

### qwen-coder (Qwen Code CLI, OpenAI-compat route via `/v1/chat/completions`)

Qwen-coder is a fork of Gemini CLI maintained by QwenLM ([github.com/QwenLM/qwen-code](https://github.com/QwenLM/qwen-code)). Harbor 0.6.4 ships the `QwenCode` agent natively; no patch is needed. The CLI reads `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` and routes through the OpenAI Node SDK against `/v1/chat/completions`.

Pinned to **0.16.0** in `src/craft_taskgen/adapters/_docker.py::QWEN_CODE_VERSION`. Harbor's `install-qwen-code.sh.j2` runs at container start: `apt-get install curl` → `nvm install 22` → `npm install -g @qwen-code/qwen-code@0.16.0`. Adds ~30s of install overhead per task vs baked agents (no prebake tarball today).

| Model | Effort | Source |
|---|---|---|
| `nvidia/qwen/qwen3.6-35b-a3b` | — | No effort kwarg; Qwen3.x thinking is server-side and default-on at the gateway |

No reasoning_effort row in `reasoning_defaults.py` for qwen-coder — the launcher passes nothing. Output is also uncapped at the agent (qwen-code's `maxOutputTokens` setting in `~/.qwen/settings.json` is not plumbed by harbor's install template).

Pairs not in the table fall through — the launcher passes no
`reasoning_effort` and whatever default the agent ships with applies
(harbor-default `"high"` for codex; nothing for claude-code+haiku-4-5 since
haiku doesn't support effort at all).

## Adding a new model

1. Run the combo once in dry-run to confirm the launcher doesn't panic:
   `scripts/run-baselines.sh --agent <a> --model <m> --dry-run ...`
2. Add a row to `REASONING_EFFORT` in `src/craft_taskgen/baselines/reasoning_defaults.py`.
3. Run a single real trial and verify reasoning is active (recipes below).

## How opencode gets reasoning

Opencode doesn't have a native `reasoning_effort` concept — it's a model-agnostic TypeScript agent that speaks to whatever OpenAI-compatible endpoint it's pointed at. Our patched `nvidia` provider in `patches/harbor-agent-patches.diff` wires reasoning in for the two model families we care about.

### Claude models (via NVIDIA gateway `/v1/chat/completions`)

Chain:

1. Launcher sees `agent=opencode`, looks up the effort (e.g. `high` for opus-4-6), and host-`export`s `OPENCODE_REASONING_EFFORT=high` (NOT via `--agent-env`, because the harbor patch reads this on the host at config-generation time, not in the container).
2. Harbor's opencode agent (patched here) reads `OPENCODE_REASONING_EFFORT` when building the opencode.jsonc config and sets `provider_cfg["models"][model_id].setdefault("options", {})["reasoningEffort"] = "high"` in the JSON it writes into the container. (The `options` sub-dict matters: opencode's schema only honors `reasoningEffort` when nested under `models.<id>.options`. See `scripts/opencode_request_capture.py` + the "Empirically-verified opencode behaviors" section below.)
3. Opencode runs inside the container, reads its config, sees `reasoningEffort: "high"` on the model entry, passes it through `@ai-sdk/openai-compatible` as a provider option.
4. The SDK translates `reasoningEffort` into `reasoning_effort` in the `/v1/chat/completions` request body.
5. NVIDIA's gateway forwards `reasoning_effort` to the upstream Claude route. Gateway accepts `low|medium|high` (not `xhigh`).
6. Claude produces reasoning tokens; the gateway surfaces them back in the response's `reasoning` field per vLLM-style OpenAI-compat conventions.

Where this can break in practice:

- If the SDK drops the `reasoning` field from the response, opencode's trajectory won't show reasoning events even though the model reasoned. Reasoning still happened (and affected the answer), just invisibly to us.
- If the gateway rejects `xhigh`, the whole request 400s — we cap opus-4-7 at `high` for opencode to avoid this.

### Qwen3.5 (via local vLLM)

Different model family, different reasoning knob. Qwen3.5 doesn't use `reasoning_effort`; it uses:

- Server flag: `--reasoning-parser qwen3` (set at vLLM launch; we don't control this).
- Request: `chat_template_kwargs.enable_thinking=true` (default is `true` for Qwen3 series; can be disabled).
- Response: `message.reasoning` surfaces as a separate field.

Today the baseline launcher does NOT extend opencode's config with an `extra_body` for Qwen's `chat_template_kwargs` — that path isn't cleanly supported by `@ai-sdk/openai-compatible`. The opt-in env `OPENCODE_ENABLE_THINKING=1` is a placeholder for future wiring; right now Qwen reasoning happens (or doesn't) based entirely on server-side defaults.

**If 80-trial Qwen rollouts show zero `message.reasoning`**, the most likely cause is the vLLM server running without `--reasoning-parser qwen3`. Verify directly:

```bash
curl -sS "$VLLM_BASE_URL/chat/completions" \
  -H "Authorization: Bearer ${VLLM_API_KEY:-EMPTY}" \
  -H 'content-type: application/json' \
  -d '{"model": "<served-name>", "messages": [{"role":"user","content":"think step by step: 2+2"}], "max_tokens": 200}' \
  | jq '.choices[0].message.reasoning'
```

If this returns `null` or omits the field, the server isn't emitting reasoning. Fix is server-side.

## Override

Set `REASONING_EFFORT_OVERRIDE=low|medium|high` to bypass the table for one
run. Set it to empty string to disable pass-through entirely.

```bash
REASONING_EFFORT_OVERRIDE=low scripts/run-baselines.sh ...
REASONING_EFFORT_OVERRIDE= scripts/run-baselines.sh ...    # no effort sent
```

## Output-token cap

Every trial also runs under a global **64000-token** per-call output cap,
sourced from `src/craft_taskgen/baselines/output_cap.py`. This is a
disclosed safety ceiling for the paper, not a per-model tuning knob, so
there's no override env. The value comes from [Anthropic's effort
docs](https://platform.claude.com/docs/en/build-with-claude/effort)
recommending 64k as a starting point for Opus 4.7 xhigh — comfortably
above [Qwen3's 32768/38912 recommendation](https://huggingface.co/Qwen/Qwen3-32B).

Per-agent plumbing:

- **claude-code**: via `CLAUDE_CODE_MAX_OUTPUT_TOKENS` env
  ([documented](https://code.claude.com/docs/en/env-vars)).
- **opencode**: via `OPENCODE_BUILD_MAX_TOKENS` + `OPENCODE_PLAN_MAX_TOKENS`
  on the host; harbor writes them into `opencode.json`
  `agent.{build,plan}.max_tokens`.
- **codex**: no working knob today. `model_max_output_tokens` is parsed
  but never applied upstream
  ([openai/codex#4138](https://github.com/openai/codex/issues/4138)). Codex
  runs uncapped; the paper should footnote this gap.

## Turn cap

- **claude-code**: `max_turns=250` via `--agent-kwarg`. Pinned in
  `scripts/run-baselines.sh`; may change in future.
- **codex** and **opencode**: harbor does not expose a turn-cap knob
  for either. Wall-clock timeout (`agent.timeout_sec=3600` in
  `task.toml`) is the only cap. Document in the paper when reporting
  numbers for these agents.

## Plan-mode disable (claude-code only)

Claude-code exposes `EnterPlanMode` / `ExitPlanMode` as agent-invokable
tools. Under non-interactive harbor runs (`claude -p
--permission-mode=bypassPermissions`), plan mode is a dead-end:

- `ExitPlanMode` is documented as "present plan for user approval" —
  with no interactive user, it terminates the trial.
- `Write` / `Edit` are blocked for the duration of plan mode, so a
  trial trapped in plan mode produces 0 code edits regardless of
  agent intent.

Haiku 4.5 in particular self-selects into plan mode frequently. Repro
evidence:

| Config | Result |
|---|---|
| claude-code + Haiku 4.5 + plan mode **on** | 0 edits on 4+ trials |
| claude-code + Haiku 4.5 + plan mode **off** | 3–9 edits per trial in first 50 turns |

The launcher therefore defaults to
`--agent-kwarg disallowed_tools=EnterPlanMode,ExitPlanMode` for
claude-code runs — harbor's `claude_code.py` already exposes this as a
CliFlag, so no harbor patch is needed. Opencode and codex have no
equivalent tool and are unaffected.

- **Override** for ablation experiments: `DISABLE_PLAN_MODE=0
  scripts/run-baselines.sh --agent claude-code ...` — the launcher
  skips the `--disallowedTools` kwarg and records `null` in the
  manifest.
- **Manifest surfacing**: `agent.disallowed_tools` is either
  `"EnterPlanMode,ExitPlanMode"` (default) or `null` (opt-out /
  opencode / codex).

Mirrors the policy in `src/craft_taskgen/runner.py` (pipeline smoke
runner already disables these tools); the launcher was the remaining
gap.

## Sanity-check runs (`oracle`, `nop`)

The launcher accepts `--agent oracle` and `--agent nop` for dataset
sanity checks. Both reuse the full launcher infrastructure (preflight,
manifest, finalize, dataset digest) but bypass model/effort/cap
machinery — they don't talk to an LLM.

| Mode | What it does | Expected result |
|---|---|---|
| `oracle` | Applies `solution/solve.sh` + `solution/changes.patch`, runs verifier | **Pass** on every well-formed task |
| `nop` | Leaves the tree unchanged, runs verifier | **Fail** on every well-formed task |

A task that fails oracle is broken — its reference solution doesn't
satisfy its own verifier. A task that passes nop is broken — the
verifier trivially passes without any code change. Reviewers expect
both checks before any agent baseline is reported, and they are
cheap (no API calls; just docker + verifier).

The launcher derives the right `harbor run` invocation automatically:
no `--model`, no `--agent-kwarg version=...`, no LLM-endpoint env
plumbing, no reasoning effort, no output cap, no plan-mode disable.
Only the determinism envs (`PYTHONHASHSEED`, `LC_ALL`) flow through.

Recipe:

```bash
# Oracle sanity over the full dataset
scripts/run-baselines.sh \
    --agent oracle \
    --tasks-dir ~/projects/craft-bench/harbor-tasks/craft-taskgen-v1a \
    --output-dir baselines/sanity-oracle
jq '.stats.evals[]' baselines/sanity-oracle/*/result.json

# Nop sanity (same flag set, agent name swapped)
scripts/run-baselines.sh --agent nop ...
```

Reading the report: harbor reports `reward=1.0` for "verifier passed,"
`reward=0.0` for "verifier failed." For oracle, you want `reward=1.0`
on every task. For nop, you want `reward=0.0` on every task — but the
manifest/result.json still calls `reward=0.0` "trial failed." Sanity
runs invert that interpretation; `failed_tasks` from harbor's view =
"trivially-passing verifiers" from yours, and vice versa.

Manifest fields that collapse to `null` for sanity runs:
`agent.{version,model,harbor_model_arg,disallowed_tools}`,
`reasoning.{effort,source,notes}`, `output_cap.{applied}`,
`compaction.*`, `backend.base_url`. The dataset digest, harbor/node
versions, launcher_argv, determinism envs, and outcomes
(harbor_rc, finalize status) all behave identically to real
baselines.

## Context-window compaction

Each agent has distinct auto-compaction behavior that affects
trajectory reproducibility. Pin what we can; document what we can't.

- **claude-code**: `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=95` sent via
  `_add_env`. Default per
  [env-vars docs](https://code.claude.com/docs/en/env-vars) is ~95%;
  setting it explicitly makes the value visible in the manifest and
  removes any ambiguity about whether it was implicit. Values above
  default have no effect, so this is the closest-to-off Anthropic
  exposes.
- **opencode**: `compaction.auto = false` written into the generated
  `opencode.json` via the harbor patch
  (`patches/harbor-agent-patches.diff`). Opencode's exact numeric
  compaction threshold is not documented for 1.4.9 (open requests:
  [#8140](https://github.com/anomalyco/opencode/issues/8140),
  [#11314](https://github.com/anomalyco/opencode/issues/11314)) —
  disabling eliminates an unverified, potentially-drifting implicit
  variable. Override via `OPENCODE_DISABLE_AUTO_COMPACT=0` on the host.
- **codex**: `model_auto_compact_token_limit` and
  `model_context_window` are documented config keys
  ([OpenAI config reference](https://developers.openai.com/codex/config-reference))
  but harbor's `CliFlag` descriptor doesn't expose them. Pinning them
  would require a harbor patch extension similar to the
  `reasoning_effort` CliFlag; not done in this MR. Codex runs with
  implicit model defaults for both; the paper should footnote this
  alongside the `openai/codex#4138` `max_output_tokens` gap.

Manifest records these under `compaction.*` with `n/a` for agents the
current run doesn't apply to.

## Run manifest

Every `scripts/run-baselines.sh` invocation writes a
`run_manifest.json` into `<output_dir>/<job_name>/` alongside harbor's
own `result.json`. The pair is the reproduction package: manifest
records every knob the launcher resolved; `result.json` records what
happened.

Schema is versioned via `run.schema_version`; see
[`src/craft_taskgen/baselines/run_manifest.py`](../src/craft_taskgen/baselines/run_manifest.py)
for the module that produces it and
[`tests/test_run_manifest.py`](../tests/test_run_manifest.py) for the
contract.

Self-resolved fields (populated by the module regardless of caller):
- `run.timestamp`, `run.hostname`
- `run.craft_taskgen_sha` + `craft_taskgen_dirty` (tri-state) + `craft_taskgen_tree_kind` (grep-friendly enum: `git-clean` / `git-dirty` / `not-git`)
- `run.harbor_version`, `run.node_version`, `run.launcher_argv`

Task dataset provenance is captured via `harness.task_dir_digest` (a
content-hash over every file under `tasks_dir`) — that's independent
of whether the dataset happens to be versioned and is what matters for
reproduction.

Launcher-populated sections: `agent`, `backend`, `reasoning`,
`output_cap`, `harness`, `outcomes`, `determinism`. Every trial runs
with `PYTHONHASHSEED=0` and `LC_ALL=C.UTF-8` forwarded into the
container so in-container pytest is reproducible. The launcher invokes
the module via `python -m craft_taskgen.baselines.run_manifest` with
one flag per
field — to add a new manifest field, add a new argparse argument in
`_cli_main` and a new `--flag "$VALUE"` line in `run-baselines.sh`.
No JSON-in-bash heredocs.

The `harness` block includes `task_dir_digest` — a sha256 over every
file under `tasks_dir` that lets a reviewer confirm their copy of the
task dataset matches ours. Task enumeration, per-task timeouts, and
per-trial outcomes are harbor's responsibility — they live in
harbor's `result.json` (pointed at from `outcomes.harbor_result_json`)
and in each trial's `config.json`. The manifest deliberately does not
replicate that state.

**Outcomes section.** `outcomes.harbor_result_json` is the absolute
path to harbor's per-job `result.json`.
`outcomes.harbor_result_json_status` starts as `"predicted"` (the
manifest is written before harbor runs) and is flipped to `"present"`
or `"missing"` by a post-run finalize step that the launcher chains
after the `nohup`'d harbor process exits. `outcomes.harbor_rc` records
harbor's exit code. A reader who sees `"predicted"` in a finished run
knows the finalize step didn't execute and should verify existence
themselves.

**vLLM serving snapshot.** When `--backend vllm`, the launcher also
probes `<base_url>/v1/models` once and records the result in
`backend.vllm_snapshot`:

```json
"vllm_snapshot": {
  "served_model_name": "model",
  "served_model_root": "/lustre/.../MiniMax-M2.5",
  "max_model_len":     196608,
  "owned_by":          "vllm"
}
```

This covers what the server is willing to share — the hosted model
path (HF root) and context window. On probe failure (server down,
connection refused, etc.) the field is `null` so the manifest still
writes.

**What's deliberately NOT in the manifest**: vLLM server-side startup
config (GPU SKU, tensor-parallel degree, vLLM version, dtype,
quantization, chat-template revision, reasoning-parser name). These
are not launcher-observable via `/v1/models`. The operator standing
up the vLLM server is responsible for recording them separately
alongside the baselines output.

## Verification recipes

### codex — reasoning_output_tokens

```bash
for r in <trial>/agent/sessions/**/rollout-*.jsonl; do
  grep reasoning_output_tokens "$r" | grep -oE '[0-9]+$' | paste -sd+ | bc
done
```

Expect non-trivial numbers (10K–500K typical) per trial. Zero means the model
catalog isn't loaded; re-check `CODEX_MODEL_CATALOG_JSON`.

### claude-code — thinking blocks

```python
import json, glob
for t in glob.glob("<job>/craft-*/agent/claude-code.txt"):
    chars = 0
    for line in open(t):
        try:
            d = json.loads(line)
            c = (d.get('message') or {}).get('content') or []
            if isinstance(c, list):
                for b in c:
                    if isinstance(b, dict) and b.get('type') == 'thinking':
                        chars += len(b.get('thinking', ''))
        except: pass
    print(t, chars)
```

Expect median >2000 chars / >3 blocks per trial with the fix applied.

### opencode + claude — reasoning parts in the stream

```bash
grep -cE '"type":"reasoning"|"type":"thinking"' <trial>/agent/opencode.txt
```

Non-zero = reasoning is flowing from the gateway. Zero after this MR is
applied means `@ai-sdk/openai-compatible` isn't surfacing the field; report.

### opencode + Qwen3.5 (vLLM) — message.reasoning

Server must run with `--reasoning-parser qwen3` or an equivalent parser. Verify
with a direct probe:

```bash
curl -sS "$VLLM_BASE_URL/chat/completions" \
  -H "Authorization: Bearer ${VLLM_API_KEY:-EMPTY}" \
  -H 'content-type: application/json' \
  -d '{"model": "<served-name>", "messages": [{"role":"user","content":"think step by step: 2+2"}], "max_tokens": 200}' \
  | jq '.choices[0].message.reasoning'
```

If that returns `null` or the key is missing, the vLLM server isn't emitting
reasoning — client-side config can't fix that.

## Known issues / open items

- **opencode drops reasoning from the trajectory — upstream bug, not ours.**
  Proven end-to-end with two direct probes of the upstream API:
  1. **vLLM :9000, Qwen3**: 200 SSE frames with `delta.reasoning` populated
     (648 chars). `usage.completion_tokens_details` is absent, so
     `tokens.reasoning` is structurally 0 on this path regardless.
  2. **NVIDIA gateway, aws/anthropic/bedrock-claude-opus-4-6 with
     `reasoning_effort=high`**: 10 SSE frames with `delta.reasoning_content`
     populated (54 chars), non-stream response has `message.reasoning_content`,
     AND `usage.completion_tokens_details.reasoning_tokens = 32`. All three
     paths deliver reasoning.

  Corresponding opencode single-trial results: **0 `reasoning-*` part events
  in opencode.txt, `tokens.reasoning` = 0 across every step_finish** for
  both vLLM-Qwen and gateway-opus. 727,813 total tokens processed on the
  opus trial; 0 reasoning tokens surfaced. The upstream *provides* the
  data, the SDK *emits* the LanguageModelV2 reasoning parts (verified in
  `@ai-sdk/openai-compatible@2.0.41` source), and opencode's own stream
  consumer drops them on the floor.

  Tracked in [sst/opencode#16963](https://github.com/sst/opencode/issues/16963)
  (exact symptom match: vLLM+Qwen3) and [#19988](https://github.com/sst/opencode/issues/19988)
  (vLLM renamed `reasoning_content` → `reasoning`, opencode's zod schema
  doesn't list it). PR [#5531](https://github.com/sst/opencode/pull/5531)
  proposed a wrapper fix; maintainer declined, said it belongs upstream.
  Bumping opencode to 1.14.20 did not fix the symptom. **The reasoning
  still happens** and still affects the model's answers; we just can't
  observe it from the trajectory.
- **`high` on codex ≠ `high` on claude-code** in strict terms. The scale is
  calibrated per-model. Don't over-interpret cross-agent comparisons.

## Open gaps / follow-up MRs

- **vLLM + opencode provider dispatch is broken on this branch** (pre-existing,
  not an MR-46 regression). `--backend vllm --agent opencode` generates an
  `opencode.json` where the `provider` key takes the raw first-segment of
  `--model` (e.g. `qwen` for `qwen/Qwen3-32B`) instead of the `nvidia`
  provider our harbor patch defines with `@ai-sdk/openai-compatible`. Net:
  opencode runs without `baseURL`/`apiKey` configured and either fails at
  startup or routes nowhere. **Blocked on local vLLM stand-up for
  verification**; follow-up MR will add a `vllm/` dispatch prefix parallel
  to the existing `nvidia/` prefix, gate the patch's
  openai-compatible-attach block on `provider in ("nvidia", "vllm")`, and
  verify with a live trial.

## Empirically-verified opencode behaviors

- **`reasoningEffort` placement in `opencode.json`.** Using
  `scripts/opencode_request_capture.py` — a stdlib HTTP echo server —
  pointed opencode 1.4.9 at a local loopback, we captured the outbound
  request body for two config shapes:

  | Placement | Outbound `reasoning_effort` |
  |---|---|
  | `provider.<name>.models.<id>.reasoningEffort` | `null` (dropped) |
  | `provider.<name>.models.<id>.options.reasoningEffort` | `"high"` (forwarded) |

  The harbor patch places `reasoningEffort` under the `options` sub-dict.
  Re-run the probe if anyone reports reasoning-effort not taking effect:
  `python scripts/opencode_request_capture.py &` then exercise opencode
  against `http://localhost:8765/v1` and inspect
  `/tmp/opencode-request-capture.log`.

## Dry-run safety

`scripts/run-baselines.sh --dry-run` redacts any `_API_KEY=*` token in
the rendered harbor command. A raw `--agent-env
ANTHROPIC_API_KEY=sk-...` would leak if pasted into an MR / Slack / bug
report, which the docs actively recommend. The real launch path
(`nohup ${CMD[@]}`) is unaffected — the container receives the real
values.

## vLLM-hosted models via opencode (Qwen3, MiniMax, others)

We host Qwen3 and MiniMax variants on local vLLM and drive them with
opencode. The reasoning/cap story for these is different from gateway
Claude routes — document the caveats explicitly so the paper doesn't
overclaim.

**What's honored by these models:**

- `max_tokens` (our 64k `OUTPUT_TOKEN_CAP`) — yes, via
  `OPENCODE_BUILD_MAX_TOKENS` → opencode.json → request body. Standard
  OpenAI-compat field.
- Sampling params (`temperature`, `top_p`, `top_k`) — yes, via
  `OPENCODE_BUILD_{TEMPERATURE,TOP_P,TOP_K}` → opencode.json. For Qwen3
  the launcher defaults these to paper-recommended values
  (T=0.6, p=0.95, k=20) whenever the model slug contains `qwen`;
  override via the env vars above.

**What's NOT honored:**

- `reasoning_effort` — **no-op for Qwen3 and most vLLM builds**. The
  Claude gateway accepts `low|medium|high`, but Qwen3 keys off
  server-side `--reasoning-parser qwen3` plus `enable_thinking` in the
  chat template, not a string effort level. Our launcher still sends
  `reasoning_effort` via the SDK; vLLM silently ignores it on these
  models.
- `thinking_budget` / `chat_template_kwargs` — our opencode harbor
  patch doesn't wire `extra_body`, so we can't force a budget from the
  client. The server-side default applies. `OPENCODE_ENABLE_THINKING=1`
  is an env-var placeholder in the launcher today, not functionally
  wired — don't claim we set it.

**Per-model status:**

- **Qwen3-\***: sampling params auto-set to paper recommendations when
  model slug matches `*[Qq]wen*` (T=0.6, p=0.95, k=20 per
  [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-32B) "Best
  Practices"); output cap 64k; reasoning invisible in opencode.txt per
  sst/opencode#16963.
- **MiniMax-M2.5**: sampling params auto-set when model slug matches
  `*[Mm]ini[Mm]ax*` (T=1.0, p=0.95, k=40 per
  [MiniMax-M2.5 card](https://huggingface.co/MiniMaxAI/MiniMax-M2.5)
  "Inference Parameters"). Thinking is server-side only — MiniMax has
  no `enable_thinking` client toggle; depends on vLLM's
  `--reasoning-parser minimax_m2` / `minimax_m2_append_think`. The
  M2.5 parser has an open bug ([vllm#38212](https://github.com/vllm-project/vllm/issues/38212))
  where `<think>` tags can leak into `content`; workaround is the
  `deepseek_r1` parser. Before running any baseline, probe with
  `scripts/inference_endpoint_audit.py --base-url … --model …` and confirm
  `delta.reasoning` frames appear in the stream.
- **NVIDIA-Nemotron-3-Super-120B-A12B**: sampling params auto-set when
  model slug matches `*[Nn]emotron*` (T=1.0, p=0.95 per the
  [Nemotron-3-Super card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16)
  "Use `temperature=1.0` and `top_p=0.95` across all tasks and serving
  backends"). No `top_k` recommended. Thinking is controlled by
  `extra_body.chat_template_kwargs.enable_thinking` (+ optional
  `low_effort: True`), which our opencode harbor patch does not wire
  today — server-side default applies. vLLM needs `--reasoning-parser
  nemotron_v3` (from the repo's `super_v3_reasoning_parser.py`; not yet
  in upstream vLLM's built-in table). SSE emits `delta.reasoning_content`
  (DeepSeek-R1 convention), which is the exact field shape that
  [lemonade-sdk#1370](https://github.com/lemonade-sdk/lemonade/issues/1370)
  documents as broken in `@ai-sdk/openai-compatible` — same opencode
  reasoning-invisibility class as MiniMax.

**Probing a new vLLM endpoint:**

```bash
scripts/inference_endpoint_audit.py \
    --base-url http://localhost:9000/v1 \
    --api-key EMPTY \
    --model model \
    --reasoning-effort high   # optional; most vLLM models ignore it
```

Three checks in one invocation: non-stream `message.reasoning` /
`reasoning_content`, non-stream `usage.completion_tokens_details.reasoning_tokens`,
and streaming `delta.reasoning` frame counts. Use this before claiming a
new model produces reasoning — the SSE shape varies by server-side
parser and several models ship with broken parsers upstream.

**What to report in the paper's methodology section** for vLLM-hosted
runs: (a) output cap 64k, (b) sampling params we set, (c) the fact
that reasoning is server-side default with our vLLM launch flags,
(d) the opencode-drops-reasoning caveat. Do NOT report `reasoning_effort`
for vLLM runs — it's a no-op we happen to send.

## <a name="evidence"></a>Production baseline evidence (2026-04-22)

Pre-plan measurements across 240 production trials on
`<craft-bench>/jobs/baseline-*-v3/`:

| Agent × Model | Trials | Reasoning active? |
|---|---|---|
| claude-code × opus-4-6 | 80 | median ~200 chars thinking per trial (barely) |
| codex × gpt-5.3-codex | 80 | 0 reasoning tokens total (`reasoning_effort=high` was sent, model didn't recognize slug) |
| opencode × haiku-4-5 | 80 | 0 reasoning events |
| opencode × Qwen3.5-397B | 80 | 0 reasoning events |

With this MR's wiring, codex alone should jump from 0 → 100K+ tokens per
trial. Claude-code should jump from ~200 chars to a couple thousand. Opencode
+ claude should start producing reasoning events where there were none.
Opencode + Qwen depends on server config.

### Post-MR single-trial smoke results (2026-04-22, craft-taskgen-v1a)

Verified with `/tmp/smoke-verify/check_reasoning.py` against one trial each:

| Agent × Model | Reasoning signal | Delta vs pre-MR |
|---|---|---|
| codex × gpt-5.3-codex | 151,918 reasoning tokens, 51/51 records non-zero | 0 → 150K+ ✅ |
| claude-code × opus-4-6 | 2 thinking blocks, 1,004 chars | ~200 → 1,004 chars (5×) ✅ |
| opencode × opus-4-6 (high) | 0 reasoning events in 28 step_finish | same as pre-MR ❌ |
| opencode × haiku-4-5 (medium) | 0 reasoning events in 33 step_finish | same as pre-MR ❌ |
| opencode × Qwen3 (vLLM :9000) | 0 reasoning events in 44 step_finish | same as pre-MR ❌ |

Opencode's 0-events result is the documented sst/opencode#16963 upstream bug
(see "Known issues" above). Request wiring is correct — opencode.json carries
`reasoningEffort`, provider points at `@ai-sdk/openai-compatible`, SSE stream
has `delta.reasoning` frames (200 of them in a direct vLLM probe). The drop
is in opencode's stream consumer, not in our wiring.
