# Security Policy: craft-taskgen

## Reporting a Vulnerability

If you discover a potential security vulnerability, please **do not open a public issue.**

* Report via: [NVIDIA Vulnerability Disclosure Program](https://www.nvidia.com/en-us/security/) (preferred)
* E-Mail: [psirt@nvidia.com](mailto:psirt@nvidia.com)
  - We encourage you to use the following PGP key for secure email communication:
    [NVIDIA public PGP Key](https://www.nvidia.com/en-us/security/pgp-key)
* GitLab: Use the repository **Security** tab to submit a private vulnerability report
  directly on this repository

Please include the following information:
- Project name and branch/commit
- Type of vulnerability
- Step-by-step reproduction instructions
- Proof-of-concept code (if available)
- Impact assessment

**Detailed reports help NVIDIA evaluate and address issues faster.**

NVIDIA's PSIRT team will acknowledge receipt, validate severity, develop fixes,
and publish security bulletins as appropriate.

## Security Architecture & Context

`craft-taskgen` is an internal pipeline that mines merged GitHub pull requests and
converts them into [Harbor](https://harborframework.com)-compatible coding-agent
evaluation tasks (instruction, `task.toml`, `environment/Dockerfile`, `solution/`,
and `tests/` verifier). It uses the Claude Code CLI (`claude -p`, routed through the
NVIDIA LiteLLM gateway) to author and fix task artifacts, builds per-task Docker
images, runs agent baselines, and scores/deep-dives the results.

This software operates at the **Service / CLI Tooling** level (a Python package plus
shell launchers run by a pipeline operator). Its primary security responsibilities are:

1. **Protect the operator host and credentials** — the pipeline invokes `claude -p`
   **unattended with auto-approved tools** on the operator's machine, builds Docker
   images from mined-PR specs, and handles gateway API keys.
2. **Protect benchmark integrity** — keep reference solutions and verification logic
   out of reach of the agent under evaluation so baseline scores reflect real capability.

### Trust boundaries

- **Untrusted:** all content derived from mined GitHub PRs — repository names, commit
  subjects, git diffs, and candidate-JSON fields (`sha`, `repo`, `src_files`,
  `spec.install`, …) — and any artifact an LLM generates from them (instruction text,
  Dockerfile bodies, `solve.sh`). Also untrusted: the agents under evaluation during
  baseline runs, which are adversarial for benchmark-integrity purposes.
- **Semi-trusted:** candidate JSON emitted by the miner, the `.env` / `mcp_servers.json`
  the operator supplies, and datasets/models pulled from HuggingFace — not authored here
  and able to change without a commit to this repo.
- **Trusted:** the pipeline operator and their host, this repository's maintainers, and
  the NVIDIA LiteLLM gateway the pipeline routes all model traffic through.

### Key interfaces

- `src/craft_taskgen/claude_cli.py` + `gateway.py` — spawn `claude -p` (gateway-routed,
  `--permission-mode auto`, tools `Bash/Read/Write/Edit/Glob/Grep`) with a restricted
  subprocess environment (`build_gateway_env` → `_filtered_host_env`).
- `src/craft_taskgen/adapters/_docker.py` (`build_dockerfile`) — generates a task
  Dockerfile from a candidate spec.
- `src/craft_taskgen/adapters/planning/converter.py` — generates `solve.sh`.
- `src/craft_taskgen/steps.py` — pipeline steps; runs `git` over candidate SHAs and
  applies the PR diff (`changes.patch`) inside task containers.
- `scripts/run-baselines.sh`, `scripts/rerun-tainted.sh` — launch agent baseline runs.
- `scripts/docker-firewall.sh` — iptables egress control for agent containers (172.17.0.0/16).

### Threat Model

The following scenarios are the primary security concerns, derived from the codebase and
an Argus security audit of the repository. Ordered by severity.

1. **Unattended agent code execution on the host (CWE-94 / CWE-732):**
   The pipeline runs `claude -p --permission-mode auto` with `Bash`/`Write`/`Edit` over
   prompts containing untrusted PR-derived data (repo names, diffs, LLM-generated
   instructions). A prompt-injection payload could drive arbitrary command execution on
   the operator host. **Accepted risk** — unattended tool use is the pipeline's core
   mechanism. *Mitigated:* the `claude -p` subprocess now receives only a curated env
   allowlist (`_filtered_host_env`), so unrelated host secrets are not exposed to the
   agent's tools; inputs are a curated mined-PR candidate set, not arbitrary attacker PRs.
2. **Dockerfile RUN injection from candidate specs (CWE-94):**
   `build_dockerfile` interpolates candidate fields (`repo`, `parent_sha`, `install`,
   `pre_install`, `test_deps`) into `RUN` directives. *Mitigated:* `repo`/`parent_sha`
   are validated against strict character sets and the free-form command fields reject
   embedded newlines, preventing shell-command and Dockerfile-directive injection.
   *Residual, accepted:* the install/test commands are themselves attacker-influenced and
   run during the image build — inherent to building tasks from PR specs; the build is
   isolated in Docker.
3. **Shell command injection in generated `solve.sh` (CWE-78):** *Fixed* —
   `merge_sha`, `repo`, and file names are `shlex.quote`d before interpolation.
4. **Host environment leaked to the Claude subprocess (CWE-201 / CWE-522):** *Fixed* —
   `build_gateway_env` forwards only an allowlist instead of the full `os.environ`.
5. **Code/command injection in helper scripts (CWE-94 / CWE-78):** *Fixed* — `$MODEL`
   and paths are passed to inline Python via `argv` (not string interpolation), `.env`
   is parsed as `KEY=VALUE` instead of `source`d, and the firewall resolver receives
   `$host` via a container env var.
6. **Path traversal in Claude binary resolution (CWE-22):** *Fixed* — the
   `claude_code_version` from a TOML profile is validated as a bare version token before
   being used to build a filesystem path.
7. **Argument injection via candidate SHAs (CWE-88):** *Fixed* — `sha`/`base_sha`/
   `merge_base_sha` are validated (must start alphanumeric, safe ref charset) before use
   in `git` argv, blocking option injection.
8. **Supply-chain code execution via remote installers (CWE-494):** *Partially
   addressed* — the `uv` installer in the openhands_sdk patch now installs via
   `pip install uv==0.7.12` (integrity-verified by PyPI) instead of `curl | sh`. See the
   accepted-risk note below for the generator/template installs that remain.
9. **Container egress-filter bypass (CWE-923):** opencode's `webfetch` tool can bypass
   `docker-firewall.sh`. **Accepted** — a benchmark-integrity limitation, not host
   compromise.
10. **Untrusted PR diff executed in a container (CWE-494):** `solve.sh` applies the PR
    `changes.patch` via `git apply`/`patch` inside the task container. **Accepted** —
    reconstructing the oracle solution from the PR diff is the pipeline's purpose; it runs
    in a sandboxed container, not on the host.
11. **XSS in generated HTML (CWE-79):** *Dashboard fixed* — timeline `title` fields are
    HTML-escaped. The static site's `benchmark_comparison_note_html` (rendered via
    `innerHTML`) is **accepted** — it is intentional citation HTML from in-repo committed
    data, exploitable only with repo write access.

### Critical Security Assumptions

- **The pipeline runs on a trusted operator host.** `claude -p` runs unattended with
  `--permission-mode auto` (Bash/Write/Edit) **by design**; this is an accepted tradeoff
  so the pipeline can generate tasks without human approval at each step. It assumes the
  operator host is trusted and that candidate inputs come from the curated PR-mining
  process, not arbitrary adversarial PRs.
- **Candidate build/install commands are executed during image build.** `spec.install` /
  `pre_install` / `test_deps` are run as shell commands inside the task Docker build;
  validation prevents Dockerfile-directive injection, but the commands themselves are
  attacker-influenced — an **accepted risk** inherent to building tasks from PR specs,
  contained by Docker isolation.
- **PR diffs are applied inside sandboxed task containers**, never on the host.
- **Egress filtering depends on the bridge network.** `docker-firewall.sh` only
  constrains 172.17.0.0/16, and opencode `webfetch` can bypass it — an **accepted**
  benchmark-integrity limitation.
- **Some generated/template task Dockerfiles still use `curl | bash` installers.** The
  Argus audit flagged and this change fixed only the `uv` installer; the NodeSource and
  agent-CLI (`claude.ai/install.sh`) installs in `adapters/_docker.py`, `search/harbor.py`,
  and the `templates/` Dockerfiles were **not** flagged and remain. They assume the
  upstream installer hosts are uncompromised — a **residual supply-chain exposure**;
  migrate them to signed apt repos + `npm install` (per harbor-datasets) when revisited.
- **Upstream models and datasets are authentic.** HuggingFace / registry pulls assume the
  upstream hosts and content are uncompromised; there is no integrity verification here.
- **The agent under evaluation is adversarial; the verifier and references are not.**

---

_Generated from codebase analysis and an Argus audit of the repository. Review for
accuracy before relying on it, and keep the Threat Model in sync as the pipeline,
adapters, and `scripts/` change._
