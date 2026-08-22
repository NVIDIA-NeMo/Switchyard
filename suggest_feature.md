# [feature] Cross-platform, declarative way to point *any* AI agent at `switchyard-server`

> Baseline: `origin/main` @ `053a61e` (post-`#501`). The proxy is now the standalone Rust binary `switchyard-server`; agents reach it over the wire by pointing their base-URL / model env at it.

## Problem

Every agent is currently wired to the server by hand. The operator must know each agent's base-URL and model environment variables or manually write per-agent configuration files (e.g., OpenCode, OpenClaw). This manual setup introduces several issues:

* Configuration errors and drift (missing `/v1` path segments, invalid model IDs, missing options like `ANTHROPIC_CUSTOM_MODEL_OPTION`).
* Cross-platform platform variances (POSIX `exec` vs. Windows process spawning, `PATHEXT` script/binary resolution).
* Resource leaks from leftover temporary configuration files when config-file-based agents terminate or encounter signal interrupts.

## Proposed solution

A cross-platform native Rust subcommand (`switchyard agent run`) that parses declarative agent specifications, renders environment variables and temporary configurations, and transparently manages agent execution.

1. **Agent spec (JSON / YAML / TOML)**
* `binary`: Executable name or path (resolved across platforms via `$PATH`).
* `env`: Environment variable map supporting `{base_url}`, `{base_url_v1}`, and `{model}` placeholders.
* `config_template` / `config_filename`: Optional temporary configuration rendering for config-file agents.
* Built-in specs shipped for standard agents (Claude Code, Codex, OpenClaw, OpenCode, Hermes, generic OpenAI/Anthropic SDKs).


2. **Cross-platform execution & lifecycle management**
* **Unix (Linux / macOS):** Uses `std::os::unix::process::CommandExt::exec` to replace the process image without overhead.
* **Windows:** Spawns a child process using `Command::spawn`, inherits standard I/O (`Stdio::inherit()`), and forwards child exit codes upon completion.
* **Binary resolution:** Uses OS-agnostic resolution handling executable extensions on Windows (`PATHEXT` for `.exe`, `.cmd`, `.bat`).
* **RAII cleanup:** Implements RAII wrappers combined with signal handlers (`SIGINT`, `SIGTERM`, `Ctrl+C`) to ensure temporary configuration files are purged on exit.



### User-facing example

```bash
# 1) Start the server
switchyard-server --config routes.toml --port 4000 &

# 2) Run any agent via a single command across Linux, macOS, or Windows
switchyard agent run --agent claude --server http://127.0.0.1:4000 --route switchyard/classified
switchyard agent run --agent ./custom-agent.json --server http://127.0.0.1:4000 --route switchyard/general

```

Custom-agent spec (`custom-agent.json`):

```json
{
  "name": "mybot",
  "binary": "mybot",
  "env": {
    "MYBOT_BASE_URL": "{base_url_v1}",
    "MYBOT_MODEL": "{model}"
  },
  "interactive_default": ["chat"]
}

```

## Verification

* **Cross-platform execution:** Verify target agents correctly resolve base URLs and reach `GET /v1/models` on Linux, macOS, and Windows.
* **Cleanup verification:** Confirm temporary config files are removed upon standard exit as well as process interrupts (`SIGINT` / `Ctrl+C`).
* **CI:** Clean `cargo check`, `cargo test`, and multi-target compilation checks.
