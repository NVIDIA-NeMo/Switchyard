Switchyard is an LLM router library. It sits between an agent's request (e.g. Claude Code, Codex CLI) and an inference server, selecting the best model for that request.

It is written in Rust with Python bindings.

Core components in `crates/`. These are layered:
- `libsy`: The core library and routing algorithms. This is the heart of Switchyard. Main entry point is `Algorithm::run_stream` method.
- `libsy-llm-client`: HTTP client that makes requests for `libsy` algorithms, and drives `run_stream`. Main entry point is `run` function in `run.rs`.
- `switchyard-runner`: Parsing TOML configuration, uses `libsy-llm-client` to run until the algorithm resolves to the selected model. The entry point is `Runner` struct.
- `switchyard-server`: A thin HTTP demo server wrapped around `switchyard-runner`. Has a TOML config file.
- `switchyard-py`: Python bindings for `libsy` and `libsy-llm-client`.

Support components (also in `crates/`):
- `protocol`: Types shared between many components.
- `switchyard-translation`: Convert between various JSON inference formats: OpenAI Chat Completions, OpenAI Responses and Anthropic Messages. We convert to/from a vendor neutral independent representation (IR). All the core components with with this IR.

Integrations:
- `crates/switchyard-nemo-relay-plugin/`: Integrate with NeMo Relay.
- `examples/litellm/`: Integrate with LiteLLM.

Write for a high-school level in short, simple sentences. Avoid jargon, analogies and metaphors. Be direct.

## Engineering guidance

- Prefer the smallest direct solution.
- Avoid abstractions, configurability, and defensive code for hypothetical needs.
- If the implementation grows unexpectedly large, reconsider and simplify it.
- For bugs, reproduce the failure before fixing it; for refactors, establish a behavioral baseline first.
- Make reasonable, reversible assumptions when consequences are small. Ask only when ambiguity would materially change the result, expand scope, or risk an irreversible acti

## Git guidance

1. Comments Explain Code, Not Project Management

Source comments are about the code. Tracking lives in the tracker.

- No issue/PLAN/step references in code (`TODO(step-6)`, "lands in step 4",
  "tracked as ISSUE-001", links to `docs/issues/`). These rot the moment the
  plan changes and leak project-management state into source.
- A plain `// TODO:` describing a concrete code gap is fine; a `// TODO`
  pointing at a tracker step is not.
- Comment what isn't obvious from the code: why a thing is done this way,
  invariants, non-obvious edge cases. Don't narrate what the code already says.
- Module doc comments should state what the module is *for*, not its build
  schedule or its "empty for now" status.

2. Commit Discipline

One step, one reviewed, one-line commit.

- One focused commit per step; every changed line traces to that step.
- Single-line commit message in Conventional Commits form
  (`type(scope): summary`). No body, no `Co-Authored-By` trailer.
- Pull request titles use the same Conventional Commits form.
- Use `git commit -s` so every commit carries the required DCO sign-off.
- Never commit unprompted. Show the diff, get approval, then commit.

