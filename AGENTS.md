Switchyard is an LLM router library. It sits between an agent's request (e.g. Claude Code, Codex CLI) and an inference server, selecting the best model for that request.

It is written in Rust with Python bindings.

Core components in `crates/`. These are layered:
- `libsy`: The core library and routing algorithms. This is the heart of Switchyard. Main entrypoing is `Algorithm::run_stream` method.
- `libsy-llm-client`: HTTP client that makes requests for `libsy` algorithms, and drive `run_stream`. Main entry point is `run` function in `run.rs`.
- `switchyard-runner`: Parsing TOML configuration, uses `libsy-llm-client` to run until the algorithm resolves to the selected model. The entry point is `Runner` struct.
- `switchyard-server`: A thin HTTP demo server wrapped around `switchyard-runner`. Has a TOML config file.
- `switchyard-py`: Python bindings for `libsy` and `libsy-llm-client`.

Support components (also in `crates/`):
- `protocol`: Types shared between many components.
- `switchyard-translation`: Convert between various JSON inference formats: OpenAI Chat Completions, OpenAI Responses and Anthropic Messages. We convert to/from a vendor neutral independent representation (IR). All the core components with with this IR. All the core components with with this IR.

Integrations:
- `crates/switchyard-nemo-relay-plugin/`: Integrate with NeMo Relay.
- `examples/litellm/`: Integrate with LiteLLM.

Write for a high-school level in short, simple sentences. Avoid jargon, analogies and metaphors. Be direct.
