# Agent-aware libsy proxy POC

This demo embeds `libsy`'s agent-aware model-pool router in the PR #17 Switchyard server. It accepts
OpenAI Chat, Anthropic Messages, and OpenAI Responses request shapes, normalizes Codex/Relay lineage
headers, asks the classifier for a model assignment, and reuses valid assignments per stable
`(session, agent, explicit task)` key.

Run it with an NVIDIA Inference Hub bearer token:

```bash
ANTHROPIC_API_KEY="$INFERENCE_HUB_SY_API_KEY" cargo run -p libsy-proxy
```

Send two buffered calls for the same Codex child thread. The first call classifies; the second reuses
the assignment:

```bash
curl http://127.0.0.1:4000/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'session-id: root-session-1' \
  -H 'thread-id: child-thread-1' \
  -H 'x-codex-parent-thread-id: root-thread-1' \
  -H 'x-openai-subagent: collab_spawn' \
  -H 'x-codex-turn-metadata: {"session_id":"root-session-1","thread_id":"child-thread-1","parent_thread_id":"root-thread-1","turn_id":"turn-1","subagent_kind":"collab_spawn"}' \
  -d '{"model":"libsy-agent-aware","messages":[{"role":"user","content":"Survey the request protocol and report evidence."}],"stream":false}'
```

The response exposes the chosen logical target and rationale in `x-model-router-*` headers. Explicit
`x-switchyard-session-id`, `x-switchyard-agent-id`, `x-switchyard-parent-agent-id`,
`x-switchyard-agent-role`, `x-switchyard-task-id`, and `x-switchyard-task-kind` headers override
harness-derived values.

The demo currently requires buffered upstream responses (`"stream": false`). Header normalization
and route selection work for Responses-shaped requests, but running an unmodified streaming Codex
CLI through this demo requires a follow-up adapter between Switchyard's SSE response and libsy's
neutral response stream.
