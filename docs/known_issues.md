# Known Issues

## 0.2.0
1. `stage_router` drops a configured tier system prompt when the inbound request and the selected target both use `openai_chat`. The call succeeds and no warning is emitted.
2. Errors returned from `/v1/messages` use the OpenAI error envelope rather than Anthropic's `{"type": "error", ...}` shape, so Anthropic SDK clients cannot dispatch on `error.type`.
3. Session state is retained without a capacity limit or eviction, so memory grows with the number of sessions a process has served.
4. Buffered upstream work continues after the client disconnects, so a cancelled request can still incur provider cost.
5. Routing-tier attribution is missing from `GET /v1/stats` and `/metrics` for fail-open, escalation, and `stage_router` fallback decisions.
6. The retry recovery counter stays at zero after a successful upstream retry.
7. `x-switchyard-session-id` is not recorded in native session stats.
8. The native server does not send the documented `X-Switchyard-Version` header upstream.
9. LLM-classifier history trimming can separate a tool result from the tool call it belongs to when `recent_turn_window` is configured.

## 0.1.0
1. Completed Codex Responses tasks may record `0` token usage in `GET /v1/stats`.
2. Tool-bearing Codex requests may fail when Switchyard routes them to an upstream that accepts only a fixed set of tool names or schemas.
