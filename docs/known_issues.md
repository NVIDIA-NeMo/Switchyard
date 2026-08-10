# Known Issues

## 0.2.0
1. Buffered upstream work continues after the client disconnects, so a cancelled request can still incur provider cost.
2. Routing-tier attribution is missing from `GET /v1/stats` and `/metrics` for LLM-classifier judge failures that route to the default target, escalation decisions, and `stage_router` fallback decisions.
3. The retry recovery counter stays at zero after a successful upstream retry.
4. `x-switchyard-session-id` is not recorded in native session stats.
5. The native server does not send the documented `X-Switchyard-Version` header upstream.

## 0.1.0
1. Completed Codex Responses tasks may record `0` token usage in `GET /v1/stats`.
2. Tool-bearing Codex requests may fail when Switchyard routes them to an upstream that accepts only a fixed set of tool names or schemas.
