# Known Issues

## 0.4.0

- When the server routes a Responses request to a Chat target, an unknown
  `previous_response_id` can start a new conversation without an error.
  Send the full conversation history when the previous response is unavailable,
  for example after a server restart.
- A route with `reasoning = false` rejects Anthropic requests that explicitly
  set thinking to disabled. Omit the `thinking` field for text-only requests.
- The translation API supports one Chat completion choice. Requests that ask
  for more than one choice return an error. Supply single-choice responses
  when you translate provider output directly.

## 0.2.0
1. Buffered upstream work continues after the client disconnects, so a cancelled request can still incur provider cost.
2. Routing-tier attribution is missing from `GET /v1/stats` and `/metrics` for LLM-classifier judge failures that route to the default target, escalation decisions, and `stage_router` fallback decisions.
3. The retry recovery counter stays at zero after a successful upstream retry.
4. `x-switchyard-session-id` is not recorded in native session stats.

## 0.1.0
1. Completed Codex Responses tasks may record `0` token usage in `GET /v1/stats`.
2. Tool-bearing Codex requests may fail when Switchyard routes them to an upstream that accepts only a fixed set of tool names or schemas.
