# Changelog

All notable changes to Switchyard are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0]

Switchyard 0.3.0 builds on the native server and Rust library introduced in
0.2.0. It adds reusable integration paths, Advisor Gate and Composite routing,
and a simpler Auto preset. It also updates the embedding API and removes the
legacy Python server and launchers. **This is a breaking upgrade for library
and Python CLI users.**

See the
[complete comparison](https://github.com/NVIDIA-NeMo/Switchyard/compare/v0.2.0...107f79853395b89eea26620cad7ca63a80b13ea0).

### Added

- **Benchmark reproduction paths** — DeepSWE v1.1 Harbor/Pier instructions,
  Stage Router and Advisor Gate profiles, and qualification settings document
  the workload, agent, budgets, scoring denominator, and provider differences.
  A NeMo Gym tutorial compares fixed-model and routed runs through LiteLLM.
  These are reproduction tools, not a claim that every configuration improves
  quality or cost. (#701, #680, #811, #812, #814)
- **Unmatched-request forwarding and input capabilities** — an optional
  `fallback_client` proxies unmatched requests; routes can declare image input
  support for client discovery and reject inputs disabled by their capabilities.
  (#547, #567, #750)
- **Experimental research tools** — checkpoint-backed Prefill Router gains
  batched inference and native route configuration. No supported checkpoint,
  exporter, or compatible encoder assets are supplied. CRAFT task generation
  is added under `experimental/`, separately from the routing runtime.
  (#539, #593, #616, #572)
- **`timeout_ms` on `[llm_clients.<name>]`** — one deadline covers all attempts,
  retry delays, and the complete response, including stream reads. Unset leaves
  the wait unbounded; `0` is rejected. A timeout returns `504` without trying another
  model, or a framed error if the final answer has already started streaming.
  Timed-out attempts are counted in metrics.
- **Per-target `reasoning_effort`** — a target can force the reasoning effort
  of every request it serves, replacing the caller's value (`reasoning.effort`
  on the Responses wire, `reasoning_effort` on Chat Completions), so a strong
  tier can run at `max` behind a client that sends `high`. `extra_body` only
  fills absent keys and could not do this. Rejected on Anthropic clients.
- **Raw Responses stream trace** — an opt-in trace of every upstream Responses
  event as received, under `RUST_LOG=switchyard_translation::responses::raw=trace`,
  for diagnosing provider-specific event shapes. (#646)
- **NeMo Relay native plugin** — a dynamically loaded integration that loads
  Switchyard's standard TOML deployment and executes its `switchyard-runner`-
  supported configured routes in process. Managed calls require NeMo Relay
  `>=0.8.0, <1.0.0`; unknown models use Relay's continuation unchanged.

- **NeMo Relay routing marks** — routing-model usage, measured routing
  overhead, and selected-model decisions are emitted as ATOF marks. The final
  serving call remains represented only by Relay's outer LLM lifecycle event to
  prevent double-counting.

- **Advisor-gate routing** — new `advisor` route type pairing the serving
  executor with a stronger judge-only advisor that reviews terminal turns:
  APPROVE releases the buffered turn, REDO discards it and feeds the advisor's
  plan back to the executor. Includes per-session review budgets scoped by
  `proxy_x_session_id`, stall checkpoints, a pattern trigger for text-protocol
  harnesses, middle-out transcript truncation, fail-open consults, and an
  `advisor_gate` block in `/v1/stats` covering verdicts, consult failures, and
  REDO-discarded turns.
- **switchyard-server container image** — a root `Dockerfile` builds the
  server container image, consolidating the benchmark Dockerfile into it.
  (#421)
- **Run-span task metadata** — `task_kind` and `agent_role` are recorded on
  the run span, so routing telemetry can be segmented by the semantic class of
  work; span fields only, no new metric labels. (#249)
- **Unified LLM-classifier bindings** — LLM-classifier routing is available
  through the native PyO3 bindings, unifying the Python-side surface. (#465)
- **Python `run_stream` aligned with Rust streaming contracts** — the surface
  mirrors `Step::CallModel` / `Step::Done(RoutingOutcome)`, adds matchable
  `LlmResponse.Agg` and `LlmResponse.Stream` variants while keeping normalized
  payloads as dictionaries, and preserves Rust response streams as Python
  async iterators without buffering. (#479)
- **Decision endpoint** — the server exposes a decision-only endpoint that
  resolves decisions from the deployment config, returns answers produced
  while routing, and rejects invalid routing outcomes with 500. (#456)
- **Release soak and routing benchmarks** — operations workflows add a release
  soak test plus routing performance reports with workload scenarios and
  realistic routing-overhead measurement. (#176)
- **Sub-agent routing decision gate** — a classifier decision gate routes
  sub-agent traffic via passthrough. (#492)
- **Subagent awareness across routing algorithms** — subagent awareness
  generalizes to all routing algorithms, with subagent UX improvements. (#505)
- **`switchyard-runner` crate** — pieces of `switchyard-server` are extracted
  into `switchyard-runner` for reuse by integrations, particularly the
  NeMo-Relay plugin. (#517)
- **Runtime-configurable fall-open tier** — a stage's fall-open tier can be
  set at runtime. (#518)
- **Transformers feature extraction for the prefill router** — the prefill
  router can extract features with Transformers models, with Qwen extraction
  parity validated. (#506)
- **Request preparation for routed targets** — a translation-layer helper
  prepares a normalized request for a routed target. (#455)
- **Safe route failure summaries** — `switchyard-runner` exposes a public,
  redaction-safe terminal-failure summary API covering route-execution
  failures before response delivery and typed failures yielded by active
  response streams. (#537)
- **Runner deployment from TOML source** — `Runner::from_toml(&str)` builds a
  configured runner from in-memory deployment text through the same
  version-1 parser and validation path as `Runner::load(path)`. (#545)
- **Composite routing** — an LLM classifier sets the Stage Router's fall-open
  tier. The shipped pairing is fixed; arbitrary nesting of algorithms is not
  supported. This was developed as hierarchical routing and renamed before
  release. (#533, #548)
- **Upstream response headers forwarded** — the LLM client records the
  upstream HTTP response headers on both the buffered and the streaming path,
  and `switchyard-server` replays an allowlisted subset to the downstream
  client: W3C tracing (`traceparent`, `tracestate`, `baggage`),
  `x-request-id`, Anthropic's `request-id`, `openai-processing-ms`, and the
  `anthropic-ratelimit-`, `x-ratelimit-`, and `x-upstream-` namespaces. Body
  description, hop-by-hop, cookie, and Switchyard-owned headers are never
  forwarded, and a header Switchyard writes itself always beats an upstream
  echo of the same name. (#571)

- **Per-target `system_prompt`** — a native target can configure
  `system_prompt`, so the prompt follows the model that actually answers the
  request instead of living only on Stage and Composite routes. (#464)
- **`auto` algorithm type** — a deployment provides just `capable` and
  `efficient` models and the recommended stage-router settings are used out
  of the box; `auto` can be repointed at new strategies or defaults in the
  future. (#654)
- **Codex freeform tools and the Responses-lite request shape** — the
  Responses codec understands the request shape Codex drives GPT-5 models
  with and its freeform (`custom`) tools, so a GPT-5.x Codex session routes
  natively and survives its first native-tool turn. (#648)
- **Custom tool semantics for the stage router** — route-scoped exact-name
  mappings classify custom tools as `observe`, `mutate`, or `plan` activity
  (neutral `new` stays unchanged), decoupling tool semantics from any one
  agent framework. (#606)
- **Restricted LLM router for Rust hosts** — `build_llm_router(ServerState)`
  exposes only the three inference endpoints (`/v1/chat/completions`,
  `/v1/messages`, `/v1/responses`), so a host can own authentication, model
  visibility, and operational routes on separate surfaces. (#586)
- **`RoutingOutcome.selected_model_ids`** — the selected model and its
  fallbacks merge into one ordered `Vec<ModelId>`, selected first, replacing
  `selected_model_id` and `fallback_models`. (#592)
- **Routing outcome metadata** — every successful outcome carries optional
  `OutcomeMetadata { outcome_id, algorithm, evidence }`, and the outcome ID
  is recorded on the matching `libsy.run` span. (#647)
- **Bounded outcome evidence** — built-in algorithms populate
  `RoutingOutcome.metadata.evidence` with bounded, structured JSON for the
  facts that determined a route, without leaking prompts, responses, raw
  errors, or arbitrary internal state. (#655)
- **Outcome metadata in OpenTelemetry and Python** — the `libsy.run` span
  records `outcome_id` and the ordered `selected_model_ids`, and Python's
  `RoutingOutcome.metadata` exposes the same identity and evidence. (#658)
- **Route attribution in the durable log** — durable routing-log records
  gain `route_id` and `algorithm`, so per-route usage stays recoverable when
  two configured routes serve the same backend model; older JSONL records
  remain readable. (#618)
- **Algorithms select a `Category`, not a specific model** — available
  models travel with each request in the `Driver`, algorithms choose a
  category such as `efficient`, and the category-to-model mapping lives in
  the `Driver`. (#630)
- **Request origin in routing logs** — the caller-supplied
  `x-switchyard-origin` header is recorded as optional `origin` in the
  durable routing JSONL for buffered answers, streamed answers, and judge
  records; missing, empty, or non-text values serialize as `null`. (#641)

### Changed

- **Python 3.10+ and a smaller package** — the core package has no declared
  Python runtime dependencies. Applications supply their own model client
  dependencies when embedding the library. Development and benchmark tools
  have separate requirements. (#391, #343, #705)
- **Host-driven routing contract** *(source-breaking)* — hosts drive
  `Step::CallModel` and `Step::Done`, supply runtime category-to-model mappings,
  and consume ordered `RoutingOutcome.selected_model_ids`. Python integrations
  should migrate to `switchyard.libsy`; old chain/profile imports are removed.
  Use the current Rust and Python examples rather than 0.2.0 signatures.
  (#340, #361, #479, #592, #630)
- **Metric vocabulary and attribution** — successful algorithm/call outcomes
  use `ok`; the legacy `routing_tier` dimension is removed. Review dashboards
  against the [OpenTelemetry reference](docs/reference/opentelemetry.md), which
  distinguishes routing decisions, logical calls, answer candidates, and HTTP
  attempts. (#554, #609, #773, #787)
- **HTTP client errors stop routing** — after the configured retries, the Rust
  runner stops the request instead of letting the routing algorithm choose a
  fallback. This also applies when `timeout_ms` is unset or an advisor has
  `fail_open = true`. The runner collects streams used during routing and preserves
  provider events for replay. Invalid judge verdicts keep their existing fallback.
- **`Algorithm::route` returns `Result<RoutingOutcome>`** — instead of the
  bare final `Result`, so callers observe the full routing outcome (see #458
  for the design). (#459)
- **`session_affinity` replaced by `classify_trigger`** — the routing config
  gains `classify_trigger = user_turn | new_session | every_request`, which
  re-runs the LLM classifier on the chosen trigger and otherwise reuses the
  previously routed model; `session_affinity` is removed from the config
  options (`classify_trigger = new_session` covers it). (#487)
- **LiteLLM integration replaced by a routing plugin** — the client
  integration becomes a routing plugin, and its example moves out of
  `experimental`. (#532)
- **`Response` gains a required `upstream_headers` field** *(source-breaking)*
  — `switchyard_protocol::Response` carries the upstream HTTP headers, so Rust
  callers that construct a `Response` with a struct literal must add
  `upstream_headers: Default::default()`. Field access and every other use are
  unaffected, and there is no wire or Python-surface change. (#571)

### Removed

- **Python coding-agent launcher CLI** — the `switchyard` command, its Claude
  Code, Codex, and OpenClaw wrappers, and the shared launcher runtime are
  removed. Connect clients directly to the standalone native server instead.
- **Deprecated Python server stack** — `switchyard serve`, YAML route bundles,
  the FastAPI endpoints and legacy chain, the `switchyard-components` crate,
  and their compatibility PyO3 bindings are removed. Use `switchyard-server`
  with native TOML deployments.
- **Packaging extras `[server]`, `[all]`, and `[cli]`** — dropped
  together with the deprecated Python server stack and launcher CLI. Install
  server functionality via the standalone `switchyard-server` binary instead.
  The legacy `[tracing]` and `[affinity-redis]` extras are also removed; the
  current package declares no optional dependency extras.

### Fixed

- **RC protocol and tool-loop compatibility** — preserve parallel tool-call
  limits, tool allowlists, native tool history, reasoning controls and
  signatures, URL citations, and terminal Responses output. Streaming fixes
  retain late response IDs, assemble fragmented tool names, and keep empty
  tool inputs. Unsupported cross-format native output fails explicitly instead
  of disappearing. (#729, #730, #771, #774, #777, #778, #779, #780, #782,
  #784, #785, #788)
- **Media re-encoding and caller identity** — keep supported image, audio,
  document, and tool-result payloads in valid provider wire shapes, including
  URL-versus-attachment handling and PDF data URLs. Stop mapping caller identity
  between provider-specific fields, which caused Hub requests to fail.
  (#755, #765, #783, #808, #810)
- **Cross-format Responses state** — materialize stored conversation history
  for Chat Completions and Anthropic targets, rather than forwarding unsupported
  Responses state handles. This retains transcripts in process memory; it is
  not stateless forwarding. See the
  [state and retention guidance](docs/routing_algorithms/llm_classifier_routing.md#responses-continuations-by-id).
  (#781, #809)
- **Routing and fallback correctness** — zero-weight Random targets are never
  fallback candidates; ambiguous same-model clients within a route are rejected;
  escalation handles first-event context overflow and preserves instruction
  anchors. Stage Router recognizes failed Anthropic tool results and Responses
  patch/shell outputs. Codex compaction stays on the parent route, while spawned
  threads use the sub-agent route. (#694, #700, #708, #722, #748, #754, #758, #789)
- **Credentials and request trust boundaries** — redact configured provider
  keys from server and Relay outputs, strip caller-controlled internal
  translation metadata, and prevent unintended credential/header forwarding.
  Relay rejects `forward_auth` because it cannot supply caller credentials.
  Redaction is per event, not a guarantee against reconstructing secrets split
  across stream events. (#690, #704, #720, #725, #738, #759)
- **Accurate failure and usage reporting** — failed or abandoned streams count
  as errors, fallback metrics identify each answer candidate, and translated
  cache-write and thinking-token usage is retained. Same-format decode failures
  produce stream errors rather than appearing successful. (#747, #769, #770,
  #772, #773)
- **Provider and integration compatibility** — retain query parameters in
  provider URLs, preserve Codex's built-in instructions during model discovery,
  repair LiteLLM routing after the library API migration, and preserve tool-free
  Responses requests with the pinned LiteLLM integration. (#693, #699, #757, #816)
- **Responses continuations stay with the provider that holds their state** — a
  request with `previous_response_id` or `conversation` could reach another
  provider and fail with `400 state_not_found`. When answer targets use different
  `[llm_clients]` entries, Switchyard records which model served each stored
  Responses ID and conversation ID. Known continuations return to that model
  without running the algorithm or trying another provider. Each route keeps up
  to 65,536 ID-to-model records per process without discarding older records.
  At capacity, it logs a warning and returns the completed reply without saving
  new state; later continuations from unsaved IDs may fail. Conflicting native
  Responses IDs return HTTP 409 with `response_state_conflict`, or a stream
  error after headers are sent. Records are lost on restart. Cross-format
  continuations also retain history, even when answer targets share one client.
- **Cross-format tool results keep image and file content** — an Anthropic
  `tool_result` carrying image or document blocks reached a Responses target
  as text only, and an image-only result became an empty `output`. In the other
  direction a Responses `function_call_output` whose `output` was an array of
  `input_text`, `input_image`, and `input_file` parts reached an Anthropic
  target as one JSON string. Both directions now carry typed text, image, and
  file blocks; plain-text results are unchanged. OpenAI `file_data` encodes as
  a valid Anthropic `document`: raw base64 with a `media_type`, the data-URI
  prefix stripped, and the file name as `title`.
- **Stored Responses tool continuations stay on the selected model** — a
  `function_call_output` sent with `previous_response_id`, where the matching
  `function_call` lives in provider state, was decoded as ordinary user text. A
  route with `classify_trigger = "user_turn"` then judged the turn again and
  could switch models mid tool loop, and a Responses upstream received a user
  message instead of the tool output. The output now stays a tool result, and
  a stored `custom_tool_call_output` keeps its type when re-encoded. An output
  with no matching call and no `previous_response_id` still degrades to
  readable user text.
- **Return HTTP 502 for failed Responses generations** — a provider's HTTP 200
  response with `status: "failed"` could appear as an empty successful answer.
  Switchyard now counts the call as an error and tries another model when the
  route supports fallback. If this failure reaches the application, Chat
  Completions, Anthropic Messages, and Responses return HTTP 502. Error replies
  preserve the provider's message and, for OpenAI applications, a nonempty
  string error code. Switchyard removes echoed caller credentials.
- **Chat tool-call index counts tool calls, not content blocks** — the OpenAI
  Chat stream encoder copied the source index into `tool_calls[].index`.
  Anthropic and Responses index the whole content array, so text ahead of the
  first tool call pushed the sole call to index 1, and the official OpenAI SDK,
  which subscripts its `tool_calls` array with that index, raised `IndexError`.
  Tool calls are now numbered within the Chat `tool_calls` array in order of
  first appearance.
- **Advisor stall checkpoint re-arms after a refunded review** — a
  stall-triggered consult that failed open or returned an unparseable verdict
  refunded the review budget but left the conversation's stall latch set, so
  every later eligible turn silently bypassed the advisor. The latch now clears
  whenever the reserved review is refunded or the budget is already spent.
- **Streamed Responses tool calls end with a tool-use stop reason** — the
  Responses stream decoder reported every `response.completed` as a plain
  completion, so a streamed `function_call` reached Anthropic clients as
  `stop_reason: "end_turn"` and Chat clients as `finish_reason: "stop"`.
  Stop-reason-driven tool loops, including the official Anthropic TypeScript
  SDK tool runner, then returned the unfinished tool-use turn without running
  the tool. The decoder now reports `tool_use` when the completed output holds
  a `function_call` or `custom_tool_call`, or when it already decoded tool
  deltas, matching the buffered decoder.
- **Harnesses without sub-agent identity on sub-agent routes** — Claude Code
  sends the child identity header (`x-claude-code-agent-id`) that sub-agent
  routing needs only from version 2.1.139. Older builds send just the session
  id, so their sub-agent requests silently routed through the parent route.
  Header normalization now flags such builds on
  `switchyard_protocol::Metadata` as `subagent_identity_unsupported`, a route
  with a `subagents` table logs one warning when it sees one, and the sub-agent
  routing guide states the version floor. The new public field means downstream
  code that builds `Metadata` with a full struct literal must add it or use
  `..Metadata::default()`.
- **Encrypted-only reasoning items open no summary part** — the Responses
  stream encoder opened a `reasoning_summary_part` for every reasoning item and
  closed it only when text had streamed, so an encrypted-only item left a part
  open with no `done`. The part now opens on the first text delta. (#671)
- **Responses reasoning through transforming routes** — reasoning that a route
  buffers or re-encodes now reaches the client in the standard `summary_text`
  shape with `reasoning_summary_*` events, encrypted-only and done-only
  reasoning items are decoded from every carrier a provider uses, and encrypted
  payloads are re-emitted under the provider's item id so the client's replay
  verifies upstream. Responses with several reasoning items keep all of them.
  (#646)
- **Unique, bounded Responses item ids** — synthesized output-item ids carry a
  per-response discriminator so replayed history no longer repeats `rs_0` and
  `fc_1` across turns, and upstream response ids longer than 40 characters are
  digested to stay within OpenAI's 64-character item-id limit. (#646)
- **Reasoning order in mixed stream chunks** — the OpenAI Chat stream decoder
  emits reasoning deltas before content deltas from the same chunk, so
  interleaved reasoning is no longer reordered. (#387)
- **Anthropic structured output in requests** — a schema arriving on
  `/v1/messages` now reaches the neutral request and the forwarded upstream
  body; unmappable output formats produce diagnostics instead of silent drops.
  (#462)
- **Responses tool arguments emitted once** — `output_item.done` repeats the
  complete function-call arguments the delta events already carried; the
  decoder suppresses the repeat when they match. (#469)
- **Content filter stops as Anthropic refusal** — `StopReason::ContentFilter`
  maps to Anthropic's `refusal` stop reason (and back) instead of `end_turn`,
  so moderation stops remain distinguishable. (#370)
- **JSON rejection statuses preserved** — request-body rejections keep the
  underlying status code instead of always returning 400. (#406)
- **Default log level** — logging defaults to `info` for all crates instead of
  discarding logs from crates without an explicit level; an unnecessary `rand`
  callback was removed on the way. (#471)
- **`router_retry_recovered` metric populated** — the counter now increments
  when a remote model call failed and needed retrying. (#474)
- **Data URI images translate to Anthropic base64 sources** — OpenAI-style
  inline images sent as `data:` URIs in `image_url.url` are encoded as
  Anthropic base64 image sources instead of being forwarded verbatim and
  rejected with "Only HTTPS URLs are supported." (#470)
- **JSON object key order preserved in proxied payloads** — `serde_json`'s
  `preserve_order` feature keeps keys in the order the client sent them;
  order is semantic for `response_format.json_schema` on order-enforcing
  structured-output backends (vLLM/xgrammar). (#439)
- **No fifth `cache_control` block** — `enable_anthropic_prompt_caching`
  counts existing breakpoints and abstains once the four-block Anthropic and
  Bedrock budget is spent, instead of failing upstream with HTTP 400. (#489)
- **Upstream error content redacted from judge warning logs** — judge warning
  logs no longer leak upstream error content. (#497)
- **Configured base URLs validated at load** — `base_url` parses into a
  validated type during `Deserialize`, so an invalid endpoint fails when the
  config loads. (#405)
- **Codex MCP namespaces preserved through translation** — Codex tool
  namespaces are carried in request extensions and survive translation,
  staying off the public API. (#384)
- **Provider extensions re-emitted in Responses encoding** — the Responses
  encoder mirrors the chat allowlist, so captured extensions such as
  `prompt_cache_key` survive any-source-to-Responses translation. (#509)
- **Routing instruction restated after windowed conversation** — libsy
  restates the routing instruction after a windowed conversation. (#520)
- **Responses instruction roles classified in the decoder** — inline system
  and developer input items route to `request.instructions` inside
  `decode_responses_input`, before reasoning and tool-call state-machine
  transitions, so an instruction item cannot flush pending reasoning or break
  tool-call grouping. (#523)
- **Incomplete upstream streams rejected** — upstream SSE that reaches EOF
  without a source-format terminal event is rejected: Anthropic requires
  `message_stop` (optional `[DONE]` stays compatible for OpenAI Chat and
  Responses), and a duplicate EOF error is no longer appended after a decoded
  in-band provider error. (#425)

- **Responses reasoning replayed as input history** — encrypted reasoning
  items survive decode/encode after exact replay is invalidated, re-emitting
  input history with `summary`/`encrypted_content` and omitting non-empty
  `content` arrays. (#645)
- **Buffered Responses input hardened for Codex upstreams** — rebuilt
  requests carry resolvable tool names for upstreams that need them, and
  parallel tool calls pair each result with its call instead of serializing
  as `call, call, result, result`. (#664)
- **Upstream URLs dropped from transport errors** — transport and timeout
  errors no longer echo the upstream request URL, whose query string can
  carry the key; the raw fallback proxy applies the same protection. (#649)
- **Upstream bodies redacted from the client-call span** — the
  `libsy.client_call` span keeps the target and HTTP status but no longer
  records the raw upstream response body. (#611)
- **Reasoning dropped from task classifier history** —
  `TaskInput::build_messages` removes reasoning blocks from the history
  handed to an LLM task classifier and drops messages left empty. (#610)
- **Escalation judge anchored to task framing; verdict reasons logged** —
  every user message that precedes the first assistant reply is anchored as
  task framing with a wider per-message budget, so a specification sent
  after Codex's environment block stays visible to the judge; verdicts keep
  their reason, logged under the escalation target. (#639)
- **OpenAI tool strictness preserved for Anthropic** — encoding an
  Anthropic Messages tool definition writes `strict` for both `true` and
  `false` and omits it when unset, so a strict tool no longer silently
  becomes a normal tool. (#600)
- **Anthropic tool strictness preserved** — decoding Anthropic Messages
  requests keeps top-level `tools[].strict`, so explicit strictness reaches
  OpenAI Chat and Responses targets instead of being replaced with an
  absent value. (#585)
- **Flat Responses `input_file` payloads decoded** — `decode_file_source`
  reads `file_data`/`filename` carried directly on the content block, not
  only nested under `file`, so flat payloads no longer fall through to
  `FileSource::Raw`. (#590)

### Limitations

- The standalone server remains a demo/evaluation component, not a production
  gateway. Release validation covers Ubuntu 24.04 on Linux x86_64; other
  platforms are outside that scope.
- The Relay integration requires `>=0.8.0, <1.0.0`. Relay 0.8.x and 0.9.0 have
  a documented native-plugin upstream-error propagation issue. Plugin bundles
  are published separately from Switchyard. See the
  [integration guide](docs/integrations/nemo_relay.md#upstream-error-compatibility).
- Prefill Router and the LiteLLM example remain experimental. Review the
  [routing overview](docs/routing_algorithms/overview.md) and
  [LiteLLM integration guide](examples/litellm/README.md) before deployment.

## [0.2.0]

Switchyard 0.2.0 introduces the native Rust server and libsy library path,
with explicit TOML deployments, provider-neutral routing algorithms, and
production-facing observability.

### Added

- **Standalone Rust server** — `switchyard-server` serves OpenAI Chat
  Completions, OpenAI Responses, and Anthropic Messages from one explicit TOML
  deployment. It includes TLS, graceful shutdown, upstream retries, token
  counting, health and model discovery, and optional durable session routing
  logs.
- **Rust library and protocol crates** — `switchyard-libsy` provides composable
  multi-LLM algorithms, `switchyard-protocol` owns the provider-neutral request
  and response contracts, `switchyard-translation` handles wire-format
  conversion, and `switchyard-llm-client` provides translated HTTP model calls.
- **Native routing algorithms** — weighted and reproducible random routing,
  capability, escalation, and custom-schema modes for LLM-classifier routing,
  multi-target policy selection, session affinity, context-window fallback, and
  signal-driven stage routing with handoff notes, per-target prompts, and an
  optional classifier fallback.
- **Python bindings for the native path** — `switchyard.libsy` runs Rust-owned
  algorithms with Python LLM clients, while `switchyard_rust.server.Server`
  hosts the Rust server in-process for the coding-agent launchers.
- **Native observability** — Prometheus metrics, GenAI OpenTelemetry spans,
  structured request logs, `/v1/stats`, `/v1/stats/reset`, and optional
  `/v1/routing/session-stats` expose request, routing, latency, token, cache,
  retry, and error data.
- **Evaluation and integration support** — native-server benchmark wiring,
  Terminal-Bench 2.1 dataset support, retry-adjusted task routing statistics,
  and an experimental LiteLLM stage-router integration.

### Changed

- **Native TOML is the primary deployment format** — LLM clients, targets, and
  routes are declared explicitly and validated by `switchyard-server`. The
  launcher path accepts the same TOML schema and includes a packaged OpenRouter
  deployment for zero-config startup.
- **Serving is built around libsy algorithms** — the native server and Python
  native-server binding construct algorithms directly instead of using the
  legacy profile and components-v2 serving stack. The Python YAML server keeps
  its existing profile APIs in this release.
- **Coding-agent launchers host the native Rust server** and use its routes,
  statistics, translation, and OpenTelemetry paths instead of constructing the
  legacy Python routing stack.
- **Cascade routing is now stage routing** — the `cascade` route and Python API
  names are replaced by `stage_router` and the native `StageRouter` algorithm.
- **The CLI is focused on serving and launching** — `switchyard serve` remains
  for Python routing-profile YAML bundles, while `switchyard launch` starts
  Claude Code, Codex CLI, or OpenClaw against a selected native route.
- **Python dependency compatibility is broader** — the supported OpenAI SDK
  floor moves from 2.34 to 2.7 while retaining the `<3.0` upper bound.
- **The Rust workspace uses Rust 1.96.1 and edition 2024.**

### Deprecated

- **The Python `switchyard serve` path** — the Python server, YAML route
  bundles, and profile APIs remain available in 0.2.0 for transition purposes
  but are deprecated. New deployments should use `switchyard-server`, native
  TOML configuration, and libsy algorithms.

### Fixed

- **Response `model` now names the model that actually served the request**, on
  every serving path and wire format. Streamed Anthropic and Responses replies,
  and every libsy-served reply, previously echoed the model id the client
  requested — for a route bundle whose key is an alias, that meant the alias
  rather than the routed target, so trajectories, dashboards, and client UIs
  labelled routed turns with the route name. The routed model was already
  reported by `x-model-router-selected-model`, `x-switchyard-selected-model`,
  `/v1/routing/stats`, and Intake's `served_model`; the response body now agrees
  with them. Streamed OpenAI Chat replies report the routed target instead of
  the provider's own id, and no longer fall back to `"unknown"` when a provider
  omits `model` on delta chunks.
- **Buffered Responses output is preserved** rather than dropping final answer
  items when translating a non-streaming response.
- **Cross-format response fidelity is improved** — Responses tool turns and
  reasoning items survive translation, raw stream events remain available, and
  Responses usage details and max-token truncation are represented correctly.
- **Known request fields are validated before translation**, so malformed
  OpenAI and Anthropic inputs return client errors instead of being silently
  coerced or omitted.
- **Anthropic interoperability is hardened** — Messages endpoints return
  Anthropic error envelopes, accept the `done` stream terminator, filter
  incompatible beta headers and OpenAI-only fields, and omit unsigned thinking
  blocks that Anthropic-compatible upstreams reject.
- **Prompt-cache usage survives format translation**, including cached and
  cache-creation token counts from OpenRouter and Anthropic-compatible
  providers; Anthropic prompt caching is enabled by default for translated
  calls.
- **Streaming stops after in-band upstream errors** instead of forwarding
  trailing events after the error.
- **Routing state and prompts remain coherent across turns** — target prompts
  and handoff notes survive same-format calls, classifier history keeps tool
  calls paired with their results, inactive session state is evicted, and
  context-overflow history is isolated by session and agent.
- **Native server model metadata is more reliable** — duplicate upstream model
  IDs produce a warning, `/v1/models` reports declared capabilities and Codex
  metadata, and streamed replies no longer fall back to an unknown model ID.

### Removed

- **Legacy Rust compatibility stacks** — the `switchyard-components-v2` and
  `switchyard-core` crates, the components-v2 profile macros, and the old PyO3
  profile and core bindings are removed. Native serving uses libsy; the Python
  profile APIs remain available in 0.2.0.
- **Legacy routing integrations** — plan-and-execute routing, RouteLLM, and the
  external OSS-router plugin path are removed. The `gpu` optional dependency
  extra is also gone with RouteLLM.
- **Latency-aware router** — the `latency_service` route type and its
  `LatencyServiceLLMBackend`, `LatencyServiceBackendConfig`,
  `LatencyServiceEndpoint`, and `LatencyServiceProfileConfig` public API are
  removed. It depended on NVIDIA Inference Hub's latency endpoint and schema.
  Deployments that need multi-endpoint, load- or latency-aware routing should
  move endpoint selection to a dedicated upstream load balancer.
- **Public `type: noop` and `type: passthrough` YAML routes** — removed from
  Python routing-profile bundles. Use an explicit `type: model` route for a
  direct target. Automatic catalog discovery from a bare `type: passthrough`
  route is also removed; list each model ID as its own `type: model` route.
- **Legacy Intake sink** — direct Intake request and response processors,
  launcher flags, and the `intake` optional dependency extra are removed. The
  native server exports telemetry through OpenTelemetry and OTLP instead.
- **Legacy CLI setup and diagnostics** — `switchyard configure`, `verify`, and
  `status`, the interactive setup and model-picker TUI, saved provider settings,
  and launcher smoke mode are removed when the CLI is narrowed to `serve` and
  `launch`. Name the credential environment variable with `api_key_env` in a
  native TOML deployment, export it, and pass the deployment to each
  `switchyard launch`. Validate a deployment with
  `switchyard-server --config <deployment.toml> --dry-run`.

### Known Issues

1. Buffered upstream work continues after the client disconnects, so a
   cancelled request can still incur provider cost.
2. Routing-tier attribution is missing from `GET /v1/stats` and `/metrics` for
   LLM-classifier judge failures that route to the default target, escalation
   decisions, and `stage_router` fallback decisions.
3. The retry recovery counter stays at zero after a successful upstream retry.
4. `x-switchyard-session-id` is not recorded in native session stats.
5. The native server does not send the documented `X-Switchyard-Version` header
   upstream.

## [0.1.0] — Initial release

First public release of Switchyard — a typed, composable control plane for LLM
traffic that sits between client applications and LLM backends.

### Added

- **Four-role chain** — `RequestProcessor → LLMBackend → ResponseProcessor →
  TranslationEngine`, executed by the Rust-backed core. See
  the [0.1.0 architecture](https://github.com/NVIDIA-NeMo/Switchyard/blob/v0.1.0/docs/architecture.md).
- **Protocol translation** — convert between OpenAI Chat Completions, Anthropic
  Messages, and OpenAI Responses wire formats, so each client keeps speaking its
  native API regardless of the upstream backend.
- **YAML route bundles** (`switchyard serve --routing-profiles`) — one bundle,
  many named routes, each its own chain. Supported route `type`s: `model`,
  `passthrough`, `random_routing`, `cascade`, `deterministic`
  (LLM-as-classifier), `latency_service`, and `noop`.
- **Routing strategies** — weighted random split, signal-driven **cascade**
  escalation (see the [0.1.0 cascade documentation](https://github.com/NVIDIA-NeMo/Switchyard/blob/v0.1.0/docs/routing_algorithms/cascade_routing.md)),
  LLM-as-classifier strong/weak routing, and latency-aware multi-endpoint
  failover.
- **One-command launchers** — `switchyard launch claude`, `launch codex`, and
  `launch openclaw` spin up a local proxy and drop you into the target CLI.
  All three **default to LLM-as-classifier routing** (validated coding-agent
  trio) with `--model` / `--routing-profiles` to opt out.
- **CLI** — `serve`, `launch`, `configure` (saved defaults, `--show`,
  `--list-models`), and `verify` / `launch --smoke` round-trip checks.
- **Observability** — Prometheus `/metrics`, a JSON `/v1/stats`
  (`/v1/routing/stats` alias), and per-request cost/token/latency stats. See
  [Metrics Reference](docs/internal/metrics_reference.md).
- **Python library** — `SwitchyardRecipes` (`passthrough_recipe`,
  `random_routing_recipe`, `cascade_recipe`, `deterministic_routing_recipe`,
  …) and typed `ChatRequest` / `ChatResponse` containers for in-process use.
- **Rust core** (PyO3) — chain execution, the latency-aware router, and the
  tool-result signal collector are implemented in Rust and re-exported to
  Python.
- **Packaging** — `pip install nemo-switchyard` with optional extras `[server]`,
  `[cli]`, `[gpu]`, `[all]`. See [Installation](INSTALLATION.md).

### Deprecated

- **`--plan-execute` launcher flag** — slated for removal; plan-execute will be
  configured through a `--routing-profiles` YAML bundle instead.

### Notes

- The `--deterministic` launcher flag was removed during pre-release
  development — LLM-as-classifier routing is now the implicit default for the
  `claude` / `codex` / `openclaw` launchers.
- Inference Hub integration docs are out of scope for this release.
