# Switchyard Architecture

Switchyard is an LLM traffic proxy that sits between clients and model
backends. It keeps the client-facing API stable while applying routing policy,
translating provider formats, and handling configured fallbacks.

## System Context

```mermaid
flowchart TB
    clients["Clients<br/>Coding agents, SDKs, applications"]
    switchyard["Switchyard<br/>Local proxy, shared service, or embedded runtime"]
    backends["Model backends<br/>Hosted providers, private endpoints, local models"]

    clients <-->|"OpenAI, Anthropic, and Gemini API formats"| switchyard
    switchyard <-->|"Provider-compatible requests and responses"| backends
```

Clients connect using supported OpenAI, Anthropic, or Gemini API formats.
Switchyard can route a request to a backend with a different native format and
still return the response shape expected by the client.

## Request Lifecycle

```mermaid
flowchart TB
    request["1. Receive<br/>Accept the client API format"]
    normalize["2. Normalize<br/>Prepare a provider-independent request"]
    route["3. Route<br/>Apply policy and select a model endpoint"]
    execute["4. Execute<br/>Call the backend and apply configured fallback"]
    respond["5. Return<br/>Translate the response or stream for the client"]

    request --> normalize --> route --> execute --> respond
```

Routing policy determines which model or endpoint receives a request. Depending
on the selected strategy, that decision can use fixed weights, a classifier,
request signals, or conversation affinity. See the
[Routing Overview](routing_algorithms/overview.md) for the available strategies.

## Backend Wire Format

`BackendFormat` controls the upstream endpoint Switchyard calls. Explicit
formats select an endpoint directly and do not run capability probes.

| Format | Upstream behavior | Use when |
|---|---|---|
| `ANTHROPIC` | Always sends to `/v1/messages`. No probe. | You know the upstream is Anthropic-native (Anthropic API, NIM Claude routes). |
| `RESPONSES` | Always sends to `/v1/responses`. No probe. | You know the upstream supports the OpenAI Responses API. Fails on NIM / non-OpenAI upstreams. |
| `OPENAI` | Always sends to `/v1/chat/completions`. No probe. | You know the upstream is OpenAI-compatible (NIM, OpenRouter, etc). Safe universal choice. |
| `GEMINI` | Always sends to `/v1beta/models/{model}:generateContent`. No probe. | You know the upstream is Gemini-native (`generativelanguage.googleapis.com`). Bare `gemini*` model IDs also fast-path here under `AUTO`. |
| `AUTO` | Probes at startup, picks best format (see below). | Upstream is unknown or varies across deployments. Used by Claude Code and Codex launchers. OpenClaw is intentionally pinned to `OPENAI`. |
| *(omitted)* | Defaults to `OPENAI` — no probe, no fast-path. Silently uses Chat Completions, which is wrong for Anthropic/Bedrock models. Always set `format:` explicitly. |  |
Claude Code and Codex launchers use `AUTO` for their single-model targets.
OpenClaw is intentionally pinned to `OPENAI` for its equivalent target.

### AUTO Decision Tree

```mermaid
flowchart TB
    auto["BackendFormat.AUTO"]
    messages{"/v1/messages works?"}
    anthropic["ANTHROPIC<br/>/v1/messages"]
    responses{"/v1/responses works?"}
    responses_format["RESPONSES<br/>/v1/responses"]
    openai["OPENAI<br/>/v1/chat/completions fallback"]

    auto -->|"Probe /v1/messages"| messages
    messages -->|"Yes"| anthropic
    messages -->|"No: probe /v1/responses"| responses
    responses -->|"Yes"| responses_format
    responses -->|"No"| openai
```

Supported inbound and response formats are handled automatically.
`TranslationEngine` converts the client's request to the resolved backend
format and translates the backend response back to the client's expected
format. When a cross-format conversion is required, both directions decode to
and re-encode from the neutral conversation IR. This lets Claude Code, Codex,
OpenClaw, and SDK clients use their native wire format with any supported
upstream format.

> Prefer an explicit format for controlled deployments. It skips capability
> probes and makes the upstream contract clear. Use `AUTO` when provider
> capabilities are unknown or vary across deployments.
## Related Documentation

- [Getting Started](getting_started.md): install Switchyard and run a first request
- [Agent Launchers](guides/agent_launchers.md): run coding agents through a local proxy
- [Routing Overview](routing_algorithms/overview.md): choose and configure a routing strategy
- [CLI Reference](cli_reference.md): configure and operate Switchyard from the command line
