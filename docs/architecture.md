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

    clients <-->|"OpenAI and Anthropic API formats"| switchyard
    switchyard <-->|"Provider-compatible requests and responses"| backends
```

Clients connect using supported OpenAI or Anthropic API formats. Switchyard can
route a request to a backend with a different native format and still return the
response shape expected by the client.

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

Each LLM client in the deployment file sets `format`. It fixes the upstream
endpoint and wire format for every target that uses that client.

| `format` | Upstream endpoint |
|---|---|
| `openai_chat` | `/v1/chat/completions` |
| `openai_responses` | `/v1/responses` |
| `anthropic_messages` | `/v1/messages` |

`format` is required. Switchyard does not probe upstreams or select a format
automatically.

Switchyard decodes every inbound request into provider-neutral types before
routing, then encodes it for the selected target's format.
`switchyard-translation` converts requests, buffered responses, and streaming
events between these formats, so Claude Code, Codex, OpenClaw, and SDK clients
keep their native wire format regardless of the upstream a route selects.

## Related Documentation

- [Getting Started](getting_started.md): install Switchyard and run a first request
- [Routing Overview](routing_algorithms/overview.md): choose and configure a routing strategy
- [CLI Reference](cli_reference.md): configure and operate Switchyard from the command line
