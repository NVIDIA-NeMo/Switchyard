<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Switchyard PreProc for Dynamo

Run Switchyard as a separate ExtProc service behind agentgateway 1.0.0. The gateway sends
an OpenAI Chat Completions request to PreProc before matching its HTTPRoute. Switchyard
chooses the model; the selected model's Dynamo EPP chooses the worker.

```mermaid
sequenceDiagram
    participant Client
    participant Gateway as agentgateway
    participant Proc as Switchyard PreProc
    participant EPP as Dynamo EPP
    participant Worker
    Client->>Gateway: Chat completion (model: auto)
    Gateway->>Proc: Request headers and body
    Proc-->>Gateway: Selected model header and updated body
    Gateway->>EPP: Route to selected InferencePool
    EPP-->>Gateway: Worker endpoint
    Gateway->>Worker: Forward request
    Worker-->>Gateway: JSON or SSE response
    Gateway->>Proc: Response headers and chunks
    Proc-->>Gateway: Pass through unchanged
    Gateway-->>Client: JSON or SSE response
```

This directory owns the service, tests and image build. The
[Dynamo deployment example](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/sglang/deploy/gaie/switchyard)
owns the Kubernetes manifests and model-pool bindings. It assumes the cluster, gateway
controller, model workers and EPPs are already installed.

## Build the image

From the Switchyard repository root, use Docker to build and publish an image to a registry
your cluster can pull from:

```bash
export PREPROC_IMAGE=registry.example.com/your-project/switchyard-preproc:example
docker build -f examples/dynamo-preproc/Dockerfile -t "$PREPROC_IMAGE" .
docker push "$PREPROC_IMAGE"
```

Set that image in Dynamo's Kustomization and follow its deployment instructions.
The image uses the workspace Switchyard crates and precompiled Envoy bindings;
it uses the repository's shared lockfile and does not require a Dynamo checkout or a protobuf compiler.

## Configure routing

The supplied [routes.toml](config/routes.toml) uses StageRouter to choose between the two
Qwen models. Send `model: "auto"`; add `X-Switchyard-Session-Id` to retain routing state.
Send `X-Switchyard-Session-Final: true` on the final request to release its session admission
slot after a successful routing decision. Otherwise, idle slots expire after one hour.

To change routing, edit the TOML in the Dynamo deployment example and reapply its
Kustomization. Keep model IDs aligned with the HTTPRoutes and InferencePools.
Policies must select a model without generating a response or rewriting the request.

This example supports text and tool history with one PreProc replica. Routing state resets
on restart. Dynamo load and cache probes are opt-in; the default StageRouter setup does not use them.

## Concurrency and timeouts

agentgateway 1.0.0 sends responses through PreProc and cannot disable those phases.
Each request holds a stream slot until its response finishes. Set `MAX_ACTIVE_STREAMS` to change
the default of 16 per replica. Preprocessing also has a separate limit of eight concurrent
requests; reaching either limit returns HTTP 503. Raising the stream limit increases memory
and response-forwarding work.

PreProc allows 120 seconds of response inactivity, resetting the deadline on each message.
It also releases stream slots if output forwarding stalls for 120 seconds.
The Dynamo example's HTTPRoute uses a separate 120-second request deadline: in agentgateway
1.0.0 it bounds the wait for upstream response headers, measured from request start, rather
than the duration of an active response stream.

## Optional cache and load probes

This requires the Dynamo EPP routing-probe implementation and Switchyard SDK changes
on `ayushag/signal-extraction-test`; it is not available in the published SDK.
Enable the EPP's internal probe listener on port 9004 and expose it only to PreProc.
The API contract in Dynamo
(`deploy/inference-gateway/ext-proc/routing-probe.md`) defines inputs, supported modes,
signal meanings and metrics.

Mount [probes.toml](config/probes.toml) and set `PROBE_CONFIG` to its path. Map each
concrete model ID to that model's EPP probe URL. `mode = "observe"` logs predictions
without changing SDK routing inputs. To use the observations, set `mode = "routing"`
and use [cache-aware.toml](config/cache-aware.toml) as `ROUTES_CONFIG`.

PreProc sends the original request with each candidate model ID and its cache namespace.
Each EPP uses its own tokenizer. Salted probes are currently unsupported and trigger
the same fallback as missing signals. Calls share a deadline and concurrency limit, have no
retries, and never reserve workers. The SDK receives fresh observations before making
one decision; the selected EPP then handles worker selection normally.

The `cache_aware` SDK policy minimizes
`(effective_prefill_tokens + active_prefill_tokens) * prefill_cost` among the route's
approved targets. Missing, stale or incomplete observations select the first target.
Choose targets that are all suitable for the task; this policy does not assess quality.
Costs need deployment-specific calibration before interpreting scores as performance.
StageRouter continues to work unchanged and does not consume these signals.

Predicted GPU, CPU-inclusive and disk-inclusive hit rates appear in PreProc logs with
model, candidate worker and request/session IDs. EPP logs actual reuse from response
usage with the same IDs and the selected worker. Predictions remain separate from
observations; missing signals are unknown. Request/session IDs never become metric labels.

On gateways with configurable ExtProc phases, use `FullDuplexStreamed` for the request
body. PreProc waits for the body before choosing the route. Response phases may be
skipped when the gateway supports it; EPP still needs response usage for observed reuse.
