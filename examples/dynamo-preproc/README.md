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
on restart. The default StageRouter configuration does not use infrastructure signals.

## Task classification and queue-based routing

[task-classifier.toml](config/task-classifier.toml) classifies the task once per session.
Its optional `infra` table lets the larger model help when every small worker reaches
`min_waiting_requests`. The larger model must have no queued requests and remain below
`max_capable_inflight`, using the greater of running and locally tracked requests.
EPP still selects the worker. This rule does not estimate waiting time or cache reuse.

Mount the routing file and [metrics.toml](config/metrics.toml), then set:

```bash
ROUTES_CONFIG=/etc/switchyard/task-classifier.toml
METRICS_CONFIG=/etc/switchyard/metrics.toml
```

Replace the model IDs and URLs with your deployed services. The classifier's judge
endpoint must bypass PreProc to avoid a routing loop. Configure a direct `/metrics`
endpoint for every worker, with labels selecting one independently scheduled engine.
Keep the worker IDs consistent in both files and update them when workers change.

The sample uses SGLang: enable `DYN_SYSTEM_PORT=8081` and `--enable-metrics` on each
worker. It reads `sglang:num_running_reqs` and `sglang:num_queue_reqs`.
For vLLM, set `backend = "vllm"`; it reads `vllm:num_requests_running` and
`vllm:num_requests_waiting`. See [Dynamo's metrics documentation](https://docs.nvidia.com/dynamo/reference/observability/metrics-comparison).

Collection runs in the background. Missing, ambiguous, invalid or stale metrics
preserve the classifier's decision. Removing `METRICS_CONFIG` or the `infra` table
also leaves task routing unchanged. StageRouter does not use this adjustment.

Use one PreProc replica. The capacity check and local increment are atomic; counts
are released when responses finish or are cancelled. Metrics can lag other traffic,
and local counts do not reserve backend capacity. Logs record the routing reason.

## Concurrency and timeouts

agentgateway 1.0.0 sends responses through PreProc and cannot disable those phases.
Each request holds a stream slot until its response finishes. Set `MAX_ACTIVE_STREAMS` to change
the default of 16 per replica. Preprocessing also has a separate limit of eight concurrent
requests; reaching either limit returns HTTP 503. Raising the stream limit increases memory
and response-forwarding work.

Preprocessing has a five-second total deadline, including the classifier call.
The example gives the judge two seconds. Metrics collection is outside this deadline.

PreProc allows 120 seconds of response inactivity, resetting the deadline on each message.
It also releases stream slots if output forwarding stalls for 120 seconds.
The Dynamo example's HTTPRoute uses a separate 120-second request deadline: in agentgateway
1.0.0 it bounds the wait for upstream response headers, measured from request start, rather
than the duration of an active response stream.
