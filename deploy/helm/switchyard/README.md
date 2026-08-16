<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Helm chart: `switchyard`

Minimal chart for deploying `switchyard-server` as an in-cluster LLM hub.

## Prerequisites

- Container image built from [`deploy/docker/Dockerfile`](../../docker/Dockerfile)
- Kubernetes secret holding the upstream API key

## Quick start

From the repository root:

```bash
docker build -f deploy/docker/Dockerfile -t switchyard-server:local .
kubectl create secret generic switchyard-api-key --from-literal=API_KEY=...
helm upgrade --install switchyard deploy/helm/switchyard \
  --set image.repository=switchyard-server \
  --set image.tag=local
# Cluster DNS `switchyard` is only reachable in-cluster; from a laptop:
kubectl port-forward svc/switchyard 4000:4000
curl -sS http://127.0.0.1:4000/health
```

If you use kind, load the image first:

```bash
kind load docker-image switchyard-server:local --name <cluster>
```

## Configuration

| Value | Default | Notes |
|-------|---------|-------|
| `image.repository` / `image.tag` | `switchyard-server` / `latest` | Point at your built image |
| `apiKeySecret` | `switchyard-api-key` / `API_KEY` | Must match `routesConfig` `api_key_env` |
| `routesConfig` | example passthrough TOML | Override with a real deployment |
| `networkPolicy.enabled` | `false` | See below |

Override `routesConfig` (or pass `-f`) with a real TOML file. See
[`deploy/docker/routes.passthrough.toml`](../../docker/routes.passthrough.toml)
and [`crates/switchyard-server/README.md`](../../../crates/switchyard-server/README.md).

## NetworkPolicy

Off by default so the chart works on clusters without NetworkPolicy support.
When enabled, set either:

- `callerPodSelectors` **and** `sandboxNamespaces`, or
- `extraIngressFrom`

Otherwise the template fails fast instead of rendering a deny-all policy
(Switchyard has no inbound authentication).
