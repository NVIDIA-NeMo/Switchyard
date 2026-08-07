# Switchyard Helm chart

Deploys the standalone `switchyard-server` proxy on Kubernetes.

The chart renders the deployment TOML into a ConfigMap, supplies upstream API
keys from a Secret, and exposes a single ClusterIP port that serves the LLM
endpoints, `/health` and `/metrics` alike.

## Prerequisites

- Kubernetes 1.27 or newer
- Helm 3.8 or newer
- A container image built from the repository root `Dockerfile`
- Nodes with an AVX2-class x86_64 CPU or a Neoverse-N1-class arm64 CPU, because
  `.cargo/config.toml` compiles with `-C target-cpu=x86-64-v3` and
  `-C target-cpu=neoverse-n1` respectively

## Install

Build and publish the image:

```bash
docker build -t ghcr.io/nvidia-nemo/switchyard/switchyard-server:0.2.0 .
docker push ghcr.io/nvidia-nemo/switchyard/switchyard-server:0.2.0
```

Put the upstream key in a Secret, then install:

```bash
kubectl create namespace switchyard

kubectl -n switchyard create secret generic switchyard-keys \
  --from-literal=OPENROUTER_API_KEY="$OPENROUTER_API_KEY"

helm install switchyard deploy/helm/switchyard \
  --namespace switchyard \
  --set apiKeySecret.name=switchyard-keys
```

The Secret's keys become environment variables, so each name must match an
`api_key_env` in the deployment TOML.

## Configuration

`config.routes` holds the deployment TOML documented in
[`crates/switchyard-server/README.md`](../../../crates/switchyard-server/README.md).
Validate it before rolling it out — the server exits non-zero on an invalid
deployment, and a bad ConfigMap otherwise surfaces as a crash-looping pod:

```bash
switchyard-server --config routes.toml --dry-run
```

A multi-target routing deployment, supplied as a values file:

```yaml
# values.routing.yaml
image:
  repository: ghcr.io/nvidia-nemo/switchyard/switchyard-server
  tag: "0.2.0"

apiKeySecret:
  name: switchyard-keys

config:
  routes: |
    schema_version = 1

    [llm_clients.openrouter]
    format = "openai_chat"
    base_url = "https://openrouter.ai/api/v1"
    api_key_env = "OPENROUTER_API_KEY"
    max_retries = 2

    [targets.strong]
    id = "anthropic/claude-sonnet-4.5"
    llm_client = "openrouter"

    [targets.weak]
    id = "openai/gpt-4o-mini"
    llm_client = "openrouter"

    [routes.classified]
    id = "switchyard/classified"
    type = "llm_classifier"
    mode = "capability"
    classifier_target = "weak"
    strong_target = "strong"
    weak_target = "weak"
    base_threshold = 0.5
```

```bash
helm upgrade --install switchyard deploy/helm/switchyard \
  --namespace switchyard -f values.routing.yaml
```

Pods carry a `checksum/config` annotation, so editing `config.routes` rolls the
Deployment automatically.

To manage the TOML outside Helm, set `config.create=false` and
`config.existingConfigMap` to a ConfigMap whose `config.key` entry holds the
document.

## Values

| Key | Default | Description |
|---|---|---|
| `replicaCount` | `1` | Replicas, ignored when `autoscaling.enabled` |
| `image.repository` | `ghcr.io/nvidia-nemo/switchyard/switchyard-server` | Image repository |
| `image.tag` | `""` | Image tag; defaults to `.Chart.AppVersion` |
| `config.create` | `true` | Render `config.routes` into a ConfigMap |
| `config.existingConfigMap` | `""` | ConfigMap to use when `config.create` is false |
| `config.key` | `routes.toml` | ConfigMap key holding the TOML |
| `config.mountPath` | `/etc/switchyard` | Mount point for the TOML |
| `config.routes` | passthrough example | Deployment TOML |
| `apiKeySecret.create` | `false` | Create a Secret from `apiKeySecret.data` |
| `apiKeySecret.name` | `""` | Existing Secret loaded with `envFrom` |
| `apiKeySecret.data` | `{}` | Key/value pairs, read only when `create` is true |
| `env` / `envFrom` | `[]` | Additional environment |
| `extraArgs` | `[]` | Extra `switchyard-server` flags |
| `service.type` / `service.port` | `ClusterIP` / `4000` | Service exposure |
| `containerPort` | `4000` | Port the server binds |
| `resources` | 200m/128Mi → 2/1Gi | Requests and limits |
| `terminationGracePeriodSeconds` | `60` | Must exceed `--shutdown-timeout` |
| `routingLog.enabled` | `false` | Enable `--routing-log-file` and session stats |
| `tls.enabled` | `false` | Terminate TLS at Switchyard |
| `metrics.podAnnotations` | `true` | Prometheus scrape annotations |
| `metrics.serviceMonitor.enabled` | `false` | Create a ServiceMonitor |
| `podDisruptionBudget.enabled` | `false` | Create a PDB |
| `autoscaling.enabled` | `false` | Create an HPA |
| `extraObjects` | `[]` | Extra manifests, templated with `tpl` |

See [`values.yaml`](values.yaml) for the full set.

## Operational notes

**Graceful shutdown.** The server drains in-flight requests for
`--shutdown-timeout` (30s by default) on SIGTERM.
`terminationGracePeriodSeconds` defaults to 60 so streaming completions finish
rather than being cut off. Raise both together if your workload streams for
longer.

**Health semantics.** `/health` reports that the process is serving. It does
not check upstream reachability, so it stays healthy during a provider outage —
watch `switchyard_errors_total` and `switchyard_upstream_attempts_total` for
that.

**Session affinity.** `llm_classifier` routes with `session_affinity = true`
keep decisions in process memory, so a given session must reach the same
replica to benefit. With more than one replica, either front the Service with
session-aware routing or accept that affinity is per-replica.

**Read-only root.** The container runs as UID 65532 with a read-only root
filesystem; `/tmp` is an emptyDir because the image sets `HOME=/tmp`. Enabling
`routingLog` adds a writable volume at the log's parent directory.

## Envoy AI Gateway

To front Switchyard with Envoy AI Gateway, or to route Switchyard's upstream
traffic through it, see
[`examples/kubernetes/`](../../../examples/kubernetes/README.md).
