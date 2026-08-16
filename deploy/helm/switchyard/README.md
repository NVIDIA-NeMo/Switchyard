# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Helm chart for deploying switchyard-server as an in-cluster LLM hub.
#
# Prerequisites:
#   - Container image built from deploy/docker/Dockerfile
#   - Kubernetes secret with the upstream API key
#
# Quick start (from repository root):
#
#   docker build -f deploy/docker/Dockerfile -t switchyard-server:local .
#   kubectl create secret generic switchyard-api-key --from-literal=API_KEY=...
#   helm upgrade --install switchyard deploy/helm/switchyard \
#     --set image.repository=switchyard-server \
#     --set image.tag=local
#   curl http://switchyard:4000/health
#
# Override routesConfig (or use -f) with a real TOML deployment. See
# deploy/docker/routes.passthrough.toml and crates/switchyard-server/README.md.
#
# NetworkPolicy is off by default. Enable only when your CNI supports it and
# you set callerPodSelectors + sandboxNamespaces (or extraIngressFrom).
