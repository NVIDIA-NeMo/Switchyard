#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Start Langfuse (:3000) + Grafana (:3001), add Opus 4.8 pricing and the Langfuse dashboard.
set -euo pipefail
cd "$(dirname "$0")"
python3 build_grafana_dashboard.py
LANGFUSE_VERSION="${LANGFUSE_VERSION:-v4.82.0}"
[ -f docker-compose.langfuse.yml ] || curl -fsSL -o docker-compose.langfuse.yml \
  "https://raw.githubusercontent.com/langfuse/langfuse/${LANGFUSE_VERSION}/docker-compose.yml"
[ -f .env ] || cp .env.example .env
set -a; source .env; set +a
export LANGFUSE_IMAGE_TAG="${LANGFUSE_IMAGE_TAG:-${LANGFUSE_VERSION#v}}"
AUTH="$LANGFUSE_INIT_PROJECT_PUBLIC_KEY:$LANGFUSE_INIT_PROJECT_SECRET_KEY"
DC="docker compose -p sy-observability -f docker-compose.langfuse.yml -f docker-compose.grafana.yml"
$DC up -d
echo "waiting for Langfuse..."; until curl -sf -o /dev/null localhost:3000/api/public/health; do sleep 3; done
sleep 5
# Bedrock model IDs are not in Langfuse's built-in price table (USD per token; Anthropic list price for Opus 4.8).
curl -s -u "$AUTH" -X POST localhost:3000/api/public/models -H 'Content-Type: application/json' -d '{
  "modelName":"bedrock-claude-opus-4-8","matchPattern":"(?i)^(.*/)?(anthropic[./-])?(bedrock-)?claude-opus-4-8.*$","unit":"TOKENS",
  "pricingTiers":[{"name":"Standard","isDefault":true,"priority":0,"conditions":[],
  "prices":{"input":0.000005,"output":0.000025,"input_cached_tokens":0.0000005,"input_cache_creation":0.00000625}}]}' >/dev/null || true
$DC exec -T postgres psql -U postgres -q -v ON_ERROR_STOP=1 < langfuse_dashboard/dashboard.sql
echo
echo "Langfuse: http://localhost:3000  (login settings in .env)"
echo "Grafana:  http://localhost:3001"
echo "Next: source ./switchyard_otel_env.sh before starting Switchyard and the agent, then: python3 forecast_cost_enricher.py"
