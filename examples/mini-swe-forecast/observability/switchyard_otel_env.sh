#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Usage: source ./switchyard_otel_env.sh  then start switchyard-server as usual.
# Sends Switchyard's OpenTelemetry traces to the local Langfuse.
_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$_dir/.env" ] && { set -a; source "$_dir/.env"; set +a; } || { set -a; source "$_dir/.env.example"; set +a; }
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:3000/api/public/otel/v1/traces
export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_TRACES_HEADERS="Authorization=Basic $(printf '%s:%s' "$LANGFUSE_INIT_PROJECT_PUBLIC_KEY" "$LANGFUSE_INIT_PROJECT_SECRET_KEY" | base64)"
export OTEL_SERVICE_NAME=switchyard
echo "OTel export -> Langfuse configured"
