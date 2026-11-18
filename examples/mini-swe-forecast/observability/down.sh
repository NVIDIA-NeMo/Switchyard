#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Stop everything. Add -v to also delete all stored data.
cd "$(dirname "$0")" && docker compose -p sy-observability -f docker-compose.langfuse.yml -f docker-compose.grafana.yml down "$@"
