#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Wipe all traces/spans/scores from the local Langfuse (keeps prices + dashboards).
# Restart forecast_cost_enricher.py afterwards so its running totals reset.
for t in events_full events_core scores traces observations observations_batch_staging; do
  curl -s -u clickhouse:clickhouse 'http://127.0.0.1:8123/' --data-binary "TRUNCATE TABLE $t"
done
echo "cleared"
