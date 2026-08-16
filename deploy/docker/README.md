<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Docker packaging for `switchyard-server`

Build a production image from this repository so the binary matches the
checked-out commit (prefer this over `cargo install` for reproducible deploys).

## Build

Use the **repository root** as the Docker build context (Cargo workspace must
resolve `Cargo.toml` / `crates/`):

```bash
docker build -f deploy/docker/Dockerfile -t switchyard-server:local .
```

## Run

```bash
docker run --rm -p 4000:4000 \
  -e API_KEY \
  -v "$PWD/deploy/docker/routes.passthrough.toml:/etc/switchyard/routes.toml:ro" \
  switchyard-server:local
```

Replace `API_KEY` and the example routes file with your upstream. Smoke checks:

```bash
curl -sS http://127.0.0.1:4000/health
curl -sS http://127.0.0.1:4000/v1/models
```

## Files

| Path | Purpose |
|------|---------|
| `Dockerfile` | Multi-stage build of `switchyard-server` |
| `routes.passthrough.toml` | Minimal example routes for local smoke tests |

For in-cluster install, see [`../helm/switchyard`](../helm/switchyard).
