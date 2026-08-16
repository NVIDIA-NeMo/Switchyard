# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Docker packaging for switchyard-server.
#
# Build from the repository root (build context must be the workspace root so
# Cargo.toml / crates/ resolve):
#
#   docker build -f deploy/docker/Dockerfile -t switchyard-server:local .
#   docker run --rm -p 4000:4000 \
#     -e API_KEY \
#     -v "$PWD/deploy/docker/routes.passthrough.toml:/etc/switchyard/routes.toml:ro" \
#     switchyard-server:local
#
# Helm chart: ../helm/switchyard
