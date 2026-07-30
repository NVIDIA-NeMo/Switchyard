#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build a codex pre-bake tarball.
#
# This produces a tarball that, when extracted into the trial container's
# `/`, places a working node runtime + @openai/codex package at
# /usr/local/bin/{node,codex} and /usr/local/lib/node_modules/@openai/codex.
#
# The trial container's patched install-codex.sh.j2 extracts the tarball
# instead of bootstrapping NVM from raw.githubusercontent.com. With
# this in place, codex installs cleanly under firewall.
#
# Pinned versions:
#   - codex: $CODEX_VERSION (default 0.121.0; matches src/craft_taskgen/adapters/_docker.py)
#   - node:  controlled by NODE_TAG (default "22-slim", a Debian glibc image)
#
# Usage:
#   bash scripts/build-codex-prebake.sh [/path/to/output.tar.gz]
#
# Then on the host running rerun-tainted.sh:
#   export CODEX_BINARY_PATH=/path/to/output.tar.gz
#   sudo -E ./scripts/rerun-tainted.sh ...
#
# Requires Docker on the host. Builds a temporary Debian image with codex
# globally installed via npm, copies node + codex out, packs them, and
# discards the builder image.

set -euo pipefail

OUT="${1:-/tmp/codex-prebake.tar.gz}"
CODEX_VERSION="${CODEX_VERSION:-0.121.0}"
NODE_TAG="${NODE_TAG:-22-slim}"

builder_tag="codex-prebake-builder:$$"

cleanup() {
    docker rmi "$builder_tag" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Building codex prebake (codex@${CODEX_VERSION} on node:${NODE_TAG})..."

# Build a one-off image that has codex installed globally via npm.
docker build -t "$builder_tag" -f - . >/dev/null <<EOF
FROM node:${NODE_TAG}
RUN npm install -g @openai/codex@${CODEX_VERSION}
EOF

# Tar the bits we need from the image and stream to the output file.
# We keep paths under usr/local/ so that extracting at / drops them in
# place. The trial container has python:3.12-slim as base which doesn't
# have node or any @openai/codex package, so there's no collision.
docker run --rm "$builder_tag" tar czf - -C / \
    usr/local/bin/node \
    usr/local/bin/codex \
    usr/local/lib/node_modules/@openai \
    > "$OUT"

size=$(du -h "$OUT" | awk '{print $1}')
echo "Built $OUT (${size})"
echo ""
echo "Set CODEX_BINARY_PATH=$OUT on the host before running rerun-tainted.sh."
echo "Bump CODEX_VERSION here if pyproject's pinned codex version changes."
