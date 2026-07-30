#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Hermetic soak rehearsal: run the soak tester against switchyard-server backed by VidaiMock
# (https://github.com/vidaiUK/VidaiMock), a mock LLM backend, with no live provider credentials
# or cost. Arguments are passed through to switchyard-soak; with none, a short rehearsal runs with
# the invalid-request canary disabled, because a permissive mock does not reject invalid input.
#
# Requires the vidaimock binary on PATH (or set VIDAIMOCK_BIN) and release builds of
# switchyard-server and switchyard-soak (cargo build --release -p switchyard-server -p switchyard-soak).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIDAIMOCK_BIN="${VIDAIMOCK_BIN:-vidaimock}"
SERVER_BIN="${SWITCHYARD_SERVER_BIN:-$REPO_ROOT/target/release/switchyard-server}"
SOAK_BIN="${SWITCHYARD_SOAK_BIN:-$REPO_ROOT/target/release/switchyard-soak}"
MOCK_PORT="${MOCK_PORT:-8100}"
SERVER_PORT="${SERVER_PORT:-4000}"
MOCK_LATENCY_MS="${MOCK_LATENCY_MS:-40}"

command -v "$VIDAIMOCK_BIN" >/dev/null 2>&1 || {
  echo "error: '$VIDAIMOCK_BIN' not found on PATH." >&2
  echo "Install VidaiMock from https://github.com/vidaiUK/VidaiMock/releases, or set VIDAIMOCK_BIN." >&2
  exit 1
}
for bin in "$SERVER_BIN" "$SOAK_BIN"; do
  [ -x "$bin" ] || {
    echo "error: '$bin' not found; run: cargo build --release -p switchyard-server -p switchyard-soak" >&2
    exit 1
  }
done

workdir="$(mktemp -d)"
mock_pid=""
server_pid=""
cleanup() {
  [ -n "$server_pid" ] && kill "$server_pid" 2>/dev/null || true
  [ -n "$mock_pid" ] && kill "$mock_pid" 2>/dev/null || true
  rm -rf "$workdir"
}
trap cleanup EXIT

wait_health() { # $1=url $2=name
  for _ in $(seq 1 60); do
    curl -sf "$1" >/dev/null 2>&1 && return 0
    sleep 0.25
  done
  echo "error: $2 did not become healthy at $1" >&2
  return 1
}

cat >"$workdir/mock.toml" <<TOML
schema_version = 1
[llm_clients.mock]
format = "openai_chat"
base_url = "http://127.0.0.1:${MOCK_PORT}/v1"
[targets.mock]
id = "gpt-4"
llm_client = "mock"
[routes.mock]
type = "passthrough"
id = "switchyard/mock"
target = "mock"
TOML

echo "Starting VidaiMock on :${MOCK_PORT} (latency ${MOCK_LATENCY_MS}ms)"
"$VIDAIMOCK_BIN" --port "$MOCK_PORT" --mode realistic --latency "$MOCK_LATENCY_MS" \
  >"$workdir/vidaimock.log" 2>&1 &
mock_pid=$!
disown 2>/dev/null || true # drop job control so cleanup does not print "Terminated"
wait_health "http://127.0.0.1:${MOCK_PORT}/health" "VidaiMock"

echo "Starting switchyard-server on :${SERVER_PORT}"
"$SERVER_BIN" --config "$workdir/mock.toml" --port "$SERVER_PORT" \
  >"$workdir/switchyard-server.log" 2>&1 &
server_pid=$!
disown 2>/dev/null || true
wait_health "http://127.0.0.1:${SERVER_PORT}/health" "switchyard-server"

if [ "$#" -gt 0 ]; then
  "$SOAK_BIN" --base-url "http://127.0.0.1:${SERVER_PORT}" --model switchyard/mock "$@"
else
  "$SOAK_BIN" --base-url "http://127.0.0.1:${SERVER_PORT}" --model switchyard/mock \
    --duration 60s --concurrency 8 --stream-ratio 0.5 --report-interval 10 \
    --invalid-canary-interval 0 --server-pid "$server_pid"
fi
