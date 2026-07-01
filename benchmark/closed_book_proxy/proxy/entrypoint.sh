#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

STATE_DIR="${SWITCHYARD_PROXY_STATE_DIR:-/etc/closed-book-proxy}"
PUBLIC_DIR="${SWITCHYARD_PROXY_PUBLIC_DIR:-/etc/proxy-public}"
ALLOWLIST_PATH="${SWITCHYARD_PROXY_ALLOWLIST:-${PUBLIC_DIR}/allowed_domains.txt}"
STRIP_LOG_PATH="${SWITCHYARD_PROXY_STRIP_LOG:-${PUBLIC_DIR}/strip.jsonl}"
REQUEST_MAP_PATH="${SWITCHYARD_PROXY_REQUEST_MAP:-${PUBLIC_DIR}/request_map.jsonl}"
VERIFIER_STATE_DIR="${STATE_DIR}/verifier"

mkdir -p \
    "${STATE_DIR}" \
    "${VERIFIER_STATE_DIR}" \
    "${PUBLIC_DIR}" \
    "$(dirname "${STRIP_LOG_PATH}")" \
    "$(dirname "${REQUEST_MAP_PATH}")"
touch "${STRIP_LOG_PATH}" "${REQUEST_MAP_PATH}"

python - "${ALLOWLIST_PATH}" <<'PY'
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

dest = Path(sys.argv[1])
base = Path("/opt/closed-book-proxy/allowlist-base.txt")
hosts: set[str] = set()

for raw in base.read_text().splitlines():
    value = raw.split("#", 1)[0].strip()
    if value:
        hosts.add(value.lower())

for env_name in (
    "OPENAI_BASE_URL",
    "ANTHROPIC_BASE_URL",
    "AZURE_OPENAI_ENDPOINT",
    "SWITCHYARD_BASE_URL",
):
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        continue
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    if parsed.hostname:
        hosts.add(parsed.hostname.lower())

for raw in os.environ.get("ALLOWED_HOSTS", "").replace(";", ",").split(","):
    value = raw.strip().lower()
    if not value:
        continue
    parsed = urlparse(value if "://" in value else f"https://{value}")
    hosts.add((parsed.hostname or value).lower())

dest.write_text("\n".join(sorted(hosts)) + "\n")
PY

(
    for _ in $(seq 1 100); do
        cert="${STATE_DIR}/mitmproxy-ca-cert.pem"
        if [[ -f "${cert}" ]]; then
            cp "${cert}" "${PUBLIC_DIR}/ca-cert.pem"
            chmod 0644 "${PUBLIC_DIR}/ca-cert.pem"
            exit 0
        fi
        sleep 0.1
    done
) &

export SWITCHYARD_PROXY_ALLOWLIST="${ALLOWLIST_PATH}"
export SWITCHYARD_PROXY_STRIP_LOG="${STRIP_LOG_PATH}"
export SWITCHYARD_PROXY_REQUEST_MAP="${REQUEST_MAP_PATH}"
export VERIFIER_PROXY_TOKEN="${VERIFIER_PROXY_TOKEN:-}"

if [[ -z "${VERIFIER_PROXY_TOKEN}" ]]; then
    VERIFIER_PROXY_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
    export VERIFIER_PROXY_TOKEN
    echo "VERIFIER_PROXY_TOKEN was not set; generated an unshared verifier proxy token." >&2
fi

mitmdump \
    --mode regular \
    --listen-host 0.0.0.0 \
    --listen-port 3129 \
    --set "confdir=${VERIFIER_STATE_DIR}" \
    --proxyauth "verifier:${VERIFIER_PROXY_TOKEN}" \
    --ignore-hosts ".*" &
verifier_proxy_pid=$!

mitmdump \
    --mode regular \
    --listen-host 0.0.0.0 \
    --listen-port 3128 \
    --set "confdir=${STATE_DIR}" \
    -s /opt/closed-book-proxy/rewriter.py &
agent_proxy_pid=$!

terminate() {
    kill "${verifier_proxy_pid}" "${agent_proxy_pid}" 2>/dev/null || true
    wait "${verifier_proxy_pid}" "${agent_proxy_pid}" 2>/dev/null || true
}

trap terminate INT TERM
wait -n "${verifier_proxy_pid}" "${agent_proxy_pid}"
status=$?
terminate
exit "${status}"
