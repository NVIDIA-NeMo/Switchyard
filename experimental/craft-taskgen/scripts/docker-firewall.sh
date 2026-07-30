#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Firewall rules for Harbor Docker containers — craft-bench variant.
# Allowlists only the hosts needed to run the inference + agent install paths
# the v2b cohort uses. Blocks all other outbound traffic from containers.
#
# Forked from harbor-datasets' `tools/docker-firewall.sh` to extend the
# allowlist for npm registry (Alpine claude-code install). The original
# is otherwise unchanged.
#
# Usage:
#   sudo bash docker-firewall.sh enable   # set rules
#   sudo bash docker-firewall.sh disable  # clear rules
#   sudo bash docker-firewall.sh status   # show current rules
#   sudo bash docker-firewall.sh test     # test from a container
#
# Must run AFTER current Harbor containers are stopped (affects all containers).
# Rules persist until cleared or system reboot.
#
# Agent compatibility under firewall:
#   - opencode:    binary pre-baked via OPENCODE_BINARY_PATH. KNOWN GAP:
#                  opencode's webfetch tool BYPASSES this iptables firewall
#                  (empirically observed during testing — webfetch returned
#                  the actual upstream content even with the destination
#                  host blocked at the network layer; the route is not yet
#                  diagnosed but appears to go through the inference gateway
#                  or some other allowlisted path). For a complete defense
#                  against opencode cheating, the webfetch tool itself
#                  needs to be disabled at the agent config level (see
#                  https://opencode.ai/docs/tools — `tools.webfetch=false`
#                  or `permission.webfetch=deny`). Mitigated for our v2b
#                  scan: 0 of 1116 opencode trials in the integrity scan
#                  attempted upstream-source fetches. Tracked as a
#                  follow-up; not blocking for v1.
#   - claude-code: works via curl https://claude.ai/install.sh which
#                  redirects to https://downloads.claude.ai (both allowlisted).
#                  Has its own WebFetch / WebSearch tools — their network
#                  routing under firewall is untested. Same potential gap
#                  as opencode webfetch if they go through the gateway.
#   - codex:       requires CODEX_BINARY_PATH (pre-baked tarball). Without
#                  it, codex's NVM bootstrap fetches from raw.githubusercontent.com
#                  which is not allowlisted (it is the dominant cheating
#                  vector — 15 of 17 incidents in the integrity scan).
#                  Build the tarball with scripts/build-codex-prebake.sh.
#                  Codex's standard fetch path is bash + curl/urllib in
#                  the trial container — this DOES go through iptables
#                  and is correctly blocked.

set -euo pipefail

# Cover the full Docker private bridge range, not just the default
# 172.17.0.0/16. Harbor's per-trial docker-compose creates user-defined
# bridge networks that Docker assigns from 172.18.0.0/16 onwards (we
# observed 172.30.x.0/24 in practice). Filtering only on 172.17.0.0/16
# left those trial networks unmatched and the DROP rule had no effect.
DOCKER_SUBNET="172.16.0.0/12"
DOCKER_HOST="172.17.0.1"

# The inference gateway is deployment-specific — derive its host from
# OPENAI_BASE_URL (e.g. https://gateway.example.com/v1) so no endpoint is
# hardcoded in the allowlist.
INFERENCE_BASE_URL="${OPENAI_BASE_URL:?OPENAI_BASE_URL required (inference gateway base URL)}"
INFERENCE_HOST="$(printf '%s' "$INFERENCE_BASE_URL" | sed -E 's#^[a-z]+://##; s#[:/].*$##')"

# Hosts the container needs to reach
ALLOWED_HOSTS=(
    "$INFERENCE_HOST"              # Inference gateway (codex / opencode / claude-code)
    "api.anthropic.com"            # Claude CLI may phone home
    "claude.ai"                    # claude-code installer entry (curl install.sh)
    "downloads.claude.ai"          # claude-code installer follow-up (binary download host)
    "statsigapi.net"               # Claude Code telemetry
    "registry.npmjs.org"           # npm install -g (Alpine claude-code)
)

# Local services on Docker host (none for craft-bench currently;
# kept commented for parity with the upstream script and easy reactivation)
ALLOWED_HOST_PORTS=()

resolve_ips() {
    local host=$1
    # Resolve from both host AND a Docker container (they may use different DNS)
    local host_ips=$(dig +short "$host" 2>/dev/null | grep -E '^[0-9]+\.' || true)
    # Pass the host via an env var (single-quoted -c, no shell interpolation) so
    # it can't break out of the Python string literal and inject code (CWE-78).
    local container_ips=$(docker run --rm -e RESOLVE_HOST="$host" python:3.12-slim python3 -c '
import os, socket
try:
    ips = set(r[4][0] for r in socket.getaddrinfo(os.environ["RESOLVE_HOST"], 443))
    print("\n".join(ips))
except Exception:
    pass
' 2>/dev/null || true)
    echo -e "${host_ips}\n${container_ips}" | grep -E '^[0-9]+\.' | sort -u
}

enable_firewall() {
    echo "Enabling Docker container firewall (craft-bench)..."

    # Clear existing custom rules
    iptables -F DOCKER-USER 2>/dev/null || true

    # Allow established connections (responses to outbound requests)
    iptables -A DOCKER-USER -m conntrack --ctstate ESTABLISHED,RELATED -j RETURN

    # Allow DNS (needed to resolve hostnames)
    iptables -A DOCKER-USER -s $DOCKER_SUBNET -p udp --dport 53 -j RETURN
    iptables -A DOCKER-USER -s $DOCKER_SUBNET -p tcp --dport 53 -j RETURN

    # Allow local services on Docker host (currently empty for craft-bench)
    for port in "${ALLOWED_HOST_PORTS[@]}"; do
        iptables -A DOCKER-USER -s $DOCKER_SUBNET -d $DOCKER_HOST -p tcp --dport "$port" -j RETURN
        echo "  Allowed: $DOCKER_HOST:$port"
    done

    # Allow specific external hosts
    for host in "${ALLOWED_HOSTS[@]}"; do
        ips=$(resolve_ips "$host")
        if [ -z "$ips" ]; then
            echo "  WARNING: could not resolve $host"
            continue
        fi
        for ip in $ips; do
            iptables -A DOCKER-USER -s $DOCKER_SUBNET -d "$ip" -j RETURN
            echo "  Allowed: $host ($ip)"
        done
    done

    # Drop everything else from Docker containers
    iptables -A DOCKER-USER -s $DOCKER_SUBNET -j DROP
    echo "  Blocked: all other outbound from $DOCKER_SUBNET"

    # Final rule: allow non-Docker traffic through
    iptables -A DOCKER-USER -j RETURN

    echo "Firewall enabled."
}

disable_firewall() {
    echo "Disabling Docker container firewall..."
    iptables -F DOCKER-USER 2>/dev/null || true
    iptables -A DOCKER-USER -j RETURN
    echo "Firewall disabled (all traffic allowed)."
}

show_status() {
    echo "Current DOCKER-USER rules:"
    iptables -L DOCKER-USER -n -v 2>/dev/null || echo "(no rules)"
}

run_test() {
    # Self-contained verification: enable firewall, run probes, disable.
    # If `enable` fails, we still try `disable` in the EXIT trap so we
    # never leave iptables in an inconsistent state.
    echo "Self-test: enabling firewall, probing from a container, then disabling."
    echo ""
    trap 'echo ""; echo "Cleanup: disabling firewall..."; iptables -F DOCKER-USER 2>/dev/null || true; iptables -A DOCKER-USER -j RETURN' EXIT
    enable_firewall
    echo ""
    echo "Testing from a Docker container..."
    echo ""

    # Test allowed: inference API. The URL goes through an env var (single-quoted
    # -c, no shell interpolation) so it cannot break out of the Python string
    # literal and inject code (CWE-78) — same pattern as resolve_ips above.
    echo -n "Inference API (https://$INFERENCE_HOST): "
    docker run --rm -e PROBE_URL="https://$INFERENCE_HOST/" python:3.12-slim python3 -c '
import os
from urllib.request import urlopen
try:
    r = urlopen(os.environ["PROBE_URL"], timeout=5)
    print(f"OK (status {r.status})")
except Exception as e:
    if "timed out" in str(e) or "Connection refused" in str(e):
        print(f"FAIL: {e}")
    else:
        print(f"OK (got error but reachable: {type(e).__name__})")
' 2>&1

    # Test allowed: npm registry (codex install needs this)
    echo -n "npm registry (https://registry.npmjs.org): "
    docker run --rm python:3.12-slim python3 -c "
from urllib.request import urlopen
try:
    r = urlopen('https://registry.npmjs.org/', timeout=5)
    print(f'OK (status {r.status})')
except Exception as e:
    if 'timed out' in str(e) or 'Connection refused' in str(e):
        print(f'FAIL: {e}')
    else:
        print(f'OK (reachable: {type(e).__name__})')
" 2>&1

    # Test blocked: raw.githubusercontent.com (cheating vector + breaks
    # codex install, which is why codex rerun support requires pre-bake)
    echo -n "raw.githubusercontent.com (should be BLOCKED): "
    docker run --rm python:3.12-slim python3 -c "
from urllib.request import urlopen
try:
    urlopen('https://raw.githubusercontent.com/', timeout=5)
    print('FAIL: raw.githubusercontent.com is reachable (should be blocked)')
except Exception as e:
    if 'timed out' in str(e) or 'Network is unreachable' in str(e) or 'Connection refused' in str(e):
        print(f'OK: blocked ({type(e).__name__})')
    else:
        print(f'MAYBE OK: {e}')
" 2>&1

    # Test blocked: github.com (cheating vector)
    echo -n "github.com (should be BLOCKED): "
    docker run --rm python:3.12-slim python3 -c "
from urllib.request import urlopen
try:
    urlopen('https://github.com/', timeout=5)
    print('FAIL: github is reachable (should be blocked)')
except Exception as e:
    if 'timed out' in str(e) or 'Network is unreachable' in str(e) or 'Connection refused' in str(e):
        print(f'OK: blocked ({type(e).__name__})')
    else:
        print(f'MAYBE OK: {e}')
" 2>&1

    # Test blocked: huggingface.co (cheating vector)
    echo -n "huggingface.co (should be BLOCKED): "
    docker run --rm python:3.12-slim python3 -c "
from urllib.request import urlopen
try:
    urlopen('https://huggingface.co/', timeout=5)
    print('FAIL: huggingface is reachable (should be blocked)')
except Exception as e:
    if 'timed out' in str(e) or 'Network is unreachable' in str(e) or 'Connection refused' in str(e):
        print(f'OK: blocked ({type(e).__name__})')
    else:
        print(f'MAYBE OK: {e}')
" 2>&1

    echo ""
    echo "Test complete."
    # EXIT trap will disable firewall.
}

case "${1:-}" in
    enable)  enable_firewall ;;
    disable) disable_firewall ;;
    status)  show_status ;;
    test)    run_test ;;
    *)
        echo "Usage: sudo bash $0 {enable|disable|status|test}"
        exit 1
        ;;
esac
