#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""This command prints Switchyard's observed session cost in a status line or a shell."""

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("SWITCHYARD_BASE_URL", "http://localhost:4000"))
    parser.add_argument("--session-id")
    args = parser.parse_args()
    try:
        session_id = args.session_id
        if not session_id:
            payload = json.load(sys.stdin)
            if not isinstance(payload, dict):
                raise ValueError("invalid status-line input")
            session_id = payload.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("missing session ID")
        url = args.base_url.rstrip("/").removesuffix("/v1") + "/v1/routing/session-stats?" + urllib.parse.urlencode({"session_id": session_id})
        with urllib.request.urlopen(url, timeout=2) as response:
            snapshot = json.load(response)
        cost = snapshot["cost"]
        if cost["total_usd"] is None:
            text = f"${cost['known_usd']:.6f} known + {cost['unknown_calls']} unpriced calls"
        else:
            label = "estimated" if cost["estimated_calls"] else "reported"
            text = f"${cost['total_usd']:.6f} {label}"
        print(f"Switchyard {text} (routing ${snapshot['routing_cost']['known_usd']:.6f} known)")
    except (OSError, ValueError, KeyError, TypeError, urllib.error.URLError):
        print("Switchyard cost unavailable")


if __name__ == "__main__":
    main()
