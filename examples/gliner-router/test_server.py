# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("server.py")
SPEC = importlib.util.spec_from_file_location("gliner_router_server", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
SERVER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SERVER
SPEC.loader.exec_module(SERVER)


class GLiNERRouterTest(unittest.TestCase):
    def setUp(self):
        routes = {
            "routine": SERVER.Route("deterministic_tool", "exact work"),
            "authority": SERVER.Route("human_review", "human authority"),
        }
        self.config = SERVER.RouterConfig(routes, "human_review", 0.55)

    def test_low_confidence_uses_safe_fallback(self):
        engine = SERVER.RoutingEngine(self.config, lambda _text: ("routine", 0.54))
        self.assertEqual(
            engine.route("calculate", {"deterministic_tool", "human_review"}),
            ("human_review", 0.54),
        )

    def test_response_schema_must_allow_every_configured_target(self):
        engine = SERVER.RoutingEngine(self.config, lambda _text: ("routine", 0.9))
        with self.assertRaisesRegex(ValueError, "human_review"):
            engine.route("calculate", {"deterministic_tool"})

    def test_http_contract(self):
        engine = SERVER.RoutingEngine(self.config, lambda _text: ("routine", 0.9))
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), SERVER.handler_for(engine))
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            request = {
                "model": "gliner-router",
                "messages": [{"role": "user", "content": "calculate 19 * 7"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "schema": {
                            "properties": {
                                "decision": {
                                    "properties": {
                                        "target": {"enum": ["deterministic_tool", "human_review"]}
                                    }
                                }
                            }
                        }
                    },
                },
            }
            data = json.dumps(request).encode()
            response = urllib.request.urlopen(
                urllib.request.Request(
                    f"http://127.0.0.1:{httpd.server_port}/v1/chat/completions",
                    data=data,
                    headers={"Content-Type": "application/json"},
                )
            )
            body = json.load(response)
            verdict = json.loads(body["choices"][0]["message"]["content"])
            self.assertEqual(verdict["decision"]["target"], "deterministic_tool")
            self.assertEqual(verdict["decision"]["confidence"], 0.9)
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
