---
title: "Known Issues in 0.1.0"
description: "Review known limitations and workarounds for NeMo Switchyard 0.1.0."
position: 2
---
## 0.1.0
1. Completed Codex Responses tasks may record `0` token usage in `GET /v1/stats` and `GET /v1/routing/stats`.
2. Tool-bearing Codex requests may fail when Switchyard routes them to an upstream that accepts only a fixed set of tool names or schemas.
