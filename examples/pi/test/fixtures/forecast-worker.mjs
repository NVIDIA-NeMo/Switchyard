// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createInterface } from "node:readline";

for await (const line of createInterface({ input: process.stdin })) {
  const request = JSON.parse(line);
  process.stdout.write(JSON.stringify({ id: request.id, result: {
    next_call: { tokens: 2000, requirements: null },
    remaining_task: { tokens: 6000, requirements: null },
    checkpoint: "fixture", inference_ms: 0,
    completed_calls: request.messages.filter(message => message.role === "assistant").length,
  } }) + "\n");
}
