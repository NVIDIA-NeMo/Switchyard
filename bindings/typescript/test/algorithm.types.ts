// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AlgorithmFactory, ConfiguredAlgorithm, ConfiguredRoute } from "../src/index.js";

declare const factory: AlgorithmFactory;

const bare = factory.build()["switchyard/coding"];
bare.runStream({}, { headers: { "x-switchyard-task-id": "task" } });
// @ts-expect-error Bare algorithms have no configured target inventory.
bare.targetModels();
// @ts-expect-error Target requirements need a configured route.
bare.runStream({}, { requiredLabels: [] });
// @ts-expect-error Trusted task IDs need a configured route.
bare.runStream({}, { taskId: "task" });

const route: ConfiguredRoute = factory.buildConfigured()["switchyard/coding"];
route.targetModels()["approved"].id;
route.runStream({}, {
  headers: { "x-switchyard-task-id": "header-task" },
  requiredLabels: ["private", "customer_data"],
  taskId: "trusted-task",
});
const compatible: ConfiguredAlgorithm = route;
compatible.runStream({});
