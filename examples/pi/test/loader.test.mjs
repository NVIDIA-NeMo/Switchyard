// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import { DefaultResourceLoader, SettingsManager } from "@earendil-works/pi-coding-agent";

test("Pi loads the extension through its public resource loader", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "switchyard-pi-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const extensionPath = fileURLToPath(new URL("../switchyard-router.ts", import.meta.url));
  const loader = new DefaultResourceLoader({
    cwd: dir,
    agentDir: join(dir, "agent"),
    settingsManager: SettingsManager.inMemory(),
    additionalExtensionPaths: [extensionPath],
    noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true,
  });
  await loader.reload();
  const { extensions, errors } = loader.getExtensions();
  assert.deepEqual(errors, []);
  assert.deepEqual(extensions.map((extension) => extension.path), [extensionPath]);
});
