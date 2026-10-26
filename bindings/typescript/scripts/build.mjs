// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { execFileSync } from "node:child_process";
import { copyFileSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { join } from "node:path";

const root = fileURLToPath(new URL("../../../", import.meta.url));
const pkg = fileURLToPath(new URL("../", import.meta.url));
execFileSync("cargo", ["build", "-p", "switchyard-node", "--locked"], { cwd: root, stdio: "inherit" });
const metadata = JSON.parse(execFileSync("cargo", ["metadata", "--no-deps", "--format-version", "1", "--locked"], { cwd: root }));
const library = process.platform === "win32" ? "switchyard_node.dll"
  : process.platform === "darwin" ? "libswitchyard_node.dylib" : "libswitchyard_node.so";
mkdirSync(join(pkg, "dist"), { recursive: true });
copyFileSync(join(metadata.target_directory, "debug", library), join(pkg, "dist", "switchyard.node"));
for (const file of ["LICENSE", "NOTICE"]) copyFileSync(join(root, file), join(pkg, file));
