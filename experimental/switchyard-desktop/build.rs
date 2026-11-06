// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// Supported desktop platforms need Tauri resources and command metadata at build time.
fn main() {
    #[cfg(any(target_os = "macos", windows))]
    tauri_build::build();
}
