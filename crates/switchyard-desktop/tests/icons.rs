// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[path = "../examples/desktop-icons/export.rs"]
mod export;

// Every host checks both platform exports, so a stale Windows icon also fails on macOS.
#[test]
fn committed_exports_match_their_png_sources() {
    let icons = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("icons");
    if let Err(error) = export::export(&icons, true) {
        panic!("{error}");
    }
}
