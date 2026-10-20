// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod export;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let check = match std::env::args().skip(1).collect::<Vec<_>>().as_slice() {
        [] => false,
        [arg] if arg == "--check" => true,
        _ => return Err("usage: cargo desktop-icons [--check]".into()),
    };
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("icons");
    export::export(&directory, check)?;
    println!(
        "Desktop icons {}.",
        if check {
            "match the app and tray PNG sources"
        } else {
            "exported for macOS and Windows"
        }
    );
    Ok(())
}
