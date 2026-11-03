// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-env-changed=OPENSHELL_PROTO_DIR");
    let proto = std::env::var("OPENSHELL_PROTO_DIR")
        .map_err(|_| "OPENSHELL_PROTO_DIR must point to the patched OpenShell proto directory")?;
    tonic_prost_build::configure()
        .compile_protos(&[format!("{proto}/supervisor_middleware.proto")], &[proto])?;
    Ok(())
}
