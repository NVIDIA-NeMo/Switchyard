// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Binary entrypoint for `switchyard-soak`.

use std::process::ExitCode;

fn main() -> ExitCode {
    switchyard_soak::cli_main()
}
