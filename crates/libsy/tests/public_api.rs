// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compile-time coverage for host-facing libsy composition APIs.

use switchyard_libsy::ToolSignalProcessor;

#[test]
fn host_can_construct_the_public_tool_signal_processor() {
    let _processor = ToolSignalProcessor::default();
}
