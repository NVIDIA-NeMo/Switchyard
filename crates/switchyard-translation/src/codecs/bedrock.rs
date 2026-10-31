// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bedrock Converse buffered bodies and ConverseStream event payloads.

mod buffered;
pub(crate) mod stream;

pub use buffered::BedrockConverseCodec;
pub use stream::BedrockConverseStreamCodec;

pub(crate) use buffered::request_projection_diagnostics;
