// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Codec pair for the Gemini generateContent wire format.

mod buffered;
mod stream;

pub use buffered::GeminiGenerateContentCodec;
pub use stream::GeminiGenerateContentStreamCodec;
