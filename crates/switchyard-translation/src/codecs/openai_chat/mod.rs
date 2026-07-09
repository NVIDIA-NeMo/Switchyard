// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenAI Chat Completions buffered and streaming codecs.

mod buffered;
mod stream;

pub use buffered::OpenAiChatCodec;
pub use stream::OpenAiChatStreamCodec;

pub(crate) use buffered::{
    decode_file_source, decode_image_source, decode_image_url, parse_arguments,
};
