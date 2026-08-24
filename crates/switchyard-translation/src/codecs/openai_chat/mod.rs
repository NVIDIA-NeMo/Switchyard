// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenAI Chat Completions buffered and streaming codecs.

mod buffered;
mod stream;

pub use buffered::OpenAiChatCodec;
pub use stream::OpenAiChatStreamCodec;

pub(crate) use buffered::{
    decode_file_source, decode_image_source, file_source_text, image_source_text, openai_file_part,
    openai_image_part,
};
