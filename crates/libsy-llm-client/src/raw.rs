// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The raw wire result of a call.
//!
//! The encoders that produce it — buffered JSON and the streamed wire-event
//! encoder — live in `switchyard-translation` alongside the rest of the neutral-IR
//! codecs; this module only holds the result type they feed.

use serde_json::Value;
use switchyard_translation::RawEventStream;

/// The wire result of
/// [`call_rewrite_model_raw`](crate::TranslatingLlmClient::call_rewrite_model_raw):
/// a buffered JSON body, or a live stream of wire events — both already in the
/// requested wire format.
pub enum RawResponse {
    /// A complete JSON response body.
    Buffered(Value),
    /// A live stream of wire-format event objects, ready for SSE framing.
    Stream(RawEventStream),
}
