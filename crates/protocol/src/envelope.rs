// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The request/response envelope: the normalized [`LlmRequest`]/[`LlmResponse`] paired
//! with the original provider payload and correlation [`Metadata`].

use crate::{LlmRequest, LlmResponse, Metadata, ModelId};

/// A request an algorithm routes: the normalized [`LlmRequest`] plus optional
/// host-owned raw data and correlation [`Metadata`].
#[derive(Clone, Default)]
pub struct Request {
    /// The normalized request an algorithm routes.
    pub llm_request: LlmRequest,
    /// An optional whole request body retained by the host. libsy and the translation
    /// codecs do not read it. Exact same-format codec replay instead uses
    /// [`LlmRequest::preservation`].
    pub raw_request: Option<serde_json::Value>,
    /// Correlation metadata carried through the request.
    pub metadata: Option<Metadata>,
}

impl Request {
    /// Returns the model currently carried by the request, when present.
    ///
    /// On ingress this is the name supplied by the client. Before an offloaded model call, the
    /// routing driver replaces it with the selected target.
    pub fn model_id(&self) -> Option<ModelId> {
        self.llm_request.model.as_deref().map(ModelId::from)
    }
}

/// A response an algorithm returns: the [`LlmResponse`] (streamed or aggregate) plus
/// optional correlation [`Metadata`].
///
/// Not `Clone` — `llm_response` may own a live stream.
pub struct Response {
    /// The neutral model response — a chunk stream or the buffered aggregate.
    pub llm_response: LlmResponse,
    /// Correlation metadata carried through the response.
    pub metadata: Option<Metadata>,
}

impl Response {
    /// Returns the model recorded by a buffered response.
    ///
    /// Returns `None` for a live stream because inspecting its `MessageStart`
    /// event would require consuming the stream.
    pub fn selected_model(&self) -> Option<&str> {
        self.llm_response.selected_model()
    }
}
