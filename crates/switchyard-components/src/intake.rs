// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Intake sink support shared by request and response processors.

pub mod client;
pub mod config;
pub mod context;
pub mod payload;

pub use client::{HttpIntakeSink, IntakeSink};
pub use config::{IntakeQueueFullPolicy, IntakeSinkConfig};
pub use context::{
    IntakeRequestMetadata, IntakeRequestState, RequestMetadata, INTAKE_APP_HEADER,
    INTAKE_ENABLED_HEADER, INTAKE_TASK_HEADER, PROXY_SESSION_ID_HEADER,
};
pub use payload::{
    anthropic_response_from_stream, gemini_response_from_stream, now_millis,
    openai_chat_response_from_stream, request_type_value, responses_response_from_stream,
    to_nvdataflow_document, IntakePayloadBuilder, IntakePayloadContext, IntakeStreamCapture,
    IntakeStreamFormat, SYNTHETIC_STREAM_RESPONSE_IDS, UNKNOWN_MODEL,
};
