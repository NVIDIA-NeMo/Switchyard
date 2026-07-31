// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Intake sink support shared by request and response processors.

pub mod client;
pub mod config;
pub mod context;
pub mod payload;

pub use client::{HttpIntakeSink, IntakeSink};
pub use config::{IntakeFormat, IntakeQueueFullPolicy, IntakeSinkConfig, IntakeTarget};
pub use context::{
    INTAKE_APP_HEADER, INTAKE_ENABLED_HEADER, INTAKE_TASK_HEADER, IntakeRequestMetadata,
    IntakeRequestState, PROXY_SESSION_ID_HEADER, RequestMetadata, SubModelCall, SubModelCalls,
};
pub use payload::{
    IntakePayloadBuilder, IntakePayloadContext, IntakeStreamCapture, IntakeStreamFormat,
    SYNTHETIC_STREAM_RESPONSE_IDS, UNKNOWN_MODEL, anthropic_response_from_stream, now_millis,
    openai_chat_response_from_stream, request_type_value, responses_response_from_stream,
    to_flat_document,
};
