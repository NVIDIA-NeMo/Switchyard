// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compatibility entry point for callers using the earlier binary restriction flag.

use switchyard_protocol::{LlmClientError, Request};

/// Adds `private` when restricted; otherwise explicitly supplies no new requirements.
/// Prefer [`crate::require_target_labels`] for deployment-defined labels.
pub fn require_restricted_targets(
    request: &mut Request,
    is_restricted: bool,
) -> Result<(), LlmClientError> {
    crate::require_target_labels(request, is_restricted.then_some("private"))
}
