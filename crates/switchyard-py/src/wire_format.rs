// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python format constants backed by the protocol's built-in identifiers.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
use switchyard_protocol::WireFormat;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    let members = [
        ("OPENAI_CHAT", WireFormat::OpenAiChat.as_str()),
        ("OPENAI_RESPONSES", WireFormat::OpenAiResponses.as_str()),
        ("ANTHROPIC_MESSAGES", WireFormat::AnthropicMessages.as_str()),
    ];
    let kwargs = PyDict::new(py);
    kwargs.set_item("type", py.get_type::<PyString>())?;
    kwargs.set_item("module", "switchyard_rust.wire_format")?;
    let formats = py
        .import("enum")?
        .getattr("Enum")?
        .call(("WireFormat", members), Some(&kwargs))?;
    formats.setattr(
        "__doc__",
        "Built-in provider protocols accepted by Switchyard.",
    )?;
    module.add("WireFormat", formats)
}
