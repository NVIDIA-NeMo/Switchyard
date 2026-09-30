// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reads the Codex tool namespaces that a Responses request decoder records.
//!
//! Codex groups tools into `namespace` containers. The Responses codec in
//! `switchyard-translation` flattens each child to `<namespace>__<tool>` and
//! stores the mapping in the request's [`ProviderExtensions`]. Code that does not
//! depend on the translation crate uses these helpers to recover the tool name.

use serde_json::{Map, Value};

use crate::ProviderExtensions;

/// Separator between a namespace and a tool name in a qualified wire name.
pub const NAMESPACE_SEPARATOR: &str = "__";

/// Request extension key holding the qualified-name to namespace mapping.
///
/// Prefixed so it cannot collide with a real provider field, and so a codec that
/// allowlists provider fields never forwards it.
pub const TOOL_NAMESPACES_KEY: &str = "switchyard_codex_tool_namespaces";

/// Reads the mapping back off a request's extensions.
pub fn tool_namespaces(extensions: &ProviderExtensions) -> Option<&Map<String, Value>> {
    extensions
        .fields
        .get(TOOL_NAMESPACES_KEY)
        .and_then(Value::as_object)
}

/// Splits a qualified wire name back into its tool name and namespace.
///
/// Returns `None` for a name the request never qualified, so an unrecognized
/// call is left alone rather than attributed to the wrong namespace. The tool
/// name may itself contain the separator, so the namespace is matched as a
/// prefix rather than by splitting on it.
pub fn split_qualified_name<'a>(
    namespaces: &'a Map<String, Value>,
    qualified: &'a str,
) -> Option<(&'a str, &'a str)> {
    let value = namespaces.get(qualified)?;
    let namespace = value
        .as_str()
        .or_else(|| value.get("namespace").and_then(Value::as_str))?;
    let tool = qualified
        .strip_prefix(namespace)?
        .strip_prefix(NAMESPACE_SEPARATOR)?;
    Some((tool, namespace))
}
