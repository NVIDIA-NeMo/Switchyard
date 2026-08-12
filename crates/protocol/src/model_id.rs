// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Model identifier used by Switchyard. See ModelId docs for details.

use std::borrow::Borrow;
use std::fmt;
use std::ops::Deref;

use serde::{Deserialize, Serialize};

/// A model name used in a request or routing decision.
///
/// It can name a provider model, such as `"openai/gpt-oss-20b"`. It can also name a
/// Switchyard route, such as `"switchyard/random"`.
///
/// A target name from server config, such as `"capable"`, is not a model ID. The server
/// resolves target names to model IDs before routing.
///
/// This type acts like a string. It can be printed, compared, and saved as a JSON string.
#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModelId(String);

/// Forwards to the wrapped string rather than deriving, so `{:?}` renders `"gpt-4"`
/// instead of `ModelId("gpt-4")`.
impl fmt::Debug for ModelId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, formatter)
    }
}

impl ModelId {
    /// Wraps a model identifier.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// The identifier as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Unwraps to the owned identifier.
    pub fn into_string(self) -> String {
        self.0
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Deref for ModelId {
    type Target = str;

    fn deref(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for ModelId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// Lets a `HashMap<ModelId, _>` or `HashSet<ModelId>` be looked up by `&str`, so
/// callers holding a borrowed id do not have to allocate one to query.
impl Borrow<str> for ModelId {
    fn borrow(&self) -> &str {
        &self.0
    }
}

impl From<String> for ModelId {
    fn from(id: String) -> Self {
        Self(id)
    }
}

impl From<&str> for ModelId {
    fn from(id: &str) -> Self {
        Self(id.to_string())
    }
}

/// Mirrors `From<&str> for String`, so a borrowed id reaches an owning
/// `impl Into<ModelId>` parameter without an explicit clone.
impl From<&ModelId> for ModelId {
    fn from(id: &ModelId) -> Self {
        id.clone()
    }
}

impl From<ModelId> for String {
    fn from(id: ModelId) -> Self {
        id.0
    }
}

impl From<&ModelId> for String {
    fn from(id: &ModelId) -> Self {
        id.0.clone()
    }
}

// Comparison against bare strings, in both directions, so an id can be checked
// against a literal or a configured name without wrapping either side.
// This is likely excessive. We can reduce when things settle.

impl PartialEq<str> for ModelId {
    fn eq(&self, other: &str) -> bool {
        self.0 == other
    }
}

impl PartialEq<&str> for ModelId {
    fn eq(&self, other: &&str) -> bool {
        self.0 == *other
    }
}

impl PartialEq<String> for ModelId {
    fn eq(&self, other: &String) -> bool {
        &self.0 == other
    }
}

impl PartialEq<ModelId> for str {
    fn eq(&self, other: &ModelId) -> bool {
        self == other.0
    }
}

impl PartialEq<ModelId> for &str {
    fn eq(&self, other: &ModelId) -> bool {
        *self == other.0
    }
}

impl PartialEq<ModelId> for String {
    fn eq(&self, other: &ModelId) -> bool {
        self == &other.0
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn it_behaves_like_the_string_it_wraps() {
        let id = ModelId::new("openai/gpt-oss-20b");

        assert_eq!(id.to_string(), "openai/gpt-oss-20b");
        assert_eq!(id, "openai/gpt-oss-20b");
        assert_eq!("openai/gpt-oss-20b", id);
        assert!(id.starts_with("openai/"));
        assert_eq!(takes_str(&id), "openai/gpt-oss-20b");
    }

    /// Deref coercion means an `&ModelId` reaches a `&str` parameter unchanged.
    fn takes_str(model: &str) -> &str {
        model
    }

    #[test]
    fn a_map_of_ids_is_queryable_by_str() {
        let by_model = HashMap::from([(ModelId::new("aws/anthropic/claude-opus-4-5"), 1)]);

        assert_eq!(by_model.get("aws/anthropic/claude-opus-4-5"), Some(&1));
    }

    #[test]
    fn it_serializes_as_a_bare_string() -> serde_json::Result<()> {
        let id = ModelId::new("openai/gpt-oss-20b");

        let json = serde_json::to_string(&id)?;
        assert_eq!(json, "\"openai/gpt-oss-20b\"");
        assert_eq!(serde_json::from_str::<ModelId>(&json)?, id);
        Ok(())
    }
}
