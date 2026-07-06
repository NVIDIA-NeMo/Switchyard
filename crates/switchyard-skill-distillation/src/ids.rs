// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Validated identifiers used by skill-distillation records and store paths.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::error::{Result, SkillDistillationError};

fn validate_path_component(value: &str, kind: &'static str) -> Result<()> {
    let reason = if value != value.trim() {
        Some("must not have leading or trailing whitespace")
    } else if value.is_empty() || matches!(value, "." | "..") {
        Some("must be a non-empty safe local path component")
    } else if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        Some("may contain only letters, numbers, dot, underscore, and hyphen")
    } else {
        None
    };

    if let Some(reason) = reason {
        return Err(SkillDistillationError::InvalidIdentifier {
            kind,
            reason: reason.to_string(),
        });
    }
    Ok(())
}

macro_rules! path_component_id {
    ($name:ident, $kind:literal, $doc:literal) => {
        #[doc = $doc]
        #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
        #[serde(try_from = "String", into = "String")]
        pub struct $name(String);

        impl $name {
            /// Creates an identifier after validating its path-component form.
            pub fn new(value: impl Into<String>) -> Result<Self> {
                let value = value.into();
                validate_path_component(&value, $kind)?;
                Ok(Self(value))
            }

            /// Returns the identifier as a borrowed string.
            pub fn as_str(&self) -> &str {
                &self.0
            }

            /// Consumes the identifier and returns its owned string.
            pub fn into_inner(self) -> String {
                self.0
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                self.as_str()
            }
        }

        impl TryFrom<String> for $name {
            type Error = SkillDistillationError;

            fn try_from(value: String) -> Result<Self> {
                Self::new(value)
            }
        }

        impl TryFrom<&str> for $name {
            type Error = SkillDistillationError;

            fn try_from(value: &str) -> Result<Self> {
                Self::new(value)
            }
        }

        impl From<$name> for String {
            fn from(value: $name) -> Self {
                value.into_inner()
            }
        }

        impl FromStr for $name {
            type Err = SkillDistillationError;

            fn from_str(value: &str) -> Result<Self> {
                Self::new(value)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }
    };
}

path_component_id!(
    SkillNamespace,
    "skill namespace",
    "Namespace for one skill that improves over time. Many separately identified \
     trajectories can contribute to it; it does not identify a session or trajectory."
);
path_component_id!(
    SkillEvidenceId,
    "skill evidence id",
    "Stable within a skill namespace for one completed run saved as distillation \
     evidence. Code that converts a session into evidence must reuse a safe session ID \
     or derive the same value each time. Requests use it to reject duplicate evidence, \
     and candidates record which evidence they used. It does not track a live session."
);
path_component_id!(
    SkillVersionId,
    "skill version id",
    "Stable identifier for one generated skill version."
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifiers_accept_safe_path_components() {
        for value in ["tooluniverse-trialqa", "a.b_c-1", "..."] {
            assert!(SkillNamespace::new(value).is_ok(), "{value}");
            assert!(SkillEvidenceId::new(value).is_ok(), "{value}");
            assert!(SkillVersionId::new(value).is_ok(), "{value}");
        }
    }

    #[test]
    fn identifiers_reject_path_traversal_and_unsafe_characters() {
        for value in [
            "",
            " ",
            " .",
            ".",
            "..",
            "../escape",
            "has/slash",
            "has space",
            "ümlaut",
        ] {
            assert!(SkillNamespace::new(value).is_err(), "{value}");
            assert!(SkillEvidenceId::new(value).is_err(), "{value}");
            assert!(SkillVersionId::new(value).is_err(), "{value}");
        }
    }
}
