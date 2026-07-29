// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Non-empty string identifiers used by compatibility component wiring.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Error returned when an identifier is empty or whitespace.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
#[error("{kind} must not be empty")]
pub struct InvalidId {
    kind: &'static str,
}

impl InvalidId {
    // Keeps the constructor private so only validated ID types can create this error.
    fn empty(kind: &'static str) -> Self {
        Self { kind }
    }
}

// Defines the repeated non-empty string ID behavior without runtime inheritance.
macro_rules! string_id {
    ($name:ident) => {
        #[doc = concat!("Validated non-empty identifier for `", stringify!($name), "`.")]
        #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
        #[serde(try_from = "String", into = "String")]
        pub struct $name(String);

        impl $name {
            const KIND: &'static str = stringify!($name);

            /// Creates a new identifier after rejecting empty or whitespace input.
            pub fn new(value: impl Into<String>) -> Result<Self, InvalidId> {
                let value = value.into();
                if value.trim().is_empty() {
                    return Err(InvalidId::empty(Self::KIND));
                }
                Ok(Self(value))
            }

            /// Creates an identifier from a compile-time string.
            pub fn from_static(value: &'static str) -> Self {
                match Self::new(value) {
                    Ok(id) => id,
                    Err(_) => panic!("static Switchyard IDs must not be empty"),
                }
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
            type Error = InvalidId;

            fn try_from(value: String) -> Result<Self, Self::Error> {
                Self::new(value)
            }
        }

        impl TryFrom<&str> for $name {
            type Error = InvalidId;

            fn try_from(value: &str) -> Result<Self, Self::Error> {
                Self::new(value)
            }
        }

        impl From<$name> for String {
            fn from(value: $name) -> Self {
                value.into_inner()
            }
        }

        impl FromStr for $name {
            type Err = InvalidId;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
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

string_id!(LlmTargetId);
string_id!(ModelId);
string_id!(RequestId);
