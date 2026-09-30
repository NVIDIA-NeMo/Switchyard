// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Category is a group of models

use std::str::FromStr;
use std::sync::Arc;

/// A group of models
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Category {
    /// When the category doesn't matter: Random, Passthrough, etc.
    Any,
    /// High accuracy and cost models.
    Capable,
    /// Lower accuracy and cost models.
    Efficient,
    /// Models the algorithm can use to decide.
    Judge,
    /// A deployment-defined group. Only a custom classifier's policy selects one,
    /// and no algorithm ascribes meaning to the name.
    Named(Arc<str>),
}

impl Category {
    /// Returns the lowercase category name used in configuration.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Any => "any",
            Self::Capable => "capable",
            Self::Efficient => "efficient",
            Self::Judge => "judge",
            Self::Named(name) => name,
        }
    }
}

impl FromStr for Category {
    type Err = String;

    /// The four reserved names are matched first. Were `"capable"` allowed to
    /// become a [`Category::Named`] there would be two keys that compare unequal
    /// but print the same, and a lookup of [`Category::Capable`] would silently
    /// miss the configured group.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let c = match s {
            "capable" => Self::Capable,
            "efficient" => Self::Efficient,
            "judge" => Self::Judge,
            "any" => Self::Any,
            "" => return Err("Category name cannot be empty".to_string()),
            name => Self::Named(Arc::from(name)),
        };
        Ok(c)
    }
}
