// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Category is a group of models

use std::{collections::HashMap, str::FromStr};

use crate::ModelId;

/// A group of models
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Category {
    /// When the category doesn't matter: Random, Passthrough, etc.
    Any,
    /// High accuracy and cost models.
    Capable,
    /// Lower accuracy and cost models.
    Efficient,
    /// Models the algorithm can use to decide.
    Judge,
}

impl Category {
    /// Convenience function to create a HashMap suitable for passing to `run_stream` and family.
    pub fn to_map(category: Category, names: &[&str]) -> HashMap<Category, Vec<ModelId>> {
        [(
            category,
            names.iter().map(|name| ModelId::from(*name)).collect(),
        )]
        .into()
    }
}

impl FromStr for Category {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let c = match s {
            "capable" => Self::Capable,
            "efficient" => Self::Efficient,
            "judge" => Self::Judge,
            "any" => Self::Any,
            x => {
                return Err(format!("Invalid Category '{x}'"));
            }
        };
        Ok(c)
    }
}
