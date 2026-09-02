// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Embedded prompts and training examples for classifiers.

pub mod gzip_knn_classifier {
    pub mod training_examples;

    pub use training_examples::create_default_classifier;
}
