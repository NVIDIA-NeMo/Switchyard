// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Training examples for GZip-kNN classifier.
//!
//! These examples are used to initialize the GZip-kNN classifier for task classification.
//! They cover six main task categories:
//! - simple_query: Basic questions and definitions
//! - code_generation: Writing or modifying code
//! - complex_reasoning: Architecture, debugging, design decisions
//! - document_analysis: Summarization, extraction, analysis
//! - creative_writing: Content creation, storytelling
//! - data_analysis: Statistics, interpretation, trend analysis

use crate::{GZipKNNClassifier, TrainingExample};

/// Create a GZip-kNN classifier with embedded training examples.
pub fn create_default_classifier() -> GZipKNNClassifier {
    let mut classifier = GZipKNNClassifier::new(5, 6);

    let examples = vec![
        // Simple Query Examples
        TrainingExample {
            text: "What is Python?".to_string(),
            label: "simple_query".to_string(),
        },
        TrainingExample {
            text: "Define machine learning".to_string(),
            label: "simple_query".to_string(),
        },
        TrainingExample {
            text: "What is the difference between lists and tuples?".to_string(),
            label: "simple_query".to_string(),
        },
        TrainingExample {
            text: "Explain what REST API means".to_string(),
            label: "simple_query".to_string(),
        },
        TrainingExample {
            text: "What is Docker?".to_string(),
            label: "simple_query".to_string(),
        },
        TrainingExample {
            text: "What does JSON stand for?".to_string(),
            label: "simple_query".to_string(),
        },
        TrainingExample {
            text: "How do I check Python version?".to_string(),
            label: "simple_query".to_string(),
        },
        TrainingExample {
            text: "What is async programming?".to_string(),
            label: "simple_query".to_string(),
        },
        // Code Generation Examples
        TrainingExample {
            text: "Write a Python function to sort an array".to_string(),
            label: "code_generation".to_string(),
        },
        TrainingExample {
            text: "Create a function that reverses a string".to_string(),
            label: "code_generation".to_string(),
        },
        TrainingExample {
            text: "Write a script to read a CSV file".to_string(),
            label: "code_generation".to_string(),
        },
        TrainingExample {
            text: "Implement binary search algorithm".to_string(),
            label: "code_generation".to_string(),
        },
        TrainingExample {
            text: "Write a REST API endpoint in Flask".to_string(),
            label: "code_generation".to_string(),
        },
        TrainingExample {
            text: "Create a class for user authentication".to_string(),
            label: "code_generation".to_string(),
        },
        TrainingExample {
            text: "Generate SQL query to find top 10 customers by sales".to_string(),
            label: "code_generation".to_string(),
        },
        TrainingExample {
            text: "Write a decorator for timing function execution".to_string(),
            label: "code_generation".to_string(),
        },
        // Complex Reasoning Examples
        TrainingExample {
            text: "Design a microservices architecture for an e-commerce platform".to_string(),
            label: "complex_reasoning".to_string(),
        },
        TrainingExample {
            text: "How would you debug a memory leak in production?".to_string(),
            label: "complex_reasoning".to_string(),
        },
        TrainingExample {
            text: "What are the trade-offs between SQL and NoSQL databases?".to_string(),
            label: "complex_reasoning".to_string(),
        },
        TrainingExample {
            text: "How would you scale a system to handle 1 million concurrent users?"
                .to_string(),
            label: "complex_reasoning".to_string(),
        },
        TrainingExample {
            text: "Explain the CAP theorem and its implications".to_string(),
            label: "complex_reasoning".to_string(),
        },
        TrainingExample {
            text: "Design a distributed cache system".to_string(),
            label: "complex_reasoning".to_string(),
        },
        TrainingExample {
            text: "How would you implement circuit breaker pattern in microservices?"
                .to_string(),
            label: "complex_reasoning".to_string(),
        },
        TrainingExample {
            text: "Analyze and optimize slow database queries".to_string(),
            label: "complex_reasoning".to_string(),
        },
        // Document Analysis Examples
        TrainingExample {
            text: "Summarize this research paper on neural networks".to_string(),
            label: "document_analysis".to_string(),
        },
        TrainingExample {
            text: "Extract key points from this technical article".to_string(),
            label: "document_analysis".to_string(),
        },
        TrainingExample {
            text: "Analyze the sentiment of customer feedback".to_string(),
            label: "document_analysis".to_string(),
        },
        TrainingExample {
            text: "Find all action items in this meeting transcript".to_string(),
            label: "document_analysis".to_string(),
        },
        TrainingExample {
            text: "Summarize the main arguments in this whitepaper".to_string(),
            label: "document_analysis".to_string(),
        },
        TrainingExample {
            text: "Extract entities from this legal document".to_string(),
            label: "document_analysis".to_string(),
        },
        TrainingExample {
            text: "Categorize these customer reviews by topic".to_string(),
            label: "document_analysis".to_string(),
        },
        TrainingExample {
            text: "Find contradictions in these statements".to_string(),
            label: "document_analysis".to_string(),
        },
        // Creative Writing Examples
        TrainingExample {
            text: "Write a blog post about machine learning trends".to_string(),
            label: "creative_writing".to_string(),
        },
        TrainingExample {
            text: "Create a short story about a time traveler".to_string(),
            label: "creative_writing".to_string(),
        },
        TrainingExample {
            text: "Write marketing copy for a new SaaS product".to_string(),
            label: "creative_writing".to_string(),
        },
        TrainingExample {
            text: "Compose a technical tutorial on Docker".to_string(),
            label: "creative_writing".to_string(),
        },
        TrainingExample {
            text: "Write a poem about artificial intelligence".to_string(),
            label: "creative_writing".to_string(),
        },
        TrainingExample {
            text: "Create a catchy tagline for our company".to_string(),
            label: "creative_writing".to_string(),
        },
        TrainingExample {
            text: "Draft an email announcing a new feature".to_string(),
            label: "creative_writing".to_string(),
        },
        TrainingExample {
            text: "Write a joke or funny anecdote about programming".to_string(),
            label: "creative_writing".to_string(),
        },
        // Data Analysis Examples
        TrainingExample {
            text: "Analyze sales trends from this dataset".to_string(),
            label: "data_analysis".to_string(),
        },
        TrainingExample {
            text: "What patterns do you see in this time series data?"
                .to_string(),
            label: "data_analysis".to_string(),
        },
        TrainingExample {
            text: "Calculate correlation between these variables".to_string(),
            label: "data_analysis".to_string(),
        },
        TrainingExample {
            text: "Interpret these statistical results".to_string(),
            label: "data_analysis".to_string(),
        },
        TrainingExample {
            text: "Identify outliers in this dataset".to_string(),
            label: "data_analysis".to_string(),
        },
        TrainingExample {
            text: "What is the average customer lifetime value?".to_string(),
            label: "data_analysis".to_string(),
        },
        TrainingExample {
            text: "Perform a cohort analysis on user engagement".to_string(),
            label: "data_analysis".to_string(),
        },
        TrainingExample {
            text: "Forecast next quarter revenue based on historical data".to_string(),
            label: "data_analysis".to_string(),
        },
    ];

    classifier.add_examples(examples);
    classifier
}
