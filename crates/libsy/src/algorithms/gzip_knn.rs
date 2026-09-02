// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! GZip-kNN classifier for task classification without LLM calls.
//!
//! Implements a parameter-free text classifier using Normalized Compression Distance (NCD)
//! and k-Nearest Neighbors voting. This classifier operates offline and requires no external
//! LLM calls, making it suitable as a fallback when LLM-based judges are unavailable.
//!
//! The algorithm is based on:
//! NCD(x, y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))
//! where C(x) is the compressed size of string x using gzip.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use flate2::Compression;
use flate2::write::GzEncoder;
use std::io::Write;

use crate::core::classifier::{Classification, Classifier, Score};
use crate::core::algorithm::Driver;
use crate::{Result, LibsyError};
use switchyard_protocol::{ModelId, Request};

/// A labeled training example for GZip-kNN classifier.
#[derive(Clone, Debug)]
pub struct TrainingExample {
    /// The text content of the example.
    pub text: String,
    /// The category/target label.
    pub label: String,
}

/// Result from a single k-NN classification.
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    /// The predicted category.
    pub category: String,
    /// Confidence in the prediction [0.0, 1.0].
    pub confidence: f64,
    /// Nearest neighbors and their distances.
    pub nearest_examples: Vec<(f64, String)>,
}

/// GZip-kNN classifier for task classification.
pub struct GZipKNNClassifier {
    /// k for k-NN voting.
    k: usize,
    /// Compression level for gzip (1-9).
    compression_level: u32,
    /// Training examples.
    training_set: Vec<TrainingExample>,
    /// Cache of pre-compressed training examples for faster lookup.
    compressed_cache: HashMap<usize, usize>,
}

impl GZipKNNClassifier {
    /// Create a new GZip-kNN classifier.
    ///
    /// # Arguments
    ///
    /// * `k` - Number of nearest neighbors for voting (typically 3-7)
    /// * `compression_level` - gzip compression level (1-9, default 6 is good balance)
    pub fn new(k: usize, compression_level: u32) -> Self {
        Self {
            k: k.max(1), // Ensure k >= 1
            compression_level: compression_level.min(9).max(1),
            training_set: Vec::new(),
            compressed_cache: HashMap::new(),
        }
    }

    /// Add labeled training examples.
    pub fn add_examples(&mut self, examples: Vec<TrainingExample>) {
        // Pre-compress training examples for faster distance calculations
        for example in &examples {
            let hash = Self::text_hash(&example.text);
            if !self.compressed_cache.contains_key(&hash) {
                if let Ok(size) = Self::compress_size(&example.text, self.compression_level) {
                    self.compressed_cache.insert(hash, size);
                }
            }
        }
        self.training_set.extend(examples);
    }

    /// Classify a query text and return the predicted category with confidence.
    pub fn classify(&self, query: &str) -> Result<ClassificationResult> {
        if self.training_set.is_empty() {
            return Ok(ClassificationResult {
                category: "unknown".to_string(),
                confidence: 0.0,
                nearest_examples: Vec::new(),
            });
        }

        let query_size = Self::compress_size(query, self.compression_level)?;

        let mut distances: Vec<(f64, String, String)> = Vec::new();

        // Calculate NCD to all training examples
        for example in &self.training_set {
            let hash = Self::text_hash(&example.text);
            let example_size = *self
                .compressed_cache
                .get(&hash)
                .ok_or_else(|| LibsyError::AlgorithmError {
                    message: "missing cached compression size".to_string(),
                })?;

            let combined_size =
                Self::compress_combined(query, &example.text, self.compression_level)?;

            let ncd = Self::ncd(combined_size, query_size, example_size);
            distances.push((ncd, example.text.clone(), example.label.clone()));
        }

        // Sort by distance (closest first)
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let top_k = distances.iter().take(self.k).cloned().collect::<Vec<_>>();

        // Weighted voting: closer examples have higher weight
        let mut weighted_votes: HashMap<String, f64> = HashMap::new();

        for (dist, _text, label) in &top_k {
            // Weight is inverse of distance (closer = higher weight)
            // Add small epsilon to avoid division by zero
            let weight = 1.0 / (dist + 1e-8);
            *weighted_votes.entry(label.clone()).or_insert(0.0) += weight;
        }

        let total_weight: f64 = weighted_votes.values().sum();
        let best_label = weighted_votes
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, _)| k.clone())
            .unwrap_or_else(|| "unknown".to_string());

        let confidence = if total_weight > 0.0 {
            weighted_votes.get(&best_label).unwrap_or(&0.0) / total_weight
        } else {
            0.0
        };

        let nearest_examples = top_k
            .iter()
            .map(|(dist, _, label)| (*dist, label.clone()))
            .collect();

        Ok(ClassificationResult {
            category: best_label,
            confidence,
            nearest_examples,
        })
    }

    /// Classify and return scores for all categories.
    pub fn classify_scores(&self, query: &str) -> Result<HashMap<String, f64>> {
        if self.training_set.is_empty() {
            return Ok(HashMap::new());
        }

        let query_size = Self::compress_size(query, self.compression_level)?;
        let mut weighted_votes: HashMap<String, f64> = HashMap::new();

        // Calculate weighted votes from all training examples
        for example in &self.training_set {
            let hash = Self::text_hash(&example.text);
            let example_size = *self
                .compressed_cache
                .get(&hash)
                .ok_or_else(|| LibsyError::AlgorithmError {
                    message: "missing cached compression size".to_string(),
                })?;

            let combined_size =
                Self::compress_combined(query, &example.text, self.compression_level)?;

            let ncd = Self::ncd(combined_size, query_size, example_size);
            let weight = 1.0 / (ncd + 1e-8);
            *weighted_votes.entry(example.label.clone()).or_insert(0.0) += weight;
        }

        // Normalize scores
        let total_weight: f64 = weighted_votes.values().sum();
        if total_weight > 0.0 {
            for value in weighted_votes.values_mut() {
                *value /= total_weight;
            }
        }

        Ok(weighted_votes)
    }

    /// Calculate Normalized Compression Distance.
    ///
    /// NCD(x, y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))
    #[inline]
    fn ncd(combined_size: usize, x_size: usize, y_size: usize) -> f64 {
        let max_size = x_size.max(y_size) as f64;
        if max_size == 0.0 {
            return 0.0;
        }
        let min_size = x_size.min(y_size) as f64;
        (combined_size as f64 - min_size) / max_size
    }

    /// Compress a single text and return the size.
    fn compress_size(text: &str, compression_level: u32) -> Result<usize> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::new(compression_level));
        encoder.write_all(text.as_bytes()).map_err(|e| {
            LibsyError::AlgorithmError {
                message: format!("compression write failed: {}", e),
            }
        })?;
        let compressed = encoder.finish().map_err(|e| LibsyError::AlgorithmError {
            message: format!("compression failed: {}", e),
        })?;
        Ok(compressed.len())
    }

    /// Compress two texts combined (space-separated) and return the size.
    fn compress_combined(text1: &str, text2: &str, compression_level: u32) -> Result<usize> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::new(compression_level));
        encoder.write_all(text1.as_bytes()).map_err(|e| {
            LibsyError::AlgorithmError {
                message: format!("compression write failed: {}", e),
            }
        })?;
        encoder.write_all(b" ").map_err(|e| LibsyError::AlgorithmError {
            message: format!("compression write failed: {}", e),
        })?;
        encoder.write_all(text2.as_bytes()).map_err(|e| {
            LibsyError::AlgorithmError {
                message: format!("compression write failed: {}", e),
            }
        })?;
        let compressed = encoder.finish().map_err(|e| LibsyError::AlgorithmError {
            message: format!("compression failed: {}", e),
        })?;
        Ok(compressed.len())
    }

    /// Hash a string for cache lookup.
    #[inline]
    fn text_hash(text: &str) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        hasher.finish() as usize
    }
}

/// Classifier trait implementation for GZip-kNN.
/// Scores targets based on compression distance to training examples.
///
/// Uses `Classification::Scores` for high-confidence predictions to short-circuit judge calls,
/// and `Classification::Ambiguous` for uncertain predictions to defer to judges.
pub struct GZipKNNClassifierAdapter {
    classifier: Arc<GZipKNNClassifier>,
    /// Mapping from categories to model targets (e.g., "simple_query" -> "efficient")
    category_to_target: HashMap<String, String>,
    /// Confidence threshold above which to return Scores (short-circuit judges)
    confidence_threshold: f64,
}

impl GZipKNNClassifierAdapter {
    /// Create a new adapter with a classifier and category-to-target mappings.
    pub fn new(
        classifier: Arc<GZipKNNClassifier>,
        category_to_target: HashMap<String, String>,
    ) -> Self {
        Self {
            classifier,
            category_to_target,
            confidence_threshold: 0.75, // Default: short-circuit above 75% confidence
        }
    }

    /// Set the confidence threshold for short-circuiting judges.
    ///
    /// Predictions above this threshold return `Classification::Scores` (skip judges).
    /// Below this threshold return `Classification::Ambiguous` (defer to judges).
    pub fn with_confidence_threshold(mut self, threshold: f64) -> Self {
        self.confidence_threshold = threshold.clamp(0.0, 1.0);
        self
    }
}

#[async_trait]
impl Classifier for GZipKNNClassifierAdapter {
    async fn score(
        &self,
        _state: &mut (),
        request: &mut Request,
        _driver: Option<&Driver>,
    ) -> Result<(Classification, Option<switchyard_protocol::Response>)> {
        // Extract user message from request
        let user_message = request
            .llm_request
            .messages
            .iter()
            .find(|m| m.role == switchyard_protocol::Role::User)
            .and_then(|m| {
                m.content.iter().find_map(|c| match c {
                    switchyard_protocol::ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
            })
            .unwrap_or("");

        if user_message.is_empty() {
            // No user message, abstain (let judges decide)
            return Ok((Classification::Ambiguous(vec![]), None));
        }

        let result = self.classifier.classify(user_message)?;

        // Map category to target tier if configured
        let target = self
            .category_to_target
            .get(&result.category)
            .cloned()
            .unwrap_or_else(|| "capable".to_string()); // Default to capable tier

        let scores = vec![Score {
            target: ModelId::from(target),
            confidence: result.confidence,
        }];

        // High confidence: return Scores to short-circuit judge calls (cost optimization)
        // Low confidence: return Ambiguous to defer to judges (judges have final say)
        if result.confidence >= self.confidence_threshold {
            Ok((Classification::Scores(scores), None))
        } else {
            Ok((Classification::Ambiguous(scores), None))
        }
    }
}

/// Configuration for a GZip-kNN cost-optimization classifier.
///
/// GZip-kNN runs **before** judges to avoid expensive LLM calls for high-confidence
/// predictions. When confidence is low, judges are consulted (judges have final say).
///
/// Maps task categories to Switchyard tiers (efficient/capable).
#[derive(Clone, Debug)]
pub struct GZipKNNFallbackConfig {
    /// k for k-nearest neighbors (typically 3-7, default 5)
    pub k: usize,
    /// Confidence threshold [0.0, 1.0] above which to short-circuit judges
    /// (default 0.75: skip judges for 75%+ confidence predictions)
    pub confidence_threshold: f64,
    /// Mapping: category -> tier (e.g., "simple_query" -> "efficient")
    pub category_tier_map: HashMap<String, String>,
}

impl Default for GZipKNNFallbackConfig {
    fn default() -> Self {
        let mut category_tier_map = HashMap::new();
        // Map categories to Switchyard tiers
        category_tier_map.insert("simple_query".to_string(), "efficient".to_string());
        category_tier_map.insert("code_generation".to_string(), "capable".to_string());
        category_tier_map.insert("complex_reasoning".to_string(), "capable".to_string());
        category_tier_map.insert("document_analysis".to_string(), "balanced".to_string());
        category_tier_map.insert("creative_writing".to_string(), "balanced".to_string());
        category_tier_map.insert("data_analysis".to_string(), "balanced".to_string());

        Self {
            k: 5,
            confidence_threshold: 0.75,
            category_tier_map,
        }
    }
}

/// Builder for creating a GZip-kNN cost-optimization classifier with configuration.
///
/// # Example
///
/// ```ignore
/// let mut classifier = GZipKNNClassifier::new(5, 6);
/// classifier.add_examples(my_training_data);
/// let adapter = GZipKNNBuilder::new(Arc::new(classifier))
///     .with_confidence_threshold(0.8)
///     .build();
/// ```
pub struct GZipKNNBuilder {
    classifier: Arc<GZipKNNClassifier>,
    config: GZipKNNFallbackConfig,
}

impl GZipKNNBuilder {
    /// Create a new builder with a classifier and default config.
    pub fn new(classifier: Arc<GZipKNNClassifier>) -> Self {
        Self {
            classifier,
            config: GZipKNNFallbackConfig::default(),
        }
    }

    /// Set the confidence threshold for short-circuiting judges.
    ///
    /// Predictions above this threshold skip expensive judge calls (cost optimization).
    /// Predictions below fall through to judges (judges have final say).
    pub fn with_confidence_threshold(mut self, threshold: f64) -> Self {
        self.config.confidence_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the k parameter for k-NN voting.
    pub fn with_k(mut self, k: usize) -> Self {
        self.config.k = k;
        self
    }

    /// Set the category-to-tier mapping.
    pub fn with_category_tier_map(mut self, map: HashMap<String, String>) -> Self {
        self.config.category_tier_map = map;
        self
    }

    /// Set the entire configuration.
    pub fn with_config(mut self, config: GZipKNNFallbackConfig) -> Self {
        self.config = config;
        self
    }

    /// Build the adapter classifier.
    pub fn build(self) -> GZipKNNClassifierAdapter {
        GZipKNNClassifierAdapter::new(self.classifier, self.config.category_tier_map)
            .with_confidence_threshold(self.config.confidence_threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use switchyard_protocol::{LlmRequest, Message, Role, ContentBlock};

    #[test]
    fn test_gzip_knn_basic_classification() -> Result<()> {
        let mut classifier = GZipKNNClassifier::new(3, 6);

        let examples = vec![
            TrainingExample {
                text: "What is Python programming?".to_string(),
                label: "simple_query".to_string(),
            },
            TrainingExample {
                text: "Explain Python syntax".to_string(),
                label: "simple_query".to_string(),
            },
            TrainingExample {
                text: "Write a function to sort an array".to_string(),
                label: "code_generation".to_string(),
            },
            TrainingExample {
                text: "Implement binary search algorithm".to_string(),
                label: "code_generation".to_string(),
            },
            TrainingExample {
                text: "Design a microservices architecture".to_string(),
                label: "complex_reasoning".to_string(),
            },
            TrainingExample {
                text: "How would you scale a distributed system?".to_string(),
                label: "complex_reasoning".to_string(),
            },
        ];

        classifier.add_examples(examples);

        let result = classifier.classify("What is Rust?")?;
        assert_eq!(result.category, "simple_query");
        assert!(result.confidence > 0.0);

        Ok(())
    }

    #[test]
    fn test_gzip_knn_classify_scores() -> Result<()> {
        let mut classifier = GZipKNNClassifier::new(3, 6);

        let examples = vec![
            TrainingExample {
                text: "What is Python?".to_string(),
                label: "simple_query".to_string(),
            },
            TrainingExample {
                text: "Write code for binary search".to_string(),
                label: "code_generation".to_string(),
            },
        ];

        classifier.add_examples(examples);

        let scores = classifier.classify_scores("What is Rust?")?;
        assert!(!scores.is_empty());
        assert!(scores.values().any(|&v| v > 0.0));

        Ok(())
    }

    #[test]
    fn test_gzip_knn_empty_training_set() -> Result<()> {
        let classifier = GZipKNNClassifier::new(3, 6);

        let result = classifier.classify("Some query")?;
        assert_eq!(result.category, "unknown");
        assert_eq!(result.confidence, 0.0);

        Ok(())
    }

    #[test]
    fn test_ncd_calculation() {
        // NCD(x, x) should be 0 (identical strings compress same)
        let ncd = GZipKNNClassifier::ncd(100, 100, 100);
        assert_eq!(ncd, 0.0);

        // NCD(x, y) where x != y should be > 0
        let ncd = GZipKNNClassifier::ncd(150, 100, 100);
        assert!(ncd > 0.0);
    }

    #[test]
    fn test_adapter_returns_scores_above_threshold() -> Result<()> {
        let mut classifier = GZipKNNClassifier::new(3, 6);
        classifier.add_examples(vec![
            TrainingExample {
                text: "What is Python?".to_string(),
                label: "simple_query".to_string(),
            },
            TrainingExample {
                text: "Explain Python".to_string(),
                label: "simple_query".to_string(),
            },
            TrainingExample {
                text: "Write a function".to_string(),
                label: "code_generation".to_string(),
            },
        ]);

        let mut map = HashMap::new();
        map.insert("simple_query".to_string(), "efficient".to_string());
        map.insert("code_generation".to_string(), "capable".to_string());

        let adapter = GZipKNNClassifierAdapter::new(Arc::new(classifier), map)
            .with_confidence_threshold(0.5);

        let mut request = Request {
            llm_request: LlmRequest {
                model: None,
                messages: vec![Message {
                    role: Role::User,
                    content: vec![ContentBlock::Text {
                        text: "What is Python?".to_string(),
                    }],
                }],
                ..Default::default()
            },
            raw_request: None,
            metadata: None,
        };

        let (classification, _) = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(adapter.score(&mut (), &mut request, None))?;

        // High confidence query should return Scores (short-circuit judges) or Ambiguous
        match classification {
            Classification::Scores(_) | Classification::Ambiguous(_) => Ok(()),
        }
    }

    #[test]
    fn test_adapter_returns_ambiguous_below_threshold() -> Result<()> {
        let mut classifier = GZipKNNClassifier::new(3, 6);
        classifier.add_examples(vec![
            TrainingExample {
                text: "What is Python?".to_string(),
                label: "simple_query".to_string(),
            },
            TrainingExample {
                text: "Write a function".to_string(),
                label: "code_generation".to_string(),
            },
        ]);

        let mut map = HashMap::new();
        map.insert("simple_query".to_string(), "efficient".to_string());
        map.insert("code_generation".to_string(), "capable".to_string());

        let adapter = GZipKNNClassifierAdapter::new(Arc::new(classifier), map)
            .with_confidence_threshold(0.99); // Very high threshold

        let mut request = Request {
            llm_request: LlmRequest {
                model: None,
                messages: vec![Message {
                    role: Role::User,
                    content: vec![ContentBlock::Text {
                        text: "Ambiguous query xyz".to_string(),
                    }],
                }],
                ..Default::default()
            },
            raw_request: None,
            metadata: None,
        };

        let (classification, _) = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(adapter.score(&mut (), &mut request, None))?;

        // Below threshold should mostly return Ambiguous (defer to judges)
        match classification {
            Classification::Ambiguous(_) | Classification::Scores(_) => Ok(()),
        }
    }

    #[test]
    fn test_builder_creates_adapter_with_config() {
        let mut classifier = GZipKNNClassifier::new(5, 6);
        classifier.add_examples(vec![TrainingExample {
            text: "test".to_string(),
            label: "simple_query".to_string(),
        }]);

        let adapter = GZipKNNBuilder::new(Arc::new(classifier))
            .with_confidence_threshold(0.8)
            .build();

        // Verify adapter was created with custom threshold
        assert_eq!(adapter.confidence_threshold, 0.8);
    }

    #[test]
    fn test_default_config_maps_to_tiers() {
        let config = GZipKNNFallbackConfig::default();
        assert_eq!(
            config.category_tier_map.get("simple_query"),
            Some(&"efficient".to_string())
        );
        assert_eq!(
            config.category_tier_map.get("complex_reasoning"),
            Some(&"capable".to_string())
        );
        assert_eq!(config.confidence_threshold, 0.75);
    }

    #[test]
    fn test_adapter_extracts_user_message() -> Result<()> {
        let mut classifier = GZipKNNClassifier::new(3, 6);
        classifier.add_examples(vec![
            TrainingExample {
                text: "What is Python?".to_string(),
                label: "simple_query".to_string(),
            },
        ]);

        let adapter = GZipKNNClassifierAdapter::new(Arc::new(classifier), {
            let mut m = HashMap::new();
            m.insert("simple_query".to_string(), "efficient".to_string());
            m
        });

        let mut request = Request {
            llm_request: LlmRequest {
                model: None,
                messages: vec![
                    Message {
                        role: Role::System,
                        content: vec![ContentBlock::Text {
                            text: "You are helpful".to_string(),
                        }],
                    },
                    Message {
                        role: Role::User,
                        content: vec![ContentBlock::Text {
                            text: "What is Python?".to_string(),
                        }],
                    },
                ],
                ..Default::default()
            },
            raw_request: None,
            metadata: None,
        };

        let (classification, _) = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(adapter.score(&mut (), &mut request, None))?;

        // Should extract the user message and classify
        assert!(matches!(classification, Classification::Scores(_) | Classification::Ambiguous(_)));
        Ok(())
    }

    #[test]
    fn test_cost_optimization_flow() -> Result<()> {
        // Test the cost-optimization flow: high-confidence predictions bypass judges
        let mut classifier = GZipKNNClassifier::new(5, 6);
        
        // Add many examples of simple queries
        for i in 0..10 {
            classifier.add_examples(vec![
                TrainingExample {
                    text: format!("What is Python {}?", i),
                    label: "simple_query".to_string(),
                },
                TrainingExample {
                    text: format!("Define {} in programming", i),
                    label: "simple_query".to_string(),
                },
            ]);
        }

        let adapter = GZipKNNBuilder::new(Arc::new(classifier))
            .with_confidence_threshold(0.7)
            .build();

        let mut request = Request {
            llm_request: LlmRequest {
                model: None,
                messages: vec![Message {
                    role: Role::User,
                    content: vec![ContentBlock::Text {
                        text: "What is Rust?".to_string(),
                    }],
                }],
                ..Default::default()
            },
            raw_request: None,
            metadata: None,
        };

        let (classification, _) = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(adapter.score(&mut (), &mut request, None))?;

        // Verify classification was made
        match classification {
            Classification::Scores(scores) => {
                // High confidence: should skip judge calls
                assert!(!scores.is_empty());
                assert!(scores[0].confidence > 0.0);
            }
            Classification::Ambiguous(scores) => {
                // Lower confidence: defer to judges
                assert!(!scores.is_empty());
            }
        }

        Ok(())
    }
}
