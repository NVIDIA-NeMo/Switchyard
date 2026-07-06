// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Switchyard library crate.

pub mod llm_class;
pub mod rand;

use async_trait::async_trait;
use std::collections::BTreeMap;
use std::error::Error;

/// AgentApiRequest internal representation of an llm request.
/// This structure is designed to be converted to a from provider specifiic structs without loosing
/// information
pub struct AgentApitRequest {
    pub prompt: String,
    pub model: String,
}

/// AgentApiRequest internal representation of an llm response.
/// This structure is designed to be converted to a from provider specifiic structs without loosing
/// information
pub struct AgentApiResponse {
    pub completion: String,
}

/// enrichement and correlation data use for routing the request
pub struct EnrichementData {
    pub session_id: Option<String>,
    pub agent_id: Option<String>,
    pub task_id: Option<String>,
    pub correlation_id: Option<String>,
    pub extra_metadata: Option<BTreeMap<String, String>>,
}

/// A Routing / optimization decision made by the AgentApiOptimizer
pub enum Decision<D> {
    ModelInference(AgentApiOptimizerResponse<D>),
    Return(),
}

/// Input to the AgentApiOptimizer, can be a request, response or metadata
pub enum AgentApiOptInput {
    Request(ChatRequest),
    Response(ChatRequest),
    Metadata(BTreeMap<String, String>),
}

/// The response from the AgentApiOptimizer, containing the optimized requests and any additional
/// data.
pub struct AgentApiOptimizerResponse<D> {
    pub requests: Vec<ChatRequest>,
    pub enrichment_data: Vec<EnrichementData>,
    pub decision_reasoning: Option<String>,
    pub decision_info: Option<D>,
}

/// AgentApiOptimizer is a stateful optimizer that is feed requests, responses and metadata and can
/// make routing / optimization decisions based on the input and the current state.
/// A single instance of an AgentApiOptimizer is created for each session and is used to optimize
/// the requests for that session.
#[async_trait]
pub trait AgentApiOptimizer<D>: Send + Sync {
    /// Feed the optimizer with a new input and enrichment data. The optimizer can use this data to
    async fn feed(
        &mut self,
        _input: AgentApiOptInput,
        _enrichment: EnrichementData,
    ) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    /// Make a routing / optimization decision based on the current state of the optimizer.
    /// Return Types:
    ///   ModelInference: The optimizer has decided to make a model inference request. The caller
    ///     should make the listed model calls and pass the responses back to the optimizer via
    ///     feed() for further optimization.
    ///  Return: The optimizer has decided to return a response to the agent (e.g. for tool
    ///    excution) the caller should pass control to the calling agent.
    async fn optimize(&mut self) -> Result<Decision<D>, Box<dyn Error>> {
        Ok(Decision::ModelInference(AgentApiOptimizerResponse {
            requests: Vec::new(),
            enrichment_data: Vec::new(),
            decision_reasoning: None,
            decision_info: None,
        }))
    }
}

/// AgentApiOptAlgorithm is a factory for creating instances of AgentApiOptimizer. It is used to
/// create a new optimizer for each session.
pub trait AgentApiOptAlgorithm<D>: Send + Sync {
    /// Create a new instance of the AgentApiOptimizer for the given session.
    fn optimizer(&self) -> Box<dyn AgentApiOptimizer<D>>;
}
