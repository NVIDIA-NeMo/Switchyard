// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Retained selection between independently configured routing strategies.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use parking_lot::Mutex;
use switchyard_protocol::{Category, Request};
use tokio::sync::Mutex as AsyncMutex;

use super::util::affinity::{evict_if_full, retention_key};
use crate::core::algorithm::RoutingIdentity;
use crate::{
    Algorithm, AnswerSummary, Classifier, ClassifyTrigger, Driver, LibsyError, LlmClassifierConfig,
    LlmTaskClassifier, OutcomeMetadata, Result, RoutingOutcome, RuntimeModels, State, StateValue,
    TaskClassifierConfig,
};

const CHOICE: &str = "composite.selected_profile";

/// A strategy instance and the model groups visible only to that instance.
pub struct RoutingProfile {
    /// Strategy that executes requests selected for this profile.
    pub algorithm: Arc<dyn Algorithm>,
    /// Profile-local model pools, including fallback order and delegated work.
    pub models: RuntimeModels,
}

/// A binary judge maps its two outcomes to named profiles.
pub struct ProfileSelector {
    /// Judge settings and selection cadence.
    pub classifier: TaskClassifierConfig,
    /// Model bindings used only by the selector.
    pub models: RuntimeModels,
    /// Profile chosen by the capable verdict.
    pub capable: String,
    /// Profile chosen by the efficient verdict.
    pub efficient: String,
}

/// Runs the selected strategy without merging its state or fallback pools with others.
pub struct ProfileRouter {
    profiles: BTreeMap<String, (Arc<dyn Algorithm>, Arc<RuntimeModels>)>,
    default: String,
    judge: LlmTaskClassifier,
    judge_models: Arc<RuntimeModels>,
    capable: String,
    efficient: String,
    fail_open: bool,
    message_hash_fallback: bool,
    choices: Mutex<HashMap<RoutingIdentity, Arc<AsyncMutex<Option<String>>>>>,
}

impl ProfileRouter {
    /// Builds a retained selector over independently constructed profiles.
    pub fn new(
        profiles: BTreeMap<String, RoutingProfile>,
        default: String,
        mut selector: ProfileSelector,
    ) -> Result<Self> {
        for name in [&default, &selector.capable, &selector.efficient] {
            if !profiles.contains_key(name) {
                return Err(LibsyError::AlgorithmError {
                    message: format!("unknown routing profile {name}"),
                });
            }
        }
        if selector.classifier.classify_trigger != ClassifyTrigger::NewSession {
            return Err(LibsyError::AlgorithmError {
                message: "profile selection requires classify_trigger = new_session".into(),
            });
        }
        let fail_open = selector.classifier.fail_open;
        let message_hash_fallback = selector.classifier.message_hash_fallback;
        selector.classifier.classify_trigger = ClassifyTrigger::EveryRequest;
        selector.classifier.message_hash_fallback = false;
        Ok(Self {
            profiles: profiles
                .into_iter()
                .map(|(name, profile)| (name, (profile.algorithm, Arc::new(profile.models))))
                .collect(),
            default,
            judge: LlmTaskClassifier::new(LlmClassifierConfig::Capability {
                config: selector.classifier,
            })?,
            judge_models: Arc::new(selector.models),
            capable: selector.capable,
            efficient: selector.efficient,
            fail_open,
            message_hash_fallback,
            choices: Mutex::default(),
        })
    }

    async fn choose(
        &self,
        request: &mut Request,
        driver: &Driver,
        choice: &mut Option<String>,
    ) -> Result<String> {
        if choice.is_none() {
            let judge =
                driver.for_profile(String::new(), self.judge_models.clone(), State::default());
            let (scores, _) = self
                .judge
                .score(&mut State::default(), request, &judge)
                .await?;
            if let Some(score) = scores.argmax(false)? {
                *choice = match score.category {
                    Some(Category::Capable) => Some(self.capable.clone()),
                    Some(Category::Efficient) => Some(self.efficient.clone()),
                    _ => choice.take(),
                };
            }
            if choice.is_none() && !self.fail_open {
                return Err(LibsyError::AlgorithmError {
                    message: "profile selector returned no valid choice".into(),
                });
            }
        }
        Ok(choice.get_or_insert_with(|| self.default.clone()).clone())
    }
}

#[async_trait::async_trait]
impl Algorithm for ProfileRouter {
    fn name(&self) -> &str {
        "composite"
    }

    fn tracks_answers(&self) -> bool {
        self.profiles
            .values()
            .any(|(algorithm, _)| algorithm.tracks_answers())
    }

    async fn observe_answer(&self, session: &str, answer: AnswerSummary) {
        let choice = self
            .choices
            .lock()
            .get(&RoutingIdentity::Session(session.into()))
            .cloned();
        if let Some(choice) = choice {
            let selected = choice.lock().await.clone();
            if let Some((algorithm, _)) = selected.as_ref().and_then(|name| self.profiles.get(name))
            {
                algorithm.observe_answer(session, answer).await;
            }
        }
    }

    async fn route(
        self: Arc<Self>,
        driver: Driver,
        mut request: Request,
    ) -> Result<RoutingOutcome> {
        let identity = retention_key(&request, self.message_hash_fallback);
        let final_request = request.metadata.as_ref().and_then(|m| m.session_final) == Some(true);
        let mut state = driver.take_state();
        let selected = if let Some(identity) = &identity {
            let cell = {
                let mut choices = self.choices.lock();
                if !choices.contains_key(identity) {
                    evict_if_full(&mut choices);
                }
                choices.entry(identity.clone()).or_default().clone()
            };
            let mut choice = cell.lock().await;
            self.choose(&mut request, &driver, &mut choice).await
        } else {
            let mut choice = match state.extra.get(CHOICE) {
                Some(StateValue::String(name)) if self.profiles.contains_key(name) => {
                    Some(name.clone())
                }
                _ => None,
            };
            let result = self.choose(&mut request, &driver, &mut choice).await;
            if final_request {
                state.extra.remove(CHOICE);
            } else if let Some(choice) = choice {
                state
                    .extra
                    .insert(CHOICE.into(), StateValue::String(choice));
            }
            result
        };
        if final_request && let Some(identity) = identity {
            self.choices.lock().remove(&identity);
        }
        let selected = match selected {
            Ok(selected) => selected,
            Err(error) => {
                driver.set_state(state);
                return Err(error);
            }
        };
        let (algorithm, models) = &self.profiles[&selected];
        let prefix = format!("composite.profile.{}:{selected}.", selected.len());
        let child_state = State {
            extra: state
                .extra
                .iter()
                .filter_map(|(key, value)| {
                    key.strip_prefix(&prefix)
                        .map(|key| (key.to_string(), value.clone()))
                })
                .collect(),
            ..State::default()
        };
        let child = driver.for_profile(selected.clone(), models.clone(), child_state);
        let result = algorithm.clone().route(child.clone(), request).await;
        state.extra.retain(|key, _| !key.starts_with(&prefix));
        state.extra.extend(
            child
                .take_state()
                .extra
                .into_iter()
                .map(|(key, value)| (format!("{prefix}{key}"), value)),
        );
        driver.set_state(state);
        let mut outcome = result?;
        let metadata = outcome
            .metadata
            .get_or_insert_with(|| OutcomeMetadata::new("composite".into(), None));
        metadata.algorithm = "composite".into();
        metadata.profile = Some(selected);
        metadata.strategy = Some(algorithm.name().into());
        if outcome.answer_observer.is_none() && algorithm.tracks_answers() {
            outcome.answer_observer = Some(algorithm.clone());
        }
        Ok(outcome)
    }
}

#[cfg(test)]
mod tests;
