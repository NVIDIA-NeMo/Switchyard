// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The `capability_router` route: composes named signals into one pool-resolved decision.
//!
//! Each configured signal (`task`, `rules`, `tool_loop`) scores the turn independently — it
//! names what the turn needs (a capability tag), not which model serves it. Every
//! signal's score is collected into one set and handed to
//! [`pool::select`](crate::core::pool::select), which picks the model. This replaces
//! the old pattern of one algorithm stacking another and writing a tier into its
//! session state: signals sit side by side, and the pool is the only place that
//! knows the catalog.
//!
//! `review` wraps the whole turn in an advisor gate instead of contributing a score:
//! the gate's own executor selection (`Category::Efficient`) happens before a model
//! is even chosen, so it cannot compose with `task`/`rules`/`tool_loop` in this
//! release — configuring it alongside any of them is rejected at construction.
//!
//! Affinity is `capability_router`'s own, and it bypasses the pool entirely: a retained
//! assignment from an earlier turn in the session is replayed outright, with no
//! signal scoring and no call to `select`. This is also why a signal's own
//! session-retention settings (e.g. `task`'s `classify_trigger`) are forced off —
//! one route should have one place deciding how long a decision lives.
//!
//! Optional escalation retention keeps balanced or capable as a temporary floor
//! for later requests with the same explicit session identity. It does not pin
//! the efficient tier or prevent a stronger selection.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use switchyard_protocol::{Category, ModelId, Request, Role};

use super::advisor_gate::{AdvisorGate, AdvisorGateConfig};
use super::llm_class::{LlmClassifierConfig, LlmTaskClassifier, TaskClassifierConfig};
use super::util::affinity::{AffinityRouter, ClassifyTrigger, evict_if_full, has_new_user_turn};
use super::util::stage::{HandoffNoteConfig, PickerMode, StageClassifier};
use super::util::tool_signals::{DEFAULT_RECENT_WINDOW, ToolSemantics, ToolSignalProcessor};
use crate::core::algorithm::{Algorithm, Driver, RoutingIdentity};
use crate::core::classifier::{Classifier, Score};
use crate::core::pool::{self, PoolContext};
use crate::core::processor::{Event, Processor};
use crate::core::state::State;
use crate::{LibsyError, Result, RoutingOutcome};

const CAPABILITY_ROUTER: &str = "capability_router";
const BALANCED: &str = "balanced";
const CAPABLE_SCORE_THRESHOLD: f64 = 0.55;

/// Tool-loop signal settings: the stage-router scorer, without its own judge
/// fallback or per-tier prompt injection (those are a route-wide concern once
/// `task` is a sibling signal rather than something `tool_loop` falls through to).
pub struct ToolLoopSignalConfig {
    /// Tier a turn defaults to when the scorer is not confident.
    pub mode: PickerMode,
    /// How much corroboration a decisive pick needs, in `[0.0, 1.0]`.
    pub confidence_threshold: f64,
    /// Trailing tool results the signal is computed over. `None` uses
    /// [`DEFAULT_RECENT_WINDOW`].
    pub recent_window: Option<usize>,
    /// Exact tool-name semantics added to the built-in coding vocabulary.
    pub tool_semantics: ToolSemantics,
    /// Note handed to the model on a signal-driven escalation or hand-back.
    pub handoff_notes: Option<HandoffNoteConfig>,
}

/// Deterministic terms matched case-insensitively against the opening user task.
pub struct RuleSignalConfig {
    /// Terms that require the capable tier.
    pub capable: Vec<String>,
    /// Terms that require the balanced tier.
    pub balanced: Vec<String>,
    /// Deployment-defined capability names and the terms that require them.
    pub named: BTreeMap<String, Vec<String>>,
}

/// One `capability_router` route's configured signals and the pool they resolve against.
pub struct CapabilityRouterConfig {
    /// The deployment's model catalog this route resolves needs against.
    pub pool: PoolContext,
    /// Judge-backed capability signal.
    pub task: Option<TaskClassifierConfig>,
    /// Deterministic opening-task text signal.
    pub rules: Option<RuleSignalConfig>,
    /// Tool-result signal.
    pub tool_loop: Option<ToolLoopSignalConfig>,
    /// Advisor review. Mutually exclusive with `task`, `rules`, and `tool_loop`.
    pub review: Option<AdvisorGateConfig>,
    /// Retains the first model chosen for a session and replays it on later
    /// turns, bypassing signal scoring and the pool entirely.
    pub affinity: bool,
}

struct ToolLoopSignal {
    processor: ToolSignalProcessor,
    classifier: StageClassifier,
}

struct RuleSignal {
    capable: Vec<String>,
    balanced: Vec<String>,
    named: Vec<(Category, Vec<String>)>,
}

impl RuleSignal {
    fn new(config: RuleSignalConfig, pool: &PoolContext) -> Result<Self> {
        let normalize = |terms: Vec<String>| {
            terms
                .into_iter()
                .map(|term| term.trim().to_lowercase())
                .filter(|term| !term.is_empty())
                .collect()
        };
        let mut named = Vec::with_capacity(config.named.len());
        for (name, terms) in config.named {
            let category =
                name.parse::<Category>()
                    .map_err(|message| LibsyError::AlgorithmError {
                        message: format!("invalid rule capability {name:?}: {message}"),
                    })?;
            named.push((category, normalize(terms)));
        }
        let signal = Self {
            capable: normalize(config.capable),
            balanced: normalize(config.balanced),
            named,
        };

        let balanced = BALANCED
            .parse::<Category>()
            .expect("the built-in balanced tag is valid");
        let mut required = signal
            .named
            .iter()
            .filter(|(_, terms)| !terms.is_empty())
            .map(|(category, _)| category.clone())
            .collect::<Vec<_>>();
        if !signal.capable.is_empty() {
            required.push(Category::Capable);
        }
        if !signal.balanced.is_empty() {
            required.push(balanced);
        }
        for category in required {
            if !pool
                .models
                .iter()
                .any(|model| model.provides.contains(&category))
            {
                return Err(LibsyError::AlgorithmError {
                    message: format!(
                        "capability_router: rule capability {category:?} is not provided by any pool model"
                    ),
                });
            }
        }

        Ok(signal)
    }

    fn score(&self, request: &Request) -> Option<Score> {
        let opening_task = request
            .llm_request
            .messages
            .iter()
            .find(|message| message.role == Role::User)?
            .text_content("\n")?
            .to_lowercase();

        let needs = if let Some((category, _)) = self
            .named
            .iter()
            .find(|(_, terms)| terms.iter().any(|term| opening_task.contains(term)))
        {
            vec![category.clone()]
        } else if self.capable.iter().any(|term| opening_task.contains(term)) {
            vec![Category::Capable]
        } else if self.balanced.iter().any(|term| opening_task.contains(term)) {
            vec![
                BALANCED
                    .parse()
                    .expect("the built-in balanced tag is valid"),
            ]
        } else {
            return None;
        };

        Some(Score::needing(needs, 1.0))
    }
}

/// Composes `task`/`rules`/`tool_loop` signals into a shared [`PoolContext`], or
/// defers the whole turn to `review`. See the module docs for how the pieces fit
/// together.
pub struct CapabilityRouter {
    pool: PoolContext,
    task: Option<Arc<LlmTaskClassifier>>,
    rules: Option<RuleSignal>,
    tool_loop: Option<ToolLoopSignal>,
    review: Option<Arc<AdvisorGate>>,
    affinity: Option<Arc<AffinityRouter>>,
    escalation_hold_turns: u32,
    escalation_holds: Mutex<HashMap<RoutingIdentity, (ModelId, u32)>>,
    capable_score_threshold: f64,
}

impl CapabilityRouter {
    /// Builds the route described by `config`.
    ///
    /// # Errors
    ///
    /// Errors when the pool is empty, when no signal is configured, when `review`
    /// is combined with another signal, or when a signal's own settings are invalid.
    pub fn new(config: CapabilityRouterConfig) -> Result<Self> {
        if config.pool.models.is_empty() {
            return Err(LibsyError::AlgorithmError {
                message: "capability_router: pool must declare at least one model".to_string(),
            });
        }
        if !config
            .pool
            .models
            .iter()
            .any(|model| model.provides.contains(&config.pool.default))
        {
            return Err(LibsyError::AlgorithmError {
                message: format!(
                    "capability_router: pool default {:?} is not provided by any pool model",
                    config.pool.default
                ),
            });
        }
        let has_review = config.review.is_some();
        let has_signals =
            config.task.is_some() || config.rules.is_some() || config.tool_loop.is_some();
        if has_review && has_signals {
            return Err(LibsyError::AlgorithmError {
                message:
                    "capability_router: review cannot be combined with task, rules, or tool_loop in this release"
                        .to_string(),
            });
        }
        if !has_review && !has_signals {
            return Err(LibsyError::AlgorithmError {
                message: "capability_router: at least one of task, rules, tool_loop, or review is required"
                    .to_string(),
            });
        }

        let task = config
            .task
            .map(|task_config| {
                LlmTaskClassifier::new(LlmClassifierConfig::Capability {
                    config: TaskClassifierConfig {
                        classify_trigger: ClassifyTrigger::EveryRequest,
                        message_hash_fallback: false,
                        ..task_config
                    },
                })
            })
            .transpose()?
            .map(Arc::new);

        let rules = config
            .rules
            .map(|rules| RuleSignal::new(rules, &config.pool))
            .transpose()?;
        let tool_loop = config.tool_loop.map(build_tool_loop_signal).transpose()?;

        let review = config
            .review
            .map(AdvisorGate::new)
            .transpose()?
            .map(Arc::new);
        let affinity = config.affinity.then(|| Arc::new(AffinityRouter::new()));

        Ok(Self {
            pool: config.pool,
            task,
            rules,
            tool_loop,
            review,
            affinity,
            escalation_hold_turns: 0,
            escalation_holds: Mutex::default(),
            capable_score_threshold: CAPABLE_SCORE_THRESHOLD,
        })
    }

    /// Keeps an elevated tier for this many following requests in its session.
    /// Zero disables retention. Requests without a stable identity do not retain.
    pub fn with_escalation_hold_turns(mut self, turns: u32) -> Self {
        self.escalation_hold_turns = turns;
        self
    }

    /// Sets the tool-loop confidence that requires capable rather than balanced.
    /// Pools without a balanced provider keep their two-tier behavior.
    pub fn with_capable_score_threshold(mut self, threshold: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(LibsyError::AlgorithmError {
                message: "capable_score_threshold must be between 0 and 1".to_string(),
            });
        }
        self.capable_score_threshold = threshold;
        Ok(self)
    }

    fn tier_rank(&self, target: &ModelId) -> Option<u8> {
        self.pool
            .models
            .iter()
            .find(|model| model.target == *target)?
            .provides
            .iter()
            .filter_map(|tag| match tag.as_str() {
                "efficient" => Some(0),
                "balanced" => Some(1),
                "capable" => Some(2),
                _ => None,
            })
            .max()
    }

    fn retain_escalation(&self, request: &Request, target: ModelId) -> ModelId {
        if self.escalation_hold_turns == 0 {
            return target;
        }
        let Some(key) = RoutingIdentity::from_request(request) else {
            return target;
        };
        let mut holds = self.escalation_holds.lock();
        if has_new_user_turn(&request.llm_request.messages) {
            holds.remove(&key);
        }
        // Named specialists stay under their own routing rules.
        let Some(rank) = self.tier_rank(&target) else {
            holds.remove(&key);
            return target;
        };
        if let Some((held, remaining)) = holds.get(&key).cloned()
            && remaining > 0
            && self
                .tier_rank(&held)
                .is_some_and(|held_rank| held_rank > rank)
        {
            if remaining == 1 {
                holds.remove(&key);
            } else {
                holds.insert(key, (held.clone(), remaining - 1));
            }
            return held;
        }
        if rank > 0 {
            evict_if_full(&mut holds);
            holds.insert(key, (target.clone(), self.escalation_hold_turns));
        } else {
            holds.remove(&key);
        }
        target
    }

    /// The pool model named `name`, if any — how an affinity pin (a literal model
    /// id) is checked against the catalog `select` would otherwise resolve against.
    fn pool_model_named(&self, name: &str) -> Option<ModelId> {
        self.pool
            .models
            .iter()
            .find(|model| model.target.as_str() == name)
            .map(|model| model.target.clone())
    }

    /// Every other pool model, in declared order — the candidates a client falls
    /// through to when `target` fails.
    fn fallback_models(&self, target: &ModelId) -> Vec<ModelId> {
        self.pool
            .models
            .iter()
            .map(|model| &model.target)
            .filter(|candidate| *candidate != target)
            .cloned()
            .collect()
    }

    /// Maps moderate positive tool-loop evidence onto the optional middle tier.
    /// Pools without a `balanced` provider keep the legacy two-tier behavior.
    fn apply_three_level_band(&self, mut score: Score) -> Score {
        let balanced: switchyard_protocol::Category = BALANCED
            .parse()
            .expect("the built-in balanced tag is valid");
        let has_balanced_provider = self
            .pool
            .models
            .iter()
            .any(|model| model.provides.contains(&balanced));

        if has_balanced_provider
            && score.needs == vec![switchyard_protocol::Category::Capable]
            && score.confidence < self.capable_score_threshold
        {
            score.needs = vec![balanced];
        }
        score
    }

    /// The model affinity is still pinning for this request, if any and if it
    /// still names a model in the pool.
    async fn affinity_pin(
        &self,
        affinity: &AffinityRouter,
        state: &mut State,
        request: &mut Request,
        driver: &Driver,
    ) -> Result<Option<ModelId>> {
        let (classification, _) = affinity.score(state, request, driver).await?;
        let Some(score) = classification.argmax(false)? else {
            return Ok(None);
        };
        Ok(score
            .pin
            .as_ref()
            .and_then(|target| self.pool_model_named(target.as_str())))
    }

    /// Lets affinity observe the turn's decision, so it retains it for later turns.
    async fn replay_decision(
        affinity: &AffinityRouter,
        state: &mut State,
        request: &mut Request,
        target: &ModelId,
        driver: &Driver,
    ) -> Result<()> {
        affinity
            .process(
                state,
                Event::Decision {
                    request,
                    selected_model_id: target,
                    category: None,
                    driver,
                },
            )
            .await
    }
}

fn build_tool_loop_signal(config: ToolLoopSignalConfig) -> Result<ToolLoopSignal> {
    config.tool_semantics.validate()?;
    if !(0.0..=1.0).contains(&config.confidence_threshold) {
        return Err(LibsyError::AlgorithmError {
            message: format!(
                "confidence_threshold must be between 0 and 1, got {}",
                config.confidence_threshold
            ),
        });
    }
    let mut classifier = StageClassifier::new(config.mode, config.confidence_threshold);
    if let Some(notes) = config.handoff_notes {
        classifier = classifier.with_handoff_notes(notes);
    }
    Ok(ToolLoopSignal {
        processor: ToolSignalProcessor {
            recent_window: config.recent_window.unwrap_or(DEFAULT_RECENT_WINDOW),
            tool_semantics: config.tool_semantics,
        },
        classifier,
    })
}

#[async_trait]
impl Algorithm for CapabilityRouter {
    fn name(&self) -> &str {
        CAPABILITY_ROUTER
    }

    async fn route(
        self: Arc<Self>,
        driver: Driver,
        mut request: Request,
    ) -> Result<RoutingOutcome> {
        if let Some(review) = &self.review {
            return review.clone().route(driver, request).await;
        }

        let mut state = State::default();

        if let Some(affinity) = &self.affinity
            && let Some(target) = self
                .affinity_pin(affinity, &mut state, &mut request, &driver)
                .await?
        {
            Self::replay_decision(affinity, &mut state, &mut request, &target, &driver).await?;
            let fallback = self.fallback_models(&target);
            return Ok(RoutingOutcome::route_to(target, fallback, request));
        }

        let mut scores = Vec::new();
        if let Some(rules) = &self.rules
            && let Some(score) = rules.score(&request)
        {
            scores.push(score);
        }
        if let Some(tool_loop) = &self.tool_loop {
            tool_loop
                .processor
                .process(
                    &mut state,
                    Event::Request {
                        request: &mut request,
                        driver: &driver,
                    },
                )
                .await?;
            let (classification, _) = tool_loop
                .classifier
                .score(&mut state, &mut request, &driver)
                .await?;
            if let Some(score) = classification.argmax(false)? {
                scores.push(self.apply_three_level_band(score));
            }
        }
        if let Some(task) = &self.task {
            let (classification, _) = task.score(&mut state, &mut request, &driver).await?;
            if let Some(score) = classification.argmax(false)? {
                scores.push(score);
            }
        }

        let target = self.retain_escalation(&request, pool::select(&scores, &self.pool)?);

        if let Some(affinity) = &self.affinity {
            Self::replay_decision(affinity, &mut state, &mut request, &target, &driver).await?;
        }

        let fallback = self.fallback_models(&target);
        Ok(RoutingOutcome::route_to(target, fallback, request))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};

    use serde_json::json;
    use switchyard_protocol::{Category, ContentBlock, Metadata};

    use super::*;
    use crate::algorithms::util::stage::PickerMode;
    use crate::algorithms::util::tier_fixtures::{JUDGE, Recorder, turn_request};
    use crate::core::testing::test_drive_with_models;

    fn pool(models: &[(&str, &[&str])], default: &str) -> PoolContext {
        PoolContext {
            models: models
                .iter()
                .map(|(target, provides)| pool::PoolModel {
                    target: ModelId::from(*target),
                    provides: provides
                        .iter()
                        .map(|tag| tag.parse::<Category>().unwrap())
                        .collect(),
                })
                .collect(),
            default: default.parse().unwrap(),
        }
    }

    fn runtime_models() -> HashMap<Category, Vec<ModelId>> {
        [
            (Category::Judge, vec![ModelId::from(JUDGE)]),
            (Category::Efficient, vec![ModelId::from("weak")]),
            (Category::Capable, vec![ModelId::from("strong")]),
            (
                Category::Any,
                vec![ModelId::from("strong"), ModelId::from("weak")],
            ),
        ]
        .into()
    }

    fn three_level_runtime_models() -> HashMap<Category, Vec<ModelId>> {
        let mut models = runtime_models();
        models.insert("balanced".parse().unwrap(), vec![ModelId::from("balanced")]);
        models
            .entry(Category::Any)
            .or_default()
            .push(ModelId::from("balanced"));
        models
    }

    fn soft_failure_request() -> Request {
        let mut request = turn_request(false);
        let ContentBlock::ToolResult(result) = &mut request.llm_request.messages[2].content[0]
        else {
            panic!("fixture must end with a tool result");
        };
        result.content = vec![ContentBlock::Text {
            text: "exit code 1".to_string(),
        }];
        result.is_error = None;
        request.raw_request = Some(json!({
            "model": "auto",
            "messages": [
                {"role": "user", "content": "fix the build"},
                {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function",
                    "function": {"name": "Bash", "arguments": "{\"command\": \"cargo test\"}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "exit code 1"},
            ],
        }));
        request
    }

    #[test]
    fn capable_threshold_rejects_values_outside_the_confidence_range() -> Result<()> {
        for threshold in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
            let router = Arc::into_inner(escalation_router(0)?).unwrap();
            assert!(router.with_capable_score_threshold(threshold).is_err());
        }
        Ok(())
    }

    #[test]
    fn capable_threshold_can_promote_moderate_evidence_without_a_judge() -> Result<()> {
        let router = CapabilityRouter::new(CapabilityRouterConfig {
            pool: pool(
                &[
                    ("strong", &["capable"]),
                    ("balanced", &["balanced"]),
                    ("weak", &["efficient"]),
                ],
                "efficient",
            ),
            task: None,
            rules: Some(RuleSignalConfig {
                capable: vec![],
                balanced: vec![],
                named: BTreeMap::new(),
            }),
            tool_loop: None,
            review: None,
            affinity: false,
        })?
        .with_capable_score_threshold(0.35)?;
        assert_eq!(
            router
                .apply_three_level_band(Score {
                    confidence: 0.40,
                    needs: vec![Category::Capable],
                    pin: None,
                })
                .needs,
            vec![Category::Capable]
        );
        assert_eq!(
            router
                .apply_three_level_band(Score {
                    confidence: 0.30,
                    needs: vec![Category::Capable],
                    pin: None,
                })
                .needs,
            vec!["balanced".parse::<Category>().unwrap()]
        );
        Ok(())
    }

    #[test]
    fn rejects_an_empty_pool() {
        assert!(matches!(
            CapabilityRouter::new(CapabilityRouterConfig {
                pool: pool(&[], "capable"),
                task: None,
                rules: None,
                tool_loop: Some(ToolLoopSignalConfig {
                    mode: PickerMode::EfficientFirst,
                    confidence_threshold: 0.5,
                    recent_window: None,
                    tool_semantics: ToolSemantics::default(),
                    handoff_notes: None,
                }),
                review: None,
                affinity: false,
            }),
            Err(LibsyError::AlgorithmError { .. })
        ));
    }

    /// A default no model provides would only surface later, as a failed request on the
    /// first turn no signal scores. The runner's `validate_pool` rejects it; so does this.
    #[test]
    fn rejects_a_pool_default_no_model_provides() {
        let error = CapabilityRouter::new(CapabilityRouterConfig {
            pool: pool(&[("sol", &["capable"])], "efficient"),
            task: None,
            rules: None,
            tool_loop: Some(ToolLoopSignalConfig {
                mode: PickerMode::EfficientFirst,
                confidence_threshold: 0.5,
                recent_window: None,
                tool_semantics: ToolSemantics::default(),
                handoff_notes: None,
            }),
            review: None,
            affinity: false,
        });
        let Err(error) = error else {
            panic!("a default with no provider must not build");
        };
        assert!(
            error.to_string().contains("pool default"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn rejects_no_signals() {
        assert!(matches!(
            CapabilityRouter::new(CapabilityRouterConfig {
                pool: pool(&[("sol", &["capable"])], "capable"),
                task: None,
                rules: None,
                tool_loop: None,
                review: None,
                affinity: false,
            }),
            Err(LibsyError::AlgorithmError { .. })
        ));
    }

    #[test]
    fn rejects_a_rule_capability_without_a_pool_provider() {
        assert!(matches!(
            CapabilityRouter::new(CapabilityRouterConfig {
                pool: pool(&[("weak", &["efficient"])], "efficient"),
                task: None,
                rules: Some(RuleSignalConfig {
                    capable: Vec::new(),
                    balanced: vec!["multi-step".to_string()],
                    named: BTreeMap::new(),
                }),
                tool_loop: None,
                review: None,
                affinity: false,
            }),
            Err(LibsyError::AlgorithmError { message })
                if message.contains("balanced") && message.contains("not provided")
        ));
    }

    #[test]
    fn rejects_review_combined_with_a_signal() {
        assert!(matches!(
            CapabilityRouter::new(CapabilityRouterConfig {
                pool: pool(&[("sol", &["capable"])], "capable"),
                task: None,
                rules: None,
                tool_loop: Some(ToolLoopSignalConfig {
                    mode: PickerMode::EfficientFirst,
                    confidence_threshold: 0.5,
                    recent_window: None,
                    tool_semantics: ToolSemantics::default(),
                    handoff_notes: None,
                }),
                review: Some(AdvisorGateConfig::default()),
                affinity: false,
            }),
            Err(LibsyError::AlgorithmError { .. })
        ));
    }

    #[tokio::test]
    async fn tool_loop_resolves_through_the_pool() -> Result<()> {
        let router = Arc::new(CapabilityRouter::new(CapabilityRouterConfig {
            pool: pool(
                &[("strong", &["capable"]), ("weak", &["efficient"])],
                "efficient",
            ),
            task: None,
            rules: None,
            tool_loop: Some(ToolLoopSignalConfig {
                mode: PickerMode::EfficientFirst,
                confidence_threshold: 0.5,
                recent_window: None,
                tool_semantics: ToolSemantics::default(),
                handoff_notes: None,
            }),
            review: None,
            affinity: false,
        })?);

        let recorder = Arc::new(Recorder::default());
        let (selected_model, _) = test_drive_with_models(
            router,
            turn_request(true),
            runtime_models(),
            recorder.serve(),
        )
        .await?;
        assert_eq!(selected_model, "strong");
        Ok(())
    }

    #[tokio::test]
    async fn quiet_signals_fall_back_to_the_pool_default() -> Result<()> {
        let router = Arc::new(CapabilityRouter::new(CapabilityRouterConfig {
            pool: pool(
                &[("strong", &["capable"]), ("weak", &["efficient"])],
                "efficient",
            ),
            task: None,
            rules: None,
            tool_loop: Some(ToolLoopSignalConfig {
                mode: PickerMode::EfficientFirst,
                confidence_threshold: 0.5,
                recent_window: None,
                tool_semantics: ToolSemantics::default(),
                handoff_notes: None,
            }),
            review: None,
            affinity: false,
        })?);

        let recorder = Arc::new(Recorder::default());
        let (selected_model, _) = test_drive_with_models(
            router,
            turn_request(false),
            runtime_models(),
            recorder.serve(),
        )
        .await?;
        assert_eq!(selected_model, "weak");
        Ok(())
    }

    fn escalation_router(turns: u32) -> Result<Arc<CapabilityRouter>> {
        Ok(Arc::new(
            CapabilityRouter::new(CapabilityRouterConfig {
                pool: pool(
                    &[
                        ("strong", &["capable"]),
                        ("balanced", &["balanced"]),
                        ("weak", &["efficient"]),
                    ],
                    "efficient",
                ),
                task: None,
                rules: None,
                tool_loop: Some(ToolLoopSignalConfig {
                    mode: PickerMode::EfficientFirst,
                    confidence_threshold: 0.20,
                    recent_window: None,
                    tool_semantics: ToolSemantics::default(),
                    handoff_notes: None,
                }),
                review: None,
                affinity: false,
            })?
            .with_escalation_hold_turns(turns),
        ))
    }

    #[tokio::test]
    async fn escalation_hold_keeps_balanced_for_three_following_requests() -> Result<()> {
        let router = escalation_router(3)?;
        let recorder = Arc::new(Recorder::default());
        for (session, request, expected) in [
            ("a", soft_failure_request(), "balanced"),
            ("b", turn_request(false), "weak"),
            ("a", turn_request(false), "balanced"),
            ("a", turn_request(false), "balanced"),
            ("a", turn_request(false), "balanced"),
            ("a", turn_request(false), "weak"),
            ("a", soft_failure_request(), "balanced"),
            ("a", turn_request(true), "strong"),
            ("a", turn_request(false), "strong"),
            ("a", turn_request(false), "strong"),
            ("a", turn_request(false), "strong"),
            ("a", turn_request(false), "weak"),
        ] {
            let request = Request {
                metadata: Some(Metadata {
                    session_id: Some(session.to_string()),
                    ..Metadata::default()
                }),
                ..request
            };
            let (selected, _) = test_drive_with_models(
                router.clone(),
                request,
                three_level_runtime_models(),
                recorder.serve(),
            )
            .await?;
            assert_eq!(selected, expected);
        }
        Ok(())
    }

    #[tokio::test]
    async fn escalation_hold_never_pins_unkeyed_requests_or_the_efficient_tier() -> Result<()> {
        let router = escalation_router(3)?;
        let recorder = Arc::new(Recorder::default());
        for (request, expected) in [
            (soft_failure_request(), "balanced"),
            (turn_request(false), "weak"),
        ] {
            let request = Request {
                metadata: None,
                ..request
            };
            let (selected, _) = test_drive_with_models(
                router.clone(),
                request,
                three_level_runtime_models(),
                recorder.serve(),
            )
            .await?;
            assert_eq!(selected, expected);
        }
        Ok(())
    }

    #[tokio::test]
    async fn zero_disables_escalation_retention() -> Result<()> {
        let router = escalation_router(0)?;
        let recorder = Arc::new(Recorder::default());
        for (request, expected) in [
            (soft_failure_request(), "balanced"),
            (turn_request(false), "weak"),
        ] {
            let request = Request {
                metadata: Some(Metadata {
                    session_id: Some("a".into()),
                    ..Metadata::default()
                }),
                ..request
            };
            let (selected, _) = test_drive_with_models(
                router.clone(),
                request,
                three_level_runtime_models(),
                recorder.serve(),
            )
            .await?;
            assert_eq!(selected, expected);
        }
        Ok(())
    }

    #[test]
    fn escalation_hold_isolates_children_and_releases_on_a_new_human_turn() -> Result<()> {
        let router = escalation_router(3)?;
        let parent = Request {
            metadata: Some(Metadata {
                session_id: Some("shared-session".into()),
                ..Metadata::default()
            }),
            ..turn_request(false)
        };
        assert_eq!(
            router.retain_escalation(&parent, "balanced".into()),
            "balanced"
        );
        let child = Request {
            metadata: Some(Metadata {
                session_id: Some("shared-session".into()),
                agent_id: Some("child".into()),
                is_subagent: true,
                ..Metadata::default()
            }),
            ..turn_request(false)
        };
        assert_eq!(router.retain_escalation(&child, "weak".into()), "weak");
        assert_eq!(router.retain_escalation(&parent, "weak".into()), "balanced");
        let mut next_task = parent;
        next_task.llm_request.messages.truncate(1);
        assert_eq!(router.retain_escalation(&next_task, "weak".into()), "weak");
        Ok(())
    }

    #[tokio::test]
    async fn three_level_tool_loop_routes_a_soft_failure_to_balanced() -> Result<()> {
        let router = Arc::new(CapabilityRouter::new(CapabilityRouterConfig {
            pool: pool(
                &[
                    ("strong", &["capable"]),
                    ("balanced", &["balanced"]),
                    ("weak", &["efficient"]),
                ],
                "efficient",
            ),
            task: None,
            rules: None,
            tool_loop: Some(ToolLoopSignalConfig {
                mode: PickerMode::EfficientFirst,
                confidence_threshold: 0.20,
                recent_window: None,
                tool_semantics: ToolSemantics::default(),
                handoff_notes: None,
            }),
            review: None,
            affinity: false,
        })?);

        let recorder = Arc::new(Recorder::default());
        let (selected_model, _) = test_drive_with_models(
            router,
            soft_failure_request(),
            three_level_runtime_models(),
            recorder.serve(),
        )
        .await?;

        assert_eq!(selected_model, "balanced");
        Ok(())
    }

    #[tokio::test]
    async fn deterministic_task_rule_routes_matching_opening_task_to_capable() -> Result<()> {
        let router = Arc::new(CapabilityRouter::new(CapabilityRouterConfig {
            pool: pool(
                &[
                    ("strong", &["capable"]),
                    ("balanced", &["balanced"]),
                    ("weak", &["efficient"]),
                ],
                "efficient",
            ),
            task: None,
            rules: Some(RuleSignalConfig {
                capable: vec!["historical tie-break".to_string()],
                balanced: vec!["multi-step".to_string()],
                named: BTreeMap::new(),
            }),
            tool_loop: None,
            review: None,
            affinity: false,
        })?);
        let mut request = turn_request(false);
        request.llm_request.messages[0].content = vec![ContentBlock::Text {
            text: "Preserve the HISTORICAL TIE-BREAK behavior.".to_string(),
        }];

        let recorder = Arc::new(Recorder::default());
        let (selected_model, _) = test_drive_with_models(
            router,
            request,
            three_level_runtime_models(),
            recorder.serve(),
        )
        .await?;

        assert_eq!(selected_model, "strong");
        Ok(())
    }

    #[tokio::test]
    async fn deterministic_task_rule_routes_matching_opening_task_to_balanced() -> Result<()> {
        let router = Arc::new(CapabilityRouter::new(CapabilityRouterConfig {
            pool: pool(
                &[
                    ("strong", &["capable"]),
                    ("balanced", &["balanced"]),
                    ("weak", &["efficient"]),
                ],
                "efficient",
            ),
            task: None,
            rules: Some(RuleSignalConfig {
                capable: vec!["historical tie-break".to_string()],
                balanced: vec!["multi-step".to_string()],
                named: BTreeMap::new(),
            }),
            tool_loop: None,
            review: None,
            affinity: false,
        })?);
        let mut request = turn_request(false);
        request.llm_request.messages[0].content = vec![ContentBlock::Text {
            text: "Complete this MULTI-STEP migration.".to_string(),
        }];

        let recorder = Arc::new(Recorder::default());
        let (selected_model, _) = test_drive_with_models(
            router,
            request,
            three_level_runtime_models(),
            recorder.serve(),
        )
        .await?;

        assert_eq!(selected_model, "balanced");
        Ok(())
    }

    #[tokio::test]
    async fn deterministic_task_rules_abstain_when_opening_task_does_not_match() -> Result<()> {
        let router = Arc::new(CapabilityRouter::new(CapabilityRouterConfig {
            pool: pool(
                &[
                    ("strong", &["capable"]),
                    ("balanced", &["balanced"]),
                    ("weak", &["efficient"]),
                ],
                "efficient",
            ),
            task: None,
            rules: Some(RuleSignalConfig {
                capable: vec!["historical tie-break".to_string()],
                balanced: vec!["multi-step".to_string()],
                named: BTreeMap::new(),
            }),
            tool_loop: None,
            review: None,
            affinity: false,
        })?);

        let recorder = Arc::new(Recorder::default());
        let (selected_model, _) = test_drive_with_models(
            router,
            turn_request(false),
            three_level_runtime_models(),
            recorder.serve(),
        )
        .await?;

        assert_eq!(selected_model, "weak");
        Ok(())
    }

    #[tokio::test]
    async fn deterministic_task_rule_routes_to_a_named_specialist() -> Result<()> {
        let router = Arc::new(CapabilityRouter::new(CapabilityRouterConfig {
            pool: pool(
                &[
                    ("specialist", &["compatibility"]),
                    ("strong", &["capable"]),
                    ("balanced", &["balanced"]),
                    ("weak", &["efficient"]),
                ],
                "efficient",
            ),
            task: None,
            rules: Some(RuleSignalConfig {
                capable: Vec::new(),
                balanced: Vec::new(),
                named: BTreeMap::from([(
                    "compatibility".to_string(),
                    vec!["historical tie-break".to_string()],
                )]),
            }),
            tool_loop: None,
            review: None,
            affinity: false,
        })?);
        let mut request = turn_request(false);
        request.llm_request.messages[0].content = vec![ContentBlock::Text {
            text: "Preserve the historical tie-break behavior.".to_string(),
        }];

        let recorder = Arc::new(Recorder::default());
        let mut runtime_models = three_level_runtime_models();
        runtime_models.insert(
            "compatibility".parse().unwrap(),
            vec![ModelId::from("specialist")],
        );
        runtime_models
            .entry(Category::Any)
            .or_default()
            .push(ModelId::from("specialist"));
        let (selected_model, _) =
            test_drive_with_models(router, request, runtime_models, recorder.serve()).await?;

        assert_eq!(selected_model, "specialist");
        Ok(())
    }

    #[tokio::test]
    async fn affinity_short_circuits_later_turns() -> Result<()> {
        let router = Arc::new(CapabilityRouter::new(CapabilityRouterConfig {
            pool: pool(
                &[("strong", &["capable"]), ("weak", &["efficient"])],
                "efficient",
            ),
            task: None,
            rules: None,
            tool_loop: Some(ToolLoopSignalConfig {
                mode: PickerMode::EfficientFirst,
                confidence_threshold: 0.5,
                recent_window: None,
                tool_semantics: ToolSemantics::default(),
                handoff_notes: None,
            }),
            review: None,
            affinity: true,
        })?);

        let recorder = Arc::new(Recorder::default());
        let session_request = Request {
            metadata: Some(Metadata {
                session_id: Some("session-1".to_string()),
                ..Metadata::default()
            }),
            ..turn_request(true)
        };
        let (first, _) = test_drive_with_models(
            router.clone(),
            session_request.clone(),
            runtime_models(),
            recorder.serve(),
        )
        .await?;
        assert_eq!(first, "strong");

        // A later, signal-quiet turn in the same session still gets "strong": affinity
        // replayed the pin instead of falling back to the pool default ("weak").
        let quiet_follow_up = Request {
            metadata: session_request.metadata.clone(),
            ..turn_request(false)
        };
        let (second, _) =
            test_drive_with_models(router, quiet_follow_up, runtime_models(), recorder.serve())
                .await?;
        assert_eq!(second, "strong");
        Ok(())
    }

    #[tokio::test]
    async fn both_signals_run_and_a_tie_keeps_the_first_pushed() -> Result<()> {
        // tool_loop scores first (an override: confidence 1.0, needs capable) and task
        // scores second (confidence 1.0, needs efficient) — both signals run regardless
        // of order, and on an exact tie the first one pushed wins, same as `argmax`.
        let recorder = Arc::new(Recorder::default());
        *recorder.judge_p_solve.lock() = 0.9;
        let router = Arc::new(CapabilityRouter::new(CapabilityRouterConfig {
            pool: pool(
                &[("strong", &["capable"]), ("weak", &["efficient"])],
                "efficient",
            ),
            task: Some(TaskClassifierConfig {
                judge: crate::CapabilityJudgeConfig::Llm(crate::LlmCapabilityConfig {
                    base_threshold: 0.5,
                    ..Default::default()
                }),
                ..TaskClassifierConfig::default()
            }),
            rules: None,
            tool_loop: Some(ToolLoopSignalConfig {
                mode: PickerMode::EfficientFirst,
                confidence_threshold: 0.5,
                recent_window: None,
                tool_semantics: ToolSemantics::default(),
                handoff_notes: None,
            }),
            review: None,
            affinity: false,
        })?);

        let (selected_model, _) = test_drive_with_models(
            router,
            turn_request(true),
            runtime_models(),
            recorder.serve(),
        )
        .await?;

        assert_eq!(selected_model, "strong");
        assert!(
            recorder
                .calls
                .lock()
                .iter()
                .any(|call| call.target == JUDGE),
            "task's judge must run even though tool_loop already decided"
        );
        Ok(())
    }

    #[tokio::test]
    async fn two_providers_for_one_tag_pick_the_first_declared() -> Result<()> {
        let router = Arc::new(CapabilityRouter::new(CapabilityRouterConfig {
            pool: pool(
                &[
                    ("sol", &["capable"]),
                    ("terra", &["capable"]),
                    ("luna", &["efficient"]),
                ],
                "efficient",
            ),
            task: None,
            rules: None,
            tool_loop: Some(ToolLoopSignalConfig {
                mode: PickerMode::EfficientFirst,
                confidence_threshold: 0.5,
                recent_window: None,
                tool_semantics: ToolSemantics::default(),
                handoff_notes: None,
            }),
            review: None,
            affinity: false,
        })?);

        let recorder = Arc::new(Recorder::default());
        let (selected_model, _) = test_drive_with_models(
            router,
            turn_request(true),
            runtime_models(),
            recorder.serve(),
        )
        .await?;
        assert_eq!(selected_model, "sol");
        Ok(())
    }
}
