// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;
use crate::core::algorithm::drive_with_state;
use crate::{Call, CapabilityJudgeConfig, DecisionJudgeConfig};
use switchyard_protocol::{
    DecisionAnswer, DecisionResponse, DecisionValue, Metadata, Probability, text_request,
};

struct Counter;

#[async_trait::async_trait]
impl Algorithm for Counter {
    fn name(&self) -> &str {
        "counter"
    }
    fn tracks_answers(&self) -> bool {
        true
    }
    async fn route(self: Arc<Self>, driver: Driver, request: Request) -> Result<RoutingOutcome> {
        let mut state = driver.take_state();
        let value = state
            .extra
            .entry("turns".into())
            .or_insert(StateValue::Count(0));
        let StateValue::Count(turns) = value else {
            panic!("counter state")
        };
        *turns += 1;
        driver.set_state(state);
        let targets = driver.models_for(&Category::Any);
        assert_eq!(targets, &["chosen", "chosen-fallback"]);
        Ok(RoutingOutcome::route_to(
            targets[0].clone(),
            targets[1..].to_vec(),
            request,
        ))
    }
}

#[tokio::test]
async fn caller_state_and_observer_stay_with_the_selected_profile() -> Result<()> {
    let child: Arc<dyn Algorithm> = Arc::new(Counter);
    let profiles = BTreeMap::from([
        (
            "selected".into(),
            RoutingProfile {
                algorithm: child.clone(),
                models: RuntimeModels::new(HashMap::from([(
                    Category::Any,
                    vec!["chosen".into(), "chosen-fallback".into()],
                )])),
            },
        ),
        (
            "other".into(),
            RoutingProfile {
                algorithm: Arc::new(crate::Passthrough),
                models: RuntimeModels::new(HashMap::from([(Category::Any, vec!["other".into()])])),
            },
        ),
    ]);
    let router = Arc::new(ProfileRouter::new(
        profiles,
        "other".into(),
        ProfileSelector {
            capable: "selected".into(),
            efficient: "other".into(),
            models: RuntimeModels::new(HashMap::from([
                (Category::Capable, vec!["capable".into()]),
                (Category::Efficient, vec!["efficient".into()]),
                (Category::Judge, vec!["judge".into()]),
            ])),
            classifier: TaskClassifierConfig {
                classify_trigger: ClassifyTrigger::NewSession,
                judge: CapabilityJudgeConfig::Decision(DecisionJudgeConfig {
                    cutoff: 0.92,
                    instructions: None,
                    criteria: None,
                    candidates: BTreeMap::from([
                        ("capable".into(), "capable".into()),
                        ("efficient".into(), "efficient".into()),
                    ]),
                    evidence: serde_json::json!({}),
                }),
                ..Default::default()
            },
        },
    )?);
    let calls = AtomicUsize::new(0);
    let serve = |call| async {
        let Call::Decision(call) = call else {
            panic!("unexpected model call")
        };
        calls.fetch_add(1, Ordering::SeqCst);
        call.respond(std::future::ready(Ok(DecisionResponse {
            id: None,
            model: None,
            usage: Default::default(),
            answers: BTreeMap::from([(
                "route".into(),
                DecisionAnswer {
                    provider_confidence: None,
                    value: DecisionValue::Choice {
                        selected: "advantage".into(),
                        probabilities: Some(BTreeMap::from([(
                            "advantage".into(),
                            Probability(1.0),
                        )])),
                    },
                },
            )]),
        })))
        .await
    };
    let mut state = State::default();
    state.extra.insert("outside".into(), StateValue::Bool(true));
    state.extra.insert(
        "composite.profile.5:other.turns".into(),
        StateValue::Count(99),
    );
    for turn in 1..=3 {
        let outcome = drive_with_state(
            router.clone(),
            Request {
                llm_request: text_request(None, "Task"),
                metadata: (turn == 2).then(|| Metadata {
                    session_final: Some(true),
                    ..Default::default()
                }),
                ..Default::default()
            },
            Arc::new(RuntimeModels::default()),
            state,
            &serve,
        )
        .await?;
        assert_eq!(
            outcome.selected_model_ids,
            vec!["chosen", "chosen-fallback"]
        );
        assert!(Arc::ptr_eq(
            outcome.answer_observer.as_ref().unwrap(),
            &child
        ));
        let metadata = outcome.metadata.unwrap();
        assert_eq!(metadata.profile.as_deref(), Some("selected"));
        assert_eq!(metadata.strategy.as_deref(), Some("counter"));
        state = outcome.state;
        assert_eq!(
            state.extra["composite.profile.8:selected.turns"],
            StateValue::Count(turn)
        );
        assert_eq!(
            state.extra["composite.profile.5:other.turns"],
            StateValue::Count(99)
        );
        assert_eq!(state.extra["outside"], StateValue::Bool(true));
        if turn == 1 {
            assert_eq!(metadata.judgments.len(), 1)
        }
        if turn == 2 {
            assert!(!state.extra.contains_key(CHOICE))
        }
    }
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    Ok(())
}
