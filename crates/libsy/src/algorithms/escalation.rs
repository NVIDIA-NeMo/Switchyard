// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Trajectory-judged escalation: serve the efficient model until an LLM judge finds the run
//! in trouble, then latch the session to the capable model for the rest of the task.
//!
//! [`EscalationRouter`] is the assembled algorithm: a [`FallThrough`] driving one classifier
//! that applies the three ordered rules the policy is made of — the affinity latch pins an
//! already-escalated session, the [`ConfirmedJudge`] escalates one whose verdicts have
//! confirmed, and the efficient tier serves everything else. Only the last rule is
//! unconditional, which keeps a judge outage from failing the turn.
//!
//! The judge picks the tier *before* the turn's model call, so a turn costs one model call and
//! the target's response — streamed or aggregated — reaches the caller untouched. A judge
//! failure fails open to the efficient tier and never latches: an outage costs quality risk,
//! never money.

use std::sync::Arc;

use async_trait::async_trait;

use super::util::escalation::ConfirmedJudge;
use super::util::AffinityRouter;
use super::FallThrough;
use crate::{
    Algorithm, Classification, Classifier, Context, Decision, Driver, Event, LlmTarget,
    LlmTargetSet, Processor, Request, Response, Result, RoutedLlmClient, State,
};

pub use super::util::escalation::EscalationJudgeConfig;

/// Telemetry label for this algorithm's spans, metrics, and logs.
const ALGORITHM_NAME: &str = "escalation";

/// Tier label reported for turns served by the capable target.
const TIER_CAPABLE: &str = "strong";

/// Tier label reported for turns served by the efficient target.
const TIER_EFFICIENT: &str = "weak";

/// The tier this router served and the rule that chose it.
///
/// Escalating and staying escalated both select the capable target, so a reason derived from
/// the target alone could not tell the two apart.
struct EscalationDecision {
    selected_model: String,
    tier: &'static str,
    reason: &'static str,
}

impl Decision for EscalationDecision {
    fn selected_model(&self) -> &str {
        &self.selected_model
    }

    fn routing_tier(&self) -> Option<&str> {
        Some(self.tier)
    }

    fn reasoning(&self) -> Option<&str> {
        Some(self.reason)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// The escalation policy as one classifier: latch, then judge, then the efficient tier.
///
/// Owns the affinity latch so its two roles — the processor that retains a decision and the
/// classifier that reads it back — cannot drift apart. The components underneath are
/// composition-agnostic (a latch does not know which tier it pins), so this is where the
/// policy's tiers and reasons are named.
struct EscalationClassifier {
    latch: AffinityRouter,
    judge: ConfirmedJudge,
    capable: String,
    efficient: String,
}

impl EscalationClassifier {
    /// The verdict for `model`, labelled with the tier it is and the rule that picked it.
    fn decided(model: &str, tier: &'static str, reason: &'static str) -> Classification {
        Classification::Decided(Arc::new(EscalationDecision {
            selected_model: model.to_string(),
            tier,
            reason,
        }))
    }
}

#[async_trait]
impl Processor<State> for EscalationClassifier {
    async fn process(&self, state: &mut State, event: Event<'_>) -> Result<()> {
        // Only the latch binds anything on a decision; the judge's streak is its own.
        self.latch.process(state, event).await
    }
}

#[async_trait]
impl Classifier<State> for EscalationClassifier {
    async fn score(
        &self,
        state: &mut State,
        request: &mut Request,
        driver: Option<&Driver>,
    ) -> Result<Classification> {
        // A pinned session stays capable without paying for another judge call.
        let latched = self.latch.score(state, request, driver).await?;
        if latched.argmax(false)?.is_some() {
            return Ok(Self::decided(
                &self.capable,
                TIER_CAPABLE,
                "session pinned to capable after prior escalation",
            ));
        }
        let judged = self.judge.score(state, request, driver).await?;
        if judged.argmax(false)?.is_some() {
            return Ok(Self::decided(
                &self.capable,
                TIER_CAPABLE,
                "judge escalated the run to the capable model",
            ));
        }
        // Unconditional, so a declined or unavailable judge costs quality risk rather than
        // the turn.
        Ok(Self::decided(
            &self.efficient,
            TIER_EFFICIENT,
            "judge has not confirmed the run is in trouble",
        ))
    }
}

/// Serves the efficient model until the judge escalates, then pins the session to the
/// capable model. See the [module docs](self).
pub struct EscalationRouter {
    route: FallThrough<State>,
}

impl EscalationRouter {
    /// Routes between `efficient_target` and `capable_target`, consulting `judge_target` on
    /// every turn that is not already latched. [`EscalationJudgeConfig::default`] is the
    /// benchmarked configuration.
    ///
    /// Errors when `config` would leave the judge nothing useful to read, or when the packaged
    /// judge prompt and schema cannot be loaded.
    pub fn new(
        efficient_target: LlmTarget,
        capable_target: LlmTarget,
        judge_target: LlmTarget,
        config: EscalationJudgeConfig,
    ) -> Result<Self> {
        let capable = capable_target.semantic_name.clone();
        let efficient = efficient_target.semantic_name.clone();
        // Capable first: `count_tokens` passes through to the first target whose client
        // supports it, and the capable tier is the one a caller is asking about. The judge is
        // called through its own target and is not a routing destination, so it stays out of
        // the set.
        let targets = LlmTargetSet::new(vec![capable_target, efficient_target]);
        let classifier = Arc::new(EscalationClassifier {
            // Latching only the capable target is what makes escalation one-way: an efficient
            // turn is never pinned.
            latch: AffinityRouter::new().with_latch_only([capable.clone()]),
            judge: ConfirmedJudge::new(judge_target, capable.clone(), config)?,
            capable,
            efficient,
        });
        Ok(Self {
            route: FallThrough::<State>::new_with_state(targets)
                .with_name(ALGORITHM_NAME)
                .with_component(classifier),
        })
    }
}

#[async_trait]
impl Algorithm for EscalationRouter {
    fn name(&self) -> &str {
        ALGORITHM_NAME
    }

    fn count_tokens_client(&self) -> Option<Arc<dyn RoutedLlmClient>> {
        self.route.count_tokens_client()
    }

    async fn create_run_task(
        self: Arc<Self>,
        ctx: Context,
        driver: Driver,
        request: Request,
    ) -> Result<Response> {
        self.route.execute(ctx, driver, request).await
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Arc;

    use futures::StreamExt;
    use parking_lot::Mutex;
    use switchyard_protocol::{completion_text, text_response, LlmClientError, LlmResponseChunk};

    use super::super::util::escalation::request_at_turn;
    use super::*;
    use crate::{Algorithm, Context, Decision, LlmResponse, LlmTarget, Request, Response};

    fn target(name: &str, client: Option<Arc<dyn RoutedLlmClient>>) -> LlmTarget {
        LlmTarget {
            semantic_name: name.to_string(),
            llm_client: client,
        }
    }

    /// A mid-conversation request: enough trajectory for the judge to have something to read.
    fn request(session_id: Option<&str>) -> Request {
        request_at_turn(session_id, 3)
    }

    /// Settings that latch on a single escalate verdict, for tests not exercising streaks.
    fn latch_immediately() -> EscalationJudgeConfig {
        EscalationJudgeConfig {
            confirmations: 1,
            ..EscalationJudgeConfig::default()
        }
    }

    fn verdict_json(escalate: bool) -> String {
        format!(r#"{{"escalate": {escalate}, "reason": "test"}}"#)
    }

    /// What a [`RecordingClient`] does when a call reaches it.
    enum Reply {
        /// Return this text as an aggregated response.
        Text(String),
        /// Return this text as a one-chunk stream, so buffering anywhere is observable.
        Stream(String),
        /// Fail the call the way a transport error would.
        Fail,
    }

    /// Convenience: a normal text reply.
    fn ok(text: &str) -> Reply {
        Reply::Text(text.to_string())
    }

    /// Records which models were called (in order) and returns pre-supplied replies.
    ///
    /// All three targets share one instance; calls are served in arrival order regardless of
    /// which target the driver is calling. This mirrors the shared `RecordingClient` pattern
    /// used throughout the libsy test suite.
    struct RecordingClient {
        /// Replies in call order.
        replies: Mutex<VecDeque<Reply>>,
        /// Model names in the order each `call()` arrived.
        calls: Arc<Mutex<Vec<String>>>,
    }

    impl RecordingClient {
        fn new(replies: impl IntoIterator<Item = Reply>) -> (Arc<Self>, Arc<Mutex<Vec<String>>>) {
            let calls = Arc::new(Mutex::new(Vec::new()));
            let client = Arc::new(Self {
                replies: Mutex::new(replies.into_iter().collect()),
                calls: calls.clone(),
            });
            (client, calls)
        }
    }

    #[async_trait::async_trait]
    impl RoutedLlmClient for RecordingClient {
        async fn call(
            &self,
            _ctx: Context,
            _request: Request,
            decision: Arc<dyn Decision>,
        ) -> std::result::Result<Response, LlmClientError> {
            self.calls
                .lock()
                .push(decision.selected_model().to_string());
            let reply = self
                .replies
                .lock()
                .pop_front()
                .unwrap_or_else(|| Reply::Text("fallback".to_string()));
            match reply {
                Reply::Text(text) => Ok(Response {
                    llm_response: LlmResponse::Agg(text_response(None, &text)),
                    metadata: None,
                }),
                Reply::Stream(text) => {
                    let chunk = LlmResponseChunk::TextDelta { index: 0, text };
                    Ok(Response {
                        llm_response: LlmResponse::Stream(
                            futures::stream::iter([Ok(chunk)]).boxed(),
                        ),
                        metadata: None,
                    })
                }
                Reply::Fail => Err(LlmClientError::Configuration {
                    message: "test error".into(),
                }),
            }
        }
    }

    /// A router paired with the call log its shared client writes to.
    type InstrumentedRouter = (Arc<EscalationRouter>, Arc<Mutex<Vec<String>>>);

    /// Router where all three targets share one `RecordingClient`. Replies are served in the
    /// order supplied, matching the call order: the judge first (when it runs), then the tier.
    fn instrumented_router(
        replies: impl IntoIterator<Item = Reply>,
    ) -> crate::Result<InstrumentedRouter> {
        instrumented_router_with(replies, EscalationJudgeConfig::default())
    }

    fn instrumented_router_with(
        replies: impl IntoIterator<Item = Reply>,
        config: EscalationJudgeConfig,
    ) -> crate::Result<InstrumentedRouter> {
        let (client, calls) = RecordingClient::new(replies);
        let client: Arc<dyn RoutedLlmClient> = client;
        let router = Arc::new(EscalationRouter::new(
            target("efficient", Some(client.clone())),
            target("capable", Some(client.clone())),
            target("judge", Some(client)),
            config,
        )?);
        Ok((router, calls))
    }

    #[tokio::test]
    async fn pass_judge_serves_efficient() -> crate::Result<()> {
        let (router, calls) = instrumented_router([
            ok(&verdict_json(false)), // judge: no escalation
            ok("4"),                  // efficient
        ])?;

        let (_, response) = router.run(Context::default(), request(Some("s1"))).await?;

        let agg = response.llm_response.as_agg();
        assert_eq!(agg.map(completion_text).as_deref(), Some("4"));
        assert_eq!(*calls.lock(), vec!["judge", "efficient"]);
        Ok(())
    }

    #[tokio::test]
    async fn confirmed_escalation_serves_capable_without_calling_efficient() -> crate::Result<()> {
        let (router, calls) = instrumented_router_with(
            [
                ok(&verdict_json(true)), // judge: escalate
                ok("The answer is 4"),   // capable
            ],
            latch_immediately(),
        )?;

        let (trace, response) = router.run(Context::default(), request(Some("s2"))).await?;

        let agg = response.llm_response.as_agg();
        assert_eq!(agg.map(completion_text).as_deref(), Some("The answer is 4"));
        // The efficient tier is never called: the judge decided before the turn's model call.
        assert_eq!(*calls.lock(), vec!["judge", "capable"]);
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].selected_model(), "capable");
        Ok(())
    }

    #[tokio::test]
    async fn pinned_session_skips_the_judge() -> crate::Result<()> {
        let (router, calls) = instrumented_router_with(
            [
                ok(&verdict_json(true)), // judge: escalate (first request)
                ok("right"),             // capable (first request)
                ok("right again"),       // capable (second request — pinned)
            ],
            latch_immediately(),
        )?;

        router
            .clone()
            .run(Context::default(), request(Some("s3")))
            .await?;
        let (trace, _) = router.run(Context::default(), request(Some("s3"))).await?;

        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].selected_model(), "capable");
        assert_eq!(*calls.lock(), vec!["judge", "capable", "capable"]);
        Ok(())
    }

    #[tokio::test]
    async fn unparseable_judge_reply_fails_open_to_efficient() -> crate::Result<()> {
        let (router, calls) = instrumented_router([
            ok("sorry, I cannot help"), // judge: unparseable
            ok("4"),                    // efficient
        ])?;

        let (_, response) = router.run(Context::default(), request(Some("s4"))).await?;

        let agg = response.llm_response.as_agg();
        assert_eq!(agg.map(completion_text).as_deref(), Some("4"));
        assert_eq!(*calls.lock(), vec!["judge", "efficient"]);
        Ok(())
    }

    #[tokio::test]
    async fn failed_judge_call_fails_open_to_efficient() -> crate::Result<()> {
        let (router, calls) = instrumented_router([
            Reply::Fail, // judge: transport failure
            ok("4"),     // efficient
        ])?;

        let (_, response) = router.run(Context::default(), request(Some("s5"))).await?;

        let agg = response.llm_response.as_agg();
        assert_eq!(agg.map(completion_text).as_deref(), Some("4"));
        assert_eq!(*calls.lock(), vec!["judge", "efficient"]);
        Ok(())
    }

    #[tokio::test]
    async fn one_escalate_verdict_does_not_latch_at_default_confirmations() -> crate::Result<()> {
        let (router, calls) = instrumented_router([
            ok(&verdict_json(true)), // judge: escalate, streak 1 of 2
            ok("4"),                 // efficient — not yet confirmed
        ])?;

        let (trace, _) = router.run(Context::default(), request(Some("c1"))).await?;

        assert_eq!(trace[0].selected_model(), "efficient");
        assert_eq!(*calls.lock(), vec!["judge", "efficient"]);
        Ok(())
    }

    #[tokio::test]
    async fn two_consecutive_escalate_verdicts_latch() -> crate::Result<()> {
        let (router, calls) = instrumented_router([
            ok(&verdict_json(true)), // judge: escalate, streak 1
            ok("4"),                 // efficient
            ok(&verdict_json(true)), // judge: escalate, streak 2 — confirmed
            ok("strong answer"),     // capable
            ok("still strong"),      // capable (third request — pinned)
        ])?;

        router
            .clone()
            .run(Context::default(), request(Some("c2")))
            .await?;
        router
            .clone()
            .run(Context::default(), request(Some("c2")))
            .await?;
        let (trace, _) = router.run(Context::default(), request(Some("c2"))).await?;

        assert_eq!(trace[0].selected_model(), "capable");
        assert_eq!(
            *calls.lock(),
            vec!["judge", "efficient", "judge", "capable", "capable"]
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_decline_between_escalate_verdicts_resets_the_streak() -> crate::Result<()> {
        let (router, calls) = instrumented_router([
            ok(&verdict_json(true)),  // streak 1
            ok("4"),                  // efficient
            ok(&verdict_json(false)), // decline — streak cleared
            ok("4"),                  // efficient
            ok(&verdict_json(true)),  // streak 1 again, not 2
            ok("4"),                  // efficient — still no latch
        ])?;

        for _ in 0..3 {
            router
                .clone()
                .run(Context::default(), request(Some("c3")))
                .await?;
        }

        assert!(
            !calls.lock().contains(&"capable".to_string()),
            "strict-consecutive confirmation must not latch across a decline"
        );
        Ok(())
    }

    #[tokio::test]
    async fn an_unavailable_judge_leaves_the_streak_intact() -> crate::Result<()> {
        let (router, _calls) = instrumented_router([
            ok(&verdict_json(true)), // streak 1
            ok("4"),                 // efficient
            Reply::Fail,             // judge unavailable — no evidence either way
            ok("4"),                 // efficient
            ok(&verdict_json(true)), // streak 2 — confirmed despite the gap
            ok("strong answer"),     // capable
        ])?;

        for _ in 0..2 {
            router
                .clone()
                .run(Context::default(), request(Some("c4")))
                .await?;
        }
        let (trace, _) = router.run(Context::default(), request(Some("c4"))).await?;

        assert_eq!(trace[0].selected_model(), "capable");
        Ok(())
    }

    #[tokio::test]
    async fn without_a_session_id_confirmations_can_never_accumulate() -> crate::Result<()> {
        let (router, calls) = instrumented_router([
            ok(&verdict_json(true)),
            ok("4"),
            ok(&verdict_json(true)),
            ok("4"),
        ])?;

        router
            .clone()
            .run(Context::default(), request(None))
            .await?;
        router.run(Context::default(), request(None)).await?;

        // No session id means no streak store, so the capable tier is unreachable at
        // confirmations > 1. Documented behaviour, asserted so it cannot change silently.
        assert_eq!(
            *calls.lock(),
            vec!["judge", "efficient", "judge", "efficient"]
        );
        Ok(())
    }

    #[tokio::test]
    async fn no_session_id_routes_without_pinning() -> crate::Result<()> {
        let (router, calls) = instrumented_router_with(
            [
                ok(&verdict_json(true)),  // judge: escalate (request 1)
                ok("4"),                  // capable (request 1)
                ok(&verdict_json(false)), // judge: no escalation (request 2 — not pinned)
                ok("4"),                  // efficient (request 2)
            ],
            latch_immediately(),
        )?;

        router
            .clone()
            .run(Context::default(), request(None))
            .await?;
        router.run(Context::default(), request(None)).await?;

        assert_eq!(
            *calls.lock(),
            vec!["judge", "capable", "judge", "efficient"]
        );
        Ok(())
    }

    #[tokio::test]
    async fn efficient_response_streams_through_unbuffered() -> crate::Result<()> {
        let (router, _) = instrumented_router([
            ok(&verdict_json(false)),              // judge: no escalation
            Reply::Stream("streamed".to_string()), // efficient
        ])?;

        let (_, response) = router.run(Context::default(), request(Some("s8"))).await?;

        // Judging before the call is what lets the tier's stream survive: nothing needs to
        // read the response to make a routing decision.
        assert!(matches!(response.llm_response, LlmResponse::Stream(_)));
        Ok(())
    }

    #[tokio::test]
    async fn each_rule_labels_its_own_decision() -> crate::Result<()> {
        let (router, _) = instrumented_router_with(
            [
                ok(&verdict_json(false)), // judge: decline
                ok("4"),                  // efficient
                ok(&verdict_json(true)),  // judge: escalate
                ok("stronger"),           // capable
                ok("stronger still"),     // capable — pinned, no judge
            ],
            latch_immediately(),
        )?;

        let mut labels = Vec::new();
        for _ in 0..3 {
            let (trace, _) = router
                .clone()
                .run(Context::default(), request(Some("r1")))
                .await?;
            labels.push(trace.first().map(|decision| {
                (
                    decision.routing_tier().unwrap_or_default().to_string(),
                    decision.reasoning().unwrap_or_default().to_string(),
                )
            }));
        }
        let labels: Vec<(String, String)> = labels.into_iter().flatten().collect();
        assert_eq!(labels.len(), 3, "{labels:?}");

        // Every rule labels its tier, the latched one included: the tier is a metric
        // dimension, so an unlabelled rule drops silently out of the strong/weak split.
        assert_eq!(labels[0].0, "weak", "{labels:?}");
        assert_eq!(labels[1].0, "strong", "{labels:?}");
        assert_eq!(labels[2].0, "strong", "{labels:?}");

        // Escalating and staying escalated both select the capable target, so a reason
        // derived from the target alone could not tell the last two turns apart.
        assert!(labels[0].1.contains("has not confirmed"), "{labels:?}");
        assert!(labels[1].1.contains("judge escalated"), "{labels:?}");
        assert!(labels[2].1.contains("pinned"), "{labels:?}");
        Ok(())
    }

    #[tokio::test]
    async fn routing_tiers_are_labelled() -> crate::Result<()> {
        let (router, _) = instrumented_router([ok(&verdict_json(false)), ok("4")])?;

        let (trace, _) = router.run(Context::default(), request(Some("s9"))).await?;

        assert_eq!(trace[0].routing_tier(), Some("weak"));
        Ok(())
    }

    #[tokio::test]
    async fn schema_and_prompt_load_at_construction() -> crate::Result<()> {
        EscalationRouter::new(
            target("a", None),
            target("b", None),
            target("c", None),
            EscalationJudgeConfig::default(),
        )?;
        Ok(())
    }

    #[test]
    fn unusable_judge_config_is_rejected_at_construction() {
        for config in [
            EscalationJudgeConfig {
                confirmations: 0,
                ..EscalationJudgeConfig::default()
            },
            EscalationJudgeConfig {
                recent_turn_window: 0,
                ..EscalationJudgeConfig::default()
            },
            EscalationJudgeConfig {
                window_message_chars: 49,
                ..EscalationJudgeConfig::default()
            },
        ] {
            assert!(
                EscalationRouter::new(
                    target("a", None),
                    target("b", None),
                    target("c", None),
                    config,
                )
                .is_err(),
                "a config that starves the judge must not build a router"
            );
        }
    }
}
