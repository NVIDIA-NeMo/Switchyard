# Escalation-Router Routing

The escalation router starts every conversation on the cheap **weak** tier. An
LLM **judge** watches the trajectory each turn and, when the run shows a clear
pattern of trouble (repeated errors, loops, false progress, drift, giving up),
escalates the conversation to the **strong** tier — one-way for the rest of
the task. A new conversation starts weak again.

Unlike [LLM Classifier Routing](llm_classifier_routing.md), which predicts
per-turn difficulty from request content, the escalation router judges whether
the work is *going well*. Hard-but-smooth runs stay on the weak tier; runs the
weak tier is fumbling get rescued by the strong tier.

## How it works

Per turn, the `EscalationJudgeRequestProcessor`:

1. Checks the session-affinity latch. A pinned conversation routes strong with
   no judge call — the latch is one-way per task.
2. Otherwise routes weak. From `judge.min_turn` on (default 3), it sends the
   judge a condensed transcript: the system + first-user anchors (individually
   capped so harness boilerplate cannot crowd out the evidence), the last
   `judge.recent_turn_window` messages (head/tail-truncated), and a coverage
   header stating how much history is not shown.
3. The judge returns `{"escalate": bool, "reason": "<one sentence>"}`. On
   `escalate: true` the conversation is pinned to strong, effective the same
   turn. The reason is logged and stamped into the request metadata
   (`_escalation_verdict`) as the audit record.
4. Any judge failure — timeout, error, invalid JSON — fails open to the weak
   tier and never pins. A judge outage costs quality risk, never money.

The latch reuses the [session-affinity](sticky_routing.md) store: conversations
are keyed on the stable prefix (system prompt + first user message), so a new
task gets a fresh key and resets to weak automatically. The weak tier is never
pinned — "not pinned" means weak-and-still-watching.

## Configuration

The CLI accepts `type: escalation_router` in a `routes:` bundle loaded with
`--routing-profiles` (same path the `deterministic` router uses):

```yaml
routes:
  switchyard:
    type: escalation_router
    fallback_target_on_evict: strong
    judge:
      model: google/gemini-3.5-flash
      api_key: ${OPENROUTER_API_KEY}
      base_url: https://openrouter.ai/api/v1
      timeout_secs: 5.0
      min_turn: 3               # first turn the judge runs on
      recent_turn_window: 14    # trailing messages shown to the judge
    strong:
      model: anthropic/claude-opus-4.7
      api_key: ${OPENROUTER_API_KEY}
      base_url: https://openrouter.ai/api/v1
    weak:
      model: deepseek/deepseek-v4-pro
      api_key: ${OPENROUTER_API_KEY}
      base_url: https://openrouter.ai/api/v1
```

| Key | Default | Meaning |
|---|---|---|
| `judge.model` / `api_key` / `base_url` | required | Judge LLM target. Pick something small and fast — it sits on the request path of every pre-escalation turn. |
| `judge.timeout_secs` | `5.0` | Judge wall-clock ceiling; fails open to weak. |
| `judge.min_turn` | `3` | First conversation turn the judge runs on. Earlier turns have no trajectory to judge. |
| `judge.recent_turn_window` | `14` | Trailing messages shown to the judge. Loops longer than the window are invisible — widen before concluding the judge misses them. |
| `judge.prompt` | built-in | Judge system-prompt override. |
| `judge.max_request_chars` | `12000` | Cap on the judge transcript; oldest window messages are dropped first. |
| `fallback_target_on_evict` | required | `strong` or `weak`; context-window-eviction reroute target. |
| `session_key_depth` | `0` | See [Repeated-trial benchmarking](#repeated-trial-benchmarking) below. |
| `tier_timeout_s` | `600` | Default per-call timeout for strong/weak targets without their own `timeout_secs`. |
| `affinity_max_sessions` | `10000` | LRU capacity of the escalation latch. |

In Python, build it from the exported config pair:

```python
from switchyard import EscalationRouterConfig, EscalationRouterProfileConfig

profile = EscalationRouterProfileConfig.from_config(
    EscalationRouterConfig.model_validate({...})
).build()
```

## Repeated-trial benchmarking (k>1)

The session key is a content hash of the system prompt + first user message,
so **k repeated trials of the same task against one long-lived server share
one latch**: trial 1's escalation makes trials 2..k start on strong from turn
1, silently corrupting pass@k and cost numbers. Two ways to keep trials
independent:

- **Run-level isolation (zero config):** run each trial set as a separate
  `benchmark/run-baseline.sh` invocation. Each run gets a fresh Switchyard
  container; latch state cannot survive between runs.
- **`session_key_depth: N` (within-run k>1):** extends the session key with
  the first `N` post-first-user messages. With nonzero sampling temperature,
  trials diverge in their early model responses and get distinct keys. The
  prefix of a conversation never changes as it grows, so the key stays stable
  within a trial; until the prefix is complete, affinity is untouched.
  Caveats: useless at temperature 0 (identical trajectories still collide),
  and mid-session history rewrites (context compaction) change the key and
  silently drop the latch — keep `0` outside benchmarks.

Also set Harbor `--max-retries 0` for escalation-router runs: a retry re-runs
the failed task against the same warm server, and failed attempts are exactly
the ones likely to have escalated, so the retry would warm-start on strong.

## Known limits and v2 levers

- **Blocking judge:** the judge call adds ~100–300 ms to each pre-escalation
  turn. If benchmarks show latency pain, the designed successor is running the
  judge concurrently with the weak call and applying the verdict next turn.
- **Window-bounded loop detection:** a repeat cycle longer than
  `recent_turn_window` is invisible. The designed successor is a per-turn
  trajectory digest distilled from the whole history.
- **No difficulty prediction:** this router rescues struggling runs; it does
  not predict hard ones. Composing it with the LLM classifier is future work.
