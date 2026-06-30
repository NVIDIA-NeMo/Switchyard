# Cascade Routing

Cascade routing picks one of two tiers (`weak` / `strong`) per request based on
tool-result history stamped onto the conversation. The strong tier handles
exploration and error recovery; the weak tier handles the mechanical
implementation phase.

The picker composes three layers in order:

1. **Hard overrides**: high-confidence signal-derived shortcuts that bypass
   the scorer (critical severity, clean tests passed).
2. **Weighted scorer**: a linear combination of normalised
   `ToolResultSignal` dimensions; produces a score in `[-1, +1]` (positive ⇒
   STRONG, negative ⇒ WEAK) and a confidence `= |score|`. Returned as the
   dimensions verdict when `confidence ≥ confidence_threshold`.
3. **Optional LLM classifier**: consulted only when the scorer is
   ambiguous (`confidence < confidence_threshold`). Fails open to the
   picker's default tier on any error.

The YAML exposes one knob (`confidence_threshold`) plus an optional
classifier sub-block. Scorer weights and hard-override thresholds live in
code as calibrated defaults. Research engineers override them by passing
`weights=...` to `score()` directly.

---

## How it fits the profile path

```
CascadeProfile
  -> extract ToolResultSignal from the request
  -> pick weak or strong
  -> rewrite request.model to the selected target model
  -> call the selected native target backend
```

`CascadeProfile` uses the same Rust `ToolResultSignal` extractor as the legacy
chain path: severity, write/edit/read counts, recent-window slices,
pure-bash streak, tests-passed, and turn depth. The profile keeps the routing
decision as typed per-call state, records the decision source as
`switchyard_routing_decisions_total{router="cascade",source=...,tier=...}`, and
records model/tier usage through the OpenTelemetry metrics served at `/metrics`.

If the selected backend returns a context-window overflow, the profile retries
once against `fallback_target_on_evict`. A second overflow returns
`ContextPoolExhausted`.

---

## Pickers

Names describe **the default tier**: the verdict returned when the scorer
is ambiguous and no classifier verdict is available.

- **`cascade_strong_default`**: STRONG is the default. WEAK only when the
  scorer is confidently negative or the classifier says WEAK. Quality-first.
- **`cascade_weak_default`**: WEAK is the default. STRONG only when the
  scorer is confidently positive or the classifier says STRONG. Cost-first.

Both share the same override path and scorer math; only the default tier
differs.

### Hard overrides

Applied *before* the scorer, in this order. Any match short-circuits.

| Override | Condition | Verdict |
|---|---|---|
| Critical severity | `signal.severity ≥ SEVERITY_CRITICAL` (1.0) | STRONG |
| Clean completion | `signal.tests_passed AND signal.turn_depth ≥ CLEAN_TESTS_MIN_TURN_DEPTH (10) AND signal.write_count ≤ CLEAN_TESTS_MAX_WRITES (1)` | WEAK |

Thresholds are module-level constants in the Rust cascade profile; retune in
one place.

### Scorer

Weighted linear sum over `CodingAgentDimensions` (a normalised view of
`ToolResultSignal`). Weights are calibrated defaults in
`switchyard/lib/processors/cascade/scorer.py::DEFAULT_WEIGHTS`; override
via `weights=...` for research. The raw sum is clipped to `[-1, +1]` and
confidence is `abs(clipped)`. Magnitudes are sized so a single high-impact
axis at maximum value (`stuck_exploring = 1.0`, `tests_passed = 1.0`, ...)
clears the recommended threshold of `0.5` on its own.

### Classifier (optional)

When `confidence < confidence_threshold` and the YAML includes a
`classifier:` sub-block, the picker calls the configured model with a
short JSON-output prompt and asks for `{"tier": "strong" | "weak"}`. On
timeout, malformed JSON, network failure, or any other classifier error,
the picker falls back to its default tier (recorded as `fall_open`).

---

## Tuning `confidence_threshold`

The threshold is the **only** dial users touch. Its effect is monotone:
raising it shifts work from the dimensions scorer to the classifier
(or, if no classifier is configured, to the picker's default tier).

!!! note "Set `0.5` explicitly"
    `0.5` is the recommended starting point and is set in the public example
    below. The implementation default still differs by configuration path:

    | Configuration path | Default when omitted |
    |---|---|
    | Profile config (`switchyard serve --config`) using the Rust `cascade` profile | **`0.7`** |
    | Deprecated route bundle (`--routing-profiles`) using the Python cascade profile | **`0.5`** |

    Set `confidence_threshold: 0.5` explicitly rather than relying on either
    schema default.

| `confidence_threshold` | Include `classifier:` block? | Typical use |
|---|---|---|
| `0.0` | no | Cost/latency-sensitive. Every scorer verdict is accepted; no per-turn LLM call. The hard-override path still catches critical errors. |
| `0.5` | no | Recommended starting point. A single high-impact axis clears the threshold deterministically; derived from SWE-Bench Pro calibration. |
| `0.7` - `0.9` | yes | Classifier-assisted. Low-confidence scorer outputs go to the LLM classifier before falling back to the default tier. The Rust schema uses `0.7` when the field is omitted. |
| `1.0` | yes (required) | Classifier-driven. Equivalent to the legacy `coding_agent` profile. |

The dimensions-vs-llm-classifier split is dataset-dependent — measure it in
production via `switchyard_routing_decisions_total{router="cascade"}` on
`/metrics` rather than relying on priors from this doc.

### Calibrating the threshold from run data

The recommended `0.5` starting point was derived from SWE-Bench Pro Python-75
calibration. To tune for a different task set or model pair, follow this
minimum-data path.

**What you need**

| Run | Coverage | Purpose |
|---|---|---|
| Pure-strong | ~40–75 representative tasks | Baseline outcomes + signal features |
| Pure-weak | ~20 tasks (sampled from strong results) | Counterfactual outcomes |

Neither run needs to cover the full task set. A few dozen strong tasks gives
enough outcome diversity; the weak probe only needs to cover the interesting
quadrant candidates identified from those strong results.

**How to sample the weak probe set**

Stratify the pure-strong results across four quadrant candidates before running weak:

| Category | Criterion | Count | Value |
|---|---|---|---|
| Easy + clean | Strong passes, small diff, clear spec | ~5 | Establishes SAFE floor |
| Easy + tricky | Strong passes, subtle logic | ~5 | Catches LOSS false-positives |
| Hard + structural | Strong fails, large multi-file diff | ~5 | HARD noise baseline |
| Hard + localized | Strong fails, small targeted fix | ~5 | Best RESCUE signal |

Sample across repos and diff sizes. Don't over-represent one project.

**Building RESCUE / LOSS quadrants**

From the overlap tasks (those with both strong and weak results):

- `RESCUE` = strong-fail ∩ weak-pass → escalation is beneficial here
- `LOSS`   = strong-pass ∩ weak-fail → do NOT escalate here
- `SAFE`   = both pass
- `HARD`   = both fail

**Running the sweep**

Three scripts in `benchmark/calibration/cascade/` form the pipeline:

| Script | Input | Output | What it does |
|---|---|---|---|
| `signal_extractor.py` | Harbor task dir (JSONL trajectory) | `ToolResultSignal` per turn | Replays a claude-code session, emitting the same signal the cascade picker would have seen at each turn (write/edit/read counts, severity, tests passed, etc.) |
| `calibrate.py` | Harbor run dirs (one per arm) | `per_task.jsonl`, `per_turn.jsonl` | Reads `result.json` + trajectory JSONL for each task in each arm. Calls `signal_extractor` to build per-turn signals, then writes one record per task (outcome + features) and one record per turn (signal snapshot). Also prints RESCUE/LOSS/SAFE/HARD quadrant counts. |
| `sweep.py` | `per_task.jsonl`, `per_turn.jsonl` | Console table | Replays the per-turn signals through a set of escalation policies and scores each: pass%, escalation rate. The best-scoring policy that keeps escalation rate reasonable is your calibrated threshold. |

```bash
cd benchmark/calibration/cascade
python calibrate.py \
  --strong-run-dir /tmp/runs/your_strong_run \
  --weak-run-dir /tmp/runs/your_weak_probe
python sweep.py
```

`calibrate.py` writes `per_task.jsonl` and `per_turn.jsonl`. `sweep.py`
prints the policy score table; pick the best row from that output.

Pick the policy whose pass% beats `always_stay` with an acceptable
escalation rate. Translate it to a `confidence_threshold` value. A policy
that escalates ~20% of tasks maps roughly to `confidence_threshold: 0.5`
with `cascade_strong_default`.

Even 15–20 probe tasks produce a stable result because signal features are
extracted from the strong-arm trajectories, which are available for all
tasks from the pure-strong run.

**Caveat on weak outcomes in cascade vs. pure-weak**

In cascade, the weak model may inherit partial context from the strong arm
(conversation history up to the escalation point). Pure-weak runs start
fresh, so RESCUE is a conservative lower bound. Weak performs at least as
well in cascade as it does alone.

### Rate-limit isolation (production caveat)

The YAML below keeps the classifier on `${OPENROUTER_API_KEY}` for a compact
example. If the classifier and weak tier share a provider bucket, each classified
turn adds one extra request to that bucket. Production deployments should use a
separate classifier credential or quota bucket, or skip the classifier sub-block
on quota-constrained deployments. Co-locating them on one bucket can produce
sustained 429s at scale.

---

## Profile configuration

```yaml
endpoints:
  openrouter:
    base_url: https://openrouter.ai/api/v1
    api_key: ${OPENROUTER_API_KEY}

targets:
  strong:
    endpoint: openrouter
    model: openai/gpt-4o
    format: openai
  weak:
    endpoint: openrouter
    model: openai/gpt-4o-mini
    format: openai

profiles:
  smart-cascade:
    type: cascade
    picker: cascade_strong_default        # or cascade_weak_default
    confidence_threshold: 0.5              # recommended; range [0.0, 1.0]
    signal_recent_window: 3                # Rust sliding-window for recent_* counts
    fallback_target_on_evict: strong       # required; see Context-Window Handling
    strong: strong                         # target id
    weak: weak                             # target id
    classifier:                            # optional
      model: openai/gpt-4o-mini
      api_key: ${OPENROUTER_API_KEY}        # use a separate key/quota in production
      base_url: https://openrouter.ai/api/v1
      timeout_secs: 30.0
      recent_turn_window: 3
    enable_stats: true                     # default true
```

Save the file as `profiles.yaml` and start it with:

```bash
switchyard serve --config profiles.yaml --port 4000
```

Omit the `classifier` block to use signals only. If `confidence_threshold` is
also omitted, this Rust profile uses its schema default of `0.7`; the example
sets the recommended `0.5` explicitly.

`fallback_target_on_evict` is required and must reference one of the
declared target ids. See [Context-Window Handling](../operations/context_window.md) for
exception types and error envelopes.

!!! note "Launcher compatibility"
    Launcher subcommands do not accept `--config`. A launcher-owned cascade
    still requires the deprecated `--routing-profiles` compatibility path and
    its legacy `routes:` schema. Use the same `type: cascade`, picker, target,
    classifier, and explicit `confidence_threshold: 0.5` values there.

---

## Observability

### Per-tier token / cost metrics (standard)

```bash
curl http://localhost:4000/metrics
```

Returns the OpenTelemetry Prometheus surface — per-model request, token,
latency, and cost series labelled by `model` and `tier`. The `tier` label
comes from the cascade decision; classifier spend is tagged `tier="classifier"`.

### Decision-source metadata (cascade-specific)

The profile records decision-source counts as
`switchyard_routing_decisions_total{router="cascade",source=...,tier=...}`.
The possible `source` values are:

| Source | When |
|---|---|
| `override` | Hard override fired (critical severity, clean tests). |
| `dimensions` | Scorer crossed `confidence_threshold`. |
| `llm-classifier` | Scorer was ambiguous and the classifier returned a verdict. |
| `fall_open` | Scorer was ambiguous and the classifier failed or wasn't configured. Default tier used. |

Harness writers snapshot the Prometheus surface with:

```bash
curl -s http://localhost:4000/metrics > routing_metrics_final.prom
```

---

## When *not* to use cascade

- **Single-model deployments.** Use
  `PassthroughProfileConfig(...).build()` wrapped by `ProfileSwitchyard`.
- **Probabilistic A/B splits.** Use
  [Random Routing](random_routing.md) (`type: random-routing` in profile configs).
  The cascade's signals are wasted on a fixed traffic ratio.
- **No tool-result history.** Cascade needs meaningful tool-call traffic to
  populate `ToolResultSignal`. For pure chat-completion workloads every
  ambiguous request lands on the picker's default tier.

---

## Related

- [Architecture](../architecture.md): the end-to-end request lifecycle and
  system boundaries.
