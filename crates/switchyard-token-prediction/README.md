# Task-completion token prediction

Build a Jev request for the total token budget needed to complete a task:

```rust
use serde_json::json;
use switchyard_token_prediction::{request, ResponseUsage};

let usage = [ResponseUsage {
    uncached_input_tokens: 800,
    output_tokens: 100,
    reasoning_tokens: 40, // Already included in output_tokens.
}];
let request = request(
    "jev-latest".into(),
    &[json!({"source": "user", "message": "Fix the parser"})],
    &usage,
);
assert_eq!(request.context["token_usage"]["tokens_used_so_far"], 900);
```

The forecast counts **uncached input plus output across the whole task**, including
usage so far. Reasoning is included once as part of output. Cache reads are
excluded; uncached cache writes count as input. Repeated conversation input counts
when uncached. Supply only the observed task prefix, including the initial task
prompt, and chronological usage for completed model calls. Empty usage predicts
from the initial task prompt.

Send the request using any `switchyard_protocol::RoutedDecisionClient`, including
`switchyard_llm_client::SystemOneClient`. Decode its response with `prediction`:

```rust,no_run
use switchyard_protocol::{DecisionRequest, RoutedDecisionClient};
use switchyard_token_prediction::{Prediction, prediction};

async fn forecast(
    client: &impl RoutedDecisionClient,
    request: DecisionRequest,
) -> Result<Prediction, Box<dyn std::error::Error>> {
    let response = client.call(request).await?;
    let forecast = prediction(&response)?;
    let estimated_total = forecast.predicted_tokens;
    let estimated_remaining = forecast.remaining_tokens(900);
    Ok(forecast)
}
```

`predicted_tokens` is the probability-weighted mean of total-budget range
midpoints, conditional on a finite budget. `expected_upper_tokens` is the mean
of range upper bounds, not a confidence bound. `selected_interval` is Jev's
chosen total-budget range. `remaining_tokens(tokens_used_so_far)` subtracts
observed usage from the estimated total, with a minimum of zero.

This ports the task-completion experiment's `summarize_for_jev` state and
`post_jev_flat.fine_request` with its default five subdivisions and `buffered`
prompt. It uses **80 budget ranges through 1.5 million tokens**, plus `impossible`
when no listed budget is likely to permit successful completion. The
`impossible_probability` is reported separately. When all probability is on
`impossible`, both means and the remaining estimate are `None`.

History entries are caller-prepared JSON steps. The caller controls truncation;
the experiments use up to 20,000 history tokens and 1,000 tokens per step.
Usage covers all completed calls even when earlier history is omitted.

The caller owns transport, credentials, and cancellation. No server or Python
runtime is needed. The crate itself makes no network calls.
