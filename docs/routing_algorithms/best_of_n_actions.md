# Best of N Actions

`best_of_n_actions` samples `N` candidate next actions from one model using the
same conversation. It currently returns sample `0` in request order, even when
another sample finishes first. Scoring, tool-action validation, and
deduplication remain TODOs.

The route provides the sampling path for a Mid-Harness-style algorithm.
With the current selection rule, extra samples add cost and latency without
improving the selected action.

## Configuration

This example uses an existing local OpenAI-compatible model server:

```toml
schema_version = 1

[llm_clients.local]
format = "openai_chat"
base_url = "http://localhost:8000/v1"

[targets.executor]
id = "your-model-id"
llm_client = "local"

[routes.sampled_actions]
id = "switchyard/sampled-actions"
type = "best_of_n_actions"
generator_target = "executor"
num_candidates = 4
temperature = 0.8
candidate_timeout_ms = 60000
```

Send harness requests to the route's `id`. Replace `your-model-id` with the
upstream model name.

| Key | Default | Meaning |
|---|---|---|
| `generator_target` | required | Existing target used for every sample. |
| `num_candidates` | `4` | Positive number of separate model calls. |
| `temperature` | `0.8` | Replaces the caller's temperature, including with `N = 1`. Must be finite and in `0..=2`; the provider may support a narrower range. |
| `candidate_timeout_ms` | `60000` | Positive deadline per candidate when `N > 1`, including stream reads and time waiting for the host. |

Other sampling settings, including a supplied seed, stay unchanged. A fixed
seed or deterministic model settings can produce repeated candidates.
Provider retry settings still apply, so HTTP attempts may exceed `N`.

## Response behavior

For `N > 1`, all samples run concurrently. The route waits for every sample to
finish or time out, then returns the complete response from sample `0`.
Discarded samples never reach the harness. The selected response retains its
tool-call IDs, signed reasoning, provider events, headers, and usage.

A candidate may contain several tool calls or a completion response with no
tool calls. The harness handles the returned response normally.

An extra sample's failure is recorded and does not replace the selected sample.
If sample `0` fails, the route returns its error even when another sample
succeeds. Cancelling the request cancels pending samples.

Streaming callers receive the selected stream after all sampling finishes.
For `N = 1`, the route uses the normal completion path with live streaming;
the LLM client's `timeout_ms` applies instead of `candidate_timeout_ms`.

## Supported requests

For `N > 1`, history must be included in the request. The route rejects
`previous_response_id` and `conversation` continuations before dispatch.
It accepts normalized function tools that execute in the harness. It rejects
provider-run or unknown provider tools, since those could execute even for
discarded candidates. The provider's native `n` field must be absent or `1`.

Use the runner or HTTP server for this route. The LiteLLM routing plugin
cannot currently serve its intermediate calls or return a sampled response.

## Observability

Routing evidence includes the selected index, each sample's duration and
status, and available token usage. The returned response reports only its own
provider usage. Count every sample when measuring total cost.

Generic `AnswerCall` observations currently match by model ID, so they can
attribute a discarded sample as the answer when several calls use that model.
Use the sample indices in routing evidence for attribution. Matching by call
ID remains a TODO.
