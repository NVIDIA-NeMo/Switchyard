# RLCD Decision Routing

RLCD ("Reinforcement Learning for Calibrated Decisions") routes by asking a
decision model for one calibrated probability per candidate target and picking
the target with the highest probability. This is the same "System One" approach
pioneered by TypeSafe's Jev. The most-liked open implementation of it is
[`AlexWortega/openjev`](https://huggingface.co/AlexWortega/openjev), a Qwen3.5
cross-encoder that scores a task against every candidate option and returns one
probability each. RLCD checkpoints such as
[`harshatheg/Qwen-2.5-1B-RLCD`](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD)
express the same contract.

A generative judge writes an answer token by token. A decision model instead
takes the task plus a list of options and returns every option's probability in
one pass. Routing with it is fast and returns a calibrated confidence for every
target, not just the winner.

Use it when you have a decision model available and want a content-aware route
with per-target confidence. The decision model is a side call, never the
turn's answer.

## Configure an RLCD route

This example serves the decision model behind its own endpoint and lets it
choose between a strong and a weak target:

```toml
schema_version = 1

[llm_clients.rlcd]
format = "openai_chat"
base_url = "http://localhost:8080/v1"
api_key_env = "RLCD_API_KEY"

[llm_clients.openrouter]
format = "openai_chat"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"

[targets.decision]
id = "Qwen-2.5-1B-RLCD"
llm_client = "rlcd"

[targets.strong]
id = "openai/gpt-5"
llm_client = "openrouter"

[targets.weak]
id = "openai/gpt-4o-mini"
llm_client = "openrouter"

[routes.best_fit]
id = "best-fit"
type = "rlcd"
classifier_target = "decision"
targets = ["weak", "strong"]
default_target = "weak"
```

The route has three moving parts:

- `classifier_target` is the model that makes routing decisions. It is a side
  call and never a completion destination.
- `targets` lists the candidates the decision model chooses among. There must
  be at least two.
- `default_target` is the candidate used when the decision model's reply
  cannot be used. It must name one of `targets`.

## How a decision is made

For each request Switchyard:

1. Enumerates every `targets` candidate as a numbered option.
2. Sends the task and the option list to `classifier_target`.
3. Parses one JSON verdict with a probability for every option:

```json
{
  "target": "model/strong",
  "probabilities": [
    {"option": "model/weak", "probability": 0.2},
    {"option": "model/strong", "probability": 0.8}
  ]
}
```

4. Routes to the option with the highest probability.
5. Keeps the remaining candidates in `targets` order as fallbacks for
   eligible non-timeout failures.

A verdict is used only when it names every candidate exactly once with a
finite probability in `[0, 1]`, the probabilities sum to about `1.00`, and
`target` is the option with the highest probability. Anything else — an
unparseable reply, a missing or duplicated option, a mismatched `target` —
is treated as an unusable verdict and routes to `default_target`.

An HTTP client failure on the decision call stops the request, exactly like a
failed classifier judge.

Decision requests use `response_format = {"type": "json_object"}` and the
verdict schema is checked locally, because self-hosted decision-model
endpoints broadly support JSON objects but not strict JSON Schema.