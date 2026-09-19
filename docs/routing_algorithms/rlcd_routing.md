# RLCD Decision Routing

RLCD ("Reinforcement Learning for Calibrated Decisions") routes by asking a
decision model for one calibrated probability per candidate target and picking
the target with the highest probability. This is the same "System One" approach
TypeSafe's Jev announcement introduced. Serve any decision model that answers
the decision prompt with one JSON object behind an OpenAI-compatible endpoint.

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
id = "decision-model"
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
3. Parses one JSON verdict with a probability for every option.
4. Routes to the option with the highest probability.
5. Keeps the remaining candidates in `targets` order as fallbacks for
   eligible non-timeout failures.

The verdict's options are the targets' resolved `id` values — the same
identifiers the upstream provider sees, not the local target names. For the
configuration above:

```json
{
  "target": "openai/gpt-5",
  "probabilities": [
    {"option": "openai/gpt-4o-mini", "probability": 0.2},
    {"option": "openai/gpt-5", "probability": 0.8}
  ]
}
```

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

## References

- TypeSafe's Jev announcement — the "System One" model class trained with
  Reinforcement Learning for Calibrated Decisions:
  [docs.typesafe.ai/concepts/system-one](https://docs.typesafe.ai/concepts/system-one)
- RLCD, Reinforcement Learning from Contrastive Distillation — the paper that
  introduced the RLCD name:
  [arXiv:2307.12950](https://arxiv.org/abs/2307.12950)