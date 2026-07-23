# Random Routing

Random routing sends a fixed percentage of traffic to a `strong` model and the
rest to a `weak` model. It does not inspect the prompt.

Use it for A/B tests, benchmark baselines, and gradual traffic ramps. If routing
should depend on task difficulty, use
[LLM Classifier Routing](llm_classifier_routing.md) or
[Stage-Router Routing](stage_router_routing.md).

## Algorithm

For each request, Switchyard flips a weighted coin:

- `strong_probability: 0.0` sends all traffic to `weak`.
- `strong_probability: 0.3` sends about 30% to `strong`.
- `strong_probability: 0.5` is an even split.
- `strong_probability: 1.0` sends all traffic to `strong`.

Each request is independent, so short runs may not match the configured
percentage exactly. `rng_seed` is optional and only useful when you need a
repeatable sequence for tests or benchmarks.

## Behavior

Random routing is per request. A multi-turn conversation can move between
models from turn to turn. If you need conversation-level stickiness, use
[Sticky Routing](sticky_routing.md) where it is supported, or select a direct
tier model instead of the routing model.

## Enable It

Use `type: random_routing` under `routes:`:

```yaml
defaults:
  api_key: ${OPENROUTER_API_KEY}
  base_url: https://openrouter.ai/api/v1
  format: openai

routes:
  ab-test:
    type: random_routing
    strong:
      model: openai/gpt-4o
    weak:
      model: openai/gpt-4o-mini
    strong_probability: 0.3
    rng_seed: 42
    fallback_target_on_evict: strong
```

Run it with:

```bash
switchyard --routing-profiles routes.yaml -- serve --port 4000
```

The route id (`ab-test`) is the model id clients select when they want the
weighted split.
