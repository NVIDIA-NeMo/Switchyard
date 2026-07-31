# Sub-Agent Override

Internal note. Not published: `mkdocs.yml` excludes `internal/**` from the site.

`subagent_target` exists only on the Python route bundle loaded by
`switchyard serve`. The `switchyard-server` TOML schema has no equivalent key.

Any bundle route may name an optional `subagent_target` in its common envelope,
alongside `type`. A request carrying a recognized sub-agent signal — Claude Code
agent-lineage headers, Codex delegated-work kinds (`x-openai-subagent:
collab_spawn` or `review`), or an explicit `x-switchyard-is-subagent: true` —
bypasses the route's own chain and runs as a direct passthrough to that target:

```yaml
routes:
  assistant:
    type: model
    target:
      model: strong-model
    subagent_target:
      model: cheaper-worker-model
```

This keeps a sub-agent loop on one intentional, cache-compatible target instead
of re-routing every worker turn. The worker may live on a different provider
entirely — give it its own `base_url` and `api_key` and one route spans two
upstreams:

```yaml
    subagent_target:
      model: my-local-model
      base_url: http://localhost:8000/v1
      api_key: dummy
      format: openai
```

Detection is the protocol crate's canonical policy, shared with the
`switchyard-libsy` `SubagentOverride` classifier, so both engines agree on what
counts as delegated work. Harness-maintenance turns (`compact`,
`memory_consolidation`) and unrecognized kinds stay on normal routing, as does
everything else when the field is absent. A worker-target failure surfaces as a
normal target error — it is never silently re-routed through the route's own
chain.

To suppress sub-agent routing for a request that carries a recognized signal,
send `x-switchyard-is-subagent: false`. This explicit header overrides Codex and
Claude Code lineage signals in either direction: `false` keeps the request on
normal routing even when delegated-work headers are present, and `true` marks a
request as a sub-agent even when no harness headers appear.

The key applies to bundle routes of type `model`, `deterministic`,
`escalation_router`, and `stage_router`. It is accepted on `random_routing`
routes but has no effect, because they expand into their table entries on a
separate path that never consumes it.
