# Context-Window Handling

When an upstream rejects a request because the prompt exceeds the model's
context window, Switchyard evicts that target for the current request,
reroutes to the configured fallback target, and retries once. If the fallback
also overflows, the request fails with a 400 in the client's inbound wire
format.

Any multi-target route (stage-router, random_routing, or deterministic) supports
this. Set `fallback_target_on_evict` on the route. Single-target routes
(`type: passthrough`, `type: model`) have no alternative target, so the original
overflow propagates unchanged.

## Configuration

`fallback_target_on_evict` is required on every multi-target route and must
match one of the route's declared target ids:

```yaml
routes:
  my-stage-router:
    type: stage_router
    picker: stage_router_capable_first
    fallback_target_on_evict: strong   # must match strong.id or weak.id
    strong:
      id: strong
      model: anthropic/claude-opus-4.7
    weak:
      id: weak
      model: moonshotai/kimi-k2.6
```

## Scope

Single eviction + single retry per request. Compaction, cool-down, and
re-insertion of evicted targets are out of scope.
