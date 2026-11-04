# The Model Pool

Every `capability_router` route shares one `[pool]`: the deployment's catalog of
models and
the capability tags each one provides. Signals never name a model directly —
they name what a request needs (`capable`, `efficient`, or any tag you choose),
and the pool resolves that need to a model.

This is what lets `task` and `tool_loop` sit side by side on one route instead
of each holding its own copy of the model list, and it is what a route's
`signals.affinity` and the pool's `default` key off.

## Configuration

```toml
[pool]
models = [
  { target = "sol",  provides = ["capable"] },
  { target = "luna", provides = ["efficient"] },
]
default = "efficient"
```

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `models` | Yes | — | The pool's models, in declared order. |
| `models[].target` | Yes | — | A name from `[targets.<name>]`. |
| `models[].provides` | Yes | — | Capability tags this model satisfies. `capable` and `efficient` are the built-in tags; any other string is a deployment-defined tag with no built-in meaning. A model may provide more than one tag. |
| `default` | Yes | — | The tag to resolve when no signal names one with confidence. Must be provided by at least one model. |

`[pool]` is global: one table, shared by every route whose `type = "capability_router"` in
the deployment. A route that does not use `capability_router` ignores it.

## How a need resolves to a model

1. Every configured signal scores the turn. The highest-confidence score wins;
   ties keep the first signal.
2. If nothing scored, or the winner named no tag, `default` is used instead.
3. The pool returns the first model, in declared order, that provides the
   winning tag.

A deployment error is raised at startup if `default` is not provided by any
pool model, or if a `models[].target` does not name a configured target.

See [Capability Router](capability_router_routing.md) for how signals produce
the scores the pool resolves.
