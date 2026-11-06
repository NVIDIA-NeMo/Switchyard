# The Model Pool

The top-level `[pool]` declares answer models once. Each model provides one or
more capability tags. Unified [Auto](auto_routing.md) signals ask for tags;
the selector resolves those tags to a target.

```toml
[pool]
models = [
  { target = "sol", provides = ["capable", "reasoning"] },
  { target = "terra", provides = ["capable"] },
  { target = "luna", provides = ["efficient"] },
]
default = "capable"
```

| Key | Meaning |
|---|---|
| `models[].target` | A target declared under `[targets.*]`. |
| `models[].provides` | Non-empty list of capability tags. |
| `default` | Capability to use when all signals abstain. Must exist in the pool. |

`capable` and `efficient` are built-in capability tags. Other strings are named
capabilities. One model can provide many tags; many models can provide the same
tag. Multiple needs on one score have OR semantics.

The selector takes the highest-confidence score, prefers a previously selected
matching model for the same explicit identity, and otherwise takes the first
matching model in declaration order. A literal model pin is distinct from a
capability tag, even when their names are identical.

The pool does not replace `[targets.*]` or `[llm_clients.*]`. Standalone legacy
routes continue to use their own target settings. See [Auto](auto_routing.md)
for signal configuration, selection rules, review, and migration.
