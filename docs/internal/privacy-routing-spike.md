# Restricted model routing

This stack replaces the unmerged generalized-label design with one trusted
`restricted` flag. Label collections are deferred.

The host decides whether a request requires approved private targets. Switchyard
runs the configured algorithm within the permitted targets. It does not inspect
content for PII. `restricted = true` declares approval, not physical location or
proof of deployment security. The standard lane permits the normal model pool.

## Configuration and Rust API

The provider connections are omitted here. They remain ordinary `llm_clients`.

```toml
[targets.efficient]
id = "cloud-efficient"
llm_client = "cloud"

[targets.capable]
id = "cloud-capable"
llm_client = "cloud"

[targets.local_efficient]
id = "local-efficient"
llm_client = "local"
restricted = true

[targets.local_capable]
id = "local-capable"
llm_client = "local"
restricted = true

[routes.coding]
id = "switchyard/coding"
type = "stage_router"
efficient_target = "efficient"
capable_target = "capable"
picker = "efficient_first"

[routes.coding.restricted]
require_explicit_signal = true
scope = "task"

[routes.coding.restricted.target_overrides]
efficient = "local_efficient"
capable = "local_capable"
```

- `targets.<name>.restricted` and `decision_targets.<name>.restricted` default to false.
- The route's optional `restricted` table enables the trusted signal. Without it, a
  supplied signal is rejected, including explicit false.
- `require_explicit_signal` defaults to false. When true, every request needs an
  assessment. False is an explicit assessment, not a missing signal.
- `scope` defaults to `"request"`. `"task"` retains true under an authorized
  `request.metadata.task_id`. Later requests cannot clear it.
- `target_overrides` is one optional flat map. It replaces existing target roles
  for the restricted lane without changing algorithm settings. Pool routers can
  filter to approved candidates without overrides. Every required fixed role must remain
  usable, including judges, fallbacks and auxiliary calls.

```rust
use switchyard_runner::require_restricted_targets;

require_restricted_targets(&mut request, restricted)?;
let output = route.execute(request, None).await?;
```

The helper writes a boolean into existing internal Rust metadata. It does not
authenticate the source, and provider-body metadata or HTTP headers cannot invoke
it. A true value cannot be cleared by a later helper call with false.

## Execution

Runner builds a standard lane and one restricted lane at startup. Each owns the
algorithm instance, client bindings, allowed model groups and auxiliary targets.
The restriction policy selects a borrowed lane before dispatch. Libsy checks model IDs;
the LLM client drives routing calls, ordered answer fallbacks and provider transport.
The protocol types and translation codecs do not change.

Task retention uses a short `parking_lot` lock and stores only restricted task IDs.
It is bounded to 4,096 entries in one route instance. There is no expiry, reset,
restart persistence or replica coordination. Existing dispatched calls are not
recalled. The host must authorize task ownership.

## Pi demo

The separate [`codex/pi-local-only` branch](https://github.com/NVIDIA-NeMo/Switchyard/tree/codex/pi-local-only/examples/pi)
updates the maintained Pi extension for the demo. Configure `local_model` in its
existing JSON file, then use `--local-only` or `/local` before reading private data.
The control is sticky for the session. A fresh session without the flag is needed
to return to normal routing.

Local-only requests select that physical local model before any cloud classifier
lookup. A missing or failed local model does not fall back to cloud. Unrestricted
requests retain the existing CompositeRouter path.

Pi calls the Node/libsy binding directly, not Runner. This demo proves host-owned
selection; separate Runner tests prove Rust admission. It applies to the managed
virtual model, not arbitrary tools, network traffic or manually selected models.

## Gateways and other consumers

An embedded Rust gateway attaches its trusted decision after constructing the
normalized request and before `Route::execute`. It can combine standing project
policy with what the application knows about the current data using boolean OR.

The native LiteLLM draft currently calls `switchyard_llm_client::run` directly.
Its maintained integration must carry trusted policy/task context into shared
admission and approved-target construction while retaining LiteLLM's clients. Calling
the boolean helper before the unchanged driver does not enforce anything. There
is no working YAML-only setup for this policy today.

The stock server and native Relay plugin also lack a trusted signal importer.
They can use fixed approved-only pools without a signal. Dynamic policy needs
maintained integration work. New HTTP inputs, Python bindings and Relay ABI changes
are outside this slice.

## Contracts and validation

`require_restricted_targets` replaces the unpublished `require_local_targets`,
`require_target_labels` and `mark_privacy_restricted` helpers. The old `local`
TOML fields and label/set TOML are also retired. These are
deliberate breaks to unmerged drafts, not removals of published APIs. Existing
`Route::execute`, decision methods and output signatures stay unchanged.
`PrivacyDecision` and its observer method keep their names and report actual
standard/restricted admission. No C ABI change is introduced. The internal metadata
key stays stable so already-attached restrictions are still recognized.

Workspace tests, strict workspace Clippy and Pi's typecheck/ten tests passed.
Real Ollama calls passed through both Runner and Pi, including retained task
restrictions and a tool continuation. A local endpoint outage did not fall back to
cloud. Three NVIDIA attempts returned 404; successful cloud inference remains
unverified. Pi's unrestricted path used an explicitly labeled fixture judge because
no authenticated Decision provider was available. This is not an accuracy or
production-performance benchmark.

The proposal format follows [#601](https://github.com/NVIDIA-NeMo/Switchyard/issues/601).
Its broader pool optimizer and unified routing API are not implemented here.
