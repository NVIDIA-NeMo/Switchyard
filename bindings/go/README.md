# NVIDIA NeMo Switchyard

NVIDIA NeMo Switchyard is a library for routing LLM requests between models.
These Go bindings run Switchyard algorithms inside your application. Your
application owns classification, credentials, provider calls, and retries.

The default backend runs libsy in WebAssembly through wazero, with no cgo
dependency. Native builds can use the C ABI instead.

## Installation

Requires Go 1.25+ and the repository's Rust toolchain to build the WASM guest.
This is a source-only module. From a checkout of this repository, build the guest
before compiling the Go package:

```sh
rustup target add wasm32-wasip1
cargo build -p switchyard-ffi --target wasm32-wasip1 --release --locked --lib
mkdir -p bindings/go/internal/wasm
cp target/wasm32-wasip1/release/switchyard_ffi.wasm bindings/go/internal/wasm/switchyard.wasm
```

These commands use the repository's pinned Rust toolchain. The generated binary
is not checked in. Rebuild it after changing Rust code.

From your application's Go module, point Go at that checkout. Replace the path
below with its absolute location:

```sh
go mod edit -replace=github.com/NVIDIA-NeMo/Switchyard/bindings/go=/absolute/path/to/Switchyard/bindings/go
go get github.com/NVIDIA-NeMo/Switchyard/bindings/go
```

For the native backend, build the Rust library for the same platform as your Go
application. This path also requires cgo and a C compiler:

```sh
cargo build -p switchyard-ffi --release --locked
```

Native builds link `target/release/libswitchyard_ffi.a` into the Go host artifact.
The native path still requires cgo and platform system libraries.
Native artifacts are not downloaded or published by this package.

The native backend uses the `switchyard_native` build tag and does not need the
WASM binary. These commands build locally; they do not publish artifacts.

## Use

```go
router, err := switchyard.New(json.RawMessage(`{
  "algorithm": {"type": "random", "targets": ["fast", "capable"]},
  "targets": {"fast": "fast-model", "capable": "capable-model"}
}`))
if err != nil {
    return err
}
defer router.Close()

outcome, err := router.Route(ctx, switchyard.Input{
    Request: json.RawMessage(`{"messages":[{
        "role":"user", "content":[{"type":"text", "text":"Fix this bug"}]
    }]}`),
    Models: map[string][]string{"any": {"fast-model", "capable-model"}},
}, nil)
```

Import `encoding/json` and
`github.com/NVIDIA-NeMo/Switchyard/bindings/go`.

The backend is selected at build time: `cgo` with `switchyard_native` uses the
native library; other builds use WASM. Both use `New(config)` and the same routing API.
Call `Backend()` to report `"native"` or `"wasm"` without creating a router.

Supply the model groups needed by the configured algorithm. Composite needs
`any` for eligible answer targets, plus `capable`, `efficient`, and `judge`.
Random and signal-only Stage need no host callback. For Composite, supply a
`HostCall` that handles `Work.Type`:

- `call_model`: call the ordered `Work.Models` candidates and return `Reply` with
  a normalized response and the target ID that answered in `ServedModel`.
- `call_decision`: send `Work.Request` to `Work.Model` and return its normalized
  decision response. `ServedModel` does not apply to decision calls.

A `HostCall` error is a failed call. The algorithm's error policy decides what
happens next, so a `fail_open` classifier can still route. Return `Abort(err)`
when the host refuses the call, such as for a budget or authorization check.
`Route` then stops and returns `err`, which keeps its Go error type.

Requests and replies use Switchyard's protocol JSON. `DecodeRequest`,
`EncodeRequest`, and `DecodeResponse` use the existing translation codecs for
`openai_chat`, `openai_responses`, and `anthropic_messages`. Typed decision
requests use the decision protocol directly.

`Outcome` contains ordered targets, the prepared request, optional routing
evidence, and an optional answer. When it has no answer, the host calls the
selected target. Routing-produced responses are buffered; the host may stream
the final selected provider response separately.

## Configured target requirements

`NewConfigured` loads a TOML route and enforces its target labels and ordered
alternatives before requesting any host calls. Use the configured route's `id`,
not its TOML table name. This works with the default WASM backend and the native
backend; both use the same Rust requirement checks.

```go
router, err := switchyard.NewConfigured(string(tomlSource), "switchyard/coding")
if err != nil {
    return err
}
defer router.Close()

targets, err := router.TargetModels()
if err != nil {
    return err
}
// Use targets[emittedModelID].Client and .ID to resolve each host call.

outcome, err := router.Route(ctx, switchyard.Input{
    Request: normalizedRequest,
    RequiredLabels: []string{"private", "customer_data"},
    Metadata: &switchyard.RequestMetadata{TaskID: trustedTaskID},
}, hostCall)
```

Leave `Models` and `SubagentModels` nil: TOML supplies those groups. `RequiredLabels`
nil means no assessment was supplied; `[]string{}` explicitly supplies no
requirements. Raw `New` routers reject both empty and nonempty requirements.

The application supplies trusted labels and tenant-scoped task IDs. Task-scope
requirements accumulate on this reused router; another router or process does
not share that retention. Each run still starts with fresh per-run routing state.

No provider clients are constructed or credentials read. The existing `HostCall`
handles approved LLM and Decision calls. LLM work and final candidates use TOML
target names; Decision work uses provider model IDs. `TargetModels()` maps those
identifiers to `TargetModel{Client, ID}` for all configured targets, including
alternatives. This inventory is not the current eligible pool: execute only calls
and final candidates returned by routing. `Reply.ServedModel` remains the emitted
target name, not its provider ID.

The host resolves the corresponding client, applies target prompts/settings, and
keeps answer retries/fallbacks within the returned candidates. This is not full
Runner execution. Rebuild the native library or WASM guest and Go package
(`go test -a` for tests) after updating; older artifacts lack configured routing
and target discovery.

## Cancellation and ownership

`Route` waits in its calling goroutine. A host that needs immediate return can
call it in a goroutine; waiting for routing steps does not occupy a blocked
native-reader thread. Raw WASM routers use a pool of `GOMAXPROCS` guest instances.
A session ID keeps the parent and all its child agents on the same instance.
Without a session ID, the first user message is used, or requests are assigned round-robin.
Guest calls are serialized within each instance; host callbacks can overlap.
A trap fails runs sharing that instance. Raw instances are replaced and their
retained algorithm state is lost.

Configured WASM routers instead use one persistent guest, retaining each task's
requirements by task ID. Guest execution is serialized, but host calls can still
overlap. This trades parallel Rust execution for consistent task retention;
native routing has no single-guest limit. A configured guest trap permanently
fails that router rather than silently forgetting requirements. Recreating it
resets local retention: existing tasks must supply their complete accumulated
requirements again, not `[]`. Otherwise, use a new task with a fresh assessment.

Each instance has its own algorithm state. VGR's circuit breaker is not shared
across the pool. Advisor's review budget is also per instance when no session ID
is supplied. Enforce gateway-wide spending limits in the host.
Message-based assignment does not give requests without session IDs persistent
session state.

While a WASM run is idle, Go re-enters it every 5 ms to advance Rust timers and
session waiters. Deadline handling can be delayed by that interval and scheduling load.

Host callbacks may run concurrently and must honor their context. Cancellation
stops routing and cancels the host context, but cannot forcibly stop Go code that
ignores it. WASM checks cancellation between guest calls. Late host results cannot
access a freed run.
Host errors cross the boundary as messages, not concrete Go error types.
Only an aborted call returns the host's original error from `Route`.

`Close` prevents new runs; existing runs retain their algorithm and finish under
their own contexts. Cancel those contexts to stop them. Router value copies
share closed state, and repeated `Close` calls are safe. Native callback handles
are released only after native cleanup has waited for callbacks to return.
Call `Close` when finished; Go garbage collection does not release the router.

`New` checks ABI generation 1 against the native library or WASM module. Scope session and agent
IDs to the host's tenant and conversation. `Input.Metadata` and
`Input.SubagentModels` carry routing context separately from provider requests.

## Test

From the repository root, after building the WASM guest:

```sh
cd bindings/go
CGO_ENABLED=0 go test ./...
CGO_ENABLED=0 go vet ./...
CGO_ENABLED=1 go test -race ./...
```

For a native build, build the native library above instead of the guest and use
the build tag:

```sh
cd bindings/go
CGO_ENABLED=1 go test -tags switchyard_native -race ./...
CGO_ENABLED=1 GOEXPERIMENT=cgocheck2 go test -tags switchyard_native -race ./...
CGO_ENABLED=1 go vet -tags switchyard_native ./...
```

Rebuild Go with `go test -a` after changing the C header or native library; Go's
cache does not track headers outside the package directory.

### Go versions

`go 1.25.0` in `go.mod` is the compatibility minimum required by wazero, not a
compiler pin. Use the latest patch of a [supported Go release](https://go.dev/doc/devel/release#policy)
for development and production; compatibility with an older release does not
make it a recommended production toolchain.

Before merging changes, run the checks above with the latest patch of the
declared minimum and the latest stable Go. Raise the minimum only when the
binding or a dependency needs newer Go, and call out that change.
