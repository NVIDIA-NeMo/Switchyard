# NVIDIA NeMo Switchyard

NVIDIA NeMo Switchyard is a library for routing LLM requests between models.
These Go bindings run Switchyard algorithms inside your application. Your
application owns model eligibility, credentials, provider calls, and retries.

A private WebAssembly runtime exercises translation codecs through wazero.
Routing and public translation functions still use the native library.

## Installation

Requires Go 1.25+, cgo, a C compiler, and the repository's Rust toolchain. This is
a source-only module. From a checkout of this repository, build the native
library for the same platform as your Go application, then build the WASM guest:

```sh
cargo build -p switchyard-ffi --release --locked
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

Native builds link `target/release/libswitchyard_ffi.a` into the Go host artifact.
The native path still requires cgo and platform system libraries.
Native artifacts are not downloaded or published by this package.

The `switchyard_native` build tag excludes the private WASM runtime and its
tests, so that build needs only the native library. The package still requires
cgo; the private runtime is not a complete WASM routing backend.

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

Supply the model groups needed by the configured algorithm. Composite needs
`any` for eligible answer targets, plus `capable`, `efficient`, and `judge`.
Random and signal-only Stage need no host callback. For Composite, supply a
`HostCall` that handles `Work.Type`:

- `call_model`: call the ordered `Work.Models` candidates and return `Reply` with
  a normalized response and the target ID that answered in `ServedModel`.
- `call_decision`: send `Work.Request` to `Work.Model` and return its normalized
  decision response. `ServedModel` does not apply to decision calls.

Requests and replies use Switchyard's protocol JSON. `DecodeRequest`,
`EncodeRequest`, and `DecodeResponse` use the existing translation codecs for
`openai_chat`, `openai_responses`, and `anthropic_messages`. Typed decision
requests use the decision protocol directly.

`Outcome` contains ordered targets, the prepared request, optional routing
evidence, and an optional answer. When it has no answer, the host calls the
selected target. Routing-produced responses are buffered; the host may stream
the final selected provider response separately.

## Cancellation and ownership

`Route` waits in its calling goroutine. A host that needs immediate return can
call it in a goroutine; waiting for routing steps does not occupy a blocked
native-reader thread. Concurrent routes share one router.

Host callbacks may run concurrently and must honor their context. Cancellation
stops the native run and cancels the host context, but cannot forcibly stop Go
code that ignores it. Late host results cannot access the freed native run.
Host errors cross the boundary as messages, not concrete Go error types.

`Close` prevents new runs; existing runs retain their algorithm and finish under
their own contexts. Cancel those contexts to stop them. Router value copies
share closed state, and repeated `Close` calls are safe. Native callback handles
are released only after native cleanup has waited for callbacks to return.
Call `Close` when finished; Go garbage collection does not release the router.

`New` checks ABI generation 1 against the linked library. Scope session and agent
IDs to the host's tenant and conversation. `Input.Metadata` and
`Input.SubagentModels` carry routing context separately from provider requests.

## Test

From the repository root, after building the native library and WASM guest:

```sh
cd bindings/go
CGO_ENABLED=1 go test -race ./...
CGO_ENABLED=1 GOEXPERIMENT=cgocheck2 go test -race ./...
CGO_ENABLED=1 go vet ./...
```

For a native-only build, skip the guest build and use the build tag:

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
