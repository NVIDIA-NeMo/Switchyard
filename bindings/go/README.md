# Embedded Go routing

Run Switchyard algorithms in a Go host that owns model eligibility, credentials,
provider calls, and retries. These bindings call libsy through the C ABI; they do
not start a server. A private WebAssembly runtime exercises translation codecs
through wazero. Routing and public translation functions still use the native library.

## Build and test

Requires Go 1.25+, cgo, a C compiler, and the repository's Rust toolchain. Build
the native library for the same platform as the Go host, then build the WASM
guest before compiling the default package or its private runtime tests.
The script installs the Rust toolchain's
`wasm32-wasip1` target. The generated binary is not checked in.

```sh
cargo build -p switchyard-ffi --release --locked
scripts/build-go-wasm.sh
cd bindings/go
CGO_ENABLED=1 go test -race ./...
CGO_ENABLED=1 GOEXPERIMENT=cgocheck2 go test -race ./...
CGO_ENABLED=1 go vet ./...
```

The `switchyard_native` build tag excludes the private WASM runtime and its
tests. This build needs only the native library:

```sh
cd bindings/go
CGO_ENABLED=1 go test -tags switchyard_native -race ./...
CGO_ENABLED=1 GOEXPERIMENT=cgocheck2 go test -tags switchyard_native -race ./...
CGO_ENABLED=1 go vet -tags switchyard_native ./...
```

Native builds link `target/release/libswitchyard_ffi.a` into the Go host artifact.
The native path still requires cgo and platform system libraries.
Rebuild Go with `go test -a` after changing the C header or native library; Go's
cache does not track headers outside the package directory. Native artifacts
are not downloaded or published by this package.

Rebuild the WASM guest after changing Rust code. These commands build and test
locally; they do not publish artifacts. The package still requires cgo; the
private runtime is not a complete WASM routing backend.

### Go versions

`go 1.25.0` in `go.mod` is the compatibility minimum required by wazero, not a
compiler pin. Use the latest patch of a [supported Go release](https://go.dev/doc/devel/release#policy)
for development and production; compatibility with an older release does not
make it a recommended production toolchain.

Before merging changes, run the checks above with the latest patch of the
declared minimum and the latest stable Go. Raise the minimum only when the
binding or a dependency needs newer Go, and call out that change.

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
`github.com/NVIDIA-NeMo/Switchyard/bindings/go`. This is a source-only module;
use a local Go module `replace` to point your host at this directory.

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

## Provider-shaped decisions

`Decide` accepts a provider request and uses the same algorithm as `Route`:

```go
decision, err := router.Decide(ctx, switchyard.WireInput{
    Input: switchyard.Input{
        Request: body,
        Models: eligibleGroups,
    },
    Format: "openai_chat",
}, nil)
```

The C bridge handles decoding and request preparation through `sy_decide_new`.
Build its matching native library before compiling this source-only Go package.

Pass a `WireHostCall` for algorithms that make routing-time calls. For
`call_model`, `WireWork.Requests` contains every ordered candidate, each with
its target model and provider-shaped request. Return the buffered provider
response in `Reply.Response`; set `ServedModel` when more than one candidate
was requested. The host still owns retries and fallback execution.

`TargetFormats` can override the input format for particular target IDs, such
as an Anthropic judge for a Chat request. Typed `call_decision` requests and
replies keep their existing decision protocol.

The result's `Requests` are prepared, ordered final candidates. Its embedded
`Outcome` retains the normalized request, evidence, and any algorithm-produced
answer. That answer remains normalized; this API does not encode responses.
No post-routing answer is sent automatically.

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
