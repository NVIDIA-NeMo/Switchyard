# Switchyard plugin for Bifrost

Experimental, checkout-only Go plugin using the shared [Switchyard Go
bindings](../../bindings/go/README.md). See the [Bifrost integration
guide](../../docs/integrations/bifrost.md) for setup, configuration, and limits.

From the repository root, build the native library, then test and build the plugin:

```sh
cargo build -p switchyard-ffi --release --locked
cd examples/bifrost
go vet -tags switchyard_native ./...
go test -tags switchyard_native -race ./...
go build -tags switchyard_native -buildmode=plugin -o ../../target/switchyard-bifrost.so .
```

This example pins Bifrost core 1.13.0 and requires Go 1.27.2 or newer, Rust,
and a C compiler. Build with the same Go version, dependency versions, platform,
and build flags as the Bifrost host. Go rejects incompatible plugins. Native
Switchyard is linked into the plugin. Algorithms that call models or judges use
the local Bifrost API; configure `bifrost_url` as shown in the guide. Prefill is
excluded. Completed routed answers can be returned as JSON or a buffered stream.
