# Switchyard plugin for Bifrost

Experimental, checkout-only Go plugin using the shared [Switchyard Go
bindings](../../bindings/go/README.md). See the [Bifrost integration
guide](../../docs/integrations/bifrost.md) for setup, configuration, and limits.

## Build with a local Bifrost host

Requires Linux or macOS, Git, Make, a C compiler, the repository's Rust toolchain,
Go 1.27.2 or newer, and npm with the Node.js version specified by Bifrost's `.nvmrc`.
Start in the directory containing your `Switchyard` checkout:

```sh
git clone --depth 1 https://github.com/maximhq/bifrost.git
export GOTOOLCHAIN=auto
export CGO_ENABLED=1

go work init ./Switchyard/bindings/go ./Switchyard/examples/bifrost ./bifrost/core ./bifrost/framework ./bifrost/transports
for module in bifrost/plugins/*; do
  if test -f "$module/go.mod"; then go work use "./$module"; fi
done
export GOWORK="$PWD/go.work"

cargo build --manifest-path Switchyard/Cargo.toml -p switchyard-ffi --release --locked
make -C bifrost build-ui
go -C bifrost/transports build -o ../../Switchyard/target/bifrost-http ./bifrost-http
go -C Switchyard/examples/bifrost build -tags switchyard_native -buildmode=plugin -o ../../target/switchyard-bifrost.so .
```

The clone uses Bifrost's current default branch. The workspace uses its local
modules for both builds, overriding the plugin's standalone Bifrost core dependency.
Go selects the toolchain required by the workspace. If a workspace already exists,
add these modules with `go work use` instead of `go work init`.

Build the host and plugin on the same platform and architecture, with the same Go
compiler, dependencies, and build flags. Go rejects incompatible plugins. The host
must support dynamic plugins; see Bifrost's [dynamic build guide](https://github.com/maximhq/bifrost/blob/HEAD/docs/plugins/building-dynamic-binary.mdx).
Rebuild both after updating Bifrost. Upstream plugin API changes may require
adapter changes. Native Switchyard is linked into the plugin.

## Configure and run

Create a Bifrost app directory and configure its providers and credentials in
`config.json`. Add the [plugin entry](../../docs/integrations/bifrost.md#configure)
to that file's `plugins` list. Use the absolute path to
`Switchyard/target/switchyard-bifrost.so`, replace the example targets with your
provider/model IDs, and set `bifrost_url` to `http://127.0.0.1:8080`.

From the directory containing both checkouts, start the host:

```sh
./Switchyard/target/bifrost-http -app-dir /absolute/path/to/bifrost-data -host 127.0.0.1 -port 8080
```

Send a request from another terminal:

```sh
curl -i http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"switchyard","messages":[{"role":"user","content":"Hello"}]}'
```

The `x-switchyard-selected-model` response header identifies the routing target.
For dashboard setup, set `BIFROST_SETUP_TOKEN` before starting the host and enter
that token in the UI at `http://127.0.0.1:8080`.

Algorithms that call models or judges use the local Bifrost API. Prefill is
excluded. Completed routed answers can be returned as JSON or a buffered stream.

## Tests

After building the native library, run these commands in the same workspace:

```sh
go -C Switchyard/examples/bifrost vet -tags switchyard_native ./...
go -C Switchyard/examples/bifrost test -tags switchyard_native -race ./...
```
