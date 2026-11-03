#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Exercise the packaged dependency, not the source checkout or a local replacement.
set -euo pipefail
package_dir=$(cd "$1" && pwd)
read -r module version escaped < <(python3 - "$package_dir/manifest.json" <<'PY'
import json
import re
import sys

with open(sys.argv[1]) as file:
    manifest = json.load(file)
module = manifest["module"]
print(module, manifest["version"], re.sub(r"[A-Z]", lambda match: "!" + match[0].lower(), module))
PY
)
work=$(mktemp -d)
archive="$package_dir/proxy/$escaped/@v/$version.zip"
mkdir -p "$work/dependencies" "$work/proxy" "$work/consumer"
unzip -p "$archive" "$module@$version/go.mod" > "$work/dependencies/go.mod"
unzip -p "$archive" "$module@$version/go.sum" > "$work/dependencies/go.sum"
export GOWORK=off GOTOOLCHAIN=local
# Fetch and verify only third-party dependencies before testing the unpublished module offline.
(cd "$work/dependencies" && GOMODCACHE="$work/seed" GOSUMDB=sum.golang.org go mod download)
cp -R "$work/seed/cache/download/." "$work/proxy/"
cp -R "$package_dir/proxy/." "$work/proxy/"
export PATH="$(go env GOROOT)/bin:/usr/bin:/bin"
if command -v cargo || command -v rustc; then
    echo "Rust must not be available in the consumer test" >&2
    exit 1
fi
unset CGO_CFLAGS CGO_CPPFLAGS CGO_CXXFLAGS CGO_LDFLAGS
export GOMODCACHE="$work/cache" GOPROXY="file://$work/proxy" GOSUMDB=off
export GOPRIVATE= GONOPROXY= GONOSUMDB=
cd "$work/consumer"
unzip -p "$archive" "$module@$version/testdata/consumer/main.go" > main.go
go mod init switchyard-consumer.invalid/check
go get "$module@$version"
CGO_ENABLED=0 go test -count=1 "$module"
CGO_ENABLED=1 GOEXPERIMENT=cgocheck2 go test -a -count=1 -race -tags switchyard_native "$module"
module_dir=$(go list -m -f '{{.Dir}}' "$module")
extension=so
if [ "$(go env GOOS)" = darwin ]; then extension=dylib; fi
CGO_ENABLED=1 go build -tags switchyard_native -buildmode=c-shared \
    -o "$work/libswitchyard_smoke.$extension" "$module/testdata/cshared"
cc "$module_dir/testdata/cshared.c" -I "$work" -L "$work" \
    -Wl,-rpath,"$work" -lswitchyard_smoke -o "$work/shared-host"
DYLD_LIBRARY_PATH="$work" "$work/shared-host"
CGO_ENABLED=1 go mod vendor
test "$(CGO_ENABLED=0 go run -mod=vendor .)" = wasm
test "$(CGO_ENABLED=1 go run -mod=vendor -tags switchyard_native .)" = native
for goos in linux darwin; do
    for goarch in amd64 arm64; do
        CGO_ENABLED=0 GOOS="$goos" GOARCH="$goarch" go build -mod=vendor \
            -o "$work/consumer-$goos-$goarch" .
    done
done
echo "Packaged bindings passed without Rust or a local replace: $work"
