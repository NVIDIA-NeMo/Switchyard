// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !cgo || !switchyard_native

package switchyard

import (
	"bytes"
	"context"
	cryptorand "crypto/rand"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/api"
	"github.com/tetratelabs/wazero/imports/wasi_snapshot_preview1"
)

//go:embed internal/wasm/switchyard.wasm
var switchyardWasm []byte

const wasmABIVersion = 1

type wasmRuntime struct {
	runtime  wazero.Runtime
	compiled wazero.CompiledModule
	close    sync.Once
}

var wasmCompilationCache = wazero.NewCompilationCache()

type wasmTrapError struct {
	function string
	err      error
	stderr   string
}

func (err *wasmTrapError) Error() string {
	if err.stderr != "" {
		return fmt.Sprintf("Switchyard wasm %s trapped: %v: %s", err.function, err.err, err.stderr)
	}
	return fmt.Sprintf("Switchyard wasm %s trapped: %v", err.function, err.err)
}
func (err *wasmTrapError) Unwrap() error { return err.err }

type wasmGuestError struct{ message string }

func (err *wasmGuestError) Error() string { return err.message }

func isWasmTrap(err error) bool {
	var target *wasmTrapError
	return errors.As(err, &target)
}

func newWasmRuntime() (*wasmRuntime, error) {
	config := wazero.NewRuntimeConfigCompiler().
		WithCompilationCache(wasmCompilationCache).
		WithMemoryLimitPages(16384).
		WithCloseOnContextDone(true)
	runtime := wazero.NewRuntimeWithConfig(context.Background(), config)
	if _, err := wasi_snapshot_preview1.Instantiate(context.Background(), runtime); err != nil {
		_ = runtime.Close(context.Background())
		return nil, fmt.Errorf("instantiate WASI: %w", err)
	}
	compiled, err := runtime.CompileModule(context.Background(), switchyardWasm)
	if err != nil {
		_ = runtime.Close(context.Background())
		return nil, fmt.Errorf("compile Switchyard wasm: %w", err)
	}
	return &wasmRuntime{runtime: runtime, compiled: compiled}, nil
}

// guestStderr keeps the guest's latest stderr output, such as a Rust panic message.
type guestStderr struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

const guestStderrLimit = 4096

func (stderr *guestStderr) Write(data []byte) (int, error) {
	stderr.mu.Lock()
	defer stderr.mu.Unlock()
	stderr.buf.Write(data)
	if extra := stderr.buf.Len() - guestStderrLimit; extra > 0 {
		stderr.buf.Next(extra)
	}
	return len(data), nil
}

func (stderr *guestStderr) take() string {
	stderr.mu.Lock()
	defer stderr.mu.Unlock()
	text := strings.TrimSpace(stderr.buf.String())
	stderr.buf.Reset()
	return text
}

func (runtime *wasmRuntime) newModule(ctx context.Context) (*guestModule, error) {
	stderr := &guestStderr{}
	config := wazero.NewModuleConfig().
		WithStderr(stderr).
		WithStartFunctions().
		WithSysWalltime().
		WithSysNanotime().
		WithRandSource(cryptorand.Reader).
		WithName("")
	module, err := runtime.runtime.InstantiateModule(ctx, runtime.compiled, config)
	if err != nil {
		return nil, err
	}
	if _, err := module.ExportedFunction("_initialize").Call(ctx); err != nil {
		_ = module.Close(context.Background())
		return nil, fmt.Errorf("initialize Switchyard wasm: %w", err)
	}
	version, err := module.ExportedFunction("sy_abi_version").Call(ctx)
	if err != nil {
		_ = module.Close(context.Background())
		return nil, fmt.Errorf("read Switchyard wasm ABI: %w", err)
	}
	if len(version) != 1 || uint32(version[0]) != wasmABIVersion {
		_ = module.Close(context.Background())
		return nil, fmt.Errorf("Switchyard wasm ABI version mismatch: module=%v, binding=%d", version, wasmABIVersion)
	}
	return &guestModule{module: module, stderr: stderr}, nil
}

func (runtime *wasmRuntime) closeAll() {
	runtime.close.Do(func() {
		_ = runtime.runtime.Close(context.Background())
	})
}

type guestModule struct {
	module api.Module
	stderr *guestStderr
}

func (guest *guestModule) call(ctx context.Context, name string, args ...uint64) ([]uint64, error) {
	function := guest.module.ExportedFunction(name)
	if function == nil {
		return nil, fmt.Errorf("Switchyard wasm export %q is missing", name)
	}
	values, err := function.Call(ctx, args...)
	if err != nil {
		return nil, &wasmTrapError{function: name, err: err, stderr: guest.stderr.take()}
	}
	return values, nil
}

func (guest *guestModule) alloc(ctx context.Context, data []byte) (uint32, error) {
	if len(data) > int(^uint32(0)) {
		return 0, errors.New("Switchyard wasm buffer exceeds wasm32 memory")
	}
	values, err := guest.call(ctx, "sy_alloc", uint64(uint32(len(data))))
	if err != nil {
		return 0, err
	}
	if len(values) != 1 || uint32(values[0]) == 0 {
		return 0, errors.New("Switchyard wasm allocation failed")
	}
	ptr := uint32(values[0])
	if !guest.module.Memory().Write(ptr, data) {
		_, _ = guest.call(context.Background(), "sy_free", uint64(ptr), uint64(uint32(len(data))))
		return 0, errors.New("Switchyard wasm allocation is outside guest memory")
	}
	return ptr, nil
}

func (guest *guestModule) free(ctx context.Context, ptr uint32, length uint32) error {
	_, err := guest.call(ctx, "sy_free", uint64(ptr), uint64(length))
	return err
}

func (guest *guestModule) cstring(ctx context.Context, data []byte) (uint32, uint32, error) {
	data = append(bytes.Clone(data), 0)
	ptr, err := guest.alloc(ctx, data)
	return ptr, uint32(len(data)), err
}

func (guest *guestModule) readCString(ptr uint32) ([]byte, error) {
	if ptr == 0 {
		return nil, errors.New("Switchyard wasm returned a null string")
	}
	memory := guest.module.Memory()
	size := memory.Size()
	if size == 0 || ptr >= size {
		return nil, errors.New("Switchyard wasm string pointer is outside guest memory")
	}
	const chunkSize = uint32(16 * 1024)
	output := make([]byte, 0, chunkSize)
	for offset := ptr; offset < size; {
		count := min(chunkSize, size-offset)
		chunk, ok := memory.Read(offset, count)
		if !ok {
			return nil, errors.New("Switchyard wasm string is outside guest memory")
		}
		if end := bytes.IndexByte(chunk, 0); end >= 0 {
			output = append(output, chunk[:end]...)
			return output, nil
		}
		output = append(output, chunk...)
		offset += count
	}
	return nil, errors.New("Switchyard wasm returned an unterminated string")
}

func (guest *guestModule) guestError(ptr uint32) error {
	if ptr == 0 {
		return nil
	}
	message, err := guest.readCString(ptr)
	_, freeErr := guest.call(context.Background(), "sy_string_free", uint64(ptr))
	if err != nil {
		return err
	}
	if freeErr != nil {
		return freeErr
	}
	return &wasmGuestError{message: string(message)}
}

func (guest *guestModule) callJSON(ctx context.Context, export string, input []byte) ([]byte, error) {
	inputPtr, inputLen, err := guest.cstring(ctx, input)
	if err != nil {
		return nil, err
	}
	defer guest.free(context.Background(), inputPtr, inputLen)
	outPtr, err := guest.alloc(ctx, make([]byte, 4))
	if err != nil {
		return nil, err
	}
	defer guest.free(context.Background(), outPtr, 4)
	values, err := guest.call(ctx, export, uint64(inputPtr), uint64(outPtr))
	if err != nil {
		return nil, err
	}
	if len(values) != 1 {
		return nil, fmt.Errorf("Switchyard wasm %s returned %d values", export, len(values))
	}
	if err := guest.guestError(uint32(values[0])); err != nil {
		return nil, err
	}
	outputPtr, ok := guest.module.Memory().ReadUint32Le(outPtr)
	if !ok || outputPtr == 0 {
		return nil, errors.New("Switchyard wasm returned no JSON output")
	}
	output, err := guest.readCString(outputPtr)
	if err != nil {
		return nil, err
	}
	if _, err := guest.call(context.Background(), "sy_string_free", uint64(outputPtr)); err != nil {
		return nil, err
	}
	return output, nil
}

type wasmTranslator struct {
	runtime *wasmRuntime
	guest   *guestModule
	err     error
	mu      sync.Mutex
}

var defaultWasmTranslator struct {
	once  sync.Once
	value *wasmTranslator
}

func wasmTranslate(export string, input []byte) (json.RawMessage, error) {
	defaultWasmTranslator.once.Do(func() {
		runtime, err := newWasmRuntime()
		defaultWasmTranslator.value = &wasmTranslator{runtime: runtime, err: err}
	})
	translator := defaultWasmTranslator.value
	if translator.err != nil {
		return nil, translator.err
	}
	translator.mu.Lock()
	defer translator.mu.Unlock()
	if translator.guest == nil {
		guest, err := translator.runtime.newModule(context.Background())
		if err != nil {
			return nil, err
		}
		translator.guest = guest
	}
	output, err := translator.guest.callJSON(context.Background(), export, input)
	if err != nil && (isWasmTrap(err) || translator.guest.module.IsClosed()) {
		_ = translator.guest.module.Close(context.Background())
		translator.guest = nil
	}
	return json.RawMessage(output), err
}
