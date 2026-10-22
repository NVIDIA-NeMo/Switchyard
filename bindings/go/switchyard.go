// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package switchyard runs embedded routing algorithms with host-owned model calls.
package switchyard

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
)

type engineBackend interface {
	route(context.Context, Input, HostCall) (Outcome, error)
	close()
}

// Router shares an algorithm across concurrent requests and value copies. Its zero value is closed.
type Router struct {
	state *routerState
}

type routerState struct {
	mu      sync.RWMutex
	backend engineBackend
}

// Input supplies a normalized request and the host-approved model groups for this run.
type Input struct {
	Request        json.RawMessage     `json:"request"`
	Models         map[string][]string `json:"models"`
	SubagentModels map[string][]string `json:"subagent_models"`
	Metadata       *RequestMetadata    `json:"metadata,omitempty"`
}

// RequestMetadata carries routing context separately from the model request.
type RequestMetadata struct {
	SessionID       string `json:"session_id,omitempty"`
	AgentID         string `json:"agent_id,omitempty"`
	IsSubagent      bool   `json:"is_subagent,omitempty"`
	IsDelegatedWork bool   `json:"is_delegated_work,omitempty"`
	SessionFinal    *bool  `json:"session_final,omitempty"`
}

// Work requests either a model call (Models) or a typed decision call (Model).
type Work struct {
	Type    string          `json:"type"`
	ID      uint64          `json:"id"`
	Models  []string        `json:"models"`
	Model   string          `json:"model"`
	Request json.RawMessage `json:"request"`
}

// Reply contains a normalized response and, for model calls, the target that answered.
type Reply struct {
	Response    json.RawMessage `json:"response"`
	ServedModel string          `json:"served_model,omitempty"`
}

// HostCall executes model or decision work. It must honor ctx and may run concurrently.
type HostCall func(context.Context, Work) (Reply, error)

// Outcome retains the ordered candidates, prepared request, and any routing-produced answer.
type Outcome struct {
	SelectedModelIDs []string         `json:"selected_model_ids"`
	Request          json.RawMessage  `json:"request"`
	Response         *json.RawMessage `json:"response"`
	Metadata         json.RawMessage  `json:"metadata"`
}

// New constructs an algorithm from {"algorithm": AlgorithmSpec, "targets": {alias: modelID}}.
func New(config json.RawMessage) (*Router, error) {
	if !json.Valid(config) {
		return nil, errors.New("Switchyard config must be valid JSON")
	}
	backend, err := newBackend(config)
	if err != nil {
		return nil, err
	}
	return &Router{state: &routerState{backend: backend}}, nil
}

// Close prevents new runs and releases this Router's backend. Repeated calls and value copies are safe.
func (router *Router) Close() {
	if router == nil || router.state == nil {
		return
	}
	state := router.state
	state.mu.Lock()
	defer state.mu.Unlock()
	if state.backend != nil {
		state.backend.close()
		state.backend = nil
	}
}

// Route waits for an outcome while the host services model and decision calls.
// Host errors follow the algorithm's configured error policy.
func (router *Router) Route(ctx context.Context, input Input, call HostCall) (Outcome, error) {
	if router == nil || router.state == nil {
		return Outcome{}, errors.New("Switchyard router is closed")
	}
	state := router.state
	state.mu.RLock()
	backend := state.backend
	state.mu.RUnlock()
	if backend == nil {
		return Outcome{}, errors.New("Switchyard router is closed")
	}
	if err := ctx.Err(); err != nil {
		return Outcome{}, err
	}
	return backend.route(ctx, input, call)
}
