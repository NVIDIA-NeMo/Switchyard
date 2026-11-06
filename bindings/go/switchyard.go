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
	targetModels() (map[string]TargetModel, error)
	close()
}

// Router shares an algorithm across concurrent requests and value copies. Its zero value is closed.
type Router struct {
	state *routerState
}

type routerState struct {
	mu         sync.RWMutex
	backend    engineBackend
	configured bool
}

// Input supplies a normalized request. Raw routers also require host-approved model groups.
type Input struct {
	Request        json.RawMessage     `json:"request"`
	Models         map[string][]string `json:"models"`
	SubagentModels map[string][]string `json:"subagent_models"`
	Metadata       *RequestMetadata    `json:"metadata,omitempty"`
	// Nil supplies no assessment; a non-nil empty slice explicitly supplies no requirements.
	RequiredLabels []string `json:"required_labels"`
}

// RequestMetadata carries routing context separately from the model request.
type RequestMetadata struct {
	SessionID       string `json:"session_id,omitempty"`
	AgentID         string `json:"agent_id,omitempty"`
	IsSubagent      bool   `json:"is_subagent,omitempty"`
	IsDelegatedWork bool   `json:"is_delegated_work,omitempty"`
	SessionFinal    *bool  `json:"session_final,omitempty"`
	TaskID          string `json:"task_id,omitempty"`
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

// TargetModel identifies the host client and provider model serving a configured target.
type TargetModel struct {
	Client string `json:"client"`
	ID     string `json:"id"`
}

// HostCall executes model or decision work. It must honor ctx and may run concurrently.
// Return an error to let the algorithm apply its error policy, or Abort(err) to stop routing.
type HostCall func(context.Context, Work) (Reply, error)

type abortError struct{ err error }

func (e abortError) Error() string { return e.err.Error() }
func (e abortError) Unwrap() error { return e.err }

// Abort marks a host call refusal. Route stops and returns err, even when the
// algorithm would recover from a failed call. A nil error remains nil.
func Abort(err error) error {
	if err == nil {
		return nil
	}
	return abortError{err}
}

// hostReplyJSON encodes a host result for sy_run_respond.
func hostReplyJSON(reply Reply, err error) ([]byte, error) {
	envelope := struct {
		Response    json.RawMessage `json:"response,omitempty"`
		ServedModel string          `json:"served_model,omitempty"`
		Error       *string         `json:"error,omitempty"`
		Abort       *string         `json:"abort,omitempty"`
	}{Response: reply.Response, ServedModel: reply.ServedModel}
	if err != nil {
		envelope.Response = nil
		message := err.Error()
		if errors.As(err, new(abortError)) {
			envelope.Abort = &message
		} else {
			envelope.Error = &message
		}
	}
	return json.Marshal(envelope)
}

// hostAbort returns the host's original error when it aborted the call.
func hostAbort(err error) error {
	var abort abortError
	if errors.As(err, &abort) {
		return abort.err
	}
	return nil
}

// Outcome retains the ordered candidates, prepared requests, and any routing-produced answer.
type Outcome struct {
	SelectedModelIDs []string         `json:"selected_model_ids"`
	Request          json.RawMessage  `json:"request"`
	FallbackRequest  *json.RawMessage `json:"fallback_request"`
	// CandidateRequests overrides Request and FallbackRequest by model ID.
	CandidateRequests map[string]json.RawMessage `json:"candidate_requests"`
	Response          *json.RawMessage           `json:"response"`
	Metadata          json.RawMessage            `json:"metadata"`
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

// NewConfigured loads one TOML route without provider clients.
func NewConfigured(source, routeID string) (*Router, error) {
	config, err := json.Marshal(struct {
		Configured configuredSource `json:"configured"`
	}{Configured: configuredSource{TOML: source, RouteID: routeID}})
	if err != nil {
		return nil, err
	}
	router, err := New(config)
	if err != nil {
		return nil, err
	}
	router.state.configured = true
	return router, nil
}

type configuredSource struct {
	TOML    string `json:"toml"`
	RouteID string `json:"route_id"`
}

// TargetModels returns all configured targets, including alternatives, not the current eligible pool.
// LLM keys are target names; Decision keys are provider model IDs. Requires NewConfigured.
func (router *Router) TargetModels() (map[string]TargetModel, error) {
	if router == nil || router.state == nil {
		return nil, errors.New("Switchyard router is closed")
	}
	state := router.state
	state.mu.RLock()
	backend := state.backend
	state.mu.RUnlock()
	if backend == nil {
		return nil, errors.New("Switchyard router is closed")
	}
	if !state.configured {
		return nil, errors.New("target discovery needs NewConfigured")
	}
	return backend.targetModels()
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
// Host errors follow the algorithm's configured error policy. Aborted calls stop routing.
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
	if input.RequiredLabels != nil && !state.configured {
		return Outcome{}, errors.New("target requirements need NewConfigured")
	}
	if state.configured && (input.Models != nil || input.SubagentModels != nil) {
		return Outcome{}, errors.New("configured routers do not accept host model groups")
	}
	return backend.route(ctx, input, call)
}
