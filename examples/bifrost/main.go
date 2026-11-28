// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"net"
	"net/http"
	"net/url"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"

	switchyard "github.com/NVIDIA-NeMo/Switchyard/bindings/go"
	"github.com/maximhq/bifrost/core/schemas"
)

const selectedModelKey schemas.BifrostContextKey = "nvidia.switchyard.selected_model"

var (
	mu     sync.RWMutex
	routes map[string]*route
)

func Init(config any) error {
	mu.Lock()
	defer mu.Unlock()
	if routes != nil {
		return errors.New("switchyard is already initialized")
	}
	data, err := json.Marshal(config)
	if err != nil {
		return err
	}
	var cfg struct {
		Routes         map[string]json.RawMessage `json:"routes"`
		BifrostURL     string                     `json:"bifrost_url"`
		TimeoutSeconds int                        `json:"timeout_seconds"`
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return err
	}
	if len(cfg.Routes) == 0 {
		return errors.New("switchyard requires at least one route")
	}
	if cfg.TimeoutSeconds == 0 {
		cfg.TimeoutSeconds = 300
	}
	if cfg.TimeoutSeconds < 0 {
		return errors.New("timeout_seconds must be positive")
	}
	if cfg.BifrostURL != "" {
		endpoint, err := url.Parse(cfg.BifrostURL)
		if err != nil || (endpoint.Scheme != "http" && endpoint.Scheme != "https") || (net.ParseIP(endpoint.Hostname()) == nil || !net.ParseIP(endpoint.Hostname()).IsLoopback()) || endpoint.User != nil || endpoint.RawQuery != "" || endpoint.ForceQuery || strings.Contains(cfg.BifrostURL, "#") {
			return errors.New("bifrost_url must use a loopback IP and have no credentials, query, or fragment")
		}
	}
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.Proxy = nil
	client := &http.Client{Transport: transport, Timeout: time.Duration(cfg.TimeoutSeconds) * time.Second, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	next := make(map[string]*route)
	defer func() {
		for _, e := range next {
			e.router.Close()
		}
	}()
	for alias, spec := range cfg.Routes {
		if alias == "" {
			return errors.New("switchyard route name must not be empty")
		}
		e, err := newRoute(spec)
		if err != nil {
			return fmt.Errorf("route %s: %w", alias, err)
		}
		e.baseURL = strings.TrimRight(cfg.BifrostURL, "/")
		e.client = client
		next[alias] = e
	}
	for _, r := range next {
		for _, target := range r.targets {
			if _, managed := next[target]; managed {
				return fmt.Errorf("target %q is a managed route and would recurse", target)
			}
		}
	}
	routes = next
	next = nil
	return nil
}

func GetName() string { return "nvidia-switchyard" }

func Cleanup() error {
	mu.Lock()
	defer mu.Unlock()
	for _, e := range routes {
		e.router.Close()
		if e.client != nil {
			e.client.CloseIdleConnections()
		}
	}
	routes = nil
	return nil
}

func HTTPTransportPreHook(ctx *schemas.BifrostContext, req *schemas.HTTPRequest) (*schemas.HTTPResponse, error) {
	if req.Method != "POST" {
		return nil, nil
	}
	switch req.Path {
	case "/v1/chat/completions", "/openai/v1/chat/completions", "/openai/chat/completions":
	default:
		return nil, nil
	}
	if req.Body == nil {
		return nil, nil
	}
	var input struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(req.Body, &input); err != nil {
		return reject(err)
	}
	mu.RLock()
	defer mu.RUnlock()
	e, ok := routes[input.Model]
	if !ok {
		return nil, nil
	}
	if err := ctx.Err(); err != nil {
		return reject(err)
	}
	decision, selected, response, err := e.execute(ctx, req)
	if err != nil {
		return reject(err)
	}
	ctx.SetValue(selectedModelKey, selected)
	if response != nil {
		response.Headers["x-switchyard-selected-model"] = selected
		return response, nil
	}
	req.Body = decision
	return nil, nil
}

func HTTPTransportResponseHeadersHook(ctx *schemas.BifrostContext, _ *schemas.HTTPRequest, resp *schemas.HTTPResponseMetadata) error {
	if selected, ok := ctx.Value(selectedModelKey).(string); ok {
		resp.SetHeader("x-switchyard-selected-model", selected)
	}
	return nil
}

func reject(err error) (*schemas.HTTPResponse, error) {
	body, _ := json.Marshal(map[string]any{"error": map[string]string{"type": "switchyard_routing_error", "message": err.Error()}})
	status := 400
	var failure *gatewayError
	if errors.As(err, &failure) {
		status = failure.status
	}
	return &schemas.HTTPResponse{StatusCode: status, Headers: map[string]string{"Content-Type": "application/json"}, Body: body}, nil
}

type route struct {
	router         *switchyard.Router
	targets        []string
	models         map[string][]string
	subagentModels map[string][]string
	baseURL        string
	client         *http.Client
}

func newRoute(config json.RawMessage) (*route, error) {
	var cfg struct {
		Targets        map[string]string   `json:"targets"`
		Models         map[string][]string `json:"models"`
		SubagentModels map[string][]string `json:"subagent_models"`
		Algorithm      struct {
			Type string `json:"type"`
		} `json:"algorithm"`
	}
	if err := json.Unmarshal(config, &cfg); err != nil {
		return nil, err
	}

	if len(cfg.Targets) == 0 && cfg.Algorithm.Type != "noop" {
		return nil, errors.New("route requires targets")
	}
	r := &route{models: cfg.Models, subagentModels: cfg.SubagentModels}
	for _, model := range cfg.Targets {
		provider, name, ok := strings.Cut(model, "/")
		if !ok || provider == "" || name == "" {
			return nil, fmt.Errorf("target %q must use provider/model", model)
		}
		r.targets = append(r.targets, model)
	}
	if len(r.models["any"]) == 0 && cfg.Algorithm.Type != "noop" {
		return nil, errors.New("route requires models.any")
	}
	for _, groups := range []map[string][]string{r.models, r.subagentModels} {
		for _, models := range groups {
			for _, model := range models {
				if !slices.Contains(r.targets, model) {
					return nil, fmt.Errorf("model %q is outside configured targets", model)
				}
			}
		}
	}
	var err error
	r.router, err = switchyard.New(config)
	return r, err
}

func (r *route) execute(ctx context.Context, req *schemas.HTTPRequest) (json.RawMessage, string, *schemas.HTTPResponse, error) {
	request, err := switchyard.DecodeRequest("openai_chat", req.Body)
	if err != nil {
		return nil, "", nil, err
	}
	original := &schemas.HTTPRequest{Headers: maps.Clone(req.Headers)}
	scope := sha256.New()
	for _, name := range authHeaders {
		scope.Write([]byte(name + ":" + original.CaseInsensitiveHeaderLookup(name) + "\n"))
	}
	outcome, err := r.router.Route(ctx, switchyard.Input{
		Request: request, Models: r.models, SubagentModels: r.subagentModels,
		Headers: original.Headers, MetadataScope: fmt.Sprintf("%x", scope.Sum(nil)),
	}, func(ctx context.Context, work switchyard.Work) (switchyard.Reply, error) {
		return r.call(ctx, original, work)
	})
	if err != nil {
		return nil, "", nil, err
	}
	if outcome.Response != nil {
		var aggregate struct{ Model string }
		if err := json.Unmarshal(*outcome.Response, &aggregate); err != nil {
			return nil, "", nil, err
		}
		selected := aggregate.Model
		if len(outcome.SelectedModelIDs) > 0 {
			selected = outcome.SelectedModelIDs[0]
		}
		contentType, body, err := switchyard.EncodeResponse("openai_chat", *outcome.Response, request)
		return nil, selected, &schemas.HTTPResponse{StatusCode: 200, Headers: map[string]string{"Content-Type": contentType}, Body: body}, err
	}
	if len(outcome.SelectedModelIDs) == 0 {
		return nil, "", nil, errors.New("Switchyard returned no targets or answer")
	}
	model := outcome.SelectedModelIDs[0]
	if !slices.Contains(r.targets, model) {
		return nil, "", nil, errors.New("selected model is outside configured targets")
	}
	if len(outcome.SelectedModelIDs) > 1 && (outcome.FallbackRequest != nil || len(outcome.CandidateRequests) > 0) {
		for index, target := range outcome.SelectedModelIDs {
			prepared := outcome.Request
			if index > 0 && outcome.FallbackRequest != nil {
				prepared = *outcome.FallbackRequest
			}
			if candidate, ok := outcome.CandidateRequests[target]; ok {
				prepared = candidate
			}
			var reply switchyard.Reply
			reply, err = r.call(ctx, original, switchyard.Work{Type: "call_model", Models: []string{target}, Request: prepared})
			if err != nil {
				if ctx.Err() != nil || admissionError(err) {
					break
				}
				continue
			}
			contentType, body, failure := switchyard.EncodeResponse("openai_chat", reply.Response, request)
			return nil, reply.ServedModel, &schemas.HTTPResponse{StatusCode: 200, Headers: map[string]string{"Content-Type": contentType}, Body: body}, failure
		}
		return nil, "", nil, err
	}

	prepared := outcome.Request
	if candidate, ok := outcome.CandidateRequests[model]; ok {
		prepared = candidate
	}
	body, err := rewrite(req.Body, request, prepared, model)
	if err != nil {
		return nil, "", nil, err
	}
	if len(outcome.SelectedModelIDs) > 1 {
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(body, &fields); err != nil {
			return nil, "", nil, err
		}
		fields["fallbacks"], err = json.Marshal(outcome.SelectedModelIDs[1:])
		if err != nil {
			return nil, "", nil, err
		}
		body, err = json.Marshal(fields)
	}
	return body, model, nil, err
}

func rewrite(body, request, prepared json.RawMessage, model string) (json.RawMessage, error) {
	before, err := switchyard.EncodeRequest("openai_chat", request, model)
	if err != nil {
		return nil, err
	}
	after, err := switchyard.EncodeRequest("openai_chat", prepared, model)
	if err != nil {
		return nil, err
	}
	var original, old, updated map[string]json.RawMessage
	if err := json.Unmarshal(body, &original); err != nil {
		return nil, err
	}
	if err := json.Unmarshal(before, &old); err != nil {
		return nil, err
	}
	if err := json.Unmarshal(after, &updated); err != nil {
		return nil, err
	}
	// Preserve caller fields that routing did not change.
	for key, value := range old {
		if _, ok := updated[key]; !ok {
			delete(original, key)
		} else if bytes.Equal(value, updated[key]) {
			delete(updated, key)
		}
	}
	for key, value := range updated {
		original[key] = value
	}
	original["model"], _ = json.Marshal(model)
	result, err := json.Marshal(original)
	return result, err
}

var authHeaders = []string{"authorization", "x-bf-vk", "x-api-key", "x-goog-api-key", "api-key", "x-bf-direct-key"}

type gatewayError struct{ status int }

func admissionError(err error) bool {
	var failure *gatewayError
	return errors.As(err, &failure) && (failure.status == 401 || failure.status == 403 || failure.status == 429)
}

func (e *gatewayError) Error() string {
	return fmt.Sprintf("Bifrost model call returned HTTP %d", e.status)
}

func (r *route) post(ctx context.Context, original *schemas.HTTPRequest, path string, body json.RawMessage) (json.RawMessage, error) {
	if r.baseURL == "" {
		return nil, switchyard.Abort(errors.New("bifrost_url is required for intermediate model calls"))
	}
	req, err := http.NewRequestWithContext(ctx, "POST", r.baseURL+path, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	// Init restricts this client to a loopback IP, with proxies and redirects disabled.
	for _, name := range authHeaders {
		if value := original.CaseInsensitiveHeaderLookup(name); value != "" {
			req.Header.Set(name, value)
		}
	}
	if value := original.CaseInsensitiveHeaderLookup("x-bf-compat"); value != "" {
		req.Header.Set("x-bf-compat", value)
	}
	response, err := r.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		failure := &gatewayError{response.StatusCode}
		if response.StatusCode == 401 || response.StatusCode == 403 || response.StatusCode == 429 {
			return nil, switchyard.Abort(failure)
		}
		return nil, failure
	}
	return io.ReadAll(response.Body)
}

func (r *route) call(ctx context.Context, original *schemas.HTTPRequest, work switchyard.Work) (switchyard.Reply, error) {
	if work.Type == "call_decision" {
		if !slices.Contains(r.targets, work.Model) {
			return switchyard.Reply{}, switchyard.Abort(errors.New("unconfigured decision target"))
		}
		response, err := r.decide(ctx, original, work)
		return switchyard.Reply{Response: response}, err
	}
	var last error
	for _, model := range work.Models {
		if !slices.Contains(r.targets, model) {
			return switchyard.Reply{}, switchyard.Abort(errors.New("unconfigured model target"))
		}
		body, err := switchyard.EncodeRequest("openai_chat", work.Request, model)
		if err != nil {
			return switchyard.Reply{}, err
		}
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(body, &fields); err != nil {
			return switchyard.Reply{}, err
		}
		fields["stream"] = json.RawMessage("false")
		delete(fields, "stream_options")
		delete(fields, "fallbacks")
		body, err = json.Marshal(fields)
		if err != nil {
			return switchyard.Reply{}, err
		}
		response, err := r.post(ctx, original, "/v1/chat/completions", body)
		var failure *gatewayError
		if errors.As(err, &failure) && (failure.status == 400 || failure.status == 422) && fields["chat_template_kwargs"] != nil {
			delete(fields, "chat_template_kwargs")
			body, err = json.Marshal(fields)
			if err == nil {
				response, err = r.post(ctx, original, "/v1/chat/completions", body)
			}
		}
		if err == nil {
			var normalized json.RawMessage
			normalized, err = switchyard.DecodeResponse("openai_chat", response)
			if err == nil {
				return switchyard.Reply{Response: normalized, ServedModel: model}, nil
			}
		}
		if admissionError(err) {
			return switchyard.Reply{}, err
		}
		last = err
		if ctx.Err() != nil {
			return switchyard.Reply{}, ctx.Err()
		}
	}
	if last == nil {
		last = errors.New("model call has no candidates")
	}
	return switchyard.Reply{}, last
}

type decisionQuestion struct {
	Instructions any `json:"instructions"`
	Kind         struct {
		Type string `json:"type"`
		Data struct {
			Options []struct {
				ID          string `json:"id"`
				Description any    `json:"description"`
			} `json:"options"`
		} `json:"data"`
	} `json:"kind"`
}

func decisionText(value any) string {
	if text, ok := value.(string); ok {
		return text
	}
	body, _ := json.Marshal(value)
	return string(body)
}

func (r *route) decide(ctx context.Context, original *schemas.HTTPRequest, work switchyard.Work) (json.RawMessage, error) {
	var input struct {
		Input     any                         `json:"input"`
		Questions map[string]decisionQuestion `json:"questions"`
	}
	if err := json.Unmarshal(work.Request, &input); err != nil {
		return nil, err
	}
	names := make([]string, 0, len(input.Questions))
	for name := range input.Questions {
		names = append(names, name)
	}
	sort.Strings(names)
	questions := make([]map[string]any, 0, len(names))
	for _, name := range names {
		q := input.Questions[name]
		wire := map[string]any{"name": name, "instructions": decisionText(q.Instructions), "type": q.Kind.Type}
		if q.Kind.Type != "choice" {
			return nil, fmt.Errorf("unsupported decision kind %q", q.Kind.Type)
		}
		choices := make([]map[string]any, 0, len(q.Kind.Data.Options))
		for _, option := range q.Kind.Data.Options {
			choice := map[string]any{"value": option.ID}
			if option.Description != nil {
				choice["description"] = decisionText(option.Description)
			}
			choices = append(choices, choice)
		}
		wire["choices"] = choices
		questions = append(questions, wire)
	}
	body, err := json.Marshal(map[string]any{"model": work.Model, "input": decisionText(input.Input), "questions": questions})
	if err != nil {
		return nil, err
	}
	body, err = r.post(ctx, original, "/v1/decisions", body)
	if err != nil {
		return nil, err
	}
	var response schemas.BifrostDecisionResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, err
	}
	answers := make(map[string]any, len(response.Answers))
	for _, answer := range response.Answers {
		if answer.Name == nil {
			return nil, fmt.Errorf("decision answer has no name")
		}
		name := *answer.Name
		q, ok := input.Questions[name]
		if !ok {
			return nil, fmt.Errorf("unexpected decision answer %q", name)
		}
		if _, duplicate := answers[name]; duplicate {
			return nil, fmt.Errorf("duplicate decision answer %q", name)
		}
		var value any
		switch {
		case answer.Type == schemas.DecisionTypeRefusal:
			value = map[string]any{"type": "refusal"}
		case q.Kind.Type == "choice" && answer.Type == "choice" && answer.Choice != nil && answer.Choice.Str != nil:
			selected := *answer.Choice.Str
			valid := false
			for _, option := range q.Kind.Data.Options {
				valid = valid || option.ID == selected
			}
			if !valid {
				return nil, fmt.Errorf("unconfigured decision choice %q", selected)
			}
			var probabilities map[string]float64
			if len(answer.Probabilities) > 0 {
				probabilities = map[string]float64{}
				for _, p := range answer.Probabilities {
					if p.Value.Str == nil || p.Probability < 0 || p.Probability > 1 {
						return nil, fmt.Errorf("invalid choice probability")
					}
					if _, duplicate := probabilities[*p.Value.Str]; duplicate {
						return nil, fmt.Errorf("duplicate choice probability")
					}
					probabilities[*p.Value.Str] = p.Probability
				}
				if len(probabilities) != len(q.Kind.Data.Options) {
					return nil, fmt.Errorf("incomplete choice probabilities")
				}
				for _, option := range q.Kind.Data.Options {
					if _, ok := probabilities[option.ID]; !ok {
						return nil, fmt.Errorf("missing choice probability")
					}
				}
			}
			value = map[string]any{"type": "choice", "data": map[string]any{"selected": selected, "probabilities": probabilities}}
		default:
			return nil, fmt.Errorf("invalid or refused decision answer %q", name)
		}
		answers[name] = map[string]any{"value": value, "provider_confidence": answer.Confidence}
	}
	if len(answers) != len(input.Questions) {
		return nil, fmt.Errorf("decision answer names do not match questions")
	}
	usage := map[string]any{}
	if u := response.Usage; u != nil {
		cached, written := 0, 0
		if u.PromptTokensDetails != nil {
			cached = u.PromptTokensDetails.CachedReadTokens
			written = u.PromptTokensDetails.CachedWriteTokens
			usage["cached_input_tokens"] = cached
			usage["cache_creation_input_tokens"] = written
		}
		usage["input_tokens"] = max(0, u.PromptTokens-cached-written)
		usage["output_tokens"] = u.CompletionTokens
		usage["total_tokens"] = u.TotalTokens
		if u.CompletionTokensDetails != nil {
			usage["reasoning_tokens"] = u.CompletionTokensDetails.ReasoningTokens
		}
	}
	return json.Marshal(map[string]any{"id": response.ID, "model": response.Model, "answers": answers, "usage": usage})
}

func main() {}
