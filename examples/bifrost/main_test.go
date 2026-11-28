// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	switchyard "github.com/NVIDIA-NeMo/Switchyard/bindings/go"
	"github.com/maximhq/bifrost/core/schemas"
)

const stage = `{"type":"stage_router","capable_target":"openai/capable","efficient_target":"openai/efficient","picker":"efficient_first","confidence_threshold":0.5}`
const request = `{"model":"switchyard","messages":[{"role":"user","content":"Hello"}],"temperature":0.25,"max_tokens":64,"stream":true,"stream_options":{"include_usage":true},"metadata":{"request_id":"test"}}`
const critical = `{"model":"switchyard","messages":[{"role":"user","content":"Fix tests"},{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"Bash","arguments":"{\"command\":\"pytest\"}"}}]},{"role":"tool","tool_call_id":"c1","content":"fatal runtime error: out of memory"}]}`

func initialize(t *testing.T, spec string) {
	t.Helper()
	if err := Init(map[string]any{"routes": map[string]json.RawMessage{"switchyard": config(spec)}}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { Cleanup() })
}

func callHook(body, path string) (*schemas.BifrostContext, *schemas.HTTPRequest, *schemas.HTTPResponse, error) {
	ctx := schemas.NewBifrostContext(context.Background(), time.Time{})
	req := &schemas.HTTPRequest{Method: "POST", Path: path, Body: []byte(body)}
	response, err := HTTPTransportPreHook(ctx, req)
	return ctx, req, response, err
}

func TestStageRoutingPreservesCallerFields(t *testing.T) {
	initialize(t, stage)
	for _, path := range []string{"/v1/chat/completions", "/openai/v1/chat/completions", "/openai/chat/completions"} {
		ctx, req, response, err := callHook(request, path)
		if response != nil || err != nil {
			t.Fatalf("%s: %v %v", path, response, err)
		}
		var before, after map[string]json.RawMessage
		json.Unmarshal([]byte(request), &before)
		json.Unmarshal(req.Body, &after)
		if string(after["model"]) != `"openai/efficient"` {
			t.Fatalf("unexpected model: %s", req.Body)
		}
		delete(before, "model")
		delete(after, "model")
		delete(after, "fallbacks")
		a, _ := json.Marshal(before)
		b, _ := json.Marshal(after)
		if string(a) != string(b) {
			t.Fatalf("caller fields changed: %s", req.Body)
		}
		headers := &schemas.HTTPResponseMetadata{}
		HTTPTransportResponseHeadersHook(ctx, req, headers)
		if headers.Header("x-switchyard-selected-model") != "openai/efficient" {
			t.Fatal("missing selected model header")
		}
	}
}

func TestStageEscalationAndHandoff(t *testing.T) {
	spec := strings.TrimSuffix(stage, "}") + `,"handoff_notes":{"escalation_note":"Recheck the failure."}}`
	initialize(t, spec)
	_, req, response, err := callHook(critical, "/v1/chat/completions")
	if response != nil || err != nil {
		t.Fatal(response, err)
	}
	var body struct {
		Model    string            `json:"model"`
		Messages []json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(req.Body, &body); err != nil {
		t.Fatal(err)
	}
	if body.Model != "openai/capable" || !strings.Contains(string(req.Body), "Recheck the failure.") {
		t.Fatalf("missing escalation: %s", req.Body)
	}
	if !strings.Contains(string(req.Body), `"tool_call_id":"c1"`) {
		t.Fatal("tool history lost")
	}
}

func TestRandomSeedAndWeights(t *testing.T) {
	spec := `{"type":"random","targets":["openai/capable","openai/efficient"],"seed":6}`
	sequence := func() []string {
		if err := Init(map[string]any{"routes": map[string]json.RawMessage{"switchyard": config(spec)}}); err != nil {
			t.Fatal(err)
		}
		defer Cleanup()
		var selected []string
		for range 32 {
			ctx, _, response, err := callHook(request, "/v1/chat/completions")
			if response != nil || err != nil {
				t.Fatal(response, err)
			}
			selected = append(selected, ctx.Value(selectedModelKey).(string))
		}
		return selected
	}
	first, second := sequence(), sequence()
	if strings.Join(first, ",") != strings.Join(second, ",") {
		t.Fatal("same seed diverged")
	}
	seen := map[string]bool{}
	for _, model := range first {
		seen[model] = true
	}
	if len(seen) != 2 {
		t.Fatal("both candidates were not exercised")
	}
	initialize(t, strings.TrimSuffix(spec, "}")+`,"weights":[0,1]}`)
	for range 8 {
		ctx, _, response, err := callHook(request, "/v1/chat/completions")
		if response != nil || err != nil || ctx.Value(selectedModelKey) != "openai/efficient" {
			t.Fatal("zero-weight candidate selected")
		}
	}
}

func TestInvalidConfigAndRequest(t *testing.T) {
	for _, spec := range []string{
		`{"type":"random","targets":["openai/a"],"weights":[-1]}`,
		`{"type":"random","targets":["openai/a"],"typo":true}`,
		strings.Replace(stage, "0.5", "2", 1),
	} {
		if r, err := newRoute(config(spec)); err == nil {
			r.router.Close()
			t.Fatalf("accepted invalid spec: %s", spec)
		}
	}
	initialize(t, stage)
	for _, body := range []string{`{`, `{"model":"switchyard","messages":[{"role":"alien","content":"bad"}]}`} {
		_, req, response, err := callHook(body, "/v1/chat/completions")
		if err != nil || response == nil || response.StatusCode != 400 || string(req.Body) != body {
			t.Fatal("request did not fail closed")
		}
	}
}

func TestBypassAndLifecycle(t *testing.T) {
	initialize(t, stage)
	for _, tc := range []struct{ body, path string }{
		{strings.Replace(request, "switchyard", "openai/direct", 1), "/v1/chat/completions"},
		{request, "/v1/responses"},
	} {
		ctx, req, response, err := callHook(tc.body, tc.path)
		if err != nil || response != nil || string(req.Body) != tc.body || ctx.Value(selectedModelKey) != nil {
			t.Fatal("bypass changed request")
		}
	}
	var wg sync.WaitGroup
	for range 16 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 20 {
				_, _, response, err := callHook(request, "/v1/chat/completions")
				if err != nil || response != nil {
					t.Errorf("concurrent route failed: %v %v", response, err)
					return
				}
			}
		}()
	}
	wg.Wait()
	if err := Cleanup(); err != nil {
		t.Fatal(err)
	}
	if err := Cleanup(); err != nil {
		t.Fatal(err)
	}
	if err := Init(map[string]any{}); err == nil {
		t.Fatal("empty config accepted")
	}
	if err := Init(map[string]any{"routes": map[string]json.RawMessage{"switchyard": config(stage)}}); err != nil {
		t.Fatal(err)
	}
}

func config(algorithm string) json.RawMessage {
	return json.RawMessage(`{"algorithm":` + algorithm + `,"targets":{"openai/capable":"openai/capable","openai/efficient":"openai/efficient","openai/a":"openai/a"},"models":{"any":["openai/capable","openai/efficient"],"capable":["openai/capable"],"efficient":["openai/efficient"],"judge":["openai/a"],"premium":["openai/capable"],"budget":["openai/efficient"]}}`)
}

func TestIntermediateCallsFailClosed(t *testing.T) {
	initialize(t, `{"type":"composite","classifier":{"target":"openai/a","base_threshold":0.5,"classify_trigger":"user_turn","fail_open":true},"stage":{"capable_target":"openai/capable","efficient_target":"openai/efficient","confidence_threshold":0.5}}`)
	_, req, response, err := callHook(request, "/v1/chat/completions")
	if err != nil || response == nil || response.StatusCode != 400 || !strings.Contains(string(response.Body), "intermediate model calls") || string(req.Body) != request {
		t.Fatalf("intermediate call was not rejected: %v %v", response, err)
	}
}

func TestConfigurationAndCancellation(t *testing.T) {
	for _, value := range []json.RawMessage{
		json.RawMessage(`{"algorithm":{"type":"random"},"targets":{"a":"unqualified"},"models":{"any":["unqualified"]}}`),
		json.RawMessage(`{"algorithm":{"type":"random"},"targets":{"a":"openai/a"}}`),
		json.RawMessage(`{"algorithm":{"type":"random"},"targets":{"a":"openai/a"},"models":{"any":["openai/other"]}}`),
	} {
		if err := Init(map[string]any{"routes": map[string]json.RawMessage{"switchyard": value}}); err == nil {
			Cleanup()
			t.Fatal("invalid route initialized")
		}
	}
	initialize(t, stage)
	if err := Init(map[string]any{}); err == nil {
		t.Fatal("second initialization accepted")
	}
	parent, cancel := context.WithCancel(context.Background())
	cancel()
	ctx := schemas.NewBifrostContext(parent, time.Time{})
	req := &schemas.HTTPRequest{Method: "POST", Path: "/v1/chat/completions", Body: []byte(request)}
	response, err := HTTPTransportPreHook(ctx, req)
	if err != nil || response == nil || response.StatusCode != 400 || string(req.Body) != request || ctx.Value(selectedModelKey) != nil {
		t.Fatal("cancelled request did not fail closed")
	}
}

func gateway(t *testing.T) (string, *atomic.Int64) {
	t.Helper()
	calls := &atomic.Int64{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
			http.Error(w, "invalid", 400)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/v1/decisions" {
			answers := []any{}
			for _, raw := range body["questions"].([]any) {
				q := raw.(map[string]any)
				answer := map[string]any{"name": q["name"], "type": q["type"]}
				switch q["type"] {
				case "choice":
					options := q["choices"].([]any)
					selected := options[0].(map[string]any)["value"]
					for _, option := range options {
						value := option.(map[string]any)["value"]
						if value == "efficient" || value == "budget" {
							selected = value
						}
					}
					probabilities := []any{}
					for _, option := range options {
						value := option.(map[string]any)["value"]
						p := 0.1 / float64(max(1, len(options)-1))
						if value == selected {
							p = 0.9
						}
						probabilities = append(probabilities, map[string]any{"value": value, "probability": p})
					}
					answer["choice"] = selected
					answer["probabilities"] = probabilities
				}
				answers = append(answers, answer)
			}
			json.NewEncoder(w).Encode(map[string]any{"model": body["model"], "answers": answers, "usage": map[string]any{"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}})
			return
		}
		if r.URL.Path != "/v1/chat/completions" || body["stream"] != false {
			t.Errorf("unexpected callback: %s %+v", r.URL.Path, body)
			http.Error(w, "invalid", 400)
			return
		}
		content := "review answer"
		raw, _ := json.Marshal(body)
		switch {
		case strings.Contains(string(raw), "APPROVE"):
			content = "APPROVE"
		case strings.Contains(string(raw), "new_evidence"):
			content = `{"escalate":true,"category":"loop","new_evidence":true,"reason":"Repeated failure"}`
		case strings.Contains(string(raw), "Choose a model group."):
			content = `{"target":"premium"}`
		case strings.Contains(string(raw), "Choose a tier."):
			content = `{"target":"efficient"}`
		case body["response_format"] != nil:
			content = `{"crux":"bounded task","primary_rule":"SUP-1","capability_boundary":"supported","p_solve":0.9}`
		}
		choice := map[string]any{"index": 0, "message": map[string]any{"role": "assistant", "content": content}, "finish_reason": "stop"}
		if body["logprobs"] == true {
			choice["message"] = map[string]any{"role": "assistant", "content": "yes"}
			choice["logprobs"] = map[string]any{"content": []any{map[string]any{"token": "yes", "logprob": math.Log(0.999), "top_logprobs": []any{map[string]any{"token": "yes", "logprob": math.Log(0.999)}, map[string]any{"token": "no", "logprob": math.Log(0.001)}}}}}
		}
		json.NewEncoder(w).Encode(map[string]any{"id": "test-completion", "object": "chat.completion", "created": 1, "model": body["model"], "choices": []any{choice}, "usage": map[string]any{"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}})
	}))
	t.Cleanup(server.Close)
	return server.URL, calls
}

func initializeGateway(t *testing.T, algorithm, url string) {
	t.Helper()
	if err := Init(map[string]any{"bifrost_url": url, "routes": map[string]json.RawMessage{"switchyard": config(algorithm)}}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { Cleanup() })
}

const tierPair = `"capable_target":"openai/capable","efficient_target":"openai/efficient"`
const classifierPair = `"classifier_target":"openai/a","strong_target":"openai/capable","weak_target":"openai/efficient"`
const customGroups = `"models":{"any":["openai/capable","openai/efficient"],"premium":["openai/capable"],"budget":["openai/efficient"],"judge":["openai/a"]},"default_target":"premium"`
const customClassifier = `"mode":"custom",` + customGroups + `,"prompt":"Choose a model group.","response_schema":"{\"type\":\"object\",\"properties\":{\"target\":{\"type\":\"string\"}},\"required\":[\"target\"],\"additionalProperties\":false}","policy":{"type":"target_selector","selector":"/target"}`

func TestAlgorithmModes(t *testing.T) {
	for _, tc := range []struct {
		name, algorithm, body, selected, rejected string
	}{
		{"noop", `{"type":"noop"}`, request, "switchyard", "no generated response"},
		{"passthrough", `{"type":"passthrough","target":"openai/capable"}`, request, "openai/capable", ""},
		{"random", `{"type":"random","targets":["openai/capable","openai/efficient"],"weights":[0,1],"seed":6}`, request, "openai/efficient", ""},
		{"auto", `{"type":"auto",` + tierPair + `}`, request, "openai/efficient", ""},
		{"stage/efficient_first", stage, request, "openai/efficient", ""},
		{"stage/capable_first", strings.Replace(stage, "efficient_first", "capable_first", 1), request, "openai/capable", ""},
		{"stage/escalation", stage, critical, "openai/capable", ""},
		{"stage/classifier", strings.TrimSuffix(stage, "}") + `,"classifier":{"target":"openai/a","base_threshold":0.5}}`, request, "openai/efficient", "intermediate model calls"},
		{"stage/decision_classifier", strings.TrimSuffix(stage, "}") + `,"classifier":{"target":"openai/a","decision":{"cutoff":0.5}}}`, request, "openai/efficient", "intermediate model calls"},
		{"classifier/capability", `{"type":"llm_classifier","mode":"capability",` + classifierPair + `,"base_threshold":0.5}`, request, "openai/efficient", "intermediate model calls"},
		{"classifier/escalation", `{"type":"llm_classifier","mode":"escalation",` + classifierPair + `,"escalation":{"confirmations":1,"recent_turn_window":28,"window_message_chars":500}}`, request, "openai/efficient", "intermediate model calls"},
		{"classifier/custom", `{"type":"llm_classifier",` + customClassifier + `}`, request, "openai/capable", "intermediate model calls"},
		{"classifier/direct", `{"type":"llm_classifier","mode":"direct",` + customGroups + `}`, request, "openai/efficient", "intermediate model calls"},
		{"classifier/capability_decision", `{"type":"llm_classifier","mode":"capability",` + classifierPair + `,"decision":{"cutoff":0.5}}`, request, "openai/efficient", "intermediate model calls"},
		{"composite/capability", `{"type":"composite","classifier":{"target":"openai/a","base_threshold":0.5,"classify_trigger":"user_turn"},"stage":{` + tierPair + `,"confidence_threshold":0.5}}`, request, "openai/efficient", "intermediate model calls"},
		{"composite/profile_targets", `{"type":"composite","classifier":{"target":"openai/a","base_threshold":0.5,"classify_trigger":"user_turn"},"stage":{` + tierPair + `,"confidence_threshold":0.5},"efficient":{"efficient_target":"openai/capable","capable_target":"openai/efficient"}}`, request, "openai/capable", "intermediate model calls"},
		{"composite/decision", `{"type":"composite","classifier":{"target":"openai/a","decision":{"cutoff":0.5},"classify_trigger":"user_turn"},"stage":{` + tierPair + `,"confidence_threshold":0.5}}`, request, "openai/efficient", "intermediate model calls"},
		{"composite/custom", `{"type":"composite","classifier":{"target":"openai/a","mode":"custom","classify_trigger":"user_turn","prompt":"Choose a tier.","response_schema":"{\"type\":\"object\",\"properties\":{\"target\":{\"type\":\"string\"}},\"required\":[\"target\"],\"additionalProperties\":false}","policy":{"type":"target_selector","selector":"/target"}},"stage":{` + tierPair + `,"confidence_threshold":0.5}}`, request, "openai/efficient", "intermediate model calls"},
		{"advisor/no_tool_call", `{"type":"advisor","executor_target":"openai/efficient","advisor_target":"openai/a"}`, request, "openai/efficient", "intermediate model calls"},
		{"advisor/pattern", `{"type":"advisor","executor_target":"openai/efficient","advisor_target":"openai/a","gate_trigger":"pattern","gate_trigger_pattern":"review"}`, request, "openai/efficient", "intermediate model calls"},
		{"vgr/off", `{"type":"vgr",` + tierPair + `,"mode":"off"}`, request, "openai/capable", ""},
		{"vgr/evaluate", `{"type":"vgr",` + tierPair + `,"mode":"evaluate","task_typing":false}`, request, "openai/efficient", "intermediate model calls"},
		{"vgr/shadow", `{"type":"vgr",` + tierPair + `,"mode":"shadow","task_typing":false}`, request, "openai/capable", "intermediate model calls"},
		{"vgr/active", `{"type":"vgr",` + tierPair + `,"mode":"active","active_approval":"prospective-validation-and-canary-approved","task_typing":false}`, request, "openai/efficient", "intermediate model calls"},
	} {
		for _, stream := range []string{"true", "false"} {
			t.Run(tc.name+"/stream="+stream, func(t *testing.T) {
				var input map[string]json.RawMessage
				if err := json.Unmarshal([]byte(tc.body), &input); err != nil {
					t.Fatal(err)
				}
				input["stream"] = json.RawMessage(stream)
				data, err := json.Marshal(input)
				if err != nil {
					t.Fatal(err)
				}
				tc.body = string(data)
				url, calls := gateway(t)
				initializeGateway(t, tc.algorithm, url)
				ctx, req, response, err := callHook(tc.body, "/v1/chat/completions")
				if err != nil {
					t.Fatal(err)
				}
				if ctx.Value(selectedModelKey) != tc.selected {
					t.Fatalf("selected=%v want=%s response=%+v", ctx.Value(selectedModelKey), tc.selected, response)
				}
				t.Logf("selected=%v calls=%d inline=%t", ctx.Value(selectedModelKey), calls.Load(), response != nil)
				if response != nil {
					if response.StatusCode != 200 {
						t.Fatalf("route failed: %d %s", response.StatusCode, response.Body)
					}
					if stream == "true" {
						if response.Headers["Content-Type"] != "text/event-stream" || !strings.HasSuffix(string(response.Body), "data: [DONE]\n\n") {
							t.Fatalf("invalid stream: %s", response.Body)
						}
					} else {
						var answer struct {
							Choices []struct{ Message struct{ Content string } }
						}
						if err := json.Unmarshal(response.Body, &answer); err != nil || len(answer.Choices) == 0 || answer.Choices[0].Message.Content == "" {
							t.Fatalf("invalid answer: %s", response.Body)
						}
					}
					if tc.name != "noop" && calls.Load() == 0 {
						t.Fatal("no model was called")
					}
					return
				}
				if tc.rejected != "" && tc.name != "noop" && calls.Load() == 0 {
					t.Fatal("classifier was not called")
				}

				var body struct {
					Model  string
					Stream bool
				}
				if err := json.Unmarshal(req.Body, &body); err != nil {
					t.Fatal(err)
				}
				if body.Stream != (stream == "true") || (tc.selected != "" && body.Model != tc.selected) || ctx.Value(selectedModelKey) != body.Model {
					t.Fatalf("selected=%s want=%s", body.Model, tc.selected)
				}
			})
		}
	}
}

func TestSubagentRouting(t *testing.T) {
	for _, tc := range []struct{ policy, expected string }{
		{`{"type":"passthrough","target":"openai/efficient"}`, "openai/efficient"},
		{`{"type":"llm_classifier",` + customClassifier + `}`, "openai/capable"},
	} {
		t.Run(tc.expected, func(t *testing.T) {
			url, _ := gateway(t)
			spec := `{"type":"passthrough","target":"openai/capable","subagents":` + tc.policy + `}`
			var cfg map[string]any
			if err := json.Unmarshal(config(spec), &cfg); err != nil {
				t.Fatal(err)
			}
			cfg["models"] = map[string][]string{"any": {"openai/capable"}}
			cfg["subagent_models"] = map[string][]string{"any": {"openai/capable", "openai/efficient"}, "premium": {"openai/capable"}, "budget": {"openai/efficient"}, "judge": {"openai/a"}}
			if tc.expected == "openai/efficient" {
				cfg["subagent_models"] = map[string][]string{"any": {"openai/efficient"}}
			}

			if err := Init(map[string]any{"bifrost_url": url, "routes": map[string]any{"switchyard": cfg}}); err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { Cleanup() })
			ctx := schemas.NewBifrostContext(context.Background(), time.Time{})
			req := &schemas.HTTPRequest{Method: "POST", Path: "/v1/chat/completions", Body: []byte(request), Headers: map[string]string{"X-Switchyard-Is-Subagent": "true", "X-Switchyard-Agent-Id": "child", "X-Switchyard-Session-Id": "session"}}
			response, err := HTTPTransportPreHook(ctx, req)
			if err != nil || response != nil || ctx.Value(selectedModelKey) != tc.expected {
				t.Fatalf("subagent route: response=%+v err=%v selected=%v", response, err, ctx.Value(selectedModelKey))
			}
		})
	}
}

func TestClassifierRetention(t *testing.T) {
	for _, trigger := range []string{"every_request", "user_turn", "new_session"} {
		for _, hash := range []bool{false, true} {
			if hash && trigger == "every_request" {
				continue
			}
			t.Run(fmt.Sprintf("%s/hash=%t", trigger, hash), func(t *testing.T) {
				url, calls := gateway(t)
				spec := `{"type":"llm_classifier","mode":"capability",` + classifierPair + `,"base_threshold":0.5,"classify_trigger":"` + trigger + `","message_hash_fallback":` + fmt.Sprint(hash) + `}`
				initializeGateway(t, spec, url)
				send := func(auth, payload string) {
					ctx := schemas.NewBifrostContext(context.Background(), time.Time{})
					headers := map[string]string{"Authorization": "Bearer " + auth}
					if !hash {
						headers["X-Switchyard-Session-Id"] = "same-session"
					}
					req := &schemas.HTTPRequest{Method: "POST", Path: "/v1/chat/completions", Body: []byte(payload), Headers: headers}
					response, err := HTTPTransportPreHook(ctx, req)
					if err != nil || response != nil || ctx.Value(selectedModelKey) != "openai/efficient" {
						t.Fatalf("retained route: %v %v", response, err)
					}
				}
				send("test-a", request)
				continuation := request
				if trigger == "user_turn" {
					continuation = strings.Replace(critical, "Fix tests", "Hello", 1)
				}
				send("test-a", continuation)
				expected := int64(1)
				if trigger == "every_request" {
					expected = 2
				}
				if calls.Load() != expected {
					t.Fatalf("calls=%d want=%d", calls.Load(), expected)
				}
				send("test-b", request)
				if calls.Load() != expected+1 {
					t.Fatal("retained decision crossed caller credentials")
				}
			})
		}
	}
}

func TestAdmissionFailureDoesNotFailOpen(t *testing.T) {
	for _, status := range []int{401, 403, 429} {
		t.Run(fmt.Sprint(status), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { http.Error(w, "private provider detail", status) }))
			defer server.Close()
			initializeGateway(t, `{"type":"llm_classifier","mode":"capability",`+classifierPair+`,"base_threshold":0.5,"fail_open":true}`, server.URL)
			ctx, req, response, err := callHook(request, "/v1/chat/completions")
			if err != nil || response == nil || response.StatusCode != status || strings.Contains(string(response.Body), "private provider detail") || string(req.Body) != request || ctx.Value(selectedModelKey) != nil {
				t.Fatalf("admission failure: %v %v", response, err)
			}
		})
	}
}

func TestCallbackRestrictions(t *testing.T) {
	for _, url := range []string{"http://example.com", "http://192.168.1.1", "http://127.0.0.1@example.com", "http://127.0.0.1?key=x", "file:///tmp/test"} {
		if err := Init(map[string]any{"bifrost_url": url, "routes": map[string]json.RawMessage{"switchyard": config(stage)}}); err == nil {
			Cleanup()
			t.Fatalf("accepted callback URL %s", url)
		}
	}
	if err := Init(map[string]any{"routes": map[string]json.RawMessage{"openai/capable": config(stage)}}); err == nil {
		Cleanup()
		t.Fatal("recursive target accepted")
	}
}

func TestDecisionChoice(t *testing.T) {
	url, _ := gateway(t)
	initializeGateway(t, stage, url)
	work := switchyard.Work{Type: "call_decision", Model: "openai/a", Request: json.RawMessage(`{"input":"test","questions":{"tier":{"instructions":"Choose","kind":{"type":"choice","data":{"options":[{"id":"efficient"},{"id":"capable"}]}}}}}`)}
	reply, err := routes["switchyard"].call(context.Background(), &schemas.HTTPRequest{}, work)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(reply.Response), `"selected":"efficient"`) || !strings.Contains(string(reply.Response), `"type":"choice"`) {
		t.Fatalf("bad answer: %s", reply.Response)
	}
}

func TestDecisionUnknownEvidence(t *testing.T) {
	for _, answer := range []string{`{"name":"tier","type":"refusal"}`, `{"name":"tier","type":"choice","choice":"efficient"}`} {
		t.Run(answer, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				fmt.Fprintf(w, `{"model":"openai/a","answers":[%s]}`, answer)
			}))
			defer server.Close()
			initializeGateway(t, stage, server.URL)
			work := switchyard.Work{Type: "call_decision", Model: "openai/a", Request: json.RawMessage(`{"input":"test","questions":{"tier":{"instructions":"Choose","kind":{"type":"choice","data":{"options":[{"id":"efficient"},{"id":"capable"}]}}}}}`)}
			reply, err := routes["switchyard"].call(context.Background(), &schemas.HTTPRequest{}, work)
			if err != nil {
				t.Fatal(err)
			}
			var response struct {
				Answers map[string]struct {
					Value struct {
						Type string
						Data struct{ Probabilities map[string]float64 }
					}
				}
			}
			if err := json.Unmarshal(reply.Response, &response); err != nil {
				t.Fatal(err)
			}
			value := response.Answers["tier"].Value
			if (strings.Contains(answer, "refusal") && value.Type != "refusal") || value.Data.Probabilities != nil {
				t.Fatalf("invented decision evidence: %s", reply.Response)
			}
		})
	}
}

func TestCallbackCancellation(t *testing.T) {
	started, release := make(chan struct{}), make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		close(started)
		select {
		case <-r.Context().Done():
		case <-release:
		}
	}))
	t.Cleanup(func() { close(release); server.Close() })
	initializeGateway(t, `{"type":"llm_classifier","mode":"capability",`+classifierPair+`,"base_threshold":0.5,"fail_open":false}`, server.URL)
	parent, cancel := context.WithCancel(context.Background())
	defer cancel()
	ctx := schemas.NewBifrostContext(parent, time.Time{})
	req := &schemas.HTTPRequest{Method: "POST", Path: "/v1/chat/completions", Body: []byte(request)}
	done := make(chan *schemas.HTTPResponse, 1)
	go func() {
		response, err := HTTPTransportPreHook(ctx, req)
		if err != nil {
			t.Error(err)
		}
		done <- response
	}()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("callback did not start")
	}
	cancel()
	select {
	case response := <-done:
		if response == nil || ctx.Value(selectedModelKey) != nil {
			t.Fatal("cancelled route succeeded")
		}
	case <-time.After(time.Second):
		t.Fatal("pending callback did not cancel")
	}
}

func TestOrderedCallsStopOnAdmissionFailure(t *testing.T) {
	attempts := &atomic.Int64{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { attempts.Add(1); http.Error(w, "denied", 403) }))
	defer server.Close()
	initializeGateway(t, stage, server.URL)
	normalized, err := switchyard.DecodeRequest("openai_chat", json.RawMessage(request))
	if err != nil {
		t.Fatal(err)
	}
	_, err = routes["switchyard"].call(context.Background(), &schemas.HTTPRequest{}, switchyard.Work{Type: "call_model", Models: []string{"openai/capable", "openai/efficient"}, Request: normalized})
	if err == nil || attempts.Load() != 1 {
		t.Fatalf("admission failed open: attempts=%d err=%v", attempts.Load(), err)
	}
}

func TestPreparedFallbackRequests(t *testing.T) {
	models := make(chan string, 2)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Model    string
			Messages []struct{ Role, Content string }
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
			return
		}
		models <- body.Model
		prompt := "efficient prompt"
		if body.Model == "openai/capable" {
			prompt = "capable prompt"
		}
		if len(body.Messages) == 0 || body.Messages[0].Role != "system" || body.Messages[0].Content != prompt {
			t.Errorf("incorrect prepared request: %+v", body)
		}
		if body.Model == "openai/efficient" {
			http.Error(w, "unavailable", 500)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"fallback","object":"chat.completion","model":"capable","choices":[{"index":0,"message":{"role":"assistant","content":"OK"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()
	spec := strings.TrimSuffix(stage, "}") + `,"efficient_system_prompt":"efficient prompt","capable_system_prompt":"capable prompt"}`
	initializeGateway(t, spec, server.URL)
	ctx, _, response, err := callHook(request, "/v1/chat/completions")
	close(models)
	var attempted []string
	for model := range models {
		attempted = append(attempted, model)
	}
	if err != nil || response == nil || response.StatusCode != 200 || ctx.Value(selectedModelKey) != "openai/capable" || strings.Join(attempted, ",") != "openai/efficient,openai/capable" {
		t.Fatalf("fallback: response=%+v err=%v models=%v", response, err, attempted)
	}
}
