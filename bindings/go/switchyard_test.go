// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package switchyard

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"runtime"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
)

const stage = `"capable_target":"strong","efficient_target":"weak","confidence_threshold":0.5`
const composite = `{"type":"composite","classifier":{"target":"judge","base_threshold":0.5,"classify_trigger":"user_turn","fail_open":false},"stage":{` + stage + `}}`
const judgeAnswer = `{"model":"provider-model","outputs":[{"role":"assistant","content":[{"type":"text","text":"{\"crux\":\"bounded task\",\"primary_rule\":\"SUP-1\",\"capability_boundary\":\"supported\",\"p_solve\":0.9}"}],"stop_reason":"end_turn"}]}`

func newRouter(t *testing.T, algorithm string) *Router {
	t.Helper()
	router, err := New(json.RawMessage(`{"algorithm":` + algorithm + `,"targets":{"weak":"weak-model","strong":"strong-model","judge":"judge-model"}}`))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(router.Close)
	return router
}

func input() Input {
	return Input{
		Request: json.RawMessage(`{"messages":[{"role":"user","content":[{"type":"text","text":"Fix this small bug"}]}]}`),
		Models: map[string][]string{
			"any": {"strong-model", "weak-model"}, "capable": {"strong-model"},
			"efficient": {"weak-model"}, "judge": {"judge-model"},
		},
	}
}

// Selection retains eligible targets and Stage's prepared request across the codecs.
func TestSelectionAndTranslation(t *testing.T) {
	type chatRequest struct {
		Model    string
		Store    *bool
		Messages []struct{ Role, Content string }
	}
	for _, tc := range []struct {
		name, algorithm, expected string
		failed                    bool
	}{
		{"random", `{"type":"random","targets":["strong","weak"]}`, "weak-model", false},
		{"stage", `{"type":"stage_router","picker":"efficient_first",` + stage + `}`, "weak-model", false},
		{"escalation", `{"type":"stage_router","picker":"efficient_first",` + stage + `,"handoff_notes":{"escalation_note":"Review the failure"}}`, "strong-model", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router := newRouter(t, tc.algorithm)
			request := input()
			if tc.name == "random" {
				request.Models["any"] = []string{"weak-model"}
			}
			if tc.failed {
				request.Request = json.RawMessage(`{"messages":[{"role":"assistant","content":[{"type":"tool_call","id":"c","name":"bash","arguments":{}}]},{"role":"tool","content":[{"type":"tool_result","tool_call_id":"c","content":[{"type":"text","text":"Out of memory"}],"is_error":true}]}]}`)
			}
			outcome, err := router.Route(context.Background(), request, nil)
			if err != nil || len(outcome.SelectedModelIDs) == 0 || outcome.SelectedModelIDs[0] != tc.expected {
				t.Fatalf("outcome=%+v err=%v", outcome, err)
			}
			wire, err := EncodeRequest("openai_chat", outcome.Request, tc.expected)
			if err != nil {
				t.Fatal(err)
			}
			var prepared chatRequest
			if err := json.Unmarshal(wire, &prepared); err != nil || prepared.Model != tc.expected {
				t.Fatalf("prepared request=%s err=%v", wire, err)
			}
			if tc.failed {
				if len(prepared.Messages) == 0 {
					t.Fatal("missing handoff message")
				}
				last := prepared.Messages[len(prepared.Messages)-1]
				if last.Role != "user" || last.Content != "Review the failure" {
					t.Fatalf("handoff message=%+v", last)
				}
			}
		})
	}
	request, err := DecodeRequest("openai_chat", json.RawMessage(`{"model":"route","messages":[{"role":"user","content":"hello"}],"store":false}`))
	if err != nil {
		t.Fatal(err)
	}
	wire, err := EncodeRequest("openai_chat", request, "actual-target")
	if err != nil {
		t.Fatal(err)
	}
	var encoded chatRequest
	if err := json.Unmarshal(wire, &encoded); err != nil || encoded.Model != "actual-target" || encoded.Store == nil || *encoded.Store {
		t.Fatalf("wire=%s err=%v", wire, err)
	}
	response, err := DecodeResponse("openai_chat", json.RawMessage(`{"choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	var decoded struct {
		Outputs []struct {
			Content []struct{ Text string }
		}
	}
	if err := json.Unmarshal(response, &decoded); err != nil || len(decoded.Outputs) != 1 || len(decoded.Outputs[0].Content) != 1 || decoded.Outputs[0].Content[0].Text != "hello" {
		t.Fatalf("response=%s err=%v", response, err)
	}
}

// Both judge protocols use Go's client; missing clients and host failures remain errors.
func TestCompositeHostCalls(t *testing.T) {
	decision := `{"type":"composite","classifier":{"target":"judge","classify_trigger":"user_turn","fail_open":false,"decision":{"cutoff":0.5,"candidates":{"a":"strong","b":"weak"},"evidence":{}}},"stage":{` + stage + `}}`
	for _, tc := range []struct {
		name, algorithm, kind, selected, response, invalidResponse string
	}{
		{"model", composite, "call_model", "weak-model", judgeAnswer, `{"outputs":"wrong"}`},
		{"decision", decision, "call_decision", "strong-model", `{"answers":{"route":{"value":{"type":"choice","data":{"selected":"advantage","probabilities":{"advantage":0.9,"no_advantage":0.1}}}}}}`, `{"answers":"wrong"}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router := newRouter(t, tc.algorithm)
			calls := 0
			outcome, err := router.Route(context.Background(), input(), func(_ context.Context, work Work) (Reply, error) {
				calls++
				isJudge := work.Model == "judge-model"
				if tc.kind == "call_model" {
					isJudge = slices.Equal(work.Models, []string{"judge-model"})
				}
				if work.Type != tc.kind || !isJudge || !json.Valid(work.Request) {
					return Reply{}, fmt.Errorf("unexpected work: %+v", work)
				}
				return Reply{Response: json.RawMessage(tc.response)}, nil
			})
			if err != nil || calls != 1 || len(outcome.SelectedModelIDs) == 0 || outcome.SelectedModelIDs[0] != tc.selected {
				t.Fatalf("calls=%d outcome=%+v err=%v", calls, outcome, err)
			}
			if _, err := router.Route(context.Background(), input(), nil); err == nil {
				t.Fatal("missing host client succeeded")
			}
			_, err = router.Route(context.Background(), input(), func(context.Context, Work) (Reply, error) {
				return Reply{}, errors.New("host unavailable")
			})
			if err == nil || !strings.Contains(err.Error(), "host unavailable") {
				t.Fatalf("host failure returned %v", err)
			}
			_, err = router.Route(context.Background(), input(), func(context.Context, Work) (Reply, error) {
				return Reply{Response: json.RawMessage(tc.invalidResponse)}, nil
			})
			if err == nil {
				t.Fatal("invalid host response succeeded")
			}
		})
	}
}

// Host failures remain subject to the algorithm's configured recovery policy.
func TestHostFailureUsesAlgorithmPolicy(t *testing.T) {
	algorithm := strings.Replace(composite, `"fail_open":false`, `"fail_open":true`, 1)
	router := newRouter(t, algorithm)
	calls := 0
	outcome, err := router.Route(context.Background(), input(), func(context.Context, Work) (Reply, error) {
		calls++
		return Reply{}, errors.New("judge unavailable")
	})
	if err != nil || calls != 1 || !slices.Equal(outcome.SelectedModelIDs, []string{"weak-model", "strong-model"}) {
		t.Fatalf("calls=%d outcome=%+v err=%v", calls, outcome, err)
	}
}

// Cancellation releases native work even if a host ignores its context and replies later.
func TestCancellationAndLateHostReply(t *testing.T) {
	router := newRouter(t, composite)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	started, release, returned := make(chan struct{}), make(chan struct{}), make(chan struct{})
	releaseHost := sync.OnceFunc(func() { close(release) })
	defer releaseHost()
	finished := make(chan error, 1)
	go func() {
		_, err := router.Route(ctx, input(), func(context.Context, Work) (Reply, error) {
			defer close(returned)
			router.Close()
			close(started)
			<-release
			return Reply{Response: json.RawMessage(judgeAnswer)}, nil
		})
		finished <- err
	}()
	select {
	case <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("host callback could not close its router")
	}
	runtime.GC()
	cancel()
	select {
	case err := <-finished:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("cancellation returned %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("cancellation waited for host code")
	}
	runtime.GC()
	// Release after Route has freed its native run, then let the late Go reply drain.
	releaseHost()
	select {
	case <-returned:
	case <-time.After(5 * time.Second):
		t.Fatal("late host callback did not finish")
	}
}

// Concurrent requests keep eligible pools separate; router copies share closed state.
func TestConcurrentRoutesAndClose(t *testing.T) {
	router := newRouter(t, `{"type":"random","targets":["weak","strong"],"seed":7}`)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := router.Route(ctx, input(), nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled context returned %v", err)
	}
	results := make(chan error, 24)
	for i := 0; i < cap(results); i++ {
		go func(i int) {
			request := input()
			model := []string{"weak-model", "strong-model"}[i%2]
			request.Models = map[string][]string{"any": {model}}
			outcome, err := router.Route(context.Background(), request, nil)
			if err == nil && (len(outcome.SelectedModelIDs) != 1 || outcome.SelectedModelIDs[0] != model) {
				err = fmt.Errorf("expected %s, got %v", model, outcome.SelectedModelIDs)
			}
			results <- err
		}(i)
	}
	for i := 0; i < cap(results); i++ {
		select {
		case err := <-results:
			if err != nil {
				t.Fatal(err)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("concurrent route did not finish")
		}
	}
	copy := *router
	for _, closed := range []*Router{&copy, router, new(Router)} {
		closed.Close()
		closed.Close()
		if _, err := closed.Route(context.Background(), input(), nil); err == nil {
			t.Fatal("closed router accepted a request")
		}
	}
	racing := newRouter(t, `{"type":"random","targets":["weak"]}`)
	racingCopy, start := *racing, make(chan struct{})
	go func() {
		<-start
		request := input()
		request.Models["any"] = []string{"weak-model"}
		outcome, err := racing.Route(context.Background(), request, nil)
		if err != nil && err.Error() == "Switchyard router is closed" {
			err = nil
		} else if err == nil && (len(outcome.SelectedModelIDs) != 1 || outcome.SelectedModelIDs[0] != "weak-model") {
			err = fmt.Errorf("close race selected %v", outcome.SelectedModelIDs)
		}
		results <- err
	}()
	go func() { <-start; racingCopy.Close(); results <- nil }()
	close(start)
	for i := 0; i < 2; i++ {
		select {
		case err := <-results:
			if err != nil {
				t.Fatal(err)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("close race did not finish")
		}
	}
	runtime.GC()
}

// Served-target attribution and agent/session metadata survive the bridge.
func TestReplyAttributionAndMetadata(t *testing.T) {
	router := newRouter(t, `{"type":"advisor","executor_target":"strong","advisor_target":"judge","gate_trigger":"pattern","gate_trigger_pattern":"NEVER"}`)
	request := input()
	request.Models["efficient"] = []string{"strong-model", "weak-model"}
	outcome, err := router.Route(context.Background(), request, func(_ context.Context, work Work) (Reply, error) {
		if len(work.Models) != 2 {
			return Reply{}, fmt.Errorf("missing fallback: %+v", work)
		}
		return Reply{Response: json.RawMessage(judgeAnswer), ServedModel: "weak-model"}, nil
	})
	if err != nil || len(outcome.SelectedModelIDs) != 1 || outcome.SelectedModelIDs[0] != "weak-model" || outcome.Response == nil {
		t.Fatalf("outcome=%+v err=%v", outcome, err)
	}
	delegated := newRouter(t, `{"type":"passthrough","target":"strong","subagents":{"type":"passthrough","target":"weak"}}`)
	for _, child := range []bool{false, true} {
		request := input()
		request.Models = map[string][]string{"any": {"strong-model"}}
		request.SubagentModels = map[string][]string{"any": {"weak-model"}}
		request.Metadata = &RequestMetadata{SessionID: "session", AgentID: "child", IsSubagent: true, IsDelegatedWork: child}
		expected := "strong-model"
		if child {
			expected = "weak-model"
		}
		outcome, err := delegated.Route(context.Background(), request, nil)
		if err != nil || len(outcome.SelectedModelIDs) != 1 || outcome.SelectedModelIDs[0] != expected {
			t.Fatalf("delegated=%v outcome=%+v err=%v", child, outcome, err)
		}
	}
	request = input()
	request.SubagentModels = map[string][]string{}
	request.Metadata = &RequestMetadata{IsSubagent: true, IsDelegatedWork: true}
	encoded, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	var envelope struct {
		SubagentModels map[string][]string `json:"subagent_models"`
	}
	if err := json.Unmarshal(encoded, &envelope); err != nil || envelope.SubagentModels == nil || len(envelope.SubagentModels) != 0 {
		t.Fatalf("explicit empty child pool was lost: %s err=%v", encoded, err)
	}
	if _, err := delegated.Route(context.Background(), request, nil); err == nil {
		t.Fatal("empty child pool selected a parent model")
	}
	plan := newRouter(t, `{"type":"plan_execute","capable_target":"strong","efficient_target":"weak"}`)
	for _, tc := range []struct {
		agent                string
		child, final, edited bool
		expected             string
	}{
		{"a", true, false, true, "weak-model"},
		{"b", true, false, false, "strong-model"},
		{"root", false, false, false, "strong-model"},
		{"a", true, true, false, "weak-model"},
		{"a", true, false, false, "strong-model"},
	} {
		request := input()
		request.Request = json.RawMessage(`{}`)
		if tc.edited {
			request.Request = json.RawMessage(`{"messages":[{"role":"assistant","content":[{"type":"tool_call","id":"edit","name":"apply_patch","arguments":{}}]}]}`)
		}
		request.Metadata = &RequestMetadata{SessionID: "session", AgentID: tc.agent, IsSubagent: tc.child, SessionFinal: &tc.final}
		outcome, err := plan.Route(context.Background(), request, nil)
		if err != nil || len(outcome.SelectedModelIDs) != 1 || outcome.SelectedModelIDs[0] != tc.expected {
			t.Fatalf("agent=%s final=%v outcome=%+v err=%v", tc.agent, tc.final, outcome, err)
		}
	}
}
