// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package switchyard

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"testing"
)

const wireChat = `{"model":"route","messages":[{"role":"user","content":"Fix this small bug"}],"store":false}`

func decisionInput(format, body string) WireInput {
	request := input()
	request.Request = json.RawMessage(body)
	return WireInput{Input: request, Format: format}
}

// Each prepared fallback has its own model stamp and retains routing-time edits.
func TestDecideFormats(t *testing.T) {
	for _, tc := range []struct{ format, body, field, value string }{
		{"openai_chat", wireChat, "store", "false"},
		{"openai_responses", `{"model":"route","input":"Fix this small bug","previous_response_id":"resp_prev"}`, "previous_response_id", `"resp_prev"`},
		{"anthropic_messages", `{"model":"route","max_tokens":64,"messages":[{"role":"user","content":"Fix this small bug"}]}`, "max_tokens", "64"},
	} {
		t.Run(tc.format, func(t *testing.T) {
			router := newRouter(t, `{"type":"stage_router","picker":"efficient_first",`+stage+`}`)
			result, err := router.Decide(context.Background(), decisionInput(tc.format, tc.body), nil)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(result.SelectedModelIDs, []string{"weak-model", "strong-model"}) || len(result.Requests) != 2 {
				t.Fatalf("result=%+v", result)
			}
			for index, prepared := range result.Requests {
				var fields map[string]json.RawMessage
				if err := json.Unmarshal(prepared.Request, &fields); err != nil {
					t.Fatal(err)
				}
				var model string
				if err := json.Unmarshal(fields["model"], &model); err != nil {
					t.Fatal(err)
				}
				if prepared.Model != result.SelectedModelIDs[index] || model != prepared.Model || prepared.Format != tc.format || string(fields[tc.field]) != tc.value {
					t.Fatalf("prepared=%+v", prepared)
				}
				normalized, err := DecodeRequest(tc.format, prepared.Request)
				if err != nil {
					t.Fatal(err)
				}
				var request struct {
					Messages []struct{ Content []struct{ Text string } }
				}
				if err := json.Unmarshal(normalized, &request); err != nil || len(request.Messages) != 1 || len(request.Messages[0].Content) != 1 || request.Messages[0].Content[0].Text != "Fix this small bug" {
					t.Fatalf("request=%s err=%v", normalized, err)
				}
			}
		})
	}
	router := newRouter(t, `{"type":"stage_router","picker":"efficient_first",`+stage+`,"handoff_notes":{"escalation_note":"Review the failure"}}`)
	request := decisionInput("openai_chat", `{"model":"route","messages":[{"role":"assistant","content":null,"tool_calls":[{"id":"c","type":"function","function":{"name":"bash","arguments":"{}"}}]},{"role":"tool","tool_call_id":"c","content":"Out of memory","is_error":true}]}`)
	result, err := router.Decide(context.Background(), request, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(result.SelectedModelIDs, []string{"strong-model", "weak-model"}) {
		t.Fatalf("targets=%v", result.SelectedModelIDs)
	}
	for _, prepared := range result.Requests {
		var body struct {
			Messages []struct {
				Role, Content string
				ToolCallID    string                `json:"tool_call_id"`
				ToolCalls     []struct{ ID string } `json:"tool_calls"`
			}
		}
		// Tool-call content may be null; the tool result and handoff are text.
		if err := json.Unmarshal(prepared.Request, &body); err != nil {
			t.Fatal(err)
		}
		if len(body.Messages) != 3 || len(body.Messages[0].ToolCalls) != 1 || body.Messages[0].ToolCalls[0].ID != "c" || body.Messages[1].ToolCallID != "c" || body.Messages[2].Role != "user" || body.Messages[2].Content != "Review the failure" {
			t.Fatalf("prepared=%s", prepared.Request)
		}
	}
}

// Provider calls use wire JSON; typed decisions keep their own protocol and error policy.
func TestDecideHostCalls(t *testing.T) {
	decision := `{"type":"composite","classifier":{"target":"judge","classify_trigger":"user_turn","fail_open":false,"decision":{"cutoff":0.5,"candidates":{"a":"strong","b":"weak"},"evidence":{}}},"stage":{` + stage + `}}`
	for _, tc := range []struct{ name, algorithm, kind, answer, target string }{
		{"model", composite, "call_model", `{"model":"judge-model","choices":[{"index":0,"message":{"role":"assistant","content":"{\"crux\":\"bounded task\",\"primary_rule\":\"SUP-1\",\"capability_boundary\":\"supported\",\"p_solve\":0.9}"},"finish_reason":"stop"}]}`, "weak-model"},
		{"decision", decision, "call_decision", `{"answers":{"route":{"value":{"type":"choice","data":{"selected":"advantage","probabilities":{"advantage":0.9,"no_advantage":0.1}}}}}}`, "strong-model"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router := newRouter(t, tc.algorithm)
			request := decisionInput("openai_chat", wireChat)
			calls := 0
			result, err := router.Decide(context.Background(), request, func(_ context.Context, work WireWork) (Reply, error) {
				calls++
				if work.Type != tc.kind {
					t.Errorf("work=%+v", work)
				}
				if tc.kind == "call_model" {
					if len(work.Requests) != 1 || work.Requests[0].Model != "judge-model" || work.Request != nil {
						t.Errorf("work=%+v", work)
					}
					var fields map[string]json.RawMessage
					if err := json.Unmarshal(work.Requests[0].Request, &fields); err != nil || string(fields["model"]) != `"judge-model"` {
						t.Errorf("wire call=%+v", work)
					}
				} else if work.Model != "judge-model" || len(work.Requests) != 0 || !json.Valid(work.Request) {
					t.Errorf("decision=%+v", work)
				}
				return Reply{Response: json.RawMessage(tc.answer)}, nil
			})
			if err != nil || calls != 1 || len(result.SelectedModelIDs) == 0 || result.SelectedModelIDs[0] != tc.target {
				t.Fatalf("calls=%d result=%+v err=%v", calls, result, err)
			}

			open := newRouter(t, strings.Replace(tc.algorithm, `"fail_open":false`, `"fail_open":true`, 1))
			refused := errors.New("budget exhausted")
			result, err = open.Decide(context.Background(), request, func(context.Context, WireWork) (Reply, error) { return Reply{}, refused })
			if err != nil || !slices.Equal(result.SelectedModelIDs, []string{"weak-model", "strong-model"}) {
				t.Fatalf("fail-open=%+v err=%v", result, err)
			}
		})
	}
}

// The host owns fallback execution; the answering target determines response decoding.
func TestDecideRetainsAnswer(t *testing.T) {
	router := newRouter(t, `{"type":"advisor","executor_target":"weak","advisor_target":"judge","gate_min_tool_results":1}`)
	request := decisionInput("openai_chat", wireChat)
	request.Models["efficient"] = []string{"weak-model", "strong-model"}
	request.TargetFormats = map[string]string{"strong-model": "anthropic_messages"}
	for _, served := range []string{"", "outside-model"} {
		_, err := router.Decide(context.Background(), request, func(context.Context, WireWork) (Reply, error) {
			return Reply{Response: json.RawMessage(`{}`), ServedModel: served}, nil
		})
		if err == nil || !strings.Contains(err.Error(), "host reply must identify a requested target") {
			t.Fatalf("served_model=%q err=%v", served, err)
		}
	}
	result, err := router.Decide(context.Background(), request, func(_ context.Context, work WireWork) (Reply, error) {
		if len(work.Requests) != 2 || work.Requests[0].Format != "openai_chat" || work.Requests[1].Format != "anthropic_messages" {
			t.Errorf("work=%+v", work)
		}
		return Reply{Response: json.RawMessage(`{"model":"provider-model","content":[{"type":"text","text":"answer"}],"stop_reason":"end_turn"}`), ServedModel: "strong-model"}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(result.SelectedModelIDs, []string{"strong-model"}) || result.Response == nil || !json.Valid(result.Metadata) {
		t.Fatalf("result=%+v", result)
	}
	var answer struct {
		Outputs []struct{ Content []struct{ Text string } }
	}
	if err := json.Unmarshal(*result.Response, &answer); err != nil || len(answer.Outputs) != 1 || len(answer.Outputs[0].Content) != 1 || answer.Outputs[0].Content[0].Text != "answer" {
		t.Fatalf("answer=%s err=%v", *result.Response, err)
	}
}

// Cancellation reaches a pending provider callback through the new facade.
func TestDecideCancellation(t *testing.T) {
	router := newRouter(t, composite)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	_, err := router.Decide(ctx, decisionInput("openai_chat", wireChat), func(ctx context.Context, _ WireWork) (Reply, error) {
		cancel()
		<-ctx.Done()
		return Reply{}, ctx.Err()
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation=%v", err)
	}
}
