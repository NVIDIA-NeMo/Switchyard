// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package switchyard

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"slices"
	"sync/atomic"
	"testing"
)

func configuredRouter(t *testing.T, routeID string) *Router {
	t.Helper()
	if _, set := os.LookupEnv("SWITCHYARD_GO_TEST_KEY_THAT_IS_NOT_SET"); set {
		t.Fatal("test credential must remain unset")
	}
	source, err := os.ReadFile("testdata/eligibility.toml")
	if err != nil {
		t.Fatal(err)
	}
	router, err := NewConfigured(string(source), routeID)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(router.Close)
	return router
}

func TestConfiguredRequirementsAndTasks(t *testing.T) {
	router := configuredRouter(t, "switchyard/coding")
	request := Input{Request: input().Request, Metadata: &RequestMetadata{TaskID: "shared"}}
	noCall := func(context.Context, Work) (Reply, error) {
		t.Error("rejected or signal-only request called a model")
		return Reply{}, Abort(errors.New("unexpected model call"))
	}
	if _, err := router.Route(context.Background(), request, noCall); err == nil {
		t.Fatal("missing assessment was accepted")
	}
	for turn, test := range []struct {
		task     string
		labels   []string
		expected []string
	}{
		{"shared", []string{}, []string{"fast", "capable"}},
		{"shared", []string{"private", "customer_data"}, []string{"private_fast", "private_capable"}},
		{"shared", []string{}, []string{"private_fast", "private_capable"}},
		{"other", []string{}, []string{"fast", "capable"}},
	} {
		request.Metadata.TaskID = test.task
		request.Metadata.SessionID = fmt.Sprintf("session-%d", turn)
		request.Request = json.RawMessage(fmt.Sprintf(`{"messages":[{"role":"user","content":[{"type":"text","text":"Summarize turn %d"}]}]}`, turn))
		request.RequiredLabels = test.labels
		outcome, err := router.Route(context.Background(), request, noCall)
		if err != nil || !slices.Equal(outcome.SelectedModelIDs, test.expected) {
			t.Fatalf("task=%s labels=%v outcome=%+v err=%v", test.task, test.labels, outcome, err)
		}
	}
	request.Models = map[string][]string{"any": {"private_fast"}}
	if _, err := router.Route(context.Background(), request, noCall); err == nil {
		t.Fatal("configured route accepted host model groups")
	}
	router.Close()
	if _, err := router.TargetModels(); err == nil {
		t.Fatal("closed router returned target models")
	}
}

func TestConfiguredJudgesUseHostClients(t *testing.T) {
	for _, test := range []struct{ route, kind, model string }{
		{"switchyard/judged", "call_decision", "decision/private"},
		{"switchyard/judged-llm", "call_model", "private_judge_llm"},
	} {
		t.Run(test.kind, func(t *testing.T) {
			router := configuredRouter(t, test.route)
			models, err := router.TargetModels()
			if err != nil {
				t.Fatal(err)
			}
			request := Input{Request: input().Request, RequiredLabels: []string{"private", "customer_data"}}
			var calls atomic.Int32
			call := func(_ context.Context, work Work) (Reply, error) {
				calls.Add(1)
				if work.Type != test.kind {
					return Reply{}, Abort(errors.New("unexpected work type"))
				}
				modelID := work.Model
				if work.Type == "call_model" {
					if !slices.Equal(work.Models, []string{test.model}) {
						return Reply{}, Abort(errors.New("unapproved LLM judge"))
					}
					modelID = work.Models[0]
				}
				target, ok := models[modelID]
				if !ok {
					return Reply{}, Abort(errors.New("unknown configured target"))
				}
				switch target.Client {
				case "provider":
					if work.Type != "call_model" || target.ID != "provider/private-judge" {
						return Reply{}, Abort(errors.New("incorrect LLM client or provider model"))
					}
					return Reply{Response: json.RawMessage(judgeAnswer), ServedModel: modelID}, nil
				case "scoring":
					var payload struct{ Model string }
					if err := json.Unmarshal(work.Request, &payload); err != nil || work.Model != test.model || payload.Model != target.ID {
						return Reply{}, Abort(errors.New("incorrect Decision client or provider model"))
					}
					return Reply{Response: json.RawMessage(`{"answers":{"route":{"value":{"type":"choice","data":{"selected":"efficient","probabilities":{"efficient":0.9,"capable":0.1}}}}}}`)}, nil
				default:
					return Reply{}, Abort(errors.New("unknown host client"))
				}
			}
			outcome, err := router.Route(context.Background(), request, call)
			if err != nil || calls.Load() != 1 || !slices.Equal(outcome.SelectedModelIDs, []string{"private_fast", "private_capable"}) {
				t.Fatalf("calls=%d outcome=%+v err=%v", calls.Load(), outcome, err)
			}
			request.RequiredLabels = []string{"unapproved"}
			if _, err := router.Route(context.Background(), request, call); err == nil || calls.Load() != 1 {
				t.Fatalf("invalid requirements reached a judge: calls=%d err=%v", calls.Load(), err)
			}
		})
	}
}
