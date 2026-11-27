// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package switchyard

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"maps"
	"runtime"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
)

const stage = `"capable_target":"strong","efficient_target":"weak","confidence_threshold":0.5`
const composite = `{"type":"composite","classifier":{"target":"judge","base_threshold":0.5,"classify_trigger":"user_turn","fail_open":false},"stage":{` + stage + `}}`
const decision = `{"type":"composite","classifier":{"target":"judge","classify_trigger":"user_turn","fail_open":false,"decision":{"cutoff":0.5}},"stage":{` + stage + `}}`
const judgeAnswer = `{"model":"provider-model","outputs":[{"role":"assistant","content":[{"type":"text","text":"{\"crux\":\"bounded task\",\"primary_rule\":\"SUP-1\",\"capability_boundary\":\"supported\",\"p_solve\":0.9}"}],"stop_reason":"end_turn"}]}`

func newRouter(t *testing.T, algorithm string) *Router {
	t.Helper()
	router, err := New(testConfig(algorithm))
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

// Completed outcomes retain fallback requests while absent requests remain optional.
func TestOutcomeFallbackRequest(t *testing.T) {
	request := input().Request
	for _, tc := range []struct {
		name, field string
		present     bool
	}{
		{"present", `"fallback_request":` + string(request), true},
		{"missing", "", false},
		{"null", `"fallback_request":null`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var outcome Outcome
			data := []byte(`{` + tc.field + `}`)
			if err := json.Unmarshal(data, &outcome); err != nil {
				t.Fatal(err)
			}
			if present := outcome.FallbackRequest != nil; present != tc.present {
				t.Fatalf("fallback request present=%v, want %v", present, tc.present)
			}
			if tc.present && string(*outcome.FallbackRequest) != string(request) {
				t.Fatalf("fallback request=%s, want %s", *outcome.FallbackRequest, request)
			}
		})
	}
}

// Stage's fallback keeps its execution prompt instead of reusing the planning prompt.
func TestCandidateRequestPreservesFallbackPrompt(t *testing.T) {
	router := newRouter(t, `{"type":"stage_router","picker":"capable_first",`+stage+`,"capable_system_prompt":"Plan the change","efficient_system_prompt":"Execute the change"}`)
	outcome, err := router.Route(context.Background(), input(), nil)
	if err != nil {
		t.Fatal(err)
	}
	wire, err := EncodeRequest("openai_chat", outcome.CandidateRequests["weak-model"], "weak-model")
	if err != nil {
		t.Fatal(err)
	}
	var encoded struct {
		Messages []struct{ Role, Content string }
	}
	if err := json.Unmarshal(wire, &encoded); err != nil {
		t.Fatal(err)
	}
	if len(encoded.Messages) == 0 || encoded.Messages[0].Role != "system" || encoded.Messages[0].Content != "Execute the change" {
		t.Fatalf("fallback request=%s, want execution prompt", wire)
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
	reply, err := EncodeResponse("anthropic_messages", response, nil)
	if err != nil {
		t.Fatal(err)
	}
	var message struct {
		Content []struct{ Text string }
	}
	if err := json.Unmarshal(reply, &message); err != nil || len(message.Content) != 1 || message.Content[0].Text != "hello" {
		t.Fatalf("reply=%s err=%v", reply, err)
	}
}

// A stream translator keeps state across events and closes the target stream on Finish.
func TestStreamTranslator(t *testing.T) {
	stream, err := NewStreamTranslator("openai_chat", "anthropic_messages", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	if _, err := stream.Translate(nil); !errors.Is(err, errEmptyEvent) {
		t.Fatalf("err=%v, want errEmptyEvent", err)
	}
	var events []json.RawMessage
	for _, event := range []string{
		`{"id":"1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"hel"}}]}`,
		`{"choices":[{"index":0,"delta":{"content":"lo"}}]}`,
		`{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
	} {
		translated, err := stream.Translate(json.RawMessage(event))
		if err != nil {
			t.Fatal(err)
		}
		events = append(events, translated...)
	}
	closing, err := stream.Finish()
	if err != nil {
		t.Fatal(err)
	}
	events = append(events, closing...)
	var types []string
	var text string
	for _, raw := range events {
		var event struct {
			Type  string
			Delta struct{ Text string }
		}
		if err := json.Unmarshal(raw, &event); err != nil {
			t.Fatal(err)
		}
		types = append(types, event.Type)
		text += event.Delta.Text
	}
	if len(types) < 2 || types[0] != "message_start" || types[len(types)-1] != "message_stop" || text != "hello" {
		t.Fatalf("types=%v text=%q", types, text)
	}
	stream.Close()
	stream.Close()
	if _, err := stream.Translate(json.RawMessage(`{}`)); err == nil {
		t.Fatal("closed stream translated")
	}
	if _, err := NewStreamTranslator("unknown", "openai_chat", nil); err == nil {
		t.Fatal("unknown format accepted")
	}

	cutOff, err := NewStreamTranslator("openai_chat", "anthropic_messages", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer cutOff.Close()
	if _, err := cutOff.Translate(json.RawMessage(`{"choices":[{"index":0,"delta":{"content":"hel"}}]}`)); err != nil {
		t.Fatal(err)
	}
	if _, err := cutOff.Finish(); err == nil || !strings.Contains(err.Error(), "ended before a terminal event") {
		t.Fatalf("cut-off stream finished with err=%v", err)
	}
}

// A "[DONE]" marker completes a stream only when the source format treats it as final.
func TestStreamTranslatorDoneMarker(t *testing.T) {
	finish := func(source, target, event string, sawDone bool) ([]json.RawMessage, error) {
		stream, err := NewStreamTranslator(source, target, nil)
		if err != nil {
			t.Fatal(err)
		}
		defer stream.Close()
		if _, err := stream.Translate(json.RawMessage(event)); err != nil {
			t.Fatal(err)
		}
		if sawDone {
			stream.DoneMarker()
		}
		return stream.Finish()
	}
	chatText := `{"id":"1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"hel"}}]}`

	closing, err := finish("openai_chat", "anthropic_messages", chatText, true)
	if err != nil {
		t.Fatalf("stream ended by [DONE] failed: %v", err)
	}
	var last struct{ Type string }
	if len(closing) == 0 || json.Unmarshal(closing[len(closing)-1], &last) != nil || last.Type != "message_stop" {
		t.Fatalf("closing events %s, want them to end with message_stop", closing)
	}
	if _, err := finish("openai_chat", "anthropic_messages", chatText, false); err == nil ||
		!strings.Contains(err.Error(), "openai_chat stream ended before a terminal event") {
		t.Fatalf("stream that hit EOF without [DONE] finished with err=%v", err)
	}

	anthropicText := `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hel"}}`
	if _, err := finish("anthropic_messages", "openai_chat", anthropicText, true); err == nil ||
		!strings.Contains(err.Error(), "anthropic_messages stream ended before a terminal event") {
		t.Fatalf("Anthropic stream without message_stop finished after [DONE] with err=%v", err)
	}
}

// toolExtensions decodes a Responses request with a top-level custom tool apply_patch and a
// namespaced function workspace.apply_patch, and returns its extensions.
func toolExtensions(t *testing.T) json.RawMessage {
	t.Helper()
	request, err := DecodeRequest("openai_responses", json.RawMessage(`{"model":"m","input":"hi","tools":[
		{"type":"custom","name":"apply_patch","description":"Apply a patch"},
		{"type":"namespace","name":"workspace","description":"Workspace tools","tools":[
			{"type":"function","name":"apply_patch","description":"Apply a structured patch",
			 "parameters":{"type":"object","properties":{"cmd":{"type":"string"}}}}]}]}`))
	if err != nil {
		t.Fatal(err)
	}
	var decoded struct{ Extensions json.RawMessage }
	if err := json.Unmarshal(request, &decoded); err != nil || len(decoded.Extensions) == 0 {
		t.Fatalf("decoded request has no extensions: %s (err=%v)", request, err)
	}
	return decoded.Extensions
}

// toolCallChunks stream an OpenAI Chat answer that calls both tools from toolExtensions, with
// arguments split across deltas. toolCallsFinish ends it.
var toolCallChunks = []string{
	`{"id":"1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"apply_patch","arguments":""}}]}}]}`,
	`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"input\":\"*** Begin"}}]}}]}`,
	`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":" Patch\"}"}}]}}]}`,
	`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_2","type":"function","function":{"name":"workspace__apply_patch","arguments":"{\"cmd\":"}}]}}]}`,
	`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"\"ls\"}"}}]}}]}`,
}

const toolCallsFinish = `{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`

// restoredTools are the client-facing item types for toolCallChunks, by tool.
var restoredTools = map[string]string{"apply_patch": "custom_tool_call", "workspace.apply_patch": "function_call"}

type toolItem struct{ Type, Name, Namespace, Input string }

// tool names an item namespace.name, or by its bare name at the top level.
func (item toolItem) tool() string {
	if item.Namespace == "" {
		return item.Name
	}
	return item.Namespace + "." + item.Name
}

// responsesStream summarizes the tool calls in a streamed OpenAI Responses answer.
type responsesStream struct {
	done      map[string]toolItem // output_item.done items, by tool
	arguments map[string]int      // argument delta and done events, by tool
	completed map[string]string   // response.completed item types, by tool
}

func readResponsesStream(t *testing.T, events []json.RawMessage) responsesStream {
	t.Helper()
	got := responsesStream{done: map[string]toolItem{}, arguments: map[string]int{}, completed: map[string]string{}}
	tools := map[int]string{} // tool by output index
	for _, raw := range events {
		var event struct {
			Type        string
			OutputIndex int `json:"output_index"`
			Item        toolItem
			Response    struct{ Output []toolItem }
		}
		if err := json.Unmarshal(raw, &event); err != nil {
			t.Fatal(err)
		}
		switch event.Type {
		case "response.output_item.added":
			tools[event.OutputIndex] = event.Item.tool()
		case "response.output_item.done":
			got.done[event.Item.tool()] = event.Item
		case "response.function_call_arguments.delta", "response.function_call_arguments.done":
			got.arguments[tools[event.OutputIndex]]++
		case "response.completed":
			for _, output := range event.Response.Output {
				got.completed[output.tool()] = output.Type
			}
		}
	}
	return got
}

// checkRestoredTools checks that a top-level custom tool came back as custom_tool_call with its
// assembled input and no argument events, and the namespaced function as a function_call with
// its namespace and argument events.
func checkRestoredTools(t *testing.T, got responsesStream) {
	t.Helper()
	if patch := got.done["apply_patch"]; patch.Type != "custom_tool_call" || patch.Input != "*** Begin Patch" {
		t.Errorf("custom tool item %+v, want custom_tool_call with the patch as input", patch)
	}
	if got.arguments["apply_patch"] != 0 {
		t.Errorf("custom tool call streamed %d argument events, want none", got.arguments["apply_patch"])
	}
	if function := got.done["workspace.apply_patch"]; function.Type != "function_call" || got.arguments["workspace.apply_patch"] == 0 {
		t.Errorf("namespaced function item %+v with %d argument events, want function_call with them",
			function, got.arguments["workspace.apply_patch"])
	}
	if !maps.Equal(got.completed, restoredTools) {
		t.Errorf("completed response items %v, want %v", got.completed, restoredTools)
	}
}

// With the request's extensions, streamed and buffered answers keep a Responses client's
// tool details: a top-level custom tool becomes custom_tool_call, and a namespaced function
// with the same bare name stays a function_call in its namespace.
func TestStreamTranslatorRestoresCustomTools(t *testing.T) {
	extensions := toolExtensions(t)
	translate := func(extensions json.RawMessage) responsesStream {
		stream, err := NewStreamTranslator("openai_chat", "openai_responses", extensions)
		if err != nil {
			t.Fatal(err)
		}
		defer stream.Close()
		var events []json.RawMessage
		for _, chunk := range append(slices.Clone(toolCallChunks), toolCallsFinish) {
			translated, err := stream.Translate(json.RawMessage(chunk))
			if err != nil {
				t.Fatal(err)
			}
			events = append(events, translated...)
		}
		closing, err := stream.Finish()
		if err != nil {
			t.Fatal(err)
		}
		return readResponsesStream(t, append(events, closing...))
	}

	checkRestoredTools(t, translate(extensions))
	plain := translate(nil)
	if want := map[string]string{"apply_patch": "function_call", "workspace__apply_patch": "function_call"}; !maps.Equal(plain.completed, want) {
		t.Errorf("without extensions, completed response items %v, want %v", plain.completed, want)
	}

	// The buffered path restores the same answer the same way.
	response, err := DecodeResponse("openai_chat", json.RawMessage(`{"id":"1","object":"chat.completion","model":"m","choices":[{"index":0,
		"message":{"role":"assistant","content":null,"tool_calls":[
			{"id":"call_1","type":"function","function":{"name":"apply_patch","arguments":"{\"input\":\"*** Begin Patch\"}"}},
			{"id":"call_2","type":"function","function":{"name":"workspace__apply_patch","arguments":"{\"cmd\":\"ls\"}"}}]},
		"finish_reason":"tool_calls"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := EncodeResponse("openai_responses", response, extensions)
	if err != nil {
		t.Fatal(err)
	}
	var buffered struct{ Output []toolItem }
	if err := json.Unmarshal(encoded, &buffered); err != nil {
		t.Fatal(err)
	}
	bufferedTypes := map[string]string{}
	for _, output := range buffered.Output {
		bufferedTypes[output.tool()] = output.Type
	}
	if !maps.Equal(bufferedTypes, restoredTools) {
		t.Errorf("buffered response items %v, want %v", bufferedTypes, restoredTools)
	}
}

// The stream fixes work together through the README's SSE loop, on a provider stream shaped
// like a real one (keep-alive, finish_reason, then [DONE]): the events keep the client's tool
// details, a copy shares the [DONE] marker, and the stream is released once.
func TestStreamTranslatorReadmeLoop(t *testing.T) {
	stream, err := NewStreamTranslator("openai_chat", "openai_responses", toolExtensions(t))
	if err != nil {
		t.Fatal(err)
	}
	counted := &countingStream{streamHandle: stream.state.handle}
	stream.state.handle = counted
	copied := *stream
	defer stream.Close()

	var body strings.Builder
	body.WriteString(": keep-alive\n\n")
	for _, chunk := range append(slices.Clone(toolCallChunks), toolCallsFinish) {
		fmt.Fprintf(&body, "data: %s\n\n", chunk)
	}
	body.WriteString("data: [DONE]\n\n")

	var events []json.RawMessage
	send := func(batch []json.RawMessage) { events = append(events, batch...) }
	// The README's loop. The copy records the marker; the original finishes.
	scanner := bufio.NewScanner(strings.NewReader(body.String()))
	scanner.Buffer(nil, 16<<20)
	for scanner.Scan() {
		data, ok := strings.CutPrefix(scanner.Text(), "data:")
		if !ok {
			continue
		}
		if data = strings.TrimSpace(data); data == "[DONE]" {
			copied.DoneMarker()
			break
		}
		translated, err := stream.Translate(json.RawMessage(data))
		if err != nil {
			t.Fatal(err)
		}
		send(translated)
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	if !stream.state.sawDone {
		t.Fatal("the original did not see the marker its copy recorded")
	}
	closing, err := stream.Finish()
	if err != nil {
		t.Fatalf("finish failed: %v", err)
	}
	send(closing)
	checkRestoredTools(t, readResponsesStream(t, events))

	copied.Close()
	stream.Close()
	if _, err := stream.Finish(); !errors.Is(err, errStreamClosed) {
		t.Fatalf("finish after close: err=%v, want errStreamClosed", err)
	}
	if counted.frees != 1 {
		t.Fatalf("stream released %d times, want 1", counted.frees)
	}
}

// Value copies share one stream: closing either releases it once and closes both.
func TestStreamTranslatorCopiesShareClose(t *testing.T) {
	stream, err := NewStreamTranslator("openai_chat", "anthropic_messages", nil)
	if err != nil {
		t.Fatal(err)
	}
	counted := &countingStream{streamHandle: stream.state.handle}
	stream.state.handle = counted
	copied := *stream
	stream.Close()
	if _, err := copied.Translate(json.RawMessage(`{}`)); !errors.Is(err, errStreamClosed) {
		t.Fatalf("copy translated after close: err=%v", err)
	}
	if _, err := copied.Finish(); !errors.Is(err, errStreamClosed) {
		t.Fatalf("copy finished after close: err=%v", err)
	}
	copied.Close()
	stream.Close()
	if counted.frees != 1 {
		t.Fatalf("stream released %d times, want 1", counted.frees)
	}
}

// A zero or nil StreamTranslator is closed, and closing it is harmless.
func TestUnopenedStreamTranslatorIsClosed(t *testing.T) {
	var zero StreamTranslator
	var missing *StreamTranslator
	for _, closed := range []*StreamTranslator{&zero, missing} {
		if _, err := closed.Translate(json.RawMessage(`{}`)); !errors.Is(err, errStreamClosed) {
			t.Fatalf("err=%v, want errStreamClosed", err)
		}
		if _, err := closed.Finish(); !errors.Is(err, errStreamClosed) {
			t.Fatalf("err=%v, want errStreamClosed", err)
		}
		closed.DoneMarker()
		closed.Close()
	}
}

// countingStream counts releases of the backend stream it wraps.
type countingStream struct {
	streamHandle
	frees int
}

func (stream *countingStream) free() {
	stream.frees++
	stream.streamHandle.free()
}

// Both judge protocols use Go's client; missing clients and host failures remain errors.
func TestCompositeHostCalls(t *testing.T) {
	for _, tc := range []struct {
		name, algorithm, kind, selected, response, invalidResponse string
	}{
		{"model", composite, "call_model", "weak-model", judgeAnswer, `{"outputs":"wrong"}`},
		{"decision", decision, "call_decision", "strong-model", `{"answers":{"route":{"value":{"type":"choice","data":{"selected":"capable","probabilities":{"capable":0.9,"efficient":0.1}}}}}}`, `{"answers":"wrong"}`},
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
	for _, algorithm := range []string{composite, decision} {
		router := newRouter(t, strings.Replace(algorithm, `"fail_open":false`, `"fail_open":true`, 1))
		calls := 0
		outcome, err := router.Route(context.Background(), input(), func(context.Context, Work) (Reply, error) {
			calls++
			return Reply{}, errors.New("judge unavailable")
		})
		if err != nil || calls != 1 || !slices.Equal(outcome.SelectedModelIDs, []string{"weak-model", "strong-model"}) {
			t.Fatalf("calls=%d outcome=%+v err=%v", calls, outcome, err)
		}
	}
}

// An aborted host call stops routing even when the algorithm recovers from host errors.
func TestHostAbortStopsRouting(t *testing.T) {
	if Abort(nil) != nil {
		t.Fatal("nil error should stay nil")
	}
	errBudget := errors.New("budget exhausted")
	refused := &fs.PathError{Op: "route", Path: "judge", Err: errBudget}
	original := fmt.Errorf("judge call refused: %w", refused)
	for _, algorithm := range []string{composite, decision} {
		router := newRouter(t, strings.Replace(algorithm, `"fail_open":false`, `"fail_open":true`, 1))
		calls := 0
		var hostCtx context.Context
		_, err := router.Route(context.Background(), input(), func(ctx context.Context, _ Work) (Reply, error) {
			calls++
			hostCtx = ctx
			return Reply{}, Abort(original)
		})
		var typed *fs.PathError
		if calls != 1 || err != original || !errors.Is(err, errBudget) || !errors.As(err, &typed) || typed != refused {
			t.Fatalf("calls=%d err=%v", calls, err)
		}
		if hostCtx.Err() != context.Canceled {
			t.Fatal("abort did not cancel host context")
		}
	}
}

// Cancellation releases routing work even if a host ignores its context and replies later.
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
	// Release after Route has freed its run, then let the late Go reply drain.
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
	// Both agents need active holds so a final request tests cleanup of the sibling's hold.
	plan := newRouter(t, `{"type":"stage_router","capable_target":"strong","efficient_target":"weak","picker":"capable_first","confidence_threshold":0,"efficient_hold_turns":-1}`)
	for _, tc := range []struct {
		agent                string
		child, final, edited bool
		expected             string
	}{
		{"a", true, false, false, "strong-model"},
		{"a", true, false, true, "weak-model"},
		{"b", true, false, false, "strong-model"},
		{"b", true, false, true, "weak-model"},
		{"root", false, false, false, "strong-model"},
		{"a", true, true, false, "weak-model"},
		{"b", true, false, false, "strong-model"},
	} {
		request := input()
		request.Request = json.RawMessage(`{}`)
		if tc.edited {
			request.Request = json.RawMessage(`{"messages":[{"role":"assistant","content":[{"type":"tool_call","id":"edit","name":"apply_patch","arguments":{}}]},{"role":"tool","content":[{"type":"tool_result","tool_call_id":"edit","content":[{"type":"text","text":"Applied patch"}],"is_error":false}]}]}`)
		}
		request.Metadata = &RequestMetadata{SessionID: "session", AgentID: tc.agent, IsSubagent: tc.child, SessionFinal: &tc.final}
		outcome, err := plan.Route(context.Background(), request, nil)
		if err != nil || len(outcome.SelectedModelIDs) == 0 || outcome.SelectedModelIDs[0] != tc.expected {
			t.Fatalf("agent=%s final=%v outcome=%+v err=%v", tc.agent, tc.final, outcome, err)
		}
	}
}
