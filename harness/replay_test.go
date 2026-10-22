package harness

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/mhersson/contextmatrix-harness/events"
	"github.com/mhersson/contextmatrix-harness/llm"
	"github.com/mhersson/contextmatrix-harness/tools"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const sigDetails = `[{"type":"reasoning.text","text":"","signature":"sig-A","format":"anthropic-claude-v1","index":0}]`

func readTurn(details string) llm.Response {
	return llm.Response{
		ToolCalls:        []llm.ToolCall{toolCall("c1", "read", `{"path":"f.txt"}`)},
		ReasoningDetails: json.RawMessage(details),
	}
}

func readReg(t *testing.T) *tools.Registry {
	t.Helper()

	root := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(root, "f.txt"), []byte("data"), 0o600))

	return tools.NewRegistry(tools.NewReadTool(root))
}

func TestRunReplayReasoning(t *testing.T) {
	const finDetails = `[{"type":"reasoning.text","text":"fin","index":0}]`

	tests := []struct {
		name      string
		replay    bool
		wantTurn2 string
		wantFinal string
	}{
		{"off sends none", false, "", ""},
		{"on carries each reply's details verbatim", true, sigDetails, finDetails},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fake := &capturingLLMSeq{responses: []llm.Response{
				readTurn(sigDetails),
				{Content: "done", FinishReason: "stop", ReasoningDetails: json.RawMessage(finDetails)},
			}}

			res, err := Run(context.Background(), fake, readReg(t), newEmitter(), "task",
				Config{MaxTurns: 5, ReplayReasoning: tt.replay})
			require.NoError(t, err)
			require.Len(t, fake.requests, 2)

			asst := fake.requests[1].Messages[1] // [0] task, [1] assistant, [2] tool
			require.Equal(t, "assistant", asst.Role)
			assert.Equal(t, tt.wantTurn2, string(asst.ReasoningDetails))

			last := res.Messages[len(res.Messages)-1]
			assert.Equal(t, tt.wantFinal, string(last.ReasoningDetails))
		})
	}
}

func TestRunReplayOffStripsSeededHistoryWithoutTouchingCaller(t *testing.T) {
	for _, replay := range []bool{false, true} {
		history := []llm.Message{
			{Role: "user", Content: "q"},
			{Role: "assistant", Content: "a", ReasoningDetails: json.RawMessage(sigDetails)},
		}
		fake := &capturingLLMSeq{}

		_, err := Run(context.Background(), fake, tools.NewRegistry(), newEmitter(), "task",
			Config{MaxTurns: 1, History: history, ReplayReasoning: replay})
		require.NoError(t, err)

		got := fake.requests[0].Messages[1].ReasoningDetails
		if replay {
			assert.JSONEq(t, sigDetails, string(got))
		} else {
			assert.Nil(t, got)
		}

		assert.JSONEq(t, sigDetails, string(history[1].ReasoningDetails), "the caller's History is never modified")
	}
}

func TestRunReplayNeverRedactsOrEmitsDetails(t *testing.T) {
	var transcript bytes.Buffer

	first := readTurn(sigDetails)
	first.Reasoning = "plan around sig-A"
	fake := &capturingLLMSeq{responses: []llm.Response{first}}

	_, err := Run(context.Background(), fake, readReg(t), events.NewEmitter(nil, &transcript), "task", Config{
		MaxTurns:         5,
		ReplayReasoning:  true,
		RedactToolOutput: func(s string) string { return strings.ReplaceAll(s, "sig-A", "***") },
	})
	require.NoError(t, err)
	require.Len(t, fake.requests, 2)
	assert.JSONEq(t, sigDetails, string(fake.requests[1].Messages[1].ReasoningDetails))
	assert.NotContains(t, transcript.String(), "sig-A", "no event carries the details or unredacted reasoning")
}

func TestWithoutReasoningNeverWritesInput(t *testing.T) {
	in := []llm.Message{{Role: "assistant", ReasoningDetails: json.RawMessage(sigDetails)}}

	out := withoutReasoning(in)

	require.Len(t, out, 1)
	assert.Nil(t, out[0].ReasoningDetails)
	assert.JSONEq(t, sigDetails, string(in[0].ReasoningDetails))
	assert.Empty(t, append([]llm.Message(nil), withoutReasoning(nil)...))
}

func TestPinProvider(t *testing.T) {
	const pin = "Claude Platform on AWS"

	tests := []struct {
		name, raw, want string // want "" = not pinned, raw returned unchanged
	}{
		{"unset becomes an order", "", `{"order":["Claude Platform on AWS"]}`},
		{"null becomes an order", "null", `{"order":["Claude Platform on AWS"]}`},
		{
			"consumer keys are kept", `{"sort":"price","allow_fallbacks":false,"require_parameters":true,"x_new":{"k":1}}`,
			`{"order":["Claude Platform on AWS"],"sort":"price","allow_fallbacks":false,"require_parameters":true,"x_new":{"k":1}}`,
		},
		{
			"pin leads the consumer order without a duplicate", `{"order":["Wafer","Claude Platform on AWS","Azure"]}`,
			`{"order":["Claude Platform on AWS","Wafer","Azure"]}`,
		},
		{
			"only that admits the pin", `{"only":["Anthropic","Claude Platform on AWS"]}`,
			`{"only":["Anthropic","Claude Platform on AWS"],"order":["Claude Platform on AWS"]}`,
		},
		{"ignore that names others", `{"ignore":["Azure"]}`, `{"ignore":["Azure"],"order":["Claude Platform on AWS"]}`},
		{"only that excludes the pin", `{"only":["Anthropic"]}`, ""},
		{"ignore that names the pin", `{"ignore":["Claude Platform on AWS"]}`, ""},
		{"names match exactly", `{"only":["claude platform on aws"]}`, ""},
		{"not an object", `["Anthropic"]`, ""},
		{"order not a string list", `{"order":"Anthropic"}`, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := pinProvider(json.RawMessage(tt.raw), pin)
			if tt.want == "" {
				assert.False(t, ok)
				assert.Equal(t, tt.raw, string(got), "an unpinned provider is sent as configured")

				return
			}

			assert.True(t, ok)
			assert.JSONEq(t, tt.want, string(got))
		})
	}
}

const (
	detailsA = `[{"type":"reasoning.text","text":"a","index":0}]`
	detailsB = `[{"type":"reasoning.text","text":"b","index":0}]`
)

func TestReplayPin(t *testing.T) {
	tests := []struct {
		name string
		msgs []llm.Message
		want string
	}{
		{"none", []llm.Message{{Role: "user", Content: "q"}}, ""},
		{"newest details decide", []llm.Message{
			{Role: "assistant", ReasoningDetails: json.RawMessage(detailsA), ReasoningProvider: "Wafer"},
			{Role: "assistant", ReasoningDetails: json.RawMessage(detailsB), ReasoningProvider: "Anthropic"},
			{Role: "tool", Content: "r"},
		}, "Anthropic"},
		{"a provider without details pins nothing", []llm.Message{
			{Role: "assistant", ReasoningDetails: json.RawMessage(detailsA), ReasoningProvider: "Wafer"},
			{Role: "assistant", ReasoningProvider: "Anthropic"},
		}, "Wafer"},
		{"newest details without a provider pin nothing", []llm.Message{
			{Role: "assistant", ReasoningDetails: json.RawMessage(detailsA), ReasoningProvider: "Wafer"},
			{Role: "assistant", ReasoningDetails: json.RawMessage(detailsB)},
		}, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, replayPin(tt.msgs))
		})
	}
}

func TestReplayRequest(t *testing.T) {
	msgs := []llm.Message{
		{Role: "assistant", Content: "1", ReasoningDetails: json.RawMessage(detailsA), ReasoningProvider: "Wafer"},
		{Role: "assistant", Content: "2", ReasoningDetails: json.RawMessage(sigDetails)},
		{Role: "assistant", Content: "3", ReasoningDetails: json.RawMessage(detailsB), ReasoningProvider: "Anthropic"},
	}
	in := llm.Request{Model: "m", Provider: json.RawMessage(`{"sort":"price"}`), Messages: msgs}

	out, pin := replayRequest(in)
	assert.Equal(t, "Anthropic", pin)
	assert.JSONEq(t, `{"order":["Anthropic"],"sort":"price"}`, string(out.Provider))
	assert.Nil(t, out.Messages[0].ReasoningDetails, "another provider's details are dropped")
	assert.Nil(t, out.Messages[1].ReasoningDetails, "details without provenance are dropped")
	assert.JSONEq(t, detailsB, string(out.Messages[2].ReasoningDetails))
	assert.JSONEq(t, detailsA, string(msgs[0].ReasoningDetails), "the caller's messages are never modified")
	assert.JSONEq(t, `{"sort":"price"}`, string(in.Provider))

	excluded := in
	excluded.Provider = json.RawMessage(`{"ignore":["Anthropic"]}`)

	out, pin = replayRequest(excluded)
	assert.Empty(t, pin)
	assert.Equal(t, excluded, out, "a declined pin leaves the request exactly as built")
}

func servedBy(resp llm.Response, provider string) llm.Response {
	resp.Provider = provider

	return resp
}

func modelRequests(t *testing.T, transcript string) []events.Event {
	t.Helper()

	var out []events.Event

	for _, ev := range parseEvents(t, transcript) {
		if ev.Kind == events.ModelRequest {
			out = append(out, ev)
		}
	}

	return out
}

func TestRunPinsReplayProvider(t *testing.T) {
	for _, replay := range []bool{false, true} {
		t.Run(fmt.Sprintf("replay=%v", replay), func(t *testing.T) {
			var transcript bytes.Buffer

			fake := &burnLLM{responses: []llm.Response{
				servedBy(readTurn(sigDetails), "Claude Platform on AWS"),
				servedBy(llm.Response{Content: "done", FinishReason: "stop"}, "Claude Platform on AWS"),
			}}

			res, err := Run(context.Background(), fake, readReg(t), events.NewEmitter(nil, &transcript), "task", Config{
				Model: "m", MaxTurns: 5, ReplayReasoning: replay, Provider: json.RawMessage(`{"sort":"price"}`),
			})
			require.NoError(t, err)
			require.Len(t, fake.requests, 2)
			assert.JSONEq(t, `{"sort":"price"}`, string(fake.requests[0].Provider), "the first request has nothing to pin")

			reqs := modelRequests(t, transcript.String())
			require.Len(t, reqs, 2)
			assert.NotContains(t, reqs[0].Data, "pinned_provider")

			if !replay {
				assert.JSONEq(t, `{"sort":"price"}`, string(fake.requests[1].Provider), "off never pins")
				assert.Empty(t, res.Messages[1].ReasoningProvider, "off records no provider")
				assert.NotContains(t, reqs[1].Data, "pinned_provider")

				return
			}

			assert.JSONEq(t, `{"order":["Claude Platform on AWS"],"sort":"price"}`, string(fake.requests[1].Provider))
			assert.Equal(t, "Claude Platform on AWS", res.Messages[1].ReasoningProvider)
			assert.Equal(t, "Claude Platform on AWS", reqs[1].Data["pinned_provider"])
		})
	}
}

func TestRunFallbackReplyMovesPinAndStripsForeignBlocks(t *testing.T) {
	fake := &burnLLM{responses: []llm.Response{
		servedBy(readTurn(detailsA), "Anthropic"),
		servedBy(readTurn(detailsB), "Azure"), // the gateway fell back
		{Content: "done", FinishReason: "stop"},
	}}

	res, err := Run(context.Background(), fake, readReg(t), newEmitter(), "task", Config{MaxTurns: 5, ReplayReasoning: true})
	require.NoError(t, err)
	require.Len(t, fake.requests, 3)

	// [0] task, [1] assistant A, [2] tool, [3] assistant B, [4] tool
	third := fake.requests[2]
	assert.JSONEq(t, `{"order":["Azure"]}`, string(third.Provider))
	assert.Nil(t, third.Messages[1].ReasoningDetails, "the earlier provider's block is dropped from the request")
	assert.JSONEq(t, detailsB, string(third.Messages[3].ReasoningDetails))
	assert.JSONEq(t, detailsA, string(res.Messages[1].ReasoningDetails), "the history keeps every block")
}

func TestRunResumedHistoryPinsFromStoredProvider(t *testing.T) {
	tests := []struct {
		name       string
		newest     llm.Message
		wantPin    string // "" = request Provider stays nil
		wantSentA  bool   // the Wafer block at index 1 reaches the wire
		wantSentV0 bool   // the provenance-less block at index 3 reaches the wire
	}{
		{"newest block names its provider", llm.Message{
			Role: "assistant", Content: "a3",
			ReasoningDetails: json.RawMessage(detailsB), ReasoningProvider: "Anthropic",
		}, "Anthropic", false, false},
		{"newest block predates provenance", llm.Message{
			Role: "assistant", Content: "a3",
			ReasoningDetails: json.RawMessage(detailsB),
		}, "", true, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stored, err := json.Marshal([]llm.Message{
				{Role: "user", Content: "q1"},
				{Role: "assistant", Content: "a1", ReasoningDetails: json.RawMessage(detailsA), ReasoningProvider: "Wafer"},
				{Role: "user", Content: "q2"},
				{Role: "assistant", Content: "a2", ReasoningDetails: json.RawMessage(sigDetails)},
				{Role: "user", Content: "q3"},
				tt.newest,
			})
			require.NoError(t, err)

			var history []llm.Message
			require.NoError(t, json.Unmarshal(stored, &history))

			fake := &burnLLM{responses: []llm.Response{{Content: "done", FinishReason: "stop"}}}

			_, err = Run(context.Background(), fake, tools.NewRegistry(), newEmitter(), "task",
				Config{MaxTurns: 1, History: history, ReplayReasoning: true})
			require.NoError(t, err)

			req := fake.requests[0]
			if tt.wantPin == "" {
				assert.Nil(t, req.Provider)
			} else {
				assert.JSONEq(t, `{"order":["`+tt.wantPin+`"]}`, string(req.Provider))
			}

			assert.Equal(t, tt.wantSentA, req.Messages[1].ReasoningDetails != nil)
			assert.Equal(t, tt.wantSentV0, req.Messages[3].ReasoningDetails != nil)
			assert.JSONEq(t, detailsB, string(req.Messages[5].ReasoningDetails))
			assert.JSONEq(t, detailsA, string(history[1].ReasoningDetails), "the caller's History is never modified")
		})
	}
}

func TestRunUnreportedProviderReplaysUnpinned(t *testing.T) {
	fake := &burnLLM{responses: []llm.Response{readTurn(sigDetails), {Content: "done", FinishReason: "stop"}}}

	res, err := Run(context.Background(), fake, readReg(t), newEmitter(), "task", Config{MaxTurns: 5, ReplayReasoning: true})
	require.NoError(t, err)
	require.Len(t, fake.requests, 2)
	assert.Nil(t, fake.requests[1].Provider)
	assert.JSONEq(t, sigDetails, string(fake.requests[1].Messages[1].ReasoningDetails))
	assert.Empty(t, res.Messages[1].ReasoningProvider)
}

func TestGraceTurnPinsReplayProvider(t *testing.T) {
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()), &graceFinishTool{})
	fake := &burnLLM{responses: []llm.Response{
		servedBy(llm.Response{ToolCalls: []llm.ToolCall{toolCall("1", "read", `{"path":"m1"}`)}, ReasoningDetails: json.RawMessage(detailsA)}, "Anthropic"),
		servedBy(llm.Response{ToolCalls: []llm.ToolCall{toolCall("2", "read", `{"path":"m2"}`)}, ReasoningDetails: json.RawMessage(detailsA)}, "Anthropic"),
		servedBy(llm.Response{ToolCalls: []llm.ToolCall{toolCall("3", "read", `{"path":"m3"}`)}, ReasoningDetails: json.RawMessage(detailsB)}, "Wafer"),
		{ToolCalls: []llm.ToolCall{toolCall("4", "finish", `{}`)}},
	}}

	res, err := Run(context.Background(), fake, reg, newEmitter(), "task",
		Config{MaxTurns: 3, GraceTurn: true, ReplayReasoning: true})
	require.NoError(t, err)
	assert.True(t, res.Completed)
	require.Len(t, fake.requests, 4)

	grace := fake.requests[3]
	assert.JSONEq(t, `{"order":["Wafer"]}`, string(grace.Provider))
	assert.NotNil(t, grace.ToolChoice)
	assert.Nil(t, grace.Messages[1].ReasoningDetails)
	assert.JSONEq(t, detailsB, string(grace.Messages[5].ReasoningDetails))
}
