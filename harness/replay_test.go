package harness

import (
	"bytes"
	"context"
	"encoding/json"
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
