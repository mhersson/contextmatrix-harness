package harness

import (
	"bytes"
	"context"
	"fmt"
	"testing"

	"github.com/mhersson/contextmatrix-harness/events"
	"github.com/mhersson/contextmatrix-harness/llm"
	"github.com/mhersson/contextmatrix-harness/tools"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// burnLLM records every request and plays back scripted responses; once the
// script is exhausted it returns a tool-call turn on every request (never
// stops on its own). An empty script burns turns from the start.
type burnLLM struct {
	responses []llm.Response
	requests  []llm.Request
}

func (w *burnLLM) Send(ctx context.Context, req llm.Request) (llm.Response, error) {
	return w.SendStream(ctx, req, nil)
}

func (w *burnLLM) SendStream(_ context.Context, req llm.Request, _ func(llm.Delta)) (llm.Response, error) {
	w.requests = append(w.requests, req)

	if len(w.responses) > 0 {
		r := w.responses[0]
		w.responses = w.responses[1:]

		return r, nil
	}

	return llm.Response{ToolCalls: []llm.ToolCall{
		toolCall(fmt.Sprintf("c%d", len(w.requests)), "read", `{"path":"missing"}`),
	}}, nil
}

// countUserMsg counts user-role messages in req whose content equals text.
func countUserMsg(req llm.Request, text string) int {
	n := 0

	for _, m := range req.Messages {
		if m.Role == "user" && m.Content == text {
			n++
		}
	}

	return n
}

func TestWrapUpNudgeInjectedOnceAtThreshold(t *testing.T) {
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()))
	w := &burnLLM{}

	var transcript bytes.Buffer

	emit := events.NewEmitter(nil, &transcript)

	res, err := Run(context.Background(), w, reg, emit, "task", Config{
		MaxTurns: 6, WrapUpTurns: 2, WrapUpMessage: "WRAP UP NOW",
	})
	require.NoError(t, err)
	assert.Equal(t, "max_turns", res.Reason)
	require.Len(t, w.requests, 6)

	// Fires at the top of the turn that leaves exactly 2 remaining: after 4
	// consumed turns, i.e. in request 5 (index 4) — and exactly once, ever.
	assert.Equal(t, 0, countUserMsg(w.requests[3], "WRAP UP NOW"), "no nudge before the threshold")
	assert.Equal(t, 1, countUserMsg(w.requests[4], "WRAP UP NOW"), "nudge lands when WrapUpTurns turns remain")
	assert.Equal(t, 1, countUserMsg(w.requests[5], "WRAP UP NOW"), "nudge is injected exactly once")

	assert.Contains(t, transcript.String(), "wrap_up_nudge", "the injection is evented for transcripts")
}

func TestWrapUpNudgeDefaultMessageNamesRemainingTurns(t *testing.T) {
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()))
	w := &burnLLM{}

	_, err := Run(context.Background(), w, reg, newEmitter(), "task", Config{
		MaxTurns: 4, WrapUpTurns: 2,
	})
	require.NoError(t, err)
	require.Len(t, w.requests, 4)

	last := w.requests[3].Messages

	var found string

	for _, m := range last {
		if m.Role == "user" && m.Content != "task" {
			found = m.Content
		}
	}

	require.NotEmpty(t, found, "a default nudge message must be injected when WrapUpMessage is empty")
	assert.Contains(t, found, "2 turn(s) left")
}

func TestWrapUpNudgeOffByDefault(t *testing.T) {
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()))
	w := &burnLLM{}

	_, err := Run(context.Background(), w, reg, newEmitter(), "task", Config{MaxTurns: 4})
	require.NoError(t, err)

	for i, req := range w.requests {
		for _, m := range req.Messages {
			if m.Role == "user" {
				assert.Equal(t, "task", m.Content, "request %d: only the seed user message exists when WrapUpTurns is 0", i+1)
			}
		}
	}
}

func TestWrapUpNudgeIgnoredInInteractiveMode(t *testing.T) {
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()))
	// 3 tool-call turns, then a plain answer; the closed inbox then ends the run.
	// The scripted burnLLM records requests so absence of the nudge is provable.
	w := &burnLLM{responses: []llm.Response{
		{ToolCalls: []llm.ToolCall{toolCall("1", "read", `{"path":"x"}`)}},
		{ToolCalls: []llm.ToolCall{toolCall("2", "read", `{"path":"x"}`)}},
		{ToolCalls: []llm.ToolCall{toolCall("3", "read", `{"path":"x"}`)}},
		{Content: "answer", FinishReason: "stop"},
	}}

	// MaxTurns 4 with WrapUpTurns 1 would fire at 3 consumed turns if the
	// interactive guard were missing.
	// scriptedInbox (harness_test.go) with only closeErr set is already-closed:
	// Drain returns nothing pending, Wait returns ErrInboxClosed immediately.
	res, err := Run(context.Background(), w, reg, newEmitter(), "task", Config{
		Interactive: true, Inbox: &scriptedInbox{closeErr: ErrInboxClosed}, MaxTurns: 4, WrapUpTurns: 1,
	})
	require.NoError(t, err)
	assert.True(t, res.Completed, "interactive run ends done on inbox close")

	for i, req := range w.requests {
		for _, m := range req.Messages {
			if m.Role == "user" {
				assert.Equal(t, "task", m.Content,
					"request %d: interactive mode must never inject the nudge", i+1)
			}
		}
	}
}
