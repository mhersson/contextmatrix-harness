package harness

import (
	"bytes"
	"context"
	"errors"
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
// stops on its own). An empty script burns turns from the start. errs, when
// set, is consumed in request order ahead of responses: a non-nil entry
// returns a zero Response and that error instead of playing back a response.
type burnLLM struct {
	responses []llm.Response
	errs      []error
	requests  []llm.Request
}

func (w *burnLLM) Send(ctx context.Context, req llm.Request) (llm.Response, error) {
	return w.SendStream(ctx, req, nil)
}

func (w *burnLLM) SendStream(_ context.Context, req llm.Request, _ func(llm.Delta)) (llm.Response, error) {
	w.requests = append(w.requests, req)

	if len(w.errs) > 0 {
		e := w.errs[0]
		w.errs = w.errs[1:]

		if e != nil {
			return llm.Response{}, e
		}
	}

	if len(w.responses) > 0 {
		r := w.responses[0]
		w.responses = w.responses[1:]

		return r, nil
	}

	// A distinct path per turn: the repeat guard would otherwise skip the second
	// and later reads, which is correct behaviour but not what these tests measure.
	return llm.Response{ToolCalls: []llm.ToolCall{
		toolCall(fmt.Sprintf("c%d", len(w.requests)), "read",
			fmt.Sprintf(`{"path":"missing%d"}`, len(w.requests))),
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
	// consumed turns, i.e. in request 5 (index 4) - and exactly once, ever.
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

// singleReadResp is one turn that spends a whole round trip on one read.
func singleReadResp(id string) llm.Response {
	return llm.Response{ToolCalls: []llm.ToolCall{toolCall(id, "read", `{"path":"missing"}`)}}
}

func TestBatchNudgeInjectedOnceAtThreshold(t *testing.T) {
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()))
	w := &burnLLM{}

	var transcript bytes.Buffer

	emit := events.NewEmitter(nil, &transcript)

	res, err := Run(context.Background(), w, reg, emit, "task", Config{
		MaxTurns: 6, BatchNudgeTurns: 3, BatchNudgeMessage: "BATCH UP",
	})
	require.NoError(t, err)
	assert.Equal(t, "max_turns", res.Reason)
	require.Len(t, w.requests, 6)

	// Three consecutive single-read turns are turns 1-3, so the nudge lands at
	// the top of turn 4 (request index 3) - and exactly once, ever.
	assert.Equal(t, 0, countUserMsg(w.requests[2], "BATCH UP"), "no nudge before the threshold")
	assert.Equal(t, 1, countUserMsg(w.requests[3], "BATCH UP"), "nudge lands on the turn after the third single call")
	assert.Equal(t, 1, countUserMsg(w.requests[5], "BATCH UP"), "never injected twice")

	var nudge *events.Event

	for i, ev := range parseEvents(t, transcript.String()) {
		if ev.Kind == events.StateChange && ev.Data["event"] == "batch_nudge" {
			nudge = &parseEvents(t, transcript.String())[i]

			break
		}
	}

	require.NotNil(t, nudge, "the nudge must be reported on the event stream")
	assert.InDelta(t, 3.0, nudge.Data["single_call_turns"], 0, "the event reports the count that tripped it")
}

func TestBatchNudgeCounterResets(t *testing.T) {
	tests := []struct {
		name    string
		breaker llm.Response
	}{
		{
			name: "a turn with two calls",
			breaker: llm.Response{ToolCalls: []llm.ToolCall{
				toolCall("m1", "read", `{"path":"missing"}`),
				toolCall("m2", "read", `{"path":"other"}`),
			}},
		},
		{
			name:    "a write call",
			breaker: llm.Response{ToolCalls: []llm.ToolCall{toolCall("w1", "write", `{"path":"f.txt","content":"x"}`)}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			root := t.TempDir()
			reg := tools.NewRegistry(tools.NewReadTool(root), tools.NewWriteTool(root))
			w := &burnLLM{responses: []llm.Response{
				singleReadResp("1"),
				singleReadResp("2"),
				tt.breaker,
				singleReadResp("4"),
				singleReadResp("5"),
			}}

			_, err := Run(context.Background(), w, reg, newEmitter(), "task", Config{
				MaxTurns: 5, BatchNudgeTurns: 3, BatchNudgeMessage: "BATCH UP",
			})
			require.NoError(t, err)
			require.Len(t, w.requests, 5)

			assert.Equal(t, 0, countUserMsg(w.requests[4], "BATCH UP"),
				"the reset means the threshold is never reached in %d turns", len(w.requests))
		})
	}
}

func TestBatchNudgeIgnoresBash(t *testing.T) {
	root := t.TempDir()
	reg := tools.NewRegistry(tools.NewBashTool(root))
	w := &burnLLM{responses: []llm.Response{
		{ToolCalls: []llm.ToolCall{toolCall("b1", "bash", `{"command":"true"}`)}},
		{ToolCalls: []llm.ToolCall{toolCall("b2", "bash", `{"command":"true"}`)}},
		{ToolCalls: []llm.ToolCall{toolCall("b3", "bash", `{"command":"true"}`)}},
		{ToolCalls: []llm.ToolCall{toolCall("b4", "bash", `{"command":"true"}`)}},
	}}

	_, err := Run(context.Background(), w, reg, newEmitter(), "task", Config{
		MaxTurns: 4, BatchNudgeTurns: 2, BatchNudgeMessage: "BATCH UP",
	})
	require.NoError(t, err)
	require.Len(t, w.requests, 4)

	assert.Equal(t, 0, countUserMsg(w.requests[3], "BATCH UP"),
		"shell commands are order-dependent - batching them is never suggested")
}

// TestBatchNudgeTerminalCallResetsCounter pins that a terminal call is another
// turn shape. It must be a FAILING terminal call: a successful one ends the run
// before the counter is touched, so the earlier version of this test - which
// used a successful finish - asserted a property the loop does not have and
// could not fail.
func TestBatchNudgeTerminalCallResetsCounter(t *testing.T) {
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()), &finishTool{execErr: errors.New("boom")})
	w := &burnLLM{responses: []llm.Response{
		singleReadResp("1"),
		singleReadResp("2"),
		{ToolCalls: []llm.ToolCall{toolCall("3", "finish", `{"commit_message":"x"}`)}},
		singleReadResp("4"),
		singleReadResp("5"),
	}}

	_, err := Run(context.Background(), w, reg, newEmitter(), "task", Config{
		MaxTurns: 5, BatchNudgeTurns: 3, BatchNudgeMessage: "BATCH UP",
	})
	require.NoError(t, err)
	require.Len(t, w.requests, 5, "an erroring terminal call does not end the run")

	assert.Equal(t, 0, countUserMsg(w.requests[4], "BATCH UP"),
		"the terminal call breaks the run of single read-only turns")
}

func TestBatchNudgeDisabledAtZero(t *testing.T) {
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()))
	w := &burnLLM{}

	var transcript bytes.Buffer

	_, err := Run(context.Background(), w, reg, events.NewEmitter(nil, &transcript), "task", Config{MaxTurns: 4})
	require.NoError(t, err)
	require.Len(t, w.requests, 4)

	for _, m := range w.requests[3].Messages {
		if m.Role == "user" {
			assert.Equal(t, "task", m.Content, "a zero threshold injects nothing")
		}
	}

	for _, ev := range parseEvents(t, transcript.String()) {
		if ev.Kind == events.StateChange {
			assert.NotEqual(t, "batch_nudge", ev.Data["event"], "a zero threshold emits nothing")
		}
	}
}

// TestBatchNudgeYieldsToWrapUp drives a run where both thresholds trip on the
// same turn: the wrap-up lands at the top of turn 4 (2 of 5 turns left), and
// three single-read turns have just put the batch counter at its threshold.
func TestBatchNudgeYieldsToWrapUp(t *testing.T) {
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()))
	w := &burnLLM{}

	_, err := Run(context.Background(), w, reg, newEmitter(), "task", Config{
		MaxTurns: 5, WrapUpTurns: 2, WrapUpMessage: "WRAP UP NOW",
		BatchNudgeTurns: 3, BatchNudgeMessage: "BATCH UP",
	})
	require.NoError(t, err)
	require.Len(t, w.requests, 5)

	assert.Equal(t, 1, countUserMsg(w.requests[3], "WRAP UP NOW"), "the wrap-up nudge wins the turn")
	assert.Equal(t, 0, countUserMsg(w.requests[3], "BATCH UP"), "no two contradictory messages on one turn")
	assert.Equal(t, 0, countUserMsg(w.requests[4], "BATCH UP"),
		"and a run already told to finish is never later told to batch its reads")
}

// D4a: a call the repeat guard skipped is not a dispatched lookup, so it must
// not advance the batching nudge's turn shape.
func TestBatchNudgeIgnoresSkippedCalls(t *testing.T) {
	look := &countingReadOnlyTool{name: "look"}

	var resp []llm.Response
	for i := 1; i <= 5; i++ {
		resp = append(resp, llm.Response{ToolCalls: []llm.ToolCall{toolCall(fmt.Sprintf("%d", i), "look", `{"path":"a.go"}`)}})
	}

	w := &burnLLM{responses: resp}

	_, err := Run(context.Background(), w, tools.NewRegistry(look), newEmitter(), "task", Config{
		MaxTurns: 5, BatchNudgeTurns: 3, BatchNudgeMessage: "BATCH UP",
	})
	require.NoError(t, err)
	require.Len(t, w.requests, 5)

	assert.Equal(t, 0, countUserMsg(w.requests[4], "BATCH UP"),
		"a model repeating one call is looping, not failing to batch")
}
