package harness

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/mhersson/contextmatrix-harness/events"
	"github.com/mhersson/contextmatrix-harness/llm"
	"github.com/mhersson/contextmatrix-harness/tools"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// scriptedCall is one scripted LLM response: either a response or an error.
type scriptedCall struct {
	resp llm.Response
	err  error
}

// scriptedLLMWithErrors is a scripted LLM that can return either responses or
// errors per call, used to test interactive transport-error recovery.
type scriptedLLMWithErrors struct {
	results []scriptedCall
	i       int
}

func (s *scriptedLLMWithErrors) Send(ctx context.Context, req llm.Request) (llm.Response, error) {
	return s.sendNext()
}

func (s *scriptedLLMWithErrors) SendStream(ctx context.Context, req llm.Request, _ func(llm.Delta)) (llm.Response, error) {
	return s.sendNext()
}

func (s *scriptedLLMWithErrors) sendNext() (llm.Response, error) {
	if s.i >= len(s.results) {
		return llm.Response{FinishReason: "stop"}, nil
	}
	r := s.results[s.i]
	s.i++
	return r.resp, r.err
}

// TestInteractiveUnbounded asserts that Interactive=true with MaxTurns=0 runs
// more than defaultMaxTurns turns without returning "max_turns".
func TestInteractiveUnbounded(t *testing.T) {
	const scriptedTurns = 35 // must exceed defaultMaxTurns (30)

	// Script scriptedTurns tool calls (with a missing path so the tool errors
	// but the loop continues), then one no-tool-call response.
	responses := make([]llm.Response, scriptedTurns+1)
	for i := 0; i < scriptedTurns; i++ {
		responses[i] = llm.Response{
			ToolCalls: []llm.ToolCall{toolCall("1", "read", `{"path":"missing"}`)},
		}
	}
	responses[scriptedTurns] = llm.Response{Content: "done", FinishReason: "stop"}

	f := &fakeLLM{responses: responses}
	// Inbox closes immediately when Wait is called; no pending messages.
	inbox := &scriptedInbox{closeErr: ErrInboxClosed}
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()))

	res, err := Run(context.Background(), f, reg, newEmitter(), "task", Config{
		Interactive: true,
		MaxTurns:    0,
		Inbox:       inbox,
	})
	require.NoError(t, err)
	assert.True(t, res.Completed)
	assert.Equal(t, "done", res.Reason)
	assert.Greater(t, res.Turns, defaultMaxTurns, "interactive=true must not stop at defaultMaxTurns (%d)", defaultMaxTurns)
	assert.NotEqual(t, "max_turns", res.Reason, "interactive=true must not produce max_turns")
}

// TestInteractiveIncapableRecovery asserts that when the incapable threshold is
// tripped in interactive mode, the session emits an error event, resets, and
// continues after the next inbox message instead of returning ReasonIncapable.
func TestInteractiveIncapableRecovery(t *testing.T) {
	// Three consecutive bad calls trip the incapable threshold (threshold=3).
	badCall := toolCall("1", "read", `{ this is not json`)
	responses := []llm.Response{
		{ToolCalls: []llm.ToolCall{badCall}},
		{ToolCalls: []llm.ToolCall{badCall}},
		{ToolCalls: []llm.ToolCall{badCall}},
		// After inbox delivers a message, the model responds normally.
		{Content: "recovered", FinishReason: "stop"},
	}

	f := &fakeLLM{responses: responses}
	// One pending message delivered via Wait only; closes after.
	inbox := &scriptedInbox{
		pending:            []UserMessage{{Content: "try again", MessageID: "m1"}},
		closeErr:           ErrInboxClosed,
		deliverViaWaitOnly: true,
	}
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()))

	var transcript bytes.Buffer
	emit := events.NewEmitter(nil, &transcript)

	res, err := Run(context.Background(), f, reg, emit, "task", Config{
		Interactive:        true,
		MaxTurns:           20,
		Inbox:              inbox,
		IncapableThreshold: 3,
	})
	require.NoError(t, err)
	assert.NotEqual(t, ReasonIncapable, res.Reason, "interactive=true must not terminate with incapable")
	assert.True(t, res.Completed)
	assert.Equal(t, "done", res.Reason)

	// The error event for "model cannot drive the tool loop" must be emitted.
	evs := parseEvents(t, transcript.String())
	var sawIncapableError bool
	for _, ev := range evs {
		if ev.Kind == events.ErrorKind {
			if msg, ok := ev.Data["error"].(string); ok && strings.Contains(msg, "model cannot drive") {
				sawIncapableError = true
				break
			}
		}
	}
	assert.True(t, sawIncapableError, "error event for incapable must be emitted before awaiting input")
}

// TestInteractiveTransportErrorRecovery asserts that a SendStream error in
// interactive mode emits the error event and awaits the next inbox message
// instead of terminating with res.Reason="error".
func TestInteractiveTransportErrorRecovery(t *testing.T) {
	transportErr := errors.New("dial tcp: connection refused")

	fakeClient := &scriptedLLMWithErrors{
		results: []scriptedCall{
			{err: transportErr}, // turn 1: transport error
			{resp: llm.Response{Content: "recovered", FinishReason: "stop"}}, // turn 2: success
		},
	}
	// One message delivered via Wait (unblocks after transport error); closes after.
	inbox := &scriptedInbox{
		pending:            []UserMessage{{Content: "retry", MessageID: "m1"}},
		closeErr:           ErrInboxClosed,
		deliverViaWaitOnly: true,
	}
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()))

	var transcript bytes.Buffer
	emit := events.NewEmitter(nil, &transcript)

	res, err := Run(context.Background(), fakeClient, reg, emit, "task", Config{
		Interactive: true,
		MaxTurns:    10,
		Inbox:       inbox,
	})
	require.NoError(t, err, "session must not terminate with an error on transport failure")
	assert.NotEqual(t, "error", res.Reason, "interactive=true must not produce reason=error on transport failure")
	assert.True(t, res.Completed)
	assert.Equal(t, "done", res.Reason)

	// The error event must have been emitted for the transport failure.
	evs := parseEvents(t, transcript.String())
	var sawTransportError bool
	for _, ev := range evs {
		if ev.Kind == events.ErrorKind {
			if msg, ok := ev.Data["error"].(string); ok && strings.Contains(msg, "connection refused") {
				sawTransportError = true
				break
			}
		}
	}
	assert.True(t, sawTransportError, "error event for transport failure must be emitted")
}
