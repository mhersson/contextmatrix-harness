package harness

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/mhersson/contextmatrix-harness/events"
	"github.com/mhersson/contextmatrix-harness/llm"
	"github.com/mhersson/contextmatrix-harness/tools"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// graceFinishTool is a self-contained Terminal double (independent of the
// finishTool double in harness_test.go).
type graceFinishTool struct{ called bool }

func (t *graceFinishTool) Name() string { return "finish" }
func (t *graceFinishTool) Schema() llm.Tool {
	return llm.Tool{Type: "function", Function: llm.ToolFunction{
		Name: "finish", Parameters: json.RawMessage(`{"type":"object"}`),
	}}
}

func (t *graceFinishTool) Execute(context.Context, map[string]any) (tools.Result, error) {
	t.called = true

	return tools.Result{Text: "ok"}, nil
}
func (t *graceFinishTool) Terminal() bool { return true }

// namedTerminalTool is a second, independently-named Terminal double, used to
// register more than one terminal tool at once.
type namedTerminalTool struct{ name string }

func (t *namedTerminalTool) Name() string { return t.name }
func (t *namedTerminalTool) Schema() llm.Tool {
	return llm.Tool{Type: "function", Function: llm.ToolFunction{
		Name: t.name, Parameters: json.RawMessage(`{"type":"object"}`),
	}}
}

func (t *namedTerminalTool) Execute(context.Context, map[string]any) (tools.Result, error) {
	return tools.Result{Text: "ok"}, nil
}
func (t *namedTerminalTool) Terminal() bool { return true }

func TestGraceTurnLandsTerminalCall(t *testing.T) {
	fin := &graceFinishTool{}
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()), fin)

	// 3 burn turns, then the grace-call response is a finish call.
	w := &burnLLM{responses: []llm.Response{
		{ToolCalls: []llm.ToolCall{toolCall("1", "read", `{"path":"missing"}`)}},
		{ToolCalls: []llm.ToolCall{toolCall("2", "read", `{"path":"missing"}`)}},
		{ToolCalls: []llm.ToolCall{toolCall("3", "read", `{"path":"missing"}`)}},
		{ToolCalls: []llm.ToolCall{toolCall("4", "finish", `{"commit_message":"done"}`)}},
	}}

	var transcript bytes.Buffer

	emit := events.NewEmitter(nil, &transcript)

	res, err := Run(context.Background(), w, reg, emit, "task", Config{MaxTurns: 3, GraceTurn: true})
	require.NoError(t, err)
	assert.True(t, res.Completed)
	assert.Equal(t, "done", res.Reason)
	assert.Equal(t, 3, res.Turns, "grace call does not count as a turn")
	assert.True(t, fin.called)
	assert.Contains(t, transcript.String(), "grace_turn")

	// The grace request offers ONLY terminal tools.
	require.Len(t, w.requests, 4)
	graceReq := w.requests[3]
	require.Len(t, graceReq.Tools, 1)
	assert.Equal(t, "finish", graceReq.Tools[0].Function.Name)

	// A single terminal tool forces the call via a named function choice.
	assert.JSONEq(t, `{"type":"function","function":{"name":"finish"}}`, string(graceReq.ToolChoice))
}

func TestGraceTurnToolChoiceRequiredWithMultipleTerminals(t *testing.T) {
	fin := &graceFinishTool{}
	other := &namedTerminalTool{name: "abort"}
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()), fin, other)

	w := &burnLLM{responses: []llm.Response{
		{ToolCalls: []llm.ToolCall{toolCall("1", "read", `{"path":"missing"}`)}},
		{ToolCalls: []llm.ToolCall{toolCall("2", "read", `{"path":"missing"}`)}},
		{ToolCalls: []llm.ToolCall{toolCall("3", "read", `{"path":"missing"}`)}},
		{ToolCalls: []llm.ToolCall{toolCall("4", "finish", `{"commit_message":"done"}`)}},
	}}

	res, err := Run(context.Background(), w, reg, newEmitter(), "task", Config{MaxTurns: 3, GraceTurn: true})
	require.NoError(t, err)
	assert.True(t, res.Completed)

	require.Len(t, w.requests, 4)
	graceReq := w.requests[3]
	require.Len(t, graceReq.Tools, 2)
	assert.JSONEq(t, `"required"`, string(graceReq.ToolChoice))
}

func TestGraceTurnToolChoiceFallbackOn400(t *testing.T) {
	fin := &graceFinishTool{}
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()), fin)

	w := &burnLLM{
		errs: []error{nil, nil, nil, errors.New("llm endpoint status 400: bad tool_choice")},
		responses: []llm.Response{
			{ToolCalls: []llm.ToolCall{toolCall("1", "read", `{"path":"missing"}`)}},
			{ToolCalls: []llm.ToolCall{toolCall("2", "read", `{"path":"missing"}`)}},
			{ToolCalls: []llm.ToolCall{toolCall("3", "read", `{"path":"missing"}`)}},
			{ToolCalls: []llm.ToolCall{toolCall("4", "finish", `{"commit_message":"done"}`)}},
		},
	}

	res, err := Run(context.Background(), w, reg, newEmitter(), "task", Config{MaxTurns: 3, GraceTurn: true})
	require.NoError(t, err)
	assert.True(t, res.Completed, "the run completes via the retried grace call")
	assert.True(t, fin.called)

	// 3 burn requests + 2 grace attempts (forced choice rejected, then retried bare).
	require.Len(t, w.requests, 5)
	assert.NotNil(t, w.requests[3].ToolChoice, "the first grace attempt still forces a choice")
	assert.Nil(t, w.requests[4].ToolChoice, "the retry falls back to the instruction-only contract")
}

func TestGraceTurnDeclinedStaysMaxTurns(t *testing.T) {
	fin := &graceFinishTool{}
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()), fin)
	w := &burnLLM{responses: []llm.Response{
		{ToolCalls: []llm.ToolCall{toolCall("1", "read", `{"path":"missing"}`)}},
		{ToolCalls: []llm.ToolCall{toolCall("2", "read", `{"path":"missing"}`)}},
		{ToolCalls: []llm.ToolCall{toolCall("3", "read", `{"path":"missing"}`)}},
		{FinishReason: "stop"}, // grace call answers with prose, no tool call
	}}

	res, err := Run(context.Background(), w, reg, newEmitter(), "task", Config{MaxTurns: 3, GraceTurn: true})
	require.NoError(t, err)
	assert.False(t, res.Completed)
	assert.Equal(t, "max_turns", res.Reason)
	assert.False(t, fin.called)
}

func TestGraceTurnSkippedWithoutTerminalTool(t *testing.T) {
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()))
	w := &burnLLM{}

	res, err := Run(context.Background(), w, reg, newEmitter(), "task", Config{MaxTurns: 3, GraceTurn: true})
	require.NoError(t, err)
	assert.Equal(t, "max_turns", res.Reason)
	assert.Len(t, w.requests, 3, "no grace call without a terminal tool")
}
