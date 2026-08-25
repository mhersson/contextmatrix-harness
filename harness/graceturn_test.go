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
type graceFinishTool struct {
	execErr error
	called  bool
}

func (t *graceFinishTool) Name() string { return "finish" }
func (t *graceFinishTool) Schema() llm.Tool {
	return llm.Tool{Type: "function", Function: llm.ToolFunction{
		Name: "finish", Parameters: json.RawMessage(`{"type":"object"}`),
	}}
}

func (t *graceFinishTool) Execute(context.Context, map[string]any) (tools.Result, error) {
	t.called = true

	if t.execErr != nil {
		return tools.Result{}, t.execErr
	}

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

func TestGraceTurnCacheBucketsAccumulate(t *testing.T) {
	fin := &graceFinishTool{}
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()), fin)

	// 3 burn turns (no usage), then the grace call lands with cache detail.
	w := &burnLLM{responses: []llm.Response{
		{ToolCalls: []llm.ToolCall{toolCall("1", "read", `{"path":"missing"}`)}},
		{ToolCalls: []llm.ToolCall{toolCall("2", "read", `{"path":"missing"}`)}},
		{ToolCalls: []llm.ToolCall{toolCall("3", "read", `{"path":"missing"}`)}},
		{
			ToolCalls: []llm.ToolCall{toolCall("4", "finish", `{"commit_message":"done"}`)},
			Usage: llm.Usage{
				PromptTokens: 100, CompletionTokens: 10,
				PromptTokensDetails:      &llm.PromptTokensDetails{CachedTokens: 80},
				CacheCreationInputTokens: 7,
			},
		},
	}}

	res, err := Run(context.Background(), w, reg, newEmitter(), "task", Config{MaxTurns: 3, GraceTurn: true})
	require.NoError(t, err)
	assert.True(t, res.Completed)
	assert.EqualValues(t, 20, res.PromptTokens, "grace call's 100 prompt tokens minus the 80 cached")
	assert.EqualValues(t, 10, res.CompletionTokens)
	assert.EqualValues(t, 80, res.CacheReadTokens)
	assert.EqualValues(t, 7, res.CacheCreationTokens)
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

func TestGraceTurnDeclinesWithoutRetryOnNon400Error(t *testing.T) {
	fin := &graceFinishTool{}
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()), fin)

	w := &burnLLM{
		errs: []error{nil, nil, nil, errors.New("llm endpoint status 500: internal error")},
	}

	res, err := Run(context.Background(), w, reg, newEmitter(), "task", Config{MaxTurns: 3, GraceTurn: true})
	require.NoError(t, err)
	assert.False(t, res.Completed)
	assert.Equal(t, "max_turns", res.Reason)
	assert.False(t, fin.called)

	// 3 burn requests + exactly ONE grace request - a non-400 error declines
	// without a retry.
	require.Len(t, w.requests, 4)
}

func TestGraceTurnFallbackRetryDeclinedStaysMaxTurns(t *testing.T) {
	fin := &graceFinishTool{}
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()), fin)

	w := &burnLLM{
		errs: []error{nil, nil, nil, errors.New("llm endpoint status 400: bad tool_choice")},
		responses: []llm.Response{
			{ToolCalls: []llm.ToolCall{toolCall("1", "read", `{"path":"missing"}`)}},
			{ToolCalls: []llm.ToolCall{toolCall("2", "read", `{"path":"missing"}`)}},
			{ToolCalls: []llm.ToolCall{toolCall("3", "read", `{"path":"missing"}`)}},
			{FinishReason: "stop"}, // the retry succeeds transport-wise but declines with prose
		},
	}

	res, err := Run(context.Background(), w, reg, newEmitter(), "task", Config{MaxTurns: 3, GraceTurn: true})
	require.NoError(t, err)
	assert.False(t, res.Completed)
	assert.Equal(t, "max_turns", res.Reason)
	assert.False(t, fin.called)

	// 3 burn requests + 2 grace attempts (forced choice rejected, retry declines too).
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

func TestGraceTurnEmitsToolCallAndToolResult(t *testing.T) {
	fin := &graceFinishTool{}
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()), fin)

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
	require.True(t, res.Completed)

	evs := parseEvents(t, transcript.String())
	require.NotEmpty(t, evs)

	// Find the grace-turn tool_call event (id "4").
	callIdx, resultIdx, stateIdx := -1, -1, -1

	for i, ev := range evs {
		switch ev.Kind {
		case events.ToolCallKind:
			if ev.Data["id"] == "4" {
				callIdx = i
			}
		case events.ToolResult:
			if ev.Data["id"] == "4" {
				resultIdx = i
			}
		case events.StateChange:
			if v, ok := ev.Data["via"]; ok && v == "grace_turn" {
				stateIdx = i
			}
		}
	}

	require.NotEqual(t, -1, callIdx, "tool_call event for the grace call must exist")
	assert.Equal(t, "finish", evs[callIdx].Data["name"])
	assert.Contains(t, evs[callIdx].Data, "raw_args")

	require.NotEqual(t, -1, resultIdx, "tool_result event for the grace call must exist")
	assert.EqualValues(t, 2, evs[resultIdx].Data["output_len"], "output_len must be 2 (\"ok\")")

	require.NotEqual(t, -1, stateIdx, "state_change event with via=grace_turn must exist")

	// tool_call must appear before tool_result, which must appear before state_change.
	assert.Less(t, callIdx, resultIdx, "tool_call must precede tool_result")
	assert.Less(t, resultIdx, stateIdx, "tool_result must precede state_change")

	assert.GreaterOrEqual(t, res.ToolCallCount, 1,
		"ToolCallCount must be at least 1 (the grace call)")
}

func TestGraceTurnToolCallRedactedArgs(t *testing.T) {
	fin := &graceFinishTool{}
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()), fin)

	w := &burnLLM{responses: []llm.Response{
		{ToolCalls: []llm.ToolCall{toolCall("1", "read", `{"path":"missing"}`)}},
		{ToolCalls: []llm.ToolCall{toolCall("2", "read", `{"path":"missing"}`)}},
		{ToolCalls: []llm.ToolCall{toolCall("3", "read", `{"path":"missing"}`)}},
		{ToolCalls: []llm.ToolCall{toolCall("4", "finish", `{"commit_message":"done"}`)}},
	}}

	var transcript bytes.Buffer

	emit := events.NewEmitter(nil, &transcript)

	redact := func(s string) string {
		return "<REDACTED>"
	}

	res, err := Run(context.Background(), w, reg, emit, "task", Config{
		MaxTurns:         3,
		GraceTurn:        true,
		RedactToolOutput: redact,
	})
	require.NoError(t, err)
	require.True(t, res.Completed)

	evs := parseEvents(t, transcript.String())

	var toolCallEv *events.Event

	for i, ev := range evs {
		if ev.Kind == events.ToolCallKind && ev.Data["id"] == "4" {
			toolCallEv = &evs[i]

			break
		}
	}

	require.NotNil(t, toolCallEv, "tool_call event for the grace call must exist")
	assert.Equal(t, "<REDACTED>", toolCallEv.Data["raw_args"],
		"raw_args must be redacted when RedactToolOutput is configured")
}

func TestGraceTurnToolCallCountIncludesGraceCall(t *testing.T) {
	fin := &graceFinishTool{}
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()), fin)

	w := &burnLLM{responses: []llm.Response{
		{ToolCalls: []llm.ToolCall{toolCall("1", "read", `{"path":"missing"}`)}},
		{ToolCalls: []llm.ToolCall{toolCall("2", "read", `{"path":"missing"}`)}},
		{ToolCalls: []llm.ToolCall{toolCall("3", "read", `{"path":"missing"}`)}},
		{ToolCalls: []llm.ToolCall{toolCall("4", "finish", `{"commit_message":"done"}`)}},
	}}

	res, err := Run(context.Background(), w, reg, newEmitter(), "task", Config{MaxTurns: 3, GraceTurn: true})
	require.NoError(t, err)
	require.True(t, res.Completed)

	// 3 burn calls (one per turn) + 1 grace call = 4 total.
	assert.Equal(t, 4, res.ToolCallCount,
		"ToolCallCount must include the 3 burn calls plus the 1 grace call")
}

func TestGraceTurnFailingTerminalToolEmitsErrorResult(t *testing.T) {
	fin := &graceFinishTool{execErr: errors.New("boom")}
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()), fin)

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

	assert.False(t, res.Completed, "a failing terminal call does not complete the run")
	assert.Equal(t, "max_turns", res.Reason)
	assert.Equal(t, 4, res.ToolCallFailures, "the 3 failing reads plus the failing grace call")

	evs := parseEvents(t, transcript.String())

	resultIdx := -1

	for i, ev := range evs {
		if ev.Kind == events.ToolResult && ev.Data["id"] == "4" {
			resultIdx = i

			break
		}
	}

	require.NotEqual(t, -1, resultIdx, "the failing grace call must still answer its tool_call")
	assert.Contains(t, evs[resultIdx].Data, "error")
	assert.Contains(t, evs[resultIdx].Data["error"], "boom")
}

func TestGraceTurnUsageEventCarriesCacheBuckets(t *testing.T) {
	fin := &graceFinishTool{}
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()), fin)

	w := &burnLLM{responses: []llm.Response{
		{ToolCalls: []llm.ToolCall{toolCall("1", "read", `{"path":"missing"}`)}},
		{ToolCalls: []llm.ToolCall{toolCall("2", "read", `{"path":"missing"}`)}},
		{ToolCalls: []llm.ToolCall{toolCall("3", "read", `{"path":"missing"}`)}},
		{
			ToolCalls: []llm.ToolCall{toolCall("4", "finish", `{"commit_message":"done"}`)},
			Usage: llm.Usage{
				PromptTokens: 100, CompletionTokens: 20, Cost: 0.5,
				PromptTokensDetails:      &llm.PromptTokensDetails{CachedTokens: 60},
				CacheCreationInputTokens: 8,
			},
		},
	}}

	var transcript bytes.Buffer

	emit := events.NewEmitter(nil, &transcript)

	res, err := Run(context.Background(), w, reg, emit, "task", Config{MaxTurns: 3, GraceTurn: true})
	require.NoError(t, err)
	require.True(t, res.Completed)

	evs := parseEvents(t, transcript.String())

	// Locate the usage event that carries the cache fields (the grace call's).
	var graceUsageEv *events.Event

	for i, ev := range evs {
		if ev.Kind == events.UsageKind {
			if cr, ok := ev.Data["cache_read_tokens"]; ok && cr != float64(0) {
				graceUsageEv = &evs[i]

				break
			}
		}
	}

	require.NotNil(t, graceUsageEv, "must have a usage event with non-zero cache_read_tokens")

	// The grace-call usage event must carry both cache fields.
	assert.InDelta(t, 60.0, graceUsageEv.Data["cache_read_tokens"], 0, "cache_read_tokens must match")
	assert.InDelta(t, 8.0, graceUsageEv.Data["cache_creation_tokens"], 0, "cache_creation_tokens must match")

	// Pre-existing keys must be unchanged.
	assert.InDelta(t, 100.0, graceUsageEv.Data["prompt_tokens"], 0, "prompt_tokens must be preserved")
	assert.InDelta(t, 20.0, graceUsageEv.Data["completion_tokens"], 0, "completion_tokens must be preserved")
	assert.InDelta(t, 0.5, graceUsageEv.Data["cost_usd"], 0, "cost_usd must be preserved")
}
