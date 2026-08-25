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

// countingReadOnlyTool counts executions and is marked side-effect-free, so the
// repeat guard may skip an identical second call to it.
type countingReadOnlyTool struct {
	name  string
	calls int
}

func (c *countingReadOnlyTool) Name() string { return c.name }

func (c *countingReadOnlyTool) Schema() llm.Tool {
	return llm.Tool{Type: "function", Function: llm.ToolFunction{Name: c.name}}
}

func (c *countingReadOnlyTool) Execute(_ context.Context, _ map[string]any) (tools.Result, error) {
	c.calls++

	return tools.Result{Text: fmt.Sprintf("result %d", c.calls)}, nil
}

func (c *countingReadOnlyTool) ReadOnly() bool { return true }

// delayedInbox delivers one message on the (after+1)th Drain, so an interjection
// can land between two turns rather than before the first.
type delayedInbox struct {
	msg   UserMessage
	after int
	seen  int
}

func (d *delayedInbox) Drain() []UserMessage {
	d.seen++

	if d.seen == d.after+1 {
		return []UserMessage{d.msg}
	}

	return nil
}

func (d *delayedInbox) Wait(context.Context) (UserMessage, error) {
	return UserMessage{}, ErrInboxClosed
}

// toolResults returns the content of every tool-role message in order.
func toolResults(msgs []llm.Message) []string {
	var out []string

	for _, m := range msgs {
		if m.Role == "tool" {
			out = append(out, m.Content)
		}
	}

	return out
}

func TestRepeatedReadOnlyCallIsSkipped(t *testing.T) {
	look := &countingReadOnlyTool{name: "look"}
	reg := tools.NewRegistry(look)

	f := &fakeLLM{responses: []llm.Response{
		{ToolCalls: []llm.ToolCall{toolCall("1", "look", `{"path":"a.go"}`)}},
		{ToolCalls: []llm.ToolCall{toolCall("2", "look", `{"path":"a.go"}`)}},
		{Content: "done", FinishReason: "stop"},
	}}

	var transcript bytes.Buffer

	res, err := Run(context.Background(), f, reg, events.NewEmitter(nil, &transcript), "task", Config{MaxTurns: 5})
	require.NoError(t, err)

	assert.Equal(t, 1, look.calls, "the identical second call must not execute")

	results := toolResults(res.Messages)
	require.Len(t, results, 2, "both calls still receive a result")
	assert.Contains(t, results[1], "turn 1", "the skip names the turn that already made the call")

	var skipped *events.Event

	evs := parseEvents(t, transcript.String())
	for i, ev := range evs {
		if ev.Kind == events.ToolResult && ev.Data["id"] == "2" {
			skipped = &evs[i]
		}
	}

	require.NotNil(t, skipped, "the skip is a normal tool_result on the transcript")
	assert.Equal(t, true, skipped.Data["skipped"])
	assert.InDelta(t, 1.0, skipped.Data["repeat_of_turn"], 0)
}

func TestRepeatedReadOnlyCallDifferentArgsExecutes(t *testing.T) {
	look := &countingReadOnlyTool{name: "look"}
	reg := tools.NewRegistry(look)

	f := &fakeLLM{responses: []llm.Response{
		{ToolCalls: []llm.ToolCall{toolCall("1", "look", `{"path":"a.go"}`)}},
		{ToolCalls: []llm.ToolCall{toolCall("2", "look", `{"path":"b.go"}`)}},
		{Content: "done", FinishReason: "stop"},
	}}

	_, err := Run(context.Background(), f, reg, newEmitter(), "task", Config{MaxTurns: 5})
	require.NoError(t, err)

	assert.Equal(t, 2, look.calls, "different arguments are a different call")
}

func TestRepeatedReadOnlyCallSurroundingWhitespaceMatches(t *testing.T) {
	look := &countingReadOnlyTool{name: "look"}
	reg := tools.NewRegistry(look)

	f := &fakeLLM{responses: []llm.Response{
		{ToolCalls: []llm.ToolCall{toolCall("1", "look", `{"path":"a.go"}`)}},
		{ToolCalls: []llm.ToolCall{toolCall("2", "look", "  {\"path\":\"a.go\"}\n")}},
		{Content: "done", FinishReason: "stop"},
	}}

	_, err := Run(context.Background(), f, reg, newEmitter(), "task", Config{MaxTurns: 5})
	require.NoError(t, err)

	assert.Equal(t, 1, look.calls, "normalizeArgs makes surrounding whitespace irrelevant")
}

func TestWriteInvalidatesTheRepeatGuard(t *testing.T) {
	look := &countingReadOnlyTool{name: "look"}
	mutate := &countingTool{name: "mutate"}
	reg := tools.NewRegistry(look, mutate)

	f := &fakeLLM{responses: []llm.Response{
		{ToolCalls: []llm.ToolCall{toolCall("1", "look", `{"path":"a.go"}`)}},
		{ToolCalls: []llm.ToolCall{toolCall("2", "mutate", `{}`)}},
		{ToolCalls: []llm.ToolCall{toolCall("3", "look", `{"path":"a.go"}`)}},
		{Content: "done", FinishReason: "stop"},
	}}

	_, err := Run(context.Background(), f, reg, newEmitter(), "task", Config{MaxTurns: 6})
	require.NoError(t, err)

	assert.Equal(t, 2, look.calls, "a read after a write must execute - the state may have changed")
}

func TestRepeatedBashCallsAlwaysExecute(t *testing.T) {
	reg := tools.NewRegistry(tools.NewBashTool(t.TempDir()))

	f := &fakeLLM{responses: []llm.Response{
		{ToolCalls: []llm.ToolCall{toolCall("1", "bash", `{"command":"echo RAN-THE-COMMAND"}`)}},
		{ToolCalls: []llm.ToolCall{toolCall("2", "bash", `{"command":"echo RAN-THE-COMMAND"}`)}},
		{Content: "done", FinishReason: "stop"},
	}}

	res, err := Run(context.Background(), f, reg, newEmitter(), "task", Config{MaxTurns: 5})
	require.NoError(t, err)

	results := toolResults(res.Messages)
	require.Len(t, results, 2)

	// An exact match, not a substring: the skip message happens to contain
	// several short words, and a loose assertion here could not fail.
	for i, r := range results {
		assert.Equal(t, "RAN-THE-COMMAND\n", r,
			"bash call %d must actually run: re-running a check after an edit is correct, not waste", i+1)
	}
}

func TestCompactionResetsTheRepeatGuard(t *testing.T) {
	look := &countingReadOnlyTool{name: "look"}
	reg := tools.NewRegistry(look)

	history := make([]llm.Message, 20)
	for i := range history {
		history[i] = llm.Message{Role: "user", Content: fmt.Sprintf("m %d", i)}
	}

	f := &capturingLLMSeq{responses: []llm.Response{
		{ToolCalls: []llm.ToolCall{toolCall("1", "look", `{"path":"a.go"}`)}},
		// Crosses the compaction threshold; its tool calls are discarded.
		{ToolCalls: []llm.ToolCall{toolCall("2", "look", `{"path":"b.go"}`)}, Usage: llm.Usage{PromptTokens: 900}},
		{Content: "SUMMARY", Usage: llm.Usage{PromptTokens: 100}},
		{ToolCalls: []llm.ToolCall{toolCall("3", "look", `{"path":"a.go"}`)}},
		{Content: "done", FinishReason: "stop"},
	}}

	_, err := Run(context.Background(), f, reg, newEmitter(), "task", Config{
		MaxTurns:      6,
		ContextWindow: 1000,
		Compaction:    &Compaction{Threshold: 0.85, KeepRecentTurns: 2},
		History:       history,
	})
	require.NoError(t, err)

	assert.Equal(t, 2, look.calls,
		"after compaction the model has genuinely lost the earlier result, so the call must execute again")
}

func TestHumanInterjectionInvalidatesTheRepeatGuard(t *testing.T) {
	look := &countingReadOnlyTool{name: "look"}
	reg := tools.NewRegistry(look)

	inbox := &delayedInbox{msg: UserMessage{MessageID: "u1", Content: "I just edited a.go - read it again"}, after: 1}

	f := &fakeLLM{responses: []llm.Response{
		{ToolCalls: []llm.ToolCall{toolCall("1", "look", `{"path":"a.go"}`)}},
		{ToolCalls: []llm.ToolCall{toolCall("2", "look", `{"path":"a.go"}`)}},
		{Content: "done", FinishReason: "stop"},
	}}

	_, err := Run(context.Background(), f, reg, newEmitter(), "task", Config{MaxTurns: 5, Inbox: inbox})
	require.NoError(t, err)

	assert.Equal(t, 2, look.calls,
		"a human who says the file changed must not be answered from a skipped call")
}

// D3: a read-only call that FAILED must not be remembered as if it produced a
// result - the retry would be told to use a result that never existed.
func TestFailedReadOnlyCallIsNotCached(t *testing.T) {
	look := &failingReadOnlyTool{}

	f := &fakeLLM{responses: []llm.Response{
		{ToolCalls: []llm.ToolCall{toolCall("1", "look", `{"path":"a.go"}`)}},
		{ToolCalls: []llm.ToolCall{toolCall("2", "look", `{"path":"a.go"}`)}},
		{Content: "done", FinishReason: "stop"},
	}}

	res, err := Run(context.Background(), f, tools.NewRegistry(look), newEmitter(), "task", Config{MaxTurns: 5})
	require.NoError(t, err)

	assert.Equal(t, 2, look.calls, "a failed call is not a result to reuse")

	for _, r := range toolResults(res.Messages) {
		assert.NotContains(t, r, "use that result", "no retry is pointed at a result that does not exist")
	}
}

// failingReadOnlyTool is a side-effect-free tool whose every call errors.
type failingReadOnlyTool struct{ calls int }

func (f *failingReadOnlyTool) Name() string { return "look" }
func (f *failingReadOnlyTool) Schema() llm.Tool {
	return llm.Tool{Type: "function", Function: llm.ToolFunction{Name: "look"}}
}

func (f *failingReadOnlyTool) Execute(context.Context, map[string]any) (tools.Result, error) {
	f.calls++

	return tools.Result{}, errors.New("file does not exist; use glob to list existing paths")
}
func (f *failingReadOnlyTool) ReadOnly() bool { return true }

// D4b: an injected synthetic nudge is not human input and must not clear the
// repeat guard.
func TestNudgeDoesNotClearTheRepeatGuard(t *testing.T) {
	look := &countingReadOnlyTool{name: "look"}
	other := &countingReadOnlyTool{name: "other"}

	// Distinct calls advance the nudge counter; the repeated one straddles the
	// nudge injection.
	w := &burnLLM{responses: []llm.Response{
		{ToolCalls: []llm.ToolCall{toolCall("1", "look", `{"path":"a.go"}`)}},
		{ToolCalls: []llm.ToolCall{toolCall("2", "other", `{"path":"b.go"}`)}},
		{ToolCalls: []llm.ToolCall{toolCall("3", "other", `{"path":"c.go"}`)}},
		{ToolCalls: []llm.ToolCall{toolCall("4", "other", `{"path":"d.go"}`)}},
		{ToolCalls: []llm.ToolCall{toolCall("5", "look", `{"path":"a.go"}`)}},
	}}

	_, err := Run(context.Background(), w, tools.NewRegistry(look, other), newEmitter(), "task", Config{
		MaxTurns: 5, BatchNudgeTurns: 3, BatchNudgeMessage: "BATCH UP",
	})
	require.NoError(t, err)
	require.Equal(t, 1, countUserMsg(w.requests[4], "BATCH UP"), "the nudge did fire")

	assert.Equal(t, 1, look.calls, "the nudge is not human input and must not invalidate the guard")
}
