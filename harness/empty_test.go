package harness

import (
	"bytes"
	"context"
	"testing"

	"github.com/mhersson/contextmatrix-harness/events"
	"github.com/mhersson/contextmatrix-harness/llm"
	"github.com/mhersson/contextmatrix-harness/tools"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// emptyReply is an empty response that still bills, so accumulation is observable.
func emptyReply() llm.Response {
	return llm.Response{FinishReason: "stop", Usage: llm.Usage{PromptTokens: 10, CompletionTokens: 1, Cost: 0.25}}
}

func retryEvents(t *testing.T, transcript string) []events.Event {
	t.Helper()

	var out []events.Event

	for _, ev := range parseEvents(t, transcript) {
		if ev.Kind == events.StateChange && ev.Data["event"] == "empty_response_retry" {
			out = append(out, ev)
		}
	}

	return out
}

func runScripted(t *testing.T, responses []llm.Response, cfg Config) (Result, *fakeLLM, string, error) {
	t.Helper()

	f := &fakeLLM{responses: responses}
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()))

	var transcript bytes.Buffer

	if cfg.MaxTurns == 0 && !cfg.Interactive {
		cfg.MaxTurns = 20
	}

	res, err := Run(context.Background(), f, reg, events.NewEmitter(nil, &transcript), "task", cfg)

	return res, f, transcript.String(), err
}

func TestRunEmptyReplyThenAnswerCompletes(t *testing.T) {
	res, f, tr, err := runScripted(t, []llm.Response{emptyReply(), {Content: "answer", FinishReason: "stop"}}, Config{})
	require.NoError(t, err)
	assert.True(t, res.Completed)
	assert.Equal(t, "done", res.Reason)
	assert.Equal(t, "answer", res.Output)
	assert.Equal(t, 2, f.i)

	retries := retryEvents(t, tr)
	require.Len(t, retries, 1)
	assert.EqualValues(t, 1, retries[0].Data["retry"])
	assert.EqualValues(t, 1, retries[0].Data["turn"])

	var assistant []string

	for _, m := range res.Messages {
		if m.Role == "assistant" {
			assistant = append(assistant, m.Content)
		}
	}

	assert.Equal(t, []string{"answer"}, assistant)
}

func TestRunEmptyRepliesExhaustRetriesAndFail(t *testing.T) {
	res, f, tr, err := runScripted(t, []llm.Response{emptyReply(), emptyReply(), emptyReply(), {Content: "late", FinishReason: "stop"}}, Config{})
	require.ErrorIs(t, err, ErrEmptyResponse)
	assert.False(t, res.Completed)
	assert.Equal(t, "error", res.Reason)
	assert.Equal(t, 3, f.i)
	assert.InDelta(t, 0.75, res.TotalCostUSD, 1e-9)
	assert.EqualValues(t, 30, res.PromptTokens)

	retries := retryEvents(t, tr)
	require.Len(t, retries, 2)
	assert.EqualValues(t, 1, retries[0].Data["retry"])
	assert.EqualValues(t, 2, retries[1].Data["retry"])

	var sawError bool

	for _, ev := range parseEvents(t, tr) {
		sawError = sawError || ev.Kind == events.ErrorKind
	}

	assert.True(t, sawError)
}

func TestRunEmptyReplyClassification(t *testing.T) {
	tests := []struct {
		name      string
		first     llm.Response
		wantCalls int
		wantRetry int
	}{
		{"whitespace content is empty", llm.Response{Content: "\n  ", FinishReason: "stop"}, 2, 1},
		{"reasoning only is a reply", llm.Response{Reasoning: "thinking", FinishReason: "stop"}, 1, 0},
		{"short answer is a reply", llm.Response{Content: "OK", FinishReason: "stop"}, 1, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			res, _, tr, err := runScripted(t, []llm.Response{tt.first}, Config{})
			require.NoError(t, err)
			assert.True(t, res.Completed)
			assert.Equal(t, "done", res.Reason)
			assert.Equal(t, tt.wantCalls, countKind(t, tr, events.ModelRequest))
			assert.Len(t, retryEvents(t, tr), tt.wantRetry)
		})
	}
}

func countKind(t *testing.T, transcript string, kind events.Kind) int {
	t.Helper()

	n := 0

	for _, ev := range parseEvents(t, transcript) {
		if ev.Kind == kind {
			n++
		}
	}

	return n
}

func TestRunEmptyCountResetsAfterNonEmptyReply(t *testing.T) {
	// Two empties, a tool turn, then two more empties: without the reset the
	// fourth empty would exceed the budget.
	call := llm.Response{ToolCalls: []llm.ToolCall{toolCall("1", "read", `{"path":"missing"}`)}}
	res, _, tr, err := runScripted(t, []llm.Response{
		emptyReply(), emptyReply(), call, emptyReply(), emptyReply(), {Content: "fin", FinishReason: "stop"},
	}, Config{})
	require.NoError(t, err)
	assert.Equal(t, "done", res.Reason)
	assert.Equal(t, "fin", res.Output)

	retries := retryEvents(t, tr)
	require.Len(t, retries, 4)
	assert.EqualValues(t, 1, retries[2].Data["retry"])
}

func TestRunEmptyRepliesStopAtCostCap(t *testing.T) {
	res, f, _, err := runScripted(t, []llm.Response{emptyReply(), emptyReply(), emptyReply()}, Config{MaxCostUSD: 0.3})
	require.NoError(t, err)
	assert.Equal(t, "max_cost", res.Reason)
	assert.False(t, res.Completed)
	assert.Equal(t, 2, f.i)
}

func TestInteractiveEmptyRepliesAwaitInputWithFreshBudget(t *testing.T) {
	inbox := &scriptedInbox{
		pending:            []UserMessage{{Content: "again", MessageID: "m1"}},
		closeErr:           ErrInboxClosed,
		deliverViaWaitOnly: true,
	}
	res, f, tr, err := runScripted(t, []llm.Response{
		emptyReply(), emptyReply(), emptyReply(), emptyReply(), {Content: "recovered", FinishReason: "stop"},
	}, Config{Interactive: true, Inbox: inbox})
	require.NoError(t, err)
	assert.True(t, res.Completed)
	assert.Equal(t, "recovered", res.Output)
	assert.Equal(t, 5, f.i)

	var sawError bool

	for _, ev := range parseEvents(t, tr) {
		sawError = sawError || ev.Kind == events.ErrorKind
	}

	assert.True(t, sawError)
	// Fresh budget after the human message: 2 retries before the failure, 1 after.
	assert.Len(t, retryEvents(t, tr), 3)
}

func TestRunEmptyReplyAtContextLimitStopsWithoutRetry(t *testing.T) {
	r := emptyReply()
	r.Usage.PromptTokens = 900
	res, f, tr, err := runScripted(t, []llm.Response{r, {Content: "late", FinishReason: "stop"}}, Config{ContextWindow: 1000})
	require.NoError(t, err)
	assert.Equal(t, "context_limit", res.Reason)
	assert.Equal(t, 1, f.i)
	assert.Empty(t, retryEvents(t, tr))
}

func TestRunEmptyReplyOverCompactionThresholdCompacts(t *testing.T) {
	history := make([]llm.Message, 20)
	for i := range history {
		history[i] = llm.Message{Role: "user", Content: "m"}
	}

	r := emptyReply()
	r.Usage.PromptTokens = 900
	res, _, tr, err := runScripted(t, []llm.Response{r, {Content: "SUMMARY"}, {Content: "final", FinishReason: "stop"}}, Config{
		ContextWindow: 1000,
		Compaction:    &Compaction{Threshold: 0.85, KeepRecentTurns: 2},
		History:       history,
	})
	require.NoError(t, err)
	assert.Equal(t, "done", res.Reason)
	assert.Equal(t, "final", res.Output)
	assert.Empty(t, retryEvents(t, tr))
}

func TestRunEmptyCountResetsWhenCompactionReplacesRequest(t *testing.T) {
	history := make([]llm.Message, 20)
	for i := range history {
		history[i] = llm.Message{Role: "user", Content: "m"}
	}

	toolOverThreshold := llm.Response{
		ToolCalls: []llm.ToolCall{toolCall("1", "read", `{"path":"missing"}`)},
		Usage:     llm.Usage{PromptTokens: 900},
	}
	res, _, tr, err := runScripted(t, []llm.Response{
		emptyReply(), emptyReply(), toolOverThreshold,
		{Content: "SUMMARY"},
		emptyReply(),
		{Content: "final", FinishReason: "stop"},
	}, Config{
		ContextWindow: 1000,
		Compaction:    &Compaction{Threshold: 0.85, KeepRecentTurns: 2},
		History:       history,
	})
	require.NoError(t, err)
	assert.Equal(t, "done", res.Reason)
	assert.Equal(t, "final", res.Output)
	assert.Len(t, retryEvents(t, tr), 3)
}
