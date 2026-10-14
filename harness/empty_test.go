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

// runEmpty runs the scripted replies and returns the result, the replies
// consumed and the number of empty_response_retry events.
func runEmpty(t *testing.T, cfg Config, replies ...llm.Response) (Result, int, int, error) {
	t.Helper()

	f := &fakeLLM{responses: replies}

	var transcript bytes.Buffer

	res, err := Run(context.Background(), f, tools.NewRegistry(), events.NewEmitter(nil, &transcript), "task", cfg)

	retries := 0

	for _, ev := range parseEvents(t, transcript.String()) {
		if ev.Data["event"] == "empty_response_retry" {
			retries++
		}
	}

	return res, f.i, retries, err
}

func TestRunEmptyRepliesRetryThenComplete(t *testing.T) {
	res, calls, _, err := runEmpty(t, Config{}, llm.Response{}, llm.Response{}, llm.Response{Content: "answer"})
	require.NoError(t, err)
	assert.True(t, res.Completed)
	assert.Equal(t, "done", res.Reason)
	assert.Equal(t, 3, calls)
}

func TestRunEmptyRepliesExhaustRetriesAndFail(t *testing.T) {
	res, calls, _, err := runEmpty(t, Config{}, llm.Response{}, llm.Response{}, llm.Response{}, llm.Response{Content: "late"})
	require.ErrorIs(t, err, ErrEmptyResponse)
	assert.Equal(t, "error", res.Reason)
	assert.Equal(t, 3, calls)
}

func TestRunReasoningOnlyReplyCompletes(t *testing.T) {
	res, calls, _, err := runEmpty(t, Config{}, llm.Response{Reasoning: "thought"}, llm.Response{Content: "late"})
	require.NoError(t, err)
	assert.Equal(t, "done", res.Reason)
	assert.Equal(t, 1, calls)
}

func TestRunContextLimitAndCompactionPrecedeEmptyRetry(t *testing.T) {
	overLimit := llm.Response{Usage: llm.Usage{PromptTokens: 900}}

	t.Run("hard stop", func(t *testing.T) {
		res, _, retries, err := runEmpty(t, Config{ContextWindow: 1000}, overLimit)
		require.NoError(t, err)
		assert.Equal(t, "context_limit", res.Reason)
		assert.Zero(t, retries)
	})

	t.Run("compaction", func(t *testing.T) {
		history := make([]llm.Message, 20)
		for i := range history {
			history[i] = llm.Message{Role: "user", Content: "m"}
		}

		res, calls, retries, err := runEmpty(t, Config{
			ContextWindow: 1000, History: history,
			Compaction: &Compaction{Threshold: 0.85, KeepRecentTurns: 2},
		}, overLimit, llm.Response{Content: "SUMMARY"}, llm.Response{Content: "answer"})
		require.NoError(t, err)
		assert.Equal(t, "done", res.Reason)
		assert.Equal(t, 3, calls)
		assert.Zero(t, retries)
	})
}

func TestInteractiveEmptyRepliesAwaitInput(t *testing.T) {
	inbox := &scriptedInbox{pending: []UserMessage{{Content: "again"}}, closeErr: ErrInboxClosed, deliverViaWaitOnly: true}
	res, calls, retries, err := runEmpty(t, Config{Interactive: true, Inbox: inbox},
		llm.Response{}, llm.Response{}, llm.Response{}, llm.Response{}, llm.Response{Content: "answer"})
	require.NoError(t, err)
	assert.True(t, res.Completed)
	assert.Equal(t, 5, calls)
	// Two retries before the failure and one after the message: the budget restarted.
	assert.Equal(t, 3, retries)
}
