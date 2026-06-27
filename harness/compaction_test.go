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

// TestRun_CompactsWhenOverThreshold verifies that when cfg.Compaction is set and
// the first-turn response exceeds the effective compaction threshold, compact() is
// called, the summarize call receives the older prefix, the resulting messages
// replace the history (system + summary + last KeepRecentTurns), and a compaction
// state_change event is emitted.
//
// ContextWindow=1000 with reservedHeadroomTokens=8192 produces a negative
// effective threshold (-7192), which means any non-negative PromptTokens (here
// 900) crosses it. That is intentional for this test: the important assertion is
// structural — the right messages end up in the post-compaction request.
func TestRun_CompactsWhenOverThreshold(t *testing.T) {
	t.Helper()

	// 20-message history gives an older prefix long enough to summarize.
	history := make([]llm.Message, 20)
	for i := range history {
		if i%2 == 0 {
			history[i] = llm.Message{Role: "user", Content: fmt.Sprintf("user %d", i)}
		} else {
			history[i] = llm.Message{Role: "assistant", Content: fmt.Sprintf("assistant %d", i)}
		}
	}

	// Scripted responses consumed in order by both Send (compact) and SendStream (loop):
	//   [0] Turn 1 SendStream: PromptTokens=900 — triggers compaction
	//   [1] Compact Send:      returns "SUMMARY"
	//   [2] Turn 2 SendStream: no tool calls → "done"
	fake := &capturingLLMSeq{responses: []llm.Response{
		{Content: "turn1", Usage: llm.Usage{PromptTokens: 900}},
		{Content: "SUMMARY"},
		{Content: "done", FinishReason: "stop"},
	}}

	var transcript bytes.Buffer
	emit := events.NewEmitter(nil, &transcript)

	cfg := Config{
		MaxTurns:      10,
		SystemPrompt:  "SYS",
		ContextWindow: 1000,
		Compaction:    &Compaction{Threshold: 0.85, KeepRecentTurns: 2},
		History:       history,
	}

	res, err := Run(context.Background(), fake, tools.NewRegistry(), emit, "go", cfg)
	require.NoError(t, err)
	assert.Equal(t, "done", res.Reason)

	// A compaction state_change event must have been emitted.
	evs := parseEvents(t, transcript.String())

	var sawCompaction bool

	for _, ev := range evs {
		if ev.Kind == events.StateChange && ev.Data["event"] == "compaction" {
			sawCompaction = true

			break
		}
	}

	require.True(t, sawCompaction, "compaction state_change event not emitted")

	// Three requests: turn-1 SendStream, compact Send, turn-2 SendStream.
	require.Len(t, fake.requests, 3)

	// The post-compaction request (index 2) must carry the compacted history:
	// system prompt → summary → last KeepRecentTurns messages → task message.
	postMsgs := fake.requests[2].Messages
	require.Equal(t, "system", postMsgs[0].Role, "first message must be the system prompt")
	require.Contains(t, postMsgs[1].Content, "SUMMARY", "second message must contain the compaction summary")
	assert.LessOrEqual(t, len(postMsgs), 5, "system + summary + KeepRecentTurns(2) + task ≤ 5")
}
