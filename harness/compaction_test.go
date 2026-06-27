package harness

import (
	"bytes"
	"context"
	"encoding/json"
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
// ContextWindow=1000, Threshold=0.85: effectiveCompactionThreshold returns 850
// (the floor window-8192 is negative so it is not applied). PromptTokens=900
// exceeds 850, triggering compaction. The important assertion is structural —
// the right messages end up in the post-compaction request.
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

// TestCompact_ForwardsModelsAndProvider verifies that the summarize request
// built inside compact carries Models and Provider from the Config, not just
// Model — a Models[]-only consumer with an empty cfg.Model would otherwise
// send no routing information and cause the summary call to fail silently.
func TestCompact_ForwardsModelsAndProvider(t *testing.T) {
	t.Helper()

	history := make([]llm.Message, 20)
	for i := range history {
		if i%2 == 0 {
			history[i] = llm.Message{Role: "user", Content: fmt.Sprintf("user %d", i)}
		} else {
			history[i] = llm.Message{Role: "assistant", Content: fmt.Sprintf("assistant %d", i)}
		}
	}

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
		Models:        []string{"m1", "m2"},
		Provider:      json.RawMessage(`{"sort":"price"}`),
	}

	_, err := Run(context.Background(), fake, tools.NewRegistry(), emit, "go", cfg)
	require.NoError(t, err)

	// Three requests: turn-1 SendStream, compact Send, turn-2 SendStream.
	require.Len(t, fake.requests, 3)

	// The summarize request (index 1) must carry Models and Provider from Config.
	summarizeReq := fake.requests[1]
	assert.Equal(t, []string{"m1", "m2"}, summarizeReq.Models, "compact must forward Models to summarize request")
	assert.JSONEq(t, `{"sort":"price"}`, string(summarizeReq.Provider), "compact must forward Provider to summarize request")
}

// TestEffectiveCompactionThreshold covers the compaction threshold helper with
// cases spanning the floor-disabled region (small window), floor-wins region
// (medium window), threshold-wins region (large window), and the boundary at
// exactly reservedHeadroomTokens (floor==0, not applied).
func TestEffectiveCompactionThreshold(t *testing.T) {
	tests := []struct {
		window    int
		threshold float64
		want      int
		desc      string
	}{
		{1000, 0.85, 850, "small window: floor negative, not applied; threshold wins"},
		{20000, 0.85, 11808, "medium window: floor (11808) < threshold (17000); floor wins"},
		{100000, 0.85, 85000, "large window: threshold (85000) < floor (91808); threshold wins"},
		{100000, 0.5, 50000, "large window, low threshold: threshold (50000) < floor (91808); threshold wins"},
		{8192, 0.85, 6963, "window==reservedHeadroomTokens: floor==0, not applied; threshold wins"},
	}

	for _, tt := range tests {
		t.Run(tt.desc, func(t *testing.T) {
			got := effectiveCompactionThreshold(tt.window, tt.threshold)
			assert.Equal(t, tt.want, got)
			assert.Positive(t, got, "result must always be positive for any positive window")
		})
	}
}
