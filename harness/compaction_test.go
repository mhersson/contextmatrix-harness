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

func TestCompactForwardsImagePrefixThenDropsIt(t *testing.T) {
	msgs := []llm.Message{
		{Role: "system", Content: "SYS"},
		{Role: "user", Content: "first"},
		{Role: "assistant", Content: "a1"},
		{Role: "user", ContentParts: []llm.ContentPart{
			{Type: "text", Text: "look"},
			{Type: "image_url", ImageURL: &llm.ImageURL{URL: "data:image/png;base64,AAAA"}},
		}},
		{Role: "assistant", Content: "a2"},
		{Role: "user", Content: "recent"},
	}

	capt := &capturingLLM{}

	out, _, err := compact(context.Background(), capt, Config{Model: "m"}, msgs, 1, newEmitter())
	require.NoError(t, err)

	// (a) The summarizer received the image-bearing prefix verbatim (not stripped).
	var sawImage bool

	for _, m := range capt.last.Messages {
		for _, p := range m.ContentParts {
			if p.Type == "image_url" && p.ImageURL != nil && p.ImageURL.URL == "data:image/png;base64,AAAA" {
				sawImage = true
			}
		}
	}

	assert.True(t, sawImage, "summarizer must receive the image-bearing prefix verbatim")

	// (b) The prefix (incl. the image message) is replaced by one text summary;
	// images do not persist past compaction. Result: system + summary + 1 kept.
	require.Len(t, out, 3)
	assert.Equal(t, "SYS", out[0].Content)
	assert.Contains(t, out[1].Content, "[Earlier conversation, summarized]")
	assert.Equal(t, "recent", out[2].Content)

	assert.Empty(t, out[1].ContentParts, "summary message must not carry image parts")
}

// assertWellPaired verifies that msgs contains no broken tool-call / tool-result
// grouping:
//   - every "tool" message has a preceding "assistant" message whose ToolCalls
//     list contains the matching ToolCallID
//   - every "assistant" message with ToolCalls is immediately followed by
//     contiguous "tool" result messages that cover every ID it issued
func assertWellPaired(t *testing.T, msgs []llm.Message) {
	t.Helper()

	for i, m := range msgs {
		if m.Role == "tool" {
			var owned bool

			for j := 0; j < i; j++ {
				for _, tc := range msgs[j].ToolCalls {
					if tc.ID == m.ToolCallID {
						owned = true

						break
					}
				}

				if owned {
					break
				}
			}

			assert.True(t, owned,
				"tool message at index %d (ToolCallID=%q) has no preceding assistant that owns it", i, m.ToolCallID)
		}

		if m.Role == "assistant" && len(m.ToolCalls) > 0 {
			needed := make(map[string]bool, len(m.ToolCalls))

			for _, tc := range m.ToolCalls {
				needed[tc.ID] = false
			}

			for j := i + 1; j < len(msgs) && msgs[j].Role == "tool"; j++ {
				needed[msgs[j].ToolCallID] = true
			}

			for id, found := range needed {
				assert.True(t, found,
					"assistant at index %d issued tool call %q but no subsequent tool result found", i, id)
			}
		}
	}
}

// TestCompactPreservesToolGroups verifies that compact() never splits a
// tool-call / tool-result group across the compaction boundary: every
// "tool" message in the kept slice must be owned by an "assistant" that is
// also in the kept slice, and the summarizer must not receive an older prefix
// that ends with an unanswered assistant tool_calls block.
func TestCompactPreservesToolGroups(t *testing.T) {
	t.Helper()

	toolCalls := []llm.ToolCall{
		{ID: "call-X", Type: "function", Function: llm.FunctionCall{Name: "foo", Arguments: "{}"}},
		{ID: "call-Y", Type: "function", Function: llm.FunctionCall{Name: "bar", Arguments: "{}"}},
		{ID: "call-Z", Type: "function", Function: llm.FunctionCall{Name: "baz", Arguments: "{}"}},
	}

	base := []llm.Message{
		{Role: "system", Content: "SYS"},
		{Role: "user", Content: "u1"},
		{Role: "assistant", Content: "a1"},
		{Role: "user", Content: "u2"},
		{Role: "assistant", ToolCalls: toolCalls},
		{Role: "tool", ToolCallID: "call-X", Content: "result-X"},
		{Role: "tool", ToolCallID: "call-Y", Content: "result-Y"},
		{Role: "tool", ToolCallID: "call-Z", Content: "result-Z"},
	}

	withTrailing := append(append([]llm.Message(nil), base...), llm.Message{Role: "user", Content: "u3"})

	tests := []struct {
		name       string
		msgs       []llm.Message
		keepRecent int
	}{
		{
			name:       "keepRecent splits mid-group no trailing message",
			msgs:       base,
			keepRecent: 2,
		},
		{
			name:       "keepRecent splits mid-group with trailing user message",
			msgs:       withTrailing,
			keepRecent: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			capt := &capturingLLM{}

			out, _, err := compact(context.Background(), capt, Config{Model: "m"}, tt.msgs, tt.keepRecent, newEmitter())
			require.NoError(t, err)

			assertWellPaired(t, out)

			// The older slice sent to the summarizer must not end with an assistant
			// that has unanswered tool_calls — that would produce a malformed request
			// and an HTTP 400 from the provider.
			if len(capt.last.Messages) > 1 {
				lastOlder := capt.last.Messages[len(capt.last.Messages)-1]
				assert.False(t, lastOlder.Role == "assistant" && len(lastOlder.ToolCalls) > 0,
					"summarizer received an older prefix ending with unanswered assistant tool_calls")
			}
		})
	}
}

func TestCompactKeepsRecentImageVerbatim(t *testing.T) {
	imgMsg := llm.Message{Role: "user", ContentParts: []llm.ContentPart{
		{Type: "text", Text: "look"},
		{Type: "image_url", ImageURL: &llm.ImageURL{URL: "data:image/png;base64,BBBB"}},
	}}
	msgs := []llm.Message{
		{Role: "system", Content: "SYS"},
		{Role: "user", Content: "old1"},
		{Role: "assistant", Content: "a1"},
		{Role: "user", Content: "old2"},
		{Role: "assistant", Content: "a2"},
		imgMsg,
	}

	out, _, err := compact(context.Background(), &capturingLLM{}, Config{Model: "m"}, msgs, 1, newEmitter())
	require.NoError(t, err)

	last := out[len(out)-1]
	require.Len(t, last.ContentParts, 2)
	require.NotNil(t, last.ContentParts[1].ImageURL)
	assert.Equal(t, "data:image/png;base64,BBBB", last.ContentParts[1].ImageURL.URL)
}
