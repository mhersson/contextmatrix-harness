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

// assertProvider checks every model_response and usage event: they carry the
// provider key when want is set and have no such key at all otherwise.
func assertProvider(t *testing.T, transcript, want string, skipPhase string) {
	t.Helper()

	seen := 0

	for _, ev := range parseEvents(t, transcript) {
		if ev.Kind != events.ModelResponse && ev.Kind != events.UsageKind {
			continue
		}

		if skipPhase != "" && ev.Data["phase"] == skipPhase {
			continue
		}

		seen++

		got, ok := ev.Data["provider"]
		if want == "" {
			assert.False(t, ok, "%s must not carry a provider key", ev.Kind)

			continue
		}

		assert.Equal(t, want, got, "%s provider", ev.Kind)
	}

	require.Positive(t, seen)
}

func TestRunEmitsProviderOnModelEvents(t *testing.T) {
	tests := []struct{ name, provider string }{
		{"reported", "DeepInfra"},
		{"not reported", ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, transcript, err := runScripted(t, []llm.Response{
				{Content: "done", FinishReason: "stop", Provider: tt.provider},
			}, Config{})
			require.NoError(t, err)
			assertProvider(t, transcript, tt.provider, "")
		})
	}
}

func TestGraceTurnEmitsProvider(t *testing.T) {
	for _, provider := range []string{"DeepInfra", ""} {
		t.Run("provider="+provider, func(t *testing.T) {
			reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()), &graceFinishTool{})
			w := &burnLLM{responses: []llm.Response{
				{ToolCalls: []llm.ToolCall{toolCall("1", "read", `{"path":"m1"}`)}, Provider: provider},
				{ToolCalls: []llm.ToolCall{toolCall("2", "read", `{"path":"m2"}`)}, Provider: provider},
				{ToolCalls: []llm.ToolCall{toolCall("3", "finish", `{"commit_message":"done"}`)}, Provider: provider},
			}}

			var transcript bytes.Buffer

			res, err := Run(context.Background(), w, reg, events.NewEmitter(nil, &transcript), "task", Config{MaxTurns: 2, GraceTurn: true})
			require.NoError(t, err)
			require.Equal(t, "done", res.Reason)
			require.Len(t, w.requests, 3, "last request is the grace call")
			assertProvider(t, transcript.String(), provider, "")
		})
	}
}

func TestCompactionUsageEmitsProvider(t *testing.T) {
	for _, provider := range []string{"DeepInfra", ""} {
		t.Run("provider="+provider, func(t *testing.T) {
			history := make([]llm.Message, 20)
			for i := range history {
				history[i] = llm.Message{Role: "user", Content: "m"}
				if i%2 == 1 {
					history[i].Role = "assistant"
				}
			}

			fake := &capturingLLMSeq{responses: []llm.Response{
				{Content: "turn1", Usage: llm.Usage{PromptTokens: 900}},
				{Content: "SUMMARY", Provider: provider},
				{Content: "done", FinishReason: "stop"},
			}}

			var transcript bytes.Buffer

			_, err := Run(context.Background(), fake, tools.NewRegistry(), events.NewEmitter(nil, &transcript), "go", Config{
				MaxTurns: 10, ContextWindow: 1000, History: history,
				Compaction: &Compaction{Threshold: 0.85, KeepRecentTurns: 2},
			})
			require.NoError(t, err)

			var found bool

			for _, ev := range parseEvents(t, transcript.String()) {
				if ev.Kind != events.UsageKind || ev.Data["phase"] != "compaction" {
					continue
				}

				found = true
				got, ok := ev.Data["provider"]
				assert.Equal(t, provider != "", ok)

				if provider != "" {
					assert.Equal(t, provider, got)
				}
			}

			require.True(t, found, "compaction usage event not emitted")
		})
	}
}
