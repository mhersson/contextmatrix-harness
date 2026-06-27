package harness

import (
	"context"
	"fmt"

	"github.com/mhersson/contextmatrix-harness/events"
	"github.com/mhersson/contextmatrix-harness/llm"
)

// reservedHeadroomTokens is the number of tokens to keep free after compaction.
// It forms a floor when computing the effective compaction threshold: if the
// context window minus this value is lower than the percentage threshold, the
// earlier (lower token count) triggers compaction first.
const reservedHeadroomTokens = 8192

// compactionPrompt instructs the model to summarize the older conversation
// prefix, preserving decisions, file paths, commands run, and current state.
const compactionPrompt = "Summarize the conversation so far into a compact briefing for continuing the work. " +
	"Preserve: decisions made, files/paths touched, commands run and their outcomes, and the current state/next step. " +
	"Omit pleasantries. Output only the briefing."

// compact summarizes msgs[firstNonSystem : len-keepRecent] into one synthetic
// message, keeping the system message and the last keepRecent messages verbatim.
// Returns an error when there is not enough context to summarize meaningfully, or
// when the summarize call fails.
func compact(ctx context.Context, client llm.LLM, model string, msgs []llm.Message, keepRecent int, emit *events.Emitter) ([]llm.Message, error) {
	sysCount := 0
	if len(msgs) > 0 && msgs[0].Role == "system" {
		sysCount = 1
	}

	if len(msgs)-sysCount-keepRecent <= 1 {
		return nil, fmt.Errorf("compaction: not enough messages to summarize (total=%d sysCount=%d keepRecent=%d)", len(msgs), sysCount, keepRecent)
	}

	older := msgs[sysCount : len(msgs)-keepRecent]

	req := llm.Request{
		Model: model,
		Messages: append([]llm.Message{
			{Role: "system", Content: compactionPrompt},
		}, older...),
	}

	resp, err := client.Send(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("compaction summarize: %w", err)
	}

	out := make([]llm.Message, 0, sysCount+1+keepRecent)
	out = append(out, msgs[:sysCount]...)
	out = append(out, llm.Message{Role: "system", Content: "[Earlier conversation, summarized]\n" + resp.Content})
	out = append(out, msgs[len(msgs)-keepRecent:]...)

	emit.Emit(events.StateChange, map[string]any{
		"event":       "compaction",
		"kept_recent": keepRecent,
		"summarized":  len(older),
	})

	return out, nil
}
