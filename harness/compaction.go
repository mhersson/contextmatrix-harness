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

// effectiveCompactionThreshold is the prompt-token count that triggers
// compaction: threshold*window, capped below window-reservedHeadroomTokens
// (so one turn's growth can't blow past the window), but the headroom floor
// is applied only when it is itself positive - a small window still gets a
// sane fractional trigger instead of a negative one.
func effectiveCompactionThreshold(window int, threshold float64) int {
	eff := int(threshold * float64(window))
	if floor := window - reservedHeadroomTokens; floor > 0 && floor < eff {
		eff = floor
	}

	return eff
}

// compact summarizes msgs[firstNonSystem : len-keepRecent] into one synthetic
// message, keeping the system message and the last keepRecent messages verbatim.
// Returns an error when there is not enough context to summarize meaningfully, or
// when the summarize call fails. The returned Usage is the cost/token accounting
// for the summarize call itself (zero value on any error path) - callers must
// fold it into their running totals since it is a real billable request.
func compact(ctx context.Context, client llm.LLM, cfg Config, msgs []llm.Message, keepRecent int, emit *events.Emitter) ([]llm.Message, llm.Usage, error) {
	if keepRecent < 0 {
		keepRecent = 0
	}

	sysCount := 0
	if len(msgs) > 0 && msgs[0].Role == "system" {
		sysCount = 1
	}

	if len(msgs)-sysCount-keepRecent <= 1 {
		return nil, llm.Usage{}, fmt.Errorf("compaction: not enough messages to summarize (total=%d sysCount=%d keepRecent=%d)", len(msgs), sysCount, keepRecent)
	}

	// Compute the split boundary, then snap it so a tool-call / tool-result
	// group is never divided across the two slices.
	b := len(msgs) - keepRecent
	// Step 1: if b lands inside a run of "tool" results, walk back to the
	// assistant that issued them so all results stay in the kept slice.
	for b > sysCount && b < len(msgs) && msgs[b].Role == "tool" {
		b--
	}
	// Step 2: if older would still end with an assistant that has unanswered
	// tool_calls, pull that assistant into kept-recent so the summarizer never
	// receives a dangling call.
	if b > sysCount && msgs[b-1].Role == "assistant" && len(msgs[b-1].ToolCalls) > 0 {
		b--
	}

	older := msgs[sysCount:b]
	if len(older) == 0 {
		return nil, llm.Usage{}, fmt.Errorf("compaction: snapped boundary left nothing to summarize (b=%d sysCount=%d keepRecent=%d)", b, sysCount, keepRecent)
	}

	req := llm.Request{
		Model:    cfg.Model,
		Models:   cfg.Models,
		Provider: cfg.Provider,
		Messages: append([]llm.Message{
			{Role: "system", Content: compactionPrompt},
		}, older...),
	}

	resp, err := client.Send(ctx, req)
	if err != nil {
		return nil, llm.Usage{}, fmt.Errorf("compaction summarize: %w", err)
	}

	out := make([]llm.Message, 0, sysCount+1+len(msgs)-b)
	out = append(out, msgs[:sysCount]...)
	out = append(out, llm.Message{Role: "system", Content: "[Earlier conversation, summarized]\n" + resp.Content})
	out = append(out, msgs[b:]...)

	emit.Emit(events.StateChange, map[string]any{
		"event":       "compaction",
		"kept_recent": keepRecent,
		"summarized":  len(older),
	})

	return out, resp.Usage, nil
}
