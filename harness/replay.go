package harness

import "github.com/mhersson/contextmatrix-harness/llm"

// withoutReasoning returns a copy of msgs with every ReasoningDetails nil.
// It never writes to msgs, which may be the caller's History.
func withoutReasoning(msgs []llm.Message) []llm.Message {
	if len(msgs) == 0 {
		return nil
	}

	out := make([]llm.Message, len(msgs))
	copy(out, msgs)

	for i := range out {
		out[i].ReasoningDetails = nil
	}

	return out
}
