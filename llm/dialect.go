package llm

import "encoding/json"

// Dialect selects the wire format the client speaks. Both dialects target the
// OpenAI-compatible /chat/completions shape; they differ only in a few
// extension fields one provider adds and another does not.
type Dialect int

const (
	// DialectOpenRouter is the default: emits the provider routing object,
	// models[] failover, the reasoning object, and usage:{include} accounting.
	DialectOpenRouter Dialect = iota
	// DialectOpenAI targets a plain OpenAI-compatible endpoint: it omits the
	// extension fields above, sends reasoning effort as the top-level
	// reasoning_effort string, and opts into streamed usage via stream_options.
	DialectOpenAI
)

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// encodeRequest renders req to the wire bytes for the given dialect.
// DialectOpenRouter marshals Request verbatim (byte-identical to the prior wire
// form). DialectOpenAI strips the OpenRouter-only extension fields (provider,
// models, usage, the reasoning object) and renders reasoning + streamed-usage in
// OpenAI-native form. Both paths marshal the SAME struct, so a future Request
// field reaches both dialects unless explicitly stripped here.
func encodeRequest(req Request, d Dialect) ([]byte, error) {
	if d == DialectOpenAI {
		req.ReasoningEffort = extractReasoningEffort(req.Reasoning)

		req.Provider, req.Models, req.Reasoning, req.Usage = nil, nil, nil, nil
		if req.Stream {
			req.StreamOptions = &streamOptions{IncludeUsage: true}
		}
	}

	return json.Marshal(req)
}

// extractReasoningEffort pulls the effort string out of an OpenRouter-shaped
// reasoning object ({"effort":"..."}); returns "" when absent so the field is
// omitted. Malformed input yields "". NOTE: only `effort` crosses to the openai
// dialect - a reasoning object carrying only max_tokens/exclude yields "" and
// those budget controls are intentionally dropped (OpenAI reasoning_effort is a
// string tier with no budget equivalent).
func extractReasoningEffort(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}

	var r struct {
		Effort string `json:"effort"`
	}

	_ = json.Unmarshal(raw, &r) //nolint:errcheck // best-effort; "" on malformed

	return r.Effort
}
