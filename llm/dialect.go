package llm

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
