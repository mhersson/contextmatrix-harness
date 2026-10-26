// Package llm is a raw-HTTP, OpenAI-compatible client for OpenRouter or any OpenAI-compatible endpoint behind a
// narrow Send/SendStream interface. No SDK: the streaming/tool-call path is
// owned end-to-end so weak-model quirks are handled explicitly.
package llm

import (
	"context"
	"encoding/json"
	"math"
)

// Message is one chat message (OpenAI-compatible).
type Message struct {
	Role         string        `json:"role"`
	Content      string        `json:"content,omitempty"`
	ContentParts []ContentPart `json:"-"`
	ToolCalls    []ToolCall    `json:"tool_calls,omitempty"`
	ToolCallID   string        `json:"tool_call_id,omitempty"`
	Name         string        `json:"name,omitempty"`
	// ReasoningDetails is the provider's reasoning_details array for an
	// assistant reply, replayed verbatim. It is opaque and signed: never modify it.
	ReasoningDetails json.RawMessage `json:"reasoning_details,omitempty"`
	// ReasoningProvider is the provider that served the reply whose
	// ReasoningDetails this message carries, as the gateway reported it;
	// persisted with the message, never sent.
	ReasoningProvider string `json:"reasoning_provider,omitempty"`
}

// HasReasoningDetails reports whether ReasoningDetails would be sent: a
// non-empty JSON array.
func (m Message) HasReasoningDetails() bool {
	return replayableDetails(m.ReasoningDetails) != nil
}

// MarshalJSON emits `content` as the parts array when ContentParts is set,
// otherwise as the string Content.
// When both are empty `content` is omitted - except for an assistant message
// with no tool calls, which emits an explicit `"content":""`: the Chat
// Completions contract requires assistant content unless tool_calls is
// present, and strict OpenAI-compatible endpoints reject a bare
// {"role":"assistant"} on every replay of the history. reasoning_details is
// emitted only when it is a non-empty JSON array. reasoning_provider is emitted
// for persistence; encodeRequest removes it.
func (m Message) MarshalJSON() ([]byte, error) {
	type wire struct {
		Role              string          `json:"role"`
		Content           any             `json:"content,omitempty"`
		ToolCalls         []ToolCall      `json:"tool_calls,omitempty"`
		ReasoningDetails  json.RawMessage `json:"reasoning_details,omitempty"`
		ReasoningProvider string          `json:"reasoning_provider,omitempty"`
		ToolCallID        string          `json:"tool_call_id,omitempty"`
		Name              string          `json:"name,omitempty"`
	}

	w := wire{
		Role: m.Role, ToolCalls: m.ToolCalls, ReasoningDetails: replayableDetails(m.ReasoningDetails),
		ReasoningProvider: m.ReasoningProvider, ToolCallID: m.ToolCallID, Name: m.Name,
	}
	switch {
	case len(m.ContentParts) > 0:
		w.Content = m.ContentParts
	case m.Content != "":
		w.Content = m.Content
	case m.Role == "assistant" && len(m.ToolCalls) == 0:
		w.Content = "" // non-nil any: emitted as "content":""
	}

	return json.Marshal(w)
}

// ContentPart is one element of a multimodal message content array
// (OpenAI Chat Completions shape). Type is "text" or "image_url".
type ContentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

// ImageURL carries an image as a data URL ("data:<mime>;base64,<data>") or a
// remote URL, per the OpenAI image_url content part.
type ImageURL struct {
	URL string `json:"url"`
}

type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

type FunctionCall struct {
	Name string `json:"name"`
	// Arguments is a JSON-encoded string; weak models may emit malformed JSON.
	Arguments string `json:"arguments"`
}

type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

type ToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// Request is the /chat/completions body. Provider, Models, and cache_control are
// OpenRouter-only extensions.
type Request struct {
	Model    string    `json:"model,omitempty"`
	Models   []string  `json:"models,omitempty"`
	Messages []Message `json:"messages"`
	Tools    []Tool    `json:"tools,omitempty"`
	// ToolChoice is the OpenAI-compatible tool_choice value: a JSON string
	// ("auto", "required") or a {"type":"function","function":{"name":...}}
	// object. Raw so callers control the exact wire shape; omitted when nil.
	ToolChoice   json.RawMessage `json:"tool_choice,omitempty"`
	Stream       bool            `json:"stream,omitempty"`
	Provider     json.RawMessage `json:"provider,omitempty"`
	Reasoning    json.RawMessage `json:"reasoning,omitempty"`
	Usage        *UsageOpt       `json:"usage,omitempty"`
	CacheControl *CacheControl   `json:"cache_control,omitempty"`
	// openai dialect only; populated by encodeRequest, never by callers.
	ReasoningEffort string         `json:"reasoning_effort,omitempty"`
	StreamOptions   *streamOptions `json:"stream_options,omitempty"`
}

// CacheControl is the prompt-caching marker (OpenRouter-only extension;
// populated by encodeRequest, never by callers). The Type is always "ephemeral".
type CacheControl struct {
	Type string `json:"type"`
}

// UsageOpt opts into OpenRouter usage accounting (token counts + the
// authoritative USD cost) on the response, including the final SSE frame for
// streamed calls. OpenRouter omits cost on streamed responses without this.
type UsageOpt struct {
	Include bool `json:"include"`
}

// PromptTokensDetails is the OpenAI-shape nested usage detail. cached_tokens
// counts tokens served from prompt cache; cache_write_tokens counts tokens
// written to the cache. Both are SUBSETS of prompt_tokens on this wire shape.
type PromptTokensDetails struct {
	CachedTokens     int `json:"cached_tokens"`
	CacheWriteTokens int `json:"cache_write_tokens"`
}

// CompletionTokensDetails carries the OpenAI-shape completion detail.
type CompletionTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens"`
}

// UnmarshalJSON never fails: reasoning_tokens is best-effort telemetry, so a
// malformed value must not fail the call. Only integral values in [0, 2^31) are kept.
func (c *CompletionTokensDetails) UnmarshalJSON(data []byte) error {
	*c = CompletionTokensDetails{}

	var raw struct {
		ReasoningTokens json.RawMessage `json:"reasoning_tokens"`
	}
	if json.Unmarshal(data, &raw) != nil {
		return nil
	}

	var f float64
	if json.Unmarshal(raw.ReasoningTokens, &f) == nil && f == math.Trunc(f) && f >= 0 && f < 1<<31 {
		c.ReasoningTokens = int(f)
	}

	return nil
}

type Usage struct {
	PromptTokens     int     `json:"prompt_tokens"`
	CompletionTokens int     `json:"completion_tokens"`
	TotalTokens      int     `json:"total_tokens"`
	Cost             float64 `json:"cost"` // OpenRouter authoritative USD for this call
	// PromptTokensDetails carries the OpenAI-shape cached-token detail
	// (cached_tokens and cache_write_tokens are subsets of prompt_tokens).
	// Emitted by OpenRouter and OpenAI-native gateways.
	PromptTokensDetails *PromptTokensDetails `json:"prompt_tokens_details,omitempty"`
	// CompletionTokensDetails.ReasoningTokens is a subset of completion_tokens,
	// not a separate bucket. OpenRouter and the endpoint gateway report it, the
	// gateway only when above 0.
	CompletionTokensDetails *CompletionTokensDetails `json:"completion_tokens_details,omitempty"`
	// CacheReadInputTokens / CacheCreationInputTokens are Anthropic-shim
	// extensions (LiteLLM-style gateways) where cache buckets are DISJOINT
	// from prompt_tokens. Zero when the gateway does not emit them.
	CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
}

// ReasoningTokens returns the reasoning tokens within CompletionTokens, or 0
// when the provider did not report them.
func (u Usage) ReasoningTokens() int {
	if u.CompletionTokensDetails == nil {
		return 0
	}

	return u.CompletionTokensDetails.ReasoningTokens
}

// Buckets normalizes per-token usage into three disjoint buckets: prompt
// (plain input tokens), cacheRead (tokens served from prompt cache), and
// cacheCreation (tokens written to the cache).
//
// The read bucket follows this precedence:
//   - If CacheReadInputTokens > 0 (Anthropic-shim shape), that value is the
//     read bucket outright and nothing is subtracted from prompt_tokens.
//   - Otherwise, PromptTokensDetails.CachedTokens (OpenAI subset shape) is
//     clamped to prompt_tokens and subtracted from it.
//
// The creation bucket mirrors the same rule independently:
//   - If CacheCreationInputTokens > 0 (Anthropic-shim shape), that value is
//     the creation bucket outright and nothing is subtracted from prompt_tokens.
//   - Otherwise, PromptTokensDetails.CacheWriteTokens (OpenAI subset shape)
//     is clamped to the prompt REMAINING after the read subtraction (order:
//     cached_tokens subtracted first, then write clamped/subtracted) and
//     subtracted from it.
//
// Absent cache info yields (prompt_tokens, 0, 0).
func (u Usage) Buckets() (prompt, cacheRead, cacheCreation int) {
	prompt = u.PromptTokens

	if u.CacheReadInputTokens > 0 {
		cacheRead = u.CacheReadInputTokens
	} else if u.PromptTokensDetails != nil && u.PromptTokensDetails.CachedTokens > 0 {
		cacheRead = min(u.PromptTokensDetails.CachedTokens, prompt)
		prompt -= cacheRead
	}

	if u.CacheCreationInputTokens > 0 {
		cacheCreation = u.CacheCreationInputTokens
	} else if u.PromptTokensDetails != nil && u.PromptTokensDetails.CacheWriteTokens > 0 {
		cacheCreation = min(u.PromptTokensDetails.CacheWriteTokens, prompt)
		prompt -= cacheCreation
	}

	return prompt, cacheRead, cacheCreation
}

// Response is the assembled result of one model call (stream or non-stream).
type Response struct {
	Content   string
	Reasoning string
	// ReasoningDetails is the reply's replayable reasoning_details array; nil when none.
	ReasoningDetails json.RawMessage
	ToolCalls        []ToolCall
	FinishReason     string
	Model            string // model actually used (after fallback)
	Provider         string // upstream provider that served the call, as reported by the gateway (OpenRouter); empty when not reported
	Usage            Usage
}

// Delta is one incremental streaming event for live display; exactly one
// field is set.
type Delta struct {
	Content   string
	Reasoning string
	// ToolCall is set when a structured tool call gains an id, a name or an
	// argument fragment. Harmony-format calls arrive as Content instead.
	ToolCall *ToolCallDelta
	// Reset voids every fragment this SendStream call delivered so far: the
	// stream failed mid-read and is replayed from the start.
	Reset bool
}

// ToolCallDelta is one step of a streamed tool call. ID and Name are the
// latest values seen for Index: a provider may send the name after the first
// argument fragments.
type ToolCallDelta struct {
	Index        int
	ID           string
	Name         string
	ArgsFragment string
}

// LLM is the narrow interface the harness depends on. An implementation reports
// a non-200 reply as *StatusError, and a wrapper must keep it reachable with %w,
// because Run's 400 handling uses errors.As.
type LLM interface {
	Send(ctx context.Context, req Request) (Response, error)
	SendStream(ctx context.Context, req Request, onDelta func(Delta)) (Response, error)
}
