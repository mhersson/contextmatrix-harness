// Package llm is a raw-HTTP, OpenAI-compatible client for OpenRouter or any OpenAI-compatible endpoint behind a
// narrow Send/SendStream interface. No SDK: the streaming/tool-call path is
// owned end-to-end so weak-model quirks are handled explicitly.
package llm

import (
	"context"
	"encoding/json"
)

// Message is one chat message (OpenAI-compatible).
type Message struct {
	Role         string        `json:"role"`
	Content      string        `json:"content,omitempty"`
	ContentParts []ContentPart `json:"-"`
	ToolCalls    []ToolCall    `json:"tool_calls,omitempty"`
	ToolCallID   string        `json:"tool_call_id,omitempty"`
	Name         string        `json:"name,omitempty"`
}

// MarshalJSON emits `content` as the parts array when ContentParts is set,
// otherwise as the string Content (byte-identical to the prior wire form).
// When both are empty `content` is omitted - except for an assistant message
// with no tool calls, which emits an explicit `"content":""`: the Chat
// Completions contract requires assistant content unless tool_calls is
// present, and strict OpenAI-compatible endpoints reject a bare
// {"role":"assistant"} on every replay of the history.
func (m Message) MarshalJSON() ([]byte, error) {
	type wire struct {
		Role       string     `json:"role"`
		Content    any        `json:"content,omitempty"`
		ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
		ToolCallID string     `json:"tool_call_id,omitempty"`
		Name       string     `json:"name,omitempty"`
	}

	w := wire{Role: m.Role, ToolCalls: m.ToolCalls, ToolCallID: m.ToolCallID, Name: m.Name}
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

// Request is the /chat/completions body. Provider/Models are OpenRouter extras.
type Request struct {
	Model    string    `json:"model,omitempty"`
	Models   []string  `json:"models,omitempty"`
	Messages []Message `json:"messages"`
	Tools    []Tool    `json:"tools,omitempty"`
	// ToolChoice is the OpenAI-compatible tool_choice value: a JSON string
	// ("auto", "required") or a {"type":"function","function":{"name":...}}
	// object. Raw so callers control the exact wire shape; omitted when nil.
	ToolChoice json.RawMessage `json:"tool_choice,omitempty"`
	Stream     bool            `json:"stream,omitempty"`
	Provider   json.RawMessage `json:"provider,omitempty"`
	Reasoning  json.RawMessage `json:"reasoning,omitempty"`
	Usage      *UsageOpt       `json:"usage,omitempty"`
	// openai dialect only; populated by encodeRequest, never by callers. Slated to
	// move to a private wire struct in the next major version.
	ReasoningEffort string         `json:"reasoning_effort,omitempty"`
	StreamOptions   *streamOptions `json:"stream_options,omitempty"`
}

// UsageOpt opts into OpenRouter usage accounting (token counts + the
// authoritative USD cost) on the response, including the final SSE frame for
// streamed calls. OpenRouter omits cost on streamed responses without this.
type UsageOpt struct {
	Include bool `json:"include"`
}

// PromptTokensDetails is the OpenAI-shape nested usage detail. cached_tokens
// counts the portion of prompt_tokens served from prompt cache - a SUBSET of
// prompt_tokens on this wire shape.
type PromptTokensDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

type Usage struct {
	PromptTokens     int     `json:"prompt_tokens"`
	CompletionTokens int     `json:"completion_tokens"`
	TotalTokens      int     `json:"total_tokens"`
	Cost             float64 `json:"cost"` // OpenRouter authoritative USD for this call
	// PromptTokensDetails carries the OpenAI-shape cached-token detail
	// (cached_tokens is a subset of prompt_tokens). Emitted by OpenRouter and
	// OpenAI-native gateways.
	PromptTokensDetails *PromptTokensDetails `json:"prompt_tokens_details,omitempty"`
	// CacheReadInputTokens / CacheCreationInputTokens are Anthropic-shim
	// extensions (LiteLLM-style gateways) where cache buckets are DISJOINT
	// from prompt_tokens. Zero when the gateway does not emit them.
	CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
}

// Buckets normalizes both cache wire shapes into disjoint token buckets
// matching the Anthropic pricing model, where prompt excludes cache traffic:
// the OpenAI subset shape has cached_tokens subtracted out of prompt; the
// Anthropic-shim shape passes through unchanged. When both appear (LiteLLM-
// family shims), the shim value wins for cacheRead and the subset
// subtraction is skipped - prompt_tokens is already disjoint on that wire.
// Absent cache info yields (prompt_tokens, 0, 0).
func (u Usage) Buckets() (prompt, cacheRead, cacheCreation int) {
	prompt = u.PromptTokens

	if u.CacheReadInputTokens > 0 {
		cacheRead = u.CacheReadInputTokens
	} else if u.PromptTokensDetails != nil && u.PromptTokensDetails.CachedTokens > 0 {
		cacheRead = min(u.PromptTokensDetails.CachedTokens, prompt)
		prompt -= cacheRead
	}

	return prompt, cacheRead, u.CacheCreationInputTokens
}

// Response is the assembled result of one model call (stream or non-stream).
type Response struct {
	Content      string
	Reasoning    string
	ToolCalls    []ToolCall
	FinishReason string
	Model        string // model actually used (after fallback)
	Usage        Usage
}

// Delta is an incremental streaming fragment for live display.
type Delta struct {
	Content   string
	Reasoning string
}

// LLM is the narrow interface the harness depends on.
type LLM interface {
	Send(ctx context.Context, req Request) (Response, error)
	SendStream(ctx context.Context, req Request, onDelta func(Delta)) (Response, error)
}
