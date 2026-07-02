package llm

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRequestMarshalsOpenRouterExtras(t *testing.T) {
	req := Request{
		Model:    "primary/model",
		Models:   []string{"primary/model", "fallback/model"},
		Provider: json.RawMessage(`{"require_parameters":true}`),
		Messages: []Message{{Role: "user", Content: "hi"}},
		Tools:    []Tool{{Type: "function", Function: ToolFunction{Name: "read", Parameters: json.RawMessage(`{"type":"object"}`)}}},
		Stream:   true,
	}
	b, err := json.Marshal(req)
	require.NoError(t, err)

	s := string(b)
	assert.Contains(t, s, `"models":["primary/model","fallback/model"]`)
	assert.Contains(t, s, `"provider":{"require_parameters":true}`)
	assert.Contains(t, s, `"stream":true`)
}

func TestResponseToleratesUnknownFields(t *testing.T) {
	// Unknown top-level + nested fields must not break decoding.
	raw := `{"model":"m","brand_new_field":42,"usage":{"prompt_tokens":3,"completion_tokens":5,"cost":0.0001,"surprise":true}}`

	var nr nonStreamResponse
	require.NoError(t, json.NewDecoder(strings.NewReader(raw)).Decode(&nr))
	assert.Equal(t, "m", nr.Model)
	assert.InEpsilon(t, 0.0001, nr.Usage.Cost, 1e-9)
}

func TestMessageMarshalJSON_TextOnly(t *testing.T) {
	b, err := json.Marshal(Message{Role: "user", Content: "hello"})
	require.NoError(t, err)
	assert.JSONEq(t, `{"role":"user","content":"hello"}`, string(b))
}

func TestMessageMarshalJSON_ToolCallsOmitEmptyContent(t *testing.T) {
	m := Message{Role: "assistant", ToolCalls: []ToolCall{
		{ID: "1", Type: "function", Function: FunctionCall{Name: "x", Arguments: "{}"}},
	}}
	b, err := json.Marshal(m)
	require.NoError(t, err)
	assert.JSONEq(t, `{"role":"assistant","tool_calls":[{"id":"1","type":"function","function":{"name":"x","arguments":"{}"}}]}`, string(b))
}

func TestMessageMarshalJSON_ContentParts(t *testing.T) {
	m := Message{Role: "user", ContentParts: []ContentPart{
		{Type: "text", Text: "describe this"},
		{Type: "image_url", ImageURL: &ImageURL{URL: "data:image/png;base64,AAAA"}},
	}}
	b, err := json.Marshal(m)
	require.NoError(t, err)
	assert.JSONEq(t, `{"role":"user","content":[
		{"type":"text","text":"describe this"},
		{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}
	]}`, string(b))
}

func TestMessageMarshalJSON_OpenAIImageShapeNotAnthropic(t *testing.T) {
	b, err := json.Marshal(Message{Role: "user", ContentParts: []ContentPart{
		{Type: "image_url", ImageURL: &ImageURL{URL: "data:image/png;base64,AAAA"}},
	}})
	require.NoError(t, err)
	assert.Contains(t, string(b), `"image_url"`)
	assert.NotContains(t, string(b), `"source"`) // not the Anthropic content-block shape
}

func TestMessageMarshalJSON_AssistantEmptyContentIsExplicit(t *testing.T) {
	tests := []struct {
		name string
		msg  Message
		want string
	}{
		{
			// Bare {"role":"assistant"} violates the Chat Completions contract
			// (assistant needs content unless tool_calls is present) and poisons
			// replayed Inbox-mode history on strict endpoints.
			name: "assistant with no content and no tool calls emits explicit empty content",
			msg:  Message{Role: "assistant"},
			want: `{"role":"assistant","content":""}`,
		},
		{
			name: "assistant with tool calls keeps omitting content",
			msg: Message{Role: "assistant", ToolCalls: []ToolCall{
				{ID: "1", Type: "function", Function: FunctionCall{Name: "x", Arguments: "{}"}},
			}},
			want: `{"role":"assistant","tool_calls":[{"id":"1","type":"function","function":{"name":"x","arguments":"{}"}}]}`,
		},
		{
			name: "non-assistant empty content is unchanged (omitted)",
			msg:  Message{Role: "user"},
			want: `{"role":"user"}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b, err := json.Marshal(tt.msg)
			require.NoError(t, err)
			assert.JSONEq(t, tt.want, string(b))
		})
	}
}
