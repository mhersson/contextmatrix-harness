package llm

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

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

func TestUsageBuckets(t *testing.T) {
	tests := []struct {
		name                             string
		raw                              string
		prompt, cacheRead, cacheCreation int
	}{
		{
			name:   "no cache info",
			raw:    `{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110}`,
			prompt: 100,
		},
		{
			name:      "openai subset shape subtracts cached from prompt",
			raw:       `{"prompt_tokens":100,"completion_tokens":10,"prompt_tokens_details":{"cached_tokens":80}}`,
			prompt:    20,
			cacheRead: 80,
		},
		{
			name:          "anthropic shim shape is disjoint",
			raw:           `{"prompt_tokens":100,"completion_tokens":10,"cache_read_input_tokens":500,"cache_creation_input_tokens":40}`,
			prompt:        100,
			cacheRead:     500,
			cacheCreation: 40,
		},
		{
			name:      "cached_tokens clamped to prompt_tokens",
			raw:       `{"prompt_tokens":50,"prompt_tokens_details":{"cached_tokens":80}}`,
			prompt:    0,
			cacheRead: 50,
		},
		{
			name:          "both shapes present counts prompt and cacheRead once",
			raw:           `{"prompt_tokens":100,"completion_tokens":10,"prompt_tokens_details":{"cached_tokens":80},"cache_read_input_tokens":500,"cache_creation_input_tokens":40}`,
			prompt:        100,
			cacheRead:     500,
			cacheCreation: 40,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var u Usage
			require.NoError(t, json.Unmarshal([]byte(tt.raw), &u))
			p, cr, cc := u.Buckets()
			assert.Equal(t, tt.prompt, p)
			assert.Equal(t, tt.cacheRead, cr)
			assert.Equal(t, tt.cacheCreation, cc)
		})
	}
}
