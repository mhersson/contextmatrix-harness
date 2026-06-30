package llm

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestWithDialectSetsDialect(t *testing.T) {
	c := NewClient("k", WithDialect(DialectOpenAI))
	assert.Equal(t, DialectOpenAI, c.dialect)
}

func TestDefaultDialectIsOpenRouter(t *testing.T) {
	c := NewClient("k")
	assert.Equal(t, DialectOpenRouter, c.dialect)
}

func fullRequest(stream bool) Request {
	return Request{
		Model:     "m",
		Models:    []string{"a", "b"},
		Messages:  []Message{{Role: "user", Content: "hi"}},
		Provider:  json.RawMessage(`{"sort":"price"}`),
		Reasoning: json.RawMessage(`{"effort":"high"}`),
		Stream:    stream,
		Usage:     &UsageOpt{Include: true},
	}
}

func keys(t *testing.T, b []byte) map[string]json.RawMessage {
	t.Helper()
	var m map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(b, &m))
	return m
}

func TestEncodeRequestOpenRouterUnchanged(t *testing.T) {
	b, err := encodeRequest(fullRequest(true), DialectOpenRouter)
	require.NoError(t, err)
	m := keys(t, b)
	for _, k := range []string{"provider", "models", "usage", "reasoning"} {
		_, ok := m[k]
		assert.True(t, ok, "openrouter must keep %q", k)
	}
	_, hasEffort := m["reasoning_effort"]
	assert.False(t, hasEffort, "openrouter must not emit reasoning_effort")
}

func TestEncodeRequestOpenAIOmitsAndTranslates(t *testing.T) {
	b, err := encodeRequest(fullRequest(true), DialectOpenAI)
	require.NoError(t, err)
	m := keys(t, b)
	for _, k := range []string{"provider", "models", "usage", "reasoning"} {
		_, ok := m[k]
		assert.False(t, ok, "openai must omit %q", k)
	}
	assert.JSONEq(t, `"high"`, string(m["reasoning_effort"]))
	assert.JSONEq(t, `{"include_usage":true}`, string(m["stream_options"]))
}

func TestEncodeRequestOpenAINonStreamHasNoStreamOptions(t *testing.T) {
	b, err := encodeRequest(fullRequest(false), DialectOpenAI)
	require.NoError(t, err)
	_, ok := keys(t, b)["stream_options"]
	assert.False(t, ok, "stream_options only on streamed calls")
}

func TestExtractReasoningEffort(t *testing.T) {
	assert.Equal(t, "low", extractReasoningEffort(json.RawMessage(`{"effort":"low"}`)))
	assert.Equal(t, "", extractReasoningEffort(nil))
	assert.Equal(t, "", extractReasoningEffort(json.RawMessage(`{"max_tokens":100}`)))
}
