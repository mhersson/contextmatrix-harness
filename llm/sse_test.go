package llm

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseStreamContentAndUsage(t *testing.T) {
	sse := strings.Join([]string{
		`: OPENROUTER PROCESSING`,
		`data: {"model":"m","choices":[{"delta":{"content":"Hel"}}]}`,
		`data: {"choices":[{"delta":{"content":"lo"}}]}`,
		`data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":2,"cost":0.0003}}`,
		`data: [DONE]`,
	}, "\n") + "\n"

	var streamed strings.Builder

	resp, err := parseStream(strings.NewReader(sse), func(d Delta) { streamed.WriteString(d.Content) })
	require.NoError(t, err)
	assert.Equal(t, "Hello", resp.Content)
	assert.Equal(t, "Hello", streamed.String())
	assert.Equal(t, "stop", resp.FinishReason)
	assert.Equal(t, "m", resp.Model)
	assert.InEpsilon(t, 0.0003, resp.Usage.Cost, 1e-9)
}

func TestParseStreamToolCallByIndexWithLateName(t *testing.T) {
	// Name arrives in a LATER delta than id/arguments; args split across frames.
	sse := strings.Join([]string{
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"arguments":"{\"pa"}}]}}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"read"}}]}}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\":\"x\"}"}}]}}]}`,
		`data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
		`data: [DONE]`,
	}, "\n") + "\n"

	resp, err := parseStream(strings.NewReader(sse), nil)
	require.NoError(t, err)
	require.Len(t, resp.ToolCalls, 1)
	assert.Equal(t, "call_1", resp.ToolCalls[0].ID)
	assert.Equal(t, "read", resp.ToolCalls[0].Function.Name)
	assert.JSONEq(t, `{"path":"x"}`, resp.ToolCalls[0].Function.Arguments)
	assert.Equal(t, "function", resp.ToolCalls[0].Type)
}

func TestParseStreamParallelToolCalls(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a","function":{"name":"read","arguments":"{}"}}]}}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"b","function":{"name":"grep","arguments":"{}"}}]}}]}`,
		`data: [DONE]`,
	}, "\n") + "\n"

	resp, err := parseStream(strings.NewReader(sse), nil)
	require.NoError(t, err)
	require.Len(t, resp.ToolCalls, 2)
	assert.Equal(t, "read", resp.ToolCalls[0].Function.Name)
	assert.Equal(t, "grep", resp.ToolCalls[1].Function.Name)
}

func TestParseStreamMidStreamError(t *testing.T) {
	sse := `data: {"error":{"code":"server_error","message":"boom"},"choices":[{"delta":{},"finish_reason":"error"}]}` + "\n"
	_, err := parseStream(strings.NewReader(sse), nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "boom")
}

func TestParseStreamRejectsOverlongLineClearly(t *testing.T) {
	// A data line longer than the (tiny, test-injected) cap must yield a clear error.
	long := `data: {"choices":[{"delta":{"content":"` + strings.Repeat("x", 256) + `"}}]}` + "\n"
	_, err := parseStreamWithLimit(strings.NewReader(long), nil, 64)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "exceeds")
}

// TestParseStreamBuilderAccumulation asserts that Content, Reasoning, and
// Function.Arguments are fully assembled from multi-frame SSE deltas.
func TestParseStreamBuilderAccumulation(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"choices":[{"delta":{"reasoning":"Let ","content":"Hel","tool_calls":[{"index":0,"id":"c1","function":{"name":"fn","arguments":"{\"a"}}]}}]}`,
		`data: {"choices":[{"delta":{"reasoning":"me.","content":"lo","tool_calls":[{"index":0,"function":{"arguments":"\":1}"}}]}}]}`,
		`data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
		`data: [DONE]`,
	}, "\n") + "\n"

	resp, err := parseStream(strings.NewReader(sse), nil)
	require.NoError(t, err)
	assert.Equal(t, "Hello", resp.Content)
	assert.Equal(t, "Let me.", resp.Reasoning)
	require.Len(t, resp.ToolCalls, 1)
	assert.Equal(t, "c1", resp.ToolCalls[0].ID)
	assert.Equal(t, "fn", resp.ToolCalls[0].Function.Name)
	assert.JSONEq(t, `{"a":1}`, resp.ToolCalls[0].Function.Arguments)
}

func TestParseStreamReasoningContent(t *testing.T) {
	tests := []struct {
		name string
		sse  string
		want string
	}{
		{
			"reasoning_content only",
			`data: {"choices":[{"delta":{"reasoning_content":"Let "}}]}` + "\n" +
				`data: {"choices":[{"delta":{"reasoning_content":"me."}}]}` + "\n" +
				`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}` + "\n",
			"Let me.",
		},
		{
			"both fields: reasoning wins, counted once per frame",
			`data: {"choices":[{"delta":{"reasoning":"a","reasoning_content":"X"}}]}` + "\n" +
				`data: {"choices":[{"delta":{"reasoning":"b","reasoning_content":"Y"}}]}` + "\n" +
				`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}` + "\n",
			"ab",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var streamed strings.Builder

			resp, err := parseStream(strings.NewReader(tt.sse), func(d Delta) {
				streamed.WriteString(d.Reasoning)
			})
			require.NoError(t, err)
			assert.Equal(t, tt.want, resp.Reasoning)
			assert.Equal(t, tt.want, streamed.String())
		})
	}
}

func TestParseStreamTruncationDetection(t *testing.T) {
	chunk := `data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}`
	noFinish := `data: {"choices":[{"delta":{"content":"hi"}}]}`

	tests := []struct {
		name    string
		body    string
		wantErr bool
	}{
		{"done marker, no finish_reason", noFinish + "\n\ndata: [DONE]\n\n", false},
		{"finish_reason, no done marker", chunk + "\n\n", false},
		{"both", chunk + "\n\ndata: [DONE]\n\n", false},
		{"neither: truncated", noFinish + "\n\n", true},
		{"empty stream", "", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := parseStream(strings.NewReader(tt.body), nil)
			if tt.wantErr {
				require.ErrorContains(t, err, "truncated")
			} else {
				require.NoError(t, err)
			}
		})
	}
}

func TestParseStreamProvider(t *testing.T) {
	tests := []struct {
		name         string
		provider     string
		wantProvider string
	}{
		{"string", `"provider":"DeepInfra",`, "DeepInfra"},
		{"absent", ``, ""},
		{"object", `"provider":{"name":"x"},`, ""},
		{"null", `"provider":null,`, ""},
		{"number", `"provider":3,`, ""},
		{"empty string", `"provider":"",`, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sse := strings.Join([]string{
				`data: {` + tt.provider + `"model":"m","choices":[{"delta":{"content":"hi"}}]}`,
				`data: {` + tt.provider + `"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a","function":{"name":"read","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`,
				`data: [DONE]`,
			}, "\n") + "\n"

			resp, err := parseStream(strings.NewReader(sse), nil)
			require.NoError(t, err)
			assert.Equal(t, tt.wantProvider, resp.Provider)
			assert.Equal(t, "hi", resp.Content)
			require.Len(t, resp.ToolCalls, 1)
		})
	}
}

func TestParseStreamProviderLastNonEmptyWins(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"provider":"A","choices":[{"delta":{"content":"x"}}]}`,
		`data: {"provider":"B","choices":[{"delta":{}}]}`,
		`data: {"provider":null,"choices":[{"delta":{},"finish_reason":"stop"}]}`,
		`data: [DONE]`,
	}, "\n") + "\n"

	resp, err := parseStream(strings.NewReader(sse), nil)
	require.NoError(t, err)
	assert.Equal(t, "B", resp.Provider)
}
