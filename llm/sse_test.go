package llm

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
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

func TestParseStreamMalformedCompletionTokensDetailsKeepsContent(t *testing.T) {
	sse := sseFrames(
		`{"choices":[{"delta":{"content":"Hi"}}]}`,
		`{"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":2,"completion_tokens_details":[]}}`,
	)

	resp, err := parseStream(strings.NewReader(sse), nil)
	require.NoError(t, err)
	assert.Equal(t, "Hi", resp.Content)
	assert.Equal(t, 2, resp.Usage.CompletionTokens)
	assert.Equal(t, 0, resp.Usage.ReasoningTokens())
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
	// A non-string provider must neither fail the decode nor clear the one already seen.
	sse := strings.Join([]string{
		`data: {"provider":"DeepInfra","choices":[{"delta":{"content":"hi"}}]}`,
		`data: {"provider":{"name":"x"},"choices":[{"delta":{},"finish_reason":"stop"}]}`,
		`data: [DONE]`,
	}, "\n") + "\n"

	resp, err := parseStream(strings.NewReader(sse), nil)
	require.NoError(t, err)
	assert.Equal(t, "DeepInfra", resp.Provider)
}

func TestParseStreamToolCallDeltas(t *testing.T) {
	tests := []struct {
		name  string
		lines []string
		want  []Delta
	}{
		{
			name: "interleaved parallel calls keep index and id",
			lines: []string{
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a","type":"function","function":{"name":"submit_artifact","arguments":""}}]}}]}`,
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"bo"}}]}}]}`,
				`data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"b","function":{"name":"read","arguments":"{}"}}]}}]}`,
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"dy\":\"x\"}"}}]}}]}`,
				`data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
				`data: [DONE]`,
			},
			want: []Delta{
				{ToolCall: &ToolCallDelta{Index: 0, ID: "a", Name: "submit_artifact"}},
				{ToolCall: &ToolCallDelta{Index: 0, ID: "a", Name: "submit_artifact", ArgsFragment: `{"bo`}},
				{ToolCall: &ToolCallDelta{Index: 1, ID: "b", Name: "read", ArgsFragment: `{}`}},
				{ToolCall: &ToolCallDelta{Index: 0, ID: "a", Name: "submit_artifact", ArgsFragment: `dy":"x"}`}},
			},
		},
		{
			name: "name after the first fragment",
			lines: []string{
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"arguments":"{\"pa"}}]}}]}`,
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"read"}}]}}]}`,
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\":\"x\"}"}}]}}]}`,
				`data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
				`data: [DONE]`,
			},
			want: []Delta{
				{ToolCall: &ToolCallDelta{Index: 0, ID: "call_1", ArgsFragment: `{"pa`}},
				{ToolCall: &ToolCallDelta{Index: 0, ID: "call_1", Name: "read"}},
				{ToolCall: &ToolCallDelta{Index: 0, ID: "call_1", Name: "read", ArgsFragment: `th":"x"}`}},
			},
		},
		{
			name: "repeated id and name emit once",
			lines: []string{
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a","function":{"name":"read"}}]}}]}`,
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a","function":{"name":"read"}}]}}]}`,
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a","function":{"name":"read","arguments":"{}"}}]}}]}`,
				`data: [DONE]`,
			},
			want: []Delta{
				{ToolCall: &ToolCallDelta{Index: 0, ID: "a", Name: "read"}},
				{ToolCall: &ToolCallDelta{Index: 0, ID: "a", Name: "read", ArgsFragment: `{}`}},
			},
		},
		{
			name: "id alone emits before the name",
			lines: []string{
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a"}]}}]}`,
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"read","arguments":"{}"}}]}}]}`,
				`data: [DONE]`,
			},
			want: []Delta{
				{ToolCall: &ToolCallDelta{Index: 0, ID: "a"}},
				{ToolCall: &ToolCallDelta{Index: 0, ID: "a", Name: "read", ArgsFragment: `{}`}},
			},
		},
		{
			name: "content in the same chunk comes first",
			lines: []string{
				`data: {"choices":[{"delta":{"content":"Writing.","tool_calls":[{"index":0,"id":"a","function":{"name":"submit_artifact","arguments":"{"}}]}}]}`,
				`data: [DONE]`,
			},
			want: []Delta{
				{Content: "Writing."},
				{ToolCall: &ToolCallDelta{Index: 0, ID: "a", Name: "submit_artifact", ArgsFragment: "{"}},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got []Delta

			sse := strings.Join(tt.lines, "\n") + "\n"
			resp, err := parseStream(strings.NewReader(sse), func(d Delta) { got = append(got, d) })
			require.NoError(t, err)
			assert.Equal(t, tt.want, got)

			frags := map[string]string{}

			for _, d := range got {
				if d.ToolCall != nil {
					frags[d.ToolCall.ID] += d.ToolCall.ArgsFragment
				}
			}

			for _, tc := range resp.ToolCalls {
				assert.Equal(t, tc.Function.Arguments, frags[tc.ID], "fragments rebuild the arguments of %s", tc.ID)
			}
		})
	}
}

func TestParseStreamTextOnlyDeltas(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"choices":[{"delta":{"reasoning":"Let me"}}]}`,
		`data: {"choices":[{"delta":{"reasoning_content":" see."}}]}`,
		`data: {"choices":[{"delta":{"content":"Hel"}}]}`,
		`data: {"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}`,
		`data: [DONE]`,
	}, "\n") + "\n"

	var got []Delta

	_, err := parseStream(strings.NewReader(sse), func(d Delta) { got = append(got, d) })
	require.NoError(t, err)
	assert.Equal(t, []Delta{{Reasoning: "Let me"}, {Reasoning: " see."}, {Content: "Hel"}, {Content: "lo"}}, got)
}

func sseFrames(frames ...string) string {
	var b strings.Builder
	for _, f := range frames {
		b.WriteString("data: " + f + "\n")
	}

	b.WriteString("data: [DONE]\n")

	return b.String()
}

func detailsFrame(entries string) string {
	return `{"choices":[{"delta":{"reasoning_details":` + entries + `}}]}`
}

func TestParseStreamAssemblesReasoningDetails(t *testing.T) {
	tests := []struct {
		name   string
		frames []string
		want   string // "" = nil
		// wantContent, when set, is the expected Response.Content.
		wantContent string
	}{
		{
			name: "fragments of one index concatenate and the late signature lands on the same entry",
			frames: []string{
				detailsFrame(`[{"type":"reasoning.text","text":"Let ","format":"anthropic-claude-v1","index":0}]`),
				detailsFrame(`[{"type":"reasoning.text","text":"me think","format":"anthropic-claude-v1","index":0}]`),
				detailsFrame(`[{"type":"reasoning.text","text":"","signature":"sig-A","format":"anthropic-claude-v1","index":0}]`),
			},
			want: `[{"type":"reasoning.text","text":"Let me think","signature":"sig-A","format":"anthropic-claude-v1","index":0}]`,
		},
		{
			name: "indexes never merge and keep first-seen order",
			frames: []string{
				detailsFrame(`[{"type":"reasoning.summary","summary":"B1","index":1}]`),
				detailsFrame(`[{"type":"reasoning.text","text":"A","index":0}]`),
				detailsFrame(`[{"type":"reasoning.summary","summary":"B2","index":1}]`),
			},
			want: `[{"type":"reasoning.summary","summary":"B1B2","index":1},{"type":"reasoning.text","text":"A","index":0}]`,
		},
		{
			name: "an entry without an index stays its own entry",
			frames: []string{
				`{"choices":[{"delta":{"reasoning":"thinking"}}]}`,
				detailsFrame(`[{"type":"reasoning.text","format":"anthropic-claude-v1","signature":"sig-B"}]`),
				detailsFrame(`[{"type":"reasoning.text","format":"anthropic-claude-v1","signature":"sig-C"}]`),
			},
			want: `[{"type":"reasoning.text","format":"anthropic-claude-v1","signature":"sig-B"},{"type":"reasoning.text","format":"anthropic-claude-v1","signature":"sig-C"}]`,
		},
		{
			name:   "empty text with a signature survives",
			frames: []string{detailsFrame(`[{"type":"reasoning.text","text":"","signature":"sig-A","index":0}]`)},
			want:   `[{"type":"reasoning.text","text":"","signature":"sig-A","index":0}]`,
		},
		{
			name: "encrypted data concatenates and unknown fields are kept",
			frames: []string{
				detailsFrame(`[{"type":"reasoning.encrypted","data":"QUJD","format":"google-gemini-v1","index":0,"x_extra":{"k":1}}]`),
				detailsFrame(`[{"type":"reasoning.encrypted","data":"REVG","index":0}]`),
			},
			want: `[{"type":"reasoning.encrypted","data":"QUJDREVG","format":"google-gemini-v1","index":0,"x_extra":{"k":1}}]`,
		},
		{
			name:   "a null fragment is skipped and a non-object one kept verbatim",
			frames: []string{detailsFrame(`[null,"opaque",{"type":"reasoning.text","text":"A","index":0}]`)},
			want:   `["opaque",{"type":"reasoning.text","text":"A","index":0}]`,
		},
		{
			name:   "a non-array reasoning_details is ignored and the rest of the chunk survives",
			frames: []string{`{"choices":[{"delta":{"content":"hi","reasoning_details":{"type":"reasoning.text","text":"x","index":0}}}]}`},
			want:   "",

			wantContent: "hi",
		},
		{
			name: "null and non-integer indexes make their own entries",
			frames: []string{
				detailsFrame(`[{"signature":"sig-B","index":null},{"signature":"sig-C","index":null},{"signature":"sig-D","index":"1"},{"signature":"sig-E","index":1.5}]`),
			},
			want: `[{"signature":"sig-B","index":null},{"signature":"sig-C","index":null},{"signature":"sig-D","index":"1"},{"signature":"sig-E","index":1.5}]`,
		},
		{
			name: "an integer fragment and a null-index fragment do not merge",
			frames: []string{
				detailsFrame(`[{"text":"A","index":0}]`),
				detailsFrame(`[{"text":"B","index":null}]`),
			},
			want: `[{"text":"A","index":0},{"text":"B","index":null}]`,
		},
		{
			name: "null on a concat key is skipped, not written as an empty string",
			frames: []string{
				detailsFrame(`[{"data":"X","index":0}]`),
				detailsFrame(`[{"data":"Y","text":null,"summary":null,"index":0}]`),
			},
			want: `[{"data":"XY","index":0}]`,
		},
		{
			name: "a non-string value on a concat key overwrites",
			frames: []string{
				detailsFrame(`[{"data":"X","index":0}]`),
				detailsFrame(`[{"data":{"k":1},"index":0}]`),
			},
			want: `[{"data":{"k":1},"index":0}]`,
		},
		{
			name: "a string after a non-string concat value replaces it",
			frames: []string{
				detailsFrame(`[{"data":"X","index":0}]`),
				detailsFrame(`[{"data":{"k":1},"index":0}]`),
				detailsFrame(`[{"data":"Z","index":0}]`),
				detailsFrame(`[{"data":"W","index":0}]`),
			},
			want: `[{"data":"ZW","index":0}]`,
		},
		{
			name:   "no reasoning_details leaves the response without them",
			frames: []string{`{"choices":[{"delta":{"content":"hi","reasoning_details":null}}]}`},
			want:   "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := parseStream(strings.NewReader(sseFrames(tt.frames...)), nil)
			require.NoError(t, err)

			if tt.wantContent != "" {
				assert.Equal(t, tt.wantContent, resp.Content)
			}

			if tt.want == "" {
				assert.Nil(t, resp.ReasoningDetails)

				return
			}

			assert.JSONEq(t, tt.want, string(resp.ReasoningDetails))
		})
	}
}

func TestParseStreamReasoningDetailsBesideToolCallsSendNoDelta(t *testing.T) {
	sse := sseFrames(
		`{"choices":[{"delta":{"reasoning":"r","reasoning_details":[{"type":"reasoning.text","text":"r","index":0}]}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"read","arguments":"{}"}}],"reasoning_details":[{"type":"reasoning.text","signature":"s","index":0}]}}]}`,
		`{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
	)

	var deltas []Delta

	resp, err := parseStream(strings.NewReader(sse), func(d Delta) { deltas = append(deltas, d) })
	require.NoError(t, err)
	assert.JSONEq(t, `[{"type":"reasoning.text","text":"r","signature":"s","index":0}]`, string(resp.ReasoningDetails))
	require.Len(t, resp.ToolCalls, 1)
	assert.Equal(t, []Delta{
		{Reasoning: "r"},
		{ToolCall: &ToolCallDelta{Index: 0, ID: "c1", Name: "read", ArgsFragment: "{}"}},
	}, deltas)
}

// allocatedBy returns the bytes allocated while n fragments are added and assembled.
func allocatedBy(t *testing.T, n int) uint64 {
	t.Helper()

	var before, after runtime.MemStats

	runtime.GC()
	runtime.ReadMemStats(&before)

	var a detailsAccumulator

	for range n {
		a.add(json.RawMessage(`{"type":"reasoning.text","text":"0123456789","index":0}`))
	}

	out, err := a.result()
	require.NoError(t, err)
	require.Contains(t, string(out), strings.Repeat("0123456789", n))

	runtime.ReadMemStats(&after)

	return after.TotalAlloc - before.TotalAlloc
}

// Quadruple the fragments: linear allocation grows about 4x, quadratic about 16x.
func TestDetailsAccumulatorLinearInFragments(t *testing.T) {
	small := allocatedBy(t, 5000)
	large := allocatedBy(t, 20000)

	assert.Less(t, large, small*8, "allocation grew faster than linearly: %d -> %d bytes", small, large)
}

// The goldens are the parser's own output that the provider accepted on
// replay; this pins that accepted form against the captured streams.
func TestParseStreamCapturedReasoningFixtures(t *testing.T) {
	paths, err := filepath.Glob("testdata/reasoning/*.sse")
	require.NoError(t, err)
	require.NotEmpty(t, paths)

	for _, p := range paths {
		t.Run(filepath.Base(p), func(t *testing.T) {
			f, err := os.Open(p)
			require.NoError(t, err)

			defer f.Close() //nolint:errcheck

			resp, err := parseStream(f, nil)
			require.NoError(t, err)
			require.NotEmpty(t, resp.ToolCalls)

			want, err := os.ReadFile(strings.TrimSuffix(p, ".sse") + ".details.json")
			require.NoError(t, err)
			assert.JSONEq(t, string(want), string(resp.ReasoningDetails))
		})
	}
}
