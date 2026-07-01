package llm

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseCatalog(t *testing.T) {
	raw := `{"data":[
		{"id":"vendor/weak","context_length":8192,"pricing":{"prompt":"0.0000002","completion":"0.0000006"},"supported_parameters":["tools","tool_choice"]},
		{"id":"vendor/notools","context_length":4096,"pricing":{"prompt":"0.0000001","completion":"0.0000001"},"supported_parameters":["temperature"]}
	]}`
	cat, err := ParseCatalog(strings.NewReader(raw))
	require.NoError(t, err)
	require.Len(t, cat, 2)

	weak, ok := cat.Find("vendor/weak")
	require.True(t, ok)
	assert.True(t, weak.SupportsTools())
	assert.Equal(t, 8192, weak.ContextLength)
	assert.InEpsilon(t, 0.0000002, weak.PromptPricePerTok, 1e-12)

	nt, ok := cat.Find("vendor/notools")
	require.True(t, ok)
	assert.False(t, nt.SupportsTools())
}

func TestParseCatalogOpenAI(t *testing.T) {
	body := `{"data":[
		{"id":"model-a","context_length":200000,
		 "pricing":{"prompt":"0.000003","completion":"0.000015"},
		 "capabilities":{"features":["streaming","tools","json_mode"]}},
		{"id":"model-b","context_length":128000,
		 "pricing":{"prompt":"0.0000007","completion":"0.000003"},
		 "capabilities":{"features":["streaming"]}}
	]}`

	cat, err := parseCatalogOpenAI(strings.NewReader(body))
	require.NoError(t, err)
	require.Len(t, cat, 2)

	a, ok := cat.Find("model-a")
	require.True(t, ok)
	assert.True(t, a.SupportsTools())
	assert.Equal(t, 200000, a.ContextLength)
	assert.InDelta(t, 0.000003, a.PromptPricePerTok, 1e-12)
	assert.InDelta(t, 0.000015, a.CompletionPricePerTok, 1e-12)

	b, ok := cat.Find("model-b")
	require.True(t, ok)
	assert.False(t, b.SupportsTools())
}

// TestFetchCatalogCapsOversizeErrorBody verifies that FetchCatalog returns a
// "too large" error when a non-200 error response body exceeds maxResponseBody.
func TestFetchCatalogCapsOversizeErrorBody(t *testing.T) {
	oversized := make([]byte, maxResponseBody+1)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		w.Write(oversized) //nolint:errcheck
	}))
	defer srv.Close()

	c := NewClient("k", WithBaseURL(srv.URL))
	_, err := c.FetchCatalog(context.Background())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "exceeds")
}
