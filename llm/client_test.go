package llm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestClientSendStream(t *testing.T) {
	var gotAuth, gotBody string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		b, _ := io.ReadAll(r.Body)
		gotBody = string(b)

		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}],\"usage\":{\"cost\":0.0002}}\ndata: [DONE]\n") //nolint:errcheck
	}))
	defer srv.Close()

	c := NewClient("secret-key", WithBaseURL(srv.URL))
	resp, err := c.SendStream(context.Background(), Request{
		Model:    "m",
		Messages: []Message{{Role: "user", Content: "x"}},
		Tools:    []Tool{{Type: "function", Function: ToolFunction{Name: "read"}}},
	}, nil)
	require.NoError(t, err)
	assert.Equal(t, "hi", resp.Content)
	assert.InEpsilon(t, 0.0002, resp.Usage.Cost, 1e-9)
	assert.Equal(t, "Bearer secret-key", gotAuth)
	assert.Contains(t, gotBody, `"stream":true`)
	assert.Contains(t, gotBody, `"model":"m"`)
	assert.Contains(t, gotBody, `"name":"read"`)
	assert.Contains(t, gotBody, `"usage":{"include":true}`)
}

func TestClientSendNonStream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"model":"m","usage":{"cost":0.001},"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"done"}}]}`) //nolint:errcheck
	}))
	defer srv.Close()

	c := NewClient("k", WithBaseURL(srv.URL))
	resp, err := c.Send(context.Background(), Request{Messages: []Message{{Role: "user", Content: "x"}}})
	require.NoError(t, err)
	assert.Equal(t, "done", resp.Content)
	assert.Equal(t, "stop", resp.FinishReason)
}

func TestClientSendStreamOpenAIDialect(t *testing.T) {
	var gotBody []byte

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotBody, _ = io.ReadAll(r.Body)

		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\ndata: [DONE]\n") //nolint:errcheck
	}))
	defer srv.Close()

	c := NewClient("k", WithBaseURL(srv.URL), WithDialect(DialectOpenAI))
	_, err := c.SendStream(context.Background(), Request{
		Model:     "m",
		Provider:  json.RawMessage(`{"sort":"price"}`),
		Models:    []string{"a", "b"},
		Messages:  []Message{{Role: "user", Content: "hi"}},
		Reasoning: json.RawMessage(`{"effort":"high"}`),
	}, nil)
	require.NoError(t, err)

	m := keys(t, gotBody)
	for _, k := range []string{"provider", "models", "usage"} {
		_, ok := m[k]
		assert.False(t, ok, "openai dialect must omit %q from the wire", k)
	}

	assert.JSONEq(t, `"high"`, string(m["reasoning_effort"]), "reasoning_effort must be wired through")
	assert.JSONEq(t, `{"include_usage":true}`, string(m["stream_options"]), "stream_options must be present for streamed openai calls")
}

func TestClientNon200(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		io.WriteString(w, `{"error":{"message":"rate limited"}}`) //nolint:errcheck
	}))
	defer srv.Close()

	c := NewClient("k", WithBaseURL(srv.URL))
	_, err := c.SendStream(context.Background(), Request{Messages: []Message{{Role: "user"}}}, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "429")
	assert.Contains(t, err.Error(), "rate limited")
}

// TestSendCapsOversizeBody verifies that Send returns a "too large" error when a
// server returns a body that exceeds maxResponseBody. Approach (b) was chosen
// (explicit error, directly assertable) over silent truncation.
func TestSendCapsOversizeBody(t *testing.T) {
	oversized := make([]byte, maxResponseBody+1)

	t.Run("oversize body returns error", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
			w.Write(oversized) //nolint:errcheck
		}))
		defer srv.Close()

		c := NewClient("k", WithBaseURL(srv.URL))
		_, err := c.Send(context.Background(), Request{Messages: []Message{{Role: "user"}}})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "exceeds")
	})

	t.Run("normal small body still works", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			io.WriteString(w, `{"model":"m","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"ok"}}]}`) //nolint:errcheck
		}))
		defer srv.Close()

		c := NewClient("k", WithBaseURL(srv.URL))
		resp, err := c.Send(context.Background(), Request{Messages: []Message{{Role: "user"}}})
		require.NoError(t, err)
		assert.Equal(t, "ok", resp.Content)
	})
}
