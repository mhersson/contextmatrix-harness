package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

const defaultBaseURL = "https://openrouter.ai/api/v1"

// maxResponseBody caps non-stream and error response bodies. 8 MiB is ample
// for any JSON payload or error page; an unbounded read is a memory-exhaustion
// vector from a misconfigured or hostile OpenAI-compatible endpoint.
const maxResponseBody = 8 << 20

type Client struct {
	http    *http.Client
	baseURL string
	apiKey  string
	dialect Dialect
	retry   RetryPolicy
	sleep   func(context.Context, time.Duration) error
}

type Option func(*Client)

func WithBaseURL(u string) Option          { return func(c *Client) { c.baseURL = u } }
func WithDialect(d Dialect) Option         { return func(c *Client) { c.dialect = d } }
func WithHTTPClient(h *http.Client) Option { return func(c *Client) { c.http = h } }

func NewClient(apiKey string, opts ...Option) *Client {
	c := &Client{http: &http.Client{}, baseURL: defaultBaseURL, apiKey: apiKey, sleep: ctxSleep}
	for _, o := range opts {
		o(c)
	}

	return c
}

func WithRetry(p RetryPolicy) Option { return func(c *Client) { c.retry = p } }

func (c *Client) SendStream(ctx context.Context, req Request, onDelta func(Delta)) (Response, error) {
	req.Stream = true
	req.Usage = &UsageOpt{Include: true}

	hr, err := c.doWithRetry(ctx, req)
	if err != nil {
		return Response{}, err
	}
	defer hr.Body.Close() //nolint:errcheck

	if hr.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(hr.Body, maxResponseBody+1))
		if len(body) > maxResponseBody {
			return Response{}, fmt.Errorf("response body exceeds %d bytes", maxResponseBody)
		}

		return Response{}, fmt.Errorf("llm endpoint status %d: %s", hr.StatusCode, string(body))
	}

	return parseStream(hr.Body, onDelta)
}

func (c *Client) Send(ctx context.Context, req Request) (Response, error) {
	req.Stream = false
	req.Usage = &UsageOpt{Include: true}

	hr, err := c.doWithRetry(ctx, req)
	if err != nil {
		return Response{}, err
	}
	defer hr.Body.Close() //nolint:errcheck

	body, _ := io.ReadAll(io.LimitReader(hr.Body, maxResponseBody+1))
	if len(body) > maxResponseBody {
		return Response{}, fmt.Errorf("response body exceeds %d bytes", maxResponseBody)
	}

	if hr.StatusCode != http.StatusOK {
		return Response{}, fmt.Errorf("llm endpoint status %d: %s", hr.StatusCode, string(body))
	}

	var nr nonStreamResponse
	if err := json.Unmarshal(body, &nr); err != nil {
		return Response{}, fmt.Errorf("decode response: %w", err)
	}

	return normalizeHarmony(nr.toResponse()), nil
}

// doWithRetry issues the request, retrying transport errors and retryable
// statuses BEFORE any body is consumed. The returned response (200, a
// non-retryable status, or the final attempt) has an intact body.
func (c *Client) doWithRetry(ctx context.Context, req Request) (*http.Response, error) {
	attempts := c.retry.MaxRetries + 1

	var (
		lastResp *http.Response
		lastErr  error
	)

	for attempt := 0; attempt < attempts; attempt++ {
		if attempt > 0 {
			if err := c.sleep(ctx, c.backoff(attempt, lastResp)); err != nil {
				return nil, err
			}
		}

		resp, err := c.do(ctx, req)
		if err != nil {
			lastErr, lastResp = err, nil

			continue
		}

		if attempt < attempts-1 && isRetryableStatus(resp.StatusCode) {
			lastResp = resp
			_, _ = io.Copy(io.Discard, resp.Body) //nolint:errcheck // drain to reuse the connection
			_ = resp.Body.Close()                 //nolint:errcheck

			continue
		}

		return resp, nil
	}

	return nil, lastErr
}

func (c *Client) do(ctx context.Context, req Request) (*http.Response, error) {
	b, err := encodeRequest(req, c.dialect)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	hr, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/chat/completions", bytes.NewReader(b))
	if err != nil {
		return nil, err
	}

	hr.Header.Set("Authorization", "Bearer "+c.apiKey)
	hr.Header.Set("Content-Type", "application/json")

	return c.http.Do(hr)
}

type nonStreamResponse struct {
	Model   string `json:"model"`
	Usage   Usage  `json:"usage"`
	Choices []struct {
		FinishReason string `json:"finish_reason"`
		Message      struct {
			Message
			Reasoning        string `json:"reasoning"`
			ReasoningContent string `json:"reasoning_content"`
		} `json:"message"`
	} `json:"choices"`
}

func (n nonStreamResponse) toResponse() Response {
	r := Response{Model: n.Model, Usage: n.Usage}
	if len(n.Choices) > 0 {
		r.Content = n.Choices[0].Message.Content
		r.ToolCalls = n.Choices[0].Message.ToolCalls
		r.FinishReason = n.Choices[0].FinishReason
		r.Reasoning = n.Choices[0].Message.Reasoning
		if r.Reasoning == "" {
			r.Reasoning = n.Choices[0].Message.ReasoningContent
		}
	}

	return r
}

// Compile-time check that Client satisfies the interface.
var _ LLM = (*Client)(nil)
