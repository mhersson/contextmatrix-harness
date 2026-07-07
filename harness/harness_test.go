package harness

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/mhersson/contextmatrix-harness/events"
	"github.com/mhersson/contextmatrix-harness/llm"
	"github.com/mhersson/contextmatrix-harness/redact"
	"github.com/mhersson/contextmatrix-harness/tools"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// fakeLLM returns scripted responses in order; after they run out it returns an
// empty response (no tool calls → loop treats it as "done").
type fakeLLM struct {
	responses []llm.Response
	i         int
}

func (f *fakeLLM) Send(ctx context.Context, req llm.Request) (llm.Response, error) {
	return f.next(), nil
}

func (f *fakeLLM) SendStream(ctx context.Context, req llm.Request, onDelta func(llm.Delta)) (llm.Response, error) {
	return f.next(), nil
}

func (f *fakeLLM) next() llm.Response {
	if f.i >= len(f.responses) {
		return llm.Response{FinishReason: "stop"}
	}

	r := f.responses[f.i]
	f.i++

	return r
}

func newEmitter() *events.Emitter { return events.NewEmitter(nil, nil) }

func toolCall(id, name, args string) llm.ToolCall {
	return llm.ToolCall{ID: id, Type: "function", Function: llm.FunctionCall{Name: name, Arguments: args}}
}

func TestRunExecutesToolThenStops(t *testing.T) {
	root := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(root, "f.txt"), []byte("data"), 0o644))
	reg := tools.NewRegistry(tools.NewReadTool(root))

	f := &fakeLLM{responses: []llm.Response{
		{ToolCalls: []llm.ToolCall{toolCall("1", "read", `{"path":"f.txt"}`)}, Usage: llm.Usage{Cost: 0.001}},
		{Content: "all done", FinishReason: "stop", Usage: llm.Usage{Cost: 0.001}},
	}}
	res, err := Run(context.Background(), f, reg, newEmitter(), "do it", Config{MaxTurns: 10})
	require.NoError(t, err)
	assert.True(t, res.Completed)
	assert.Equal(t, "done", res.Reason)
	assert.Equal(t, 1, res.ToolCallCount)
	assert.Equal(t, 0, res.RepairCount)
	assert.InEpsilon(t, 0.002, res.TotalCostUSD, 1e-9)
}

func TestRunRepairsMalformedArgs(t *testing.T) {
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()))
	f := &fakeLLM{responses: []llm.Response{
		{ToolCalls: []llm.ToolCall{toolCall("1", "read", `{"path":`)}}, // malformed
		{Content: "ok", FinishReason: "stop"},
	}}
	res, err := Run(context.Background(), f, reg, newEmitter(), "task", Config{MaxTurns: 10})
	require.NoError(t, err)
	assert.Equal(t, 1, res.RepairCount)
	assert.Equal(t, 1, res.ToolCallCount)
}

func TestRunToolCallsBeatLyingFinishReason(t *testing.T) {
	root := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(root, "f.txt"), []byte("x"), 0o644))
	reg := tools.NewRegistry(tools.NewReadTool(root))
	f := &fakeLLM{responses: []llm.Response{
		// finish_reason "stop" but tool_calls present — must still execute.
		{FinishReason: "stop", ToolCalls: []llm.ToolCall{toolCall("1", "read", `{"path":"f.txt"}`)}},
		{Content: "fin", FinishReason: "stop"},
	}}
	res, err := Run(context.Background(), f, reg, newEmitter(), "task", Config{MaxTurns: 10})
	require.NoError(t, err)
	assert.Equal(t, 1, res.ToolCallCount)
	assert.True(t, res.Completed)
}

func TestRunMaxTurns(t *testing.T) {
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()))
	// Always asks for an (unknown-path) read → never stops on its own.
	loop := []llm.Response{{ToolCalls: []llm.ToolCall{toolCall("1", "read", `{"path":"missing"}`)}}}
	f := &fakeLLM{responses: append(append(append([]llm.Response{}, loop...), loop...), loop...)}
	res, err := Run(context.Background(), f, reg, newEmitter(), "task", Config{MaxTurns: 3})
	require.NoError(t, err)
	assert.False(t, res.Completed)
	assert.Equal(t, "max_turns", res.Reason)
	assert.Equal(t, 3, res.Turns)
}

func TestToolResultMsgNeverEmptyContent(t *testing.T) {
	m := toolResultMsg("call_1", "")
	assert.Equal(t, "tool", m.Role)
	assert.Equal(t, "call_1", m.ToolCallID)
	assert.NotEmpty(t, m.Content) // empty tool output must not drop the wire `content` field

	m2 := toolResultMsg("call_2", "hello")
	assert.Equal(t, "hello", m2.Content)
}

func TestRunMaxCost(t *testing.T) {
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()))
	f := &fakeLLM{responses: []llm.Response{
		{Content: "thinking", Usage: llm.Usage{Cost: 0.6}, ToolCalls: []llm.ToolCall{toolCall("1", "read", `{"path":"x"}`)}},
		{Content: "more", Usage: llm.Usage{Cost: 0.6}, ToolCalls: []llm.ToolCall{toolCall("2", "read", `{"path":"x"}`)}},
	}}
	res, err := Run(context.Background(), f, reg, newEmitter(), "task", Config{MaxTurns: 10, MaxCostUSD: 0.5})
	require.NoError(t, err)
	assert.Equal(t, "max_cost", res.Reason)
	assert.False(t, res.Completed)
}

type capturingLLM struct{ last llm.Request }

func (c *capturingLLM) Send(ctx context.Context, req llm.Request) (llm.Response, error) {
	c.last = req

	return llm.Response{FinishReason: "stop"}, nil
}

func (c *capturingLLM) SendStream(ctx context.Context, req llm.Request, onDelta func(llm.Delta)) (llm.Response, error) {
	c.last = req

	return llm.Response{FinishReason: "stop"}, nil
}

func TestRunForwardsProviderAndReasoning(t *testing.T) {
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()))
	capt := &capturingLLM{}
	_, err := Run(context.Background(), capt, reg, newEmitter(), "task", Config{
		MaxTurns:  1,
		Models:    []string{"primary/m", "fallback/m"},
		Provider:  json.RawMessage(`{"sort":"price"}`),
		Reasoning: json.RawMessage(`{"effort":"high"}`),
	})
	require.NoError(t, err)
	assert.Equal(t, []string{"primary/m", "fallback/m"}, capt.last.Models) // models[] failover forwarded
	assert.JSONEq(t, `{"sort":"price"}`, string(capt.last.Provider))
	assert.JSONEq(t, `{"effort":"high"}`, string(capt.last.Reasoning))
}

func TestRunMaxTurnsZeroUsesDefault(t *testing.T) {
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()))
	// Always asks for a (missing-path) read → never stops on its own.
	resp := llm.Response{ToolCalls: []llm.ToolCall{toolCall("1", "read", `{"path":"missing"}`)}}

	many := make([]llm.Response, 0, defaultMaxTurns)
	for i := 0; i < defaultMaxTurns; i++ {
		many = append(many, resp)
	}

	f := &fakeLLM{responses: many}

	// MaxTurns 0 must NOT mean "run zero turns and silently complete".
	res, err := Run(context.Background(), f, reg, newEmitter(), "task", Config{MaxTurns: 0})
	require.NoError(t, err)
	assert.False(t, res.Completed)
	assert.Equal(t, "max_turns", res.Reason)
	assert.Equal(t, defaultMaxTurns, res.Turns)
}

func TestRunContextLimitReturnsIncomplete(t *testing.T) {
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()))
	f := &fakeLLM{responses: []llm.Response{
		{
			Content: "thinking", Usage: llm.Usage{PromptTokens: 900},
			ToolCalls: []llm.ToolCall{toolCall("1", "read", `{"path":"x"}`)},
		},
	}}
	res, err := Run(context.Background(), f, reg, newEmitter(), "task", Config{MaxTurns: 10, ContextWindow: 1000})
	require.NoError(t, err)
	assert.False(t, res.Completed)
	assert.Equal(t, "context_limit", res.Reason)
	assert.Equal(t, 1, res.Turns)
}

func TestRunContextLimitDisabledWhenWindowZero(t *testing.T) {
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()))
	f := &fakeLLM{responses: []llm.Response{
		{Content: "done", FinishReason: "stop", Usage: llm.Usage{PromptTokens: 999999}},
	}}
	res, err := Run(context.Background(), f, reg, newEmitter(), "task", Config{MaxTurns: 10}) // window 0 = disabled
	require.NoError(t, err)
	assert.True(t, res.Completed)
	assert.Equal(t, "done", res.Reason)
}

func TestRun_CompactionNoProgressFallsThrough(t *testing.T) {
	// A single assistant+3-tool group as history: snapping pulls the whole group
	// into the kept tail, older collapses to [user "old"], and out == in (no shrink).
	tcs := []llm.ToolCall{
		{ID: "c1", Type: "function", Function: llm.FunctionCall{Name: "foo", Arguments: "{}"}},
		{ID: "c2", Type: "function", Function: llm.FunctionCall{Name: "foo", Arguments: "{}"}},
		{ID: "c3", Type: "function", Function: llm.FunctionCall{Name: "foo", Arguments: "{}"}},
	}
	history := []llm.Message{
		{Role: "user", Content: "old"},
		{Role: "assistant", ToolCalls: tcs},
		{Role: "tool", ToolCallID: "c1", Content: "r1"},
		{Role: "tool", ToolCallID: "c2", Content: "r2"},
		{Role: "tool", ToolCallID: "c3", Content: "r3"},
	}
	fake := &capturingLLMSeq{responses: []llm.Response{
		{Content: "turn1", Usage: llm.Usage{PromptTokens: 900}}, // >850 → triggers compaction
		{Content: "SUMMARY"}, // compact Send
	}}

	var transcript bytes.Buffer

	emit := events.NewEmitter(nil, &transcript)

	res, err := Run(context.Background(), fake, tools.NewRegistry(), emit, "go", Config{
		MaxTurns:      10,
		SystemPrompt:  "SYS",
		ContextWindow: 1000,
		Compaction:    &Compaction{Threshold: 0.85, KeepRecentTurns: 4},
		History:       history,
	})
	require.NoError(t, err)
	// A no-op compaction must fall through to the hard stop, not loop or "complete".
	assert.Equal(t, "context_limit", res.Reason)
	require.Len(t, fake.requests, 2) // turn-1 SendStream + one compact Send; no turn-2

	sawFailed := false

	for _, ev := range parseEvents(t, transcript.String()) {
		if ev.Kind == events.StateChange && ev.Data["event"] == "compaction_failed" {
			sawFailed = true
		}
	}

	assert.True(t, sawFailed, "no-progress compaction must emit compaction_failed")
}

func TestRun_CompactionCostIsCounted(t *testing.T) {
	history := make([]llm.Message, 20)
	for i := range history {
		if i%2 == 0 {
			history[i] = llm.Message{Role: "user", Content: fmt.Sprintf("user %d", i)}
		} else {
			history[i] = llm.Message{Role: "assistant", Content: fmt.Sprintf("assistant %d", i)}
		}
	}

	fake := &capturingLLMSeq{responses: []llm.Response{
		{Content: "turn1", Usage: llm.Usage{PromptTokens: 900, Cost: 0.01}},                         // triggers compaction
		{Content: "SUMMARY", Usage: llm.Usage{PromptTokens: 500, CompletionTokens: 20, Cost: 0.10}}, // compact Send
		{Content: "done", FinishReason: "stop", Usage: llm.Usage{Cost: 0.02}},
	}}
	res, err := Run(context.Background(), fake, tools.NewRegistry(), newEmitter(), "go", Config{
		MaxTurns:      10,
		SystemPrompt:  "SYS",
		ContextWindow: 1000,
		Compaction:    &Compaction{Threshold: 0.85, KeepRecentTurns: 2},
		History:       history,
	})
	require.NoError(t, err)
	assert.InEpsilon(t, 0.13, res.TotalCostUSD, 1e-9) // 0.01 + 0.10 + 0.02
	assert.EqualValues(t, 1400, res.PromptTokens)     // 900 + 500
	assert.EqualValues(t, 20, res.CompletionTokens)
}

func TestRun_CompactionThresholdAbove85DoesNotHardStopEarly(t *testing.T) {
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()))
	f := &fakeLLM{responses: []llm.Response{
		{Content: "ok", FinishReason: "stop", Usage: llm.Usage{PromptTokens: 175000}},
	}}
	// window 200000, Threshold 0.95 → effective compaction 190000; old hard stop 170000.
	// 175000 is in the dead band: must NOT hard-stop.
	res, err := Run(context.Background(), f, reg, newEmitter(), "task", Config{
		MaxTurns:      10,
		ContextWindow: 200000,
		Compaction:    &Compaction{Threshold: 0.95, KeepRecentTurns: 2},
	})
	require.NoError(t, err)
	assert.Equal(t, "done", res.Reason)
	assert.True(t, res.Completed)
}

// bigTool is a fake tool that returns a large string with distinct head/tail content.
type bigTool struct{ output string }

func (b *bigTool) Name() string { return "big" }
func (b *bigTool) Schema() llm.Tool {
	return llm.Tool{Type: "function", Function: llm.ToolFunction{Name: "big"}}
}

func (b *bigTool) Execute(_ context.Context, _ map[string]any) (tools.Result, error) {
	return tools.Result{Text: b.output}, nil
}

// capturingLLMSeq records all requests; scripted responses are returned in order.
type capturingLLMSeq struct {
	responses []llm.Response
	requests  []llm.Request
	i         int
}

func (c *capturingLLMSeq) Send(_ context.Context, req llm.Request) (llm.Response, error) {
	return c.next(req), nil
}

func (c *capturingLLMSeq) SendStream(_ context.Context, req llm.Request, _ func(llm.Delta)) (llm.Response, error) {
	return c.next(req), nil
}

func (c *capturingLLMSeq) next(req llm.Request) llm.Response {
	c.requests = append(c.requests, req)
	if c.i >= len(c.responses) {
		return llm.Response{FinishReason: "stop"}
	}

	r := c.responses[c.i]
	c.i++

	return r
}

func TestRunToolOutputCapTruncates(t *testing.T) {
	const maxBytes = 1000

	head := strings.Repeat("H", 60000)
	tail := strings.Repeat("T", 40000)
	large := head + tail // 100 KiB, clearly distinct head/tail

	bt := &bigTool{output: large}
	reg := tools.NewRegistry(bt)

	capt := &capturingLLMSeq{responses: []llm.Response{
		{ToolCalls: []llm.ToolCall{toolCall("1", "big", `{}`)}},
		{Content: "done", FinishReason: "stop"},
	}}

	_, err := Run(context.Background(), capt, reg, newEmitter(), "task", Config{
		MaxTurns:           10,
		ToolOutputMaxBytes: maxBytes,
	})
	require.NoError(t, err)

	// The second request carries the tool-result message from the first turn.
	require.Len(t, capt.requests, 2)
	secondReq := capt.requests[1]

	// Find the tool-result message.
	var toolResultContent string

	for _, m := range secondReq.Messages {
		if m.Role == "tool" && m.ToolCallID == "1" {
			toolResultContent = m.Content

			break
		}
	}

	require.NotEmpty(t, toolResultContent, "tool-result message not found in second request")

	assert.Contains(t, toolResultContent, "bytes truncated")
	assert.True(t, strings.HasPrefix(toolResultContent, "HH"), "head content preserved")
	assert.True(t, strings.HasSuffix(toolResultContent, "TT"), "tail content preserved")
	assert.LessOrEqual(t, len(toolResultContent), maxBytes+80) // marker allowance
}

// secretTool is a fake tool that embeds a known secret in its output.
type secretTool struct{ secret string }

func (s *secretTool) Name() string { return "secret" }
func (s *secretTool) Schema() llm.Tool {
	return llm.Tool{Type: "function", Function: llm.ToolFunction{Name: "secret"}}
}

func (s *secretTool) Execute(_ context.Context, _ map[string]any) (tools.Result, error) {
	return tools.Result{Text: "output contains " + s.secret + " end"}, nil
}

func TestRunRedactToolOutput(t *testing.T) {
	const seedSecret = "sk-or-v1-supersecretkey99"

	st := &secretTool{secret: seedSecret}
	reg := tools.NewRegistry(st)

	capt := &capturingLLMSeq{responses: []llm.Response{
		{ToolCalls: []llm.ToolCall{toolCall("1", "secret", `{}`)}},
		{Content: "done", FinishReason: "stop"},
	}}

	var transcript bytes.Buffer

	emit := events.NewEmitter(nil, &transcript)

	r := redact.New([]string{seedSecret})

	_, err := Run(context.Background(), capt, reg, emit, "task", Config{
		MaxTurns:         10,
		RedactToolOutput: r.Apply,
	})
	require.NoError(t, err)

	// The second request must carry the tool-result message with redacted content.
	require.Len(t, capt.requests, 2)

	var toolResultContent string

	for _, m := range capt.requests[1].Messages {
		if m.Role == "tool" && m.ToolCallID == "1" {
			toolResultContent = m.Content

			break
		}
	}

	require.NotEmpty(t, toolResultContent, "tool-result message not found in second request")
	assert.Contains(t, toolResultContent, "[REDACTED]", "secret must be masked in tool message")
	assert.NotContains(t, toolResultContent, seedSecret, "raw secret must not appear in tool message")

	// No event in the JSONL transcript may contain the raw secret.
	assert.NotContains(t, transcript.String(), seedSecret, "raw secret must not appear in any event")
}

// erroringTool is a fake tool that fails with an error carrying a large,
// secret-bearing message — models a subprocess error that dumps full output.
type erroringTool struct{ msg string }

func (e *erroringTool) Name() string { return "boom" }
func (e *erroringTool) Schema() llm.Tool {
	return llm.Tool{Type: "function", Function: llm.ToolFunction{Name: "boom"}}
}

func (e *erroringTool) Execute(_ context.Context, _ map[string]any) (tools.Result, error) {
	return tools.Result{}, errors.New(e.msg)
}

func TestRunToolErrorOutputCappedAndRedacted(t *testing.T) {
	const (
		maxBytes   = 1000
		seedSecret = "sk-or-v1-supersecretkey99"
	)

	// Error message far exceeds the cap and embeds a secret.
	big := strings.Repeat("E", 50000) + " " + seedSecret + " " + strings.Repeat("F", 50000)
	et := &erroringTool{msg: big}
	reg := tools.NewRegistry(et)

	capt := &capturingLLMSeq{responses: []llm.Response{
		{ToolCalls: []llm.ToolCall{toolCall("1", "boom", `{}`)}},
		{Content: "done", FinishReason: "stop"},
	}}

	r := redact.New([]string{seedSecret})

	_, err := Run(context.Background(), capt, reg, newEmitter(), "task", Config{
		MaxTurns:           10,
		ToolOutputMaxBytes: maxBytes,
		RedactToolOutput:   r.Apply,
	})
	require.NoError(t, err)

	require.Len(t, capt.requests, 2)

	var toolResultContent string

	for _, m := range capt.requests[1].Messages {
		if m.Role == "tool" && m.ToolCallID == "1" {
			toolResultContent = m.Content

			break
		}
	}

	require.NotEmpty(t, toolResultContent, "tool-result message not found in second request")
	assert.Contains(t, toolResultContent, "bytes truncated", "oversized tool error must be size-capped")
	assert.LessOrEqual(t, len(toolResultContent), maxBytes+80, "tool error must be bounded by the cap") // marker allowance
	assert.NotContains(t, toolResultContent, seedSecret, "secret must be redacted on the error path")
}

// TestEmittedContentIsRedacted covers the ModelResponse and Thinking emit
// sites: cfg.RedactToolOutput must scrub resp.Content and resp.Reasoning
// before they reach the event stream/transcript, not just tool results.
func TestEmittedContentIsRedacted(t *testing.T) {
	f := &fakeLLM{responses: []llm.Response{
		{Content: "leak SECRET here", Reasoning: "think SECRET", FinishReason: "stop"},
	}}

	var transcript bytes.Buffer

	emit := events.NewEmitter(nil, &transcript)

	_, err := Run(context.Background(), f, tools.NewRegistry(), emit, "task", Config{
		MaxTurns:         1,
		RedactToolOutput: func(s string) string { return strings.ReplaceAll(s, "SECRET", "***") },
	})
	require.NoError(t, err)
	assert.NotContains(t, transcript.String(), "SECRET")
	assert.Contains(t, transcript.String(), "***")
}

// TestEmittedToolCallArgsAreRedacted covers the third emit site: ToolCallKind
// emits tc.Function.Arguments as "raw_args" before the call is dispatched, so
// it must be routed through cfg.RedactToolOutput independently of the
// tool-result redaction already covered by TestRunRedactToolOutput.
func TestEmittedToolCallArgsAreRedacted(t *testing.T) {
	root := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(root, "f.txt"), []byte("data"), 0o644))
	reg := tools.NewRegistry(tools.NewReadTool(root))

	f := &fakeLLM{responses: []llm.Response{
		{ToolCalls: []llm.ToolCall{toolCall("1", "read", `{"path":"f.txt","note":"SECRET"}`)}},
		{Content: "done", FinishReason: "stop"},
	}}

	var transcript bytes.Buffer

	emit := events.NewEmitter(nil, &transcript)

	_, err := Run(context.Background(), f, reg, emit, "task", Config{
		MaxTurns:         10,
		RedactToolOutput: func(s string) string { return strings.ReplaceAll(s, "SECRET", "***") },
	})
	require.NoError(t, err)
	assert.NotContains(t, transcript.String(), "SECRET")
	assert.Contains(t, transcript.String(), "***")
}

// scriptedInbox: queue of messages; Drain pops all pending; Wait pops one or
// returns closeErr when the queue is empty. When deliverViaWaitOnly is set,
// Drain returns nothing so the pending queue is delivered exclusively through
// Wait (models a message that only arrives while the loop is parked at Wait).
type scriptedInbox struct {
	mu                 sync.Mutex
	pending            []UserMessage
	closeErr           error // ErrInboxClosed once exhausted, or nil to block forever
	deliverViaWaitOnly bool
}

func (s *scriptedInbox) Drain() []UserMessage {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.deliverViaWaitOnly || len(s.pending) == 0 {
		return nil
	}

	out := s.pending
	s.pending = nil

	return out
}

func (s *scriptedInbox) Wait(ctx context.Context) (UserMessage, error) {
	s.mu.Lock()

	if len(s.pending) > 0 {
		um := s.pending[0]
		s.pending = s.pending[1:]
		s.mu.Unlock()

		return um, nil
	}

	closeErr := s.closeErr
	s.mu.Unlock()

	if closeErr != nil {
		return UserMessage{}, closeErr
	}

	// Block forever unless ctx fires.
	<-ctx.Done()

	return UserMessage{}, ctx.Err()
}

func (s *scriptedInbox) push(um UserMessage) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.pending = append(s.pending, um)
}

// parseEvents decodes JSONL transcript events for assertions.
func parseEvents(t *testing.T, transcript string) []events.Event {
	t.Helper()

	var out []events.Event

	for _, line := range strings.Split(strings.TrimSpace(transcript), "\n") {
		if line == "" {
			continue
		}

		var ev events.Event
		require.NoError(t, json.Unmarshal([]byte(line), &ev))
		out = append(out, ev)
	}

	return out
}

// findToolResult locates the tool-result message for a given tool-call id.
func findToolResult(msgs []llm.Message, id string) (string, bool) {
	for _, m := range msgs {
		if m.Role == "tool" && m.ToolCallID == id {
			return m.Content, true
		}
	}

	return "", false
}

// firstUserMessageIndex returns the index of the first user message at-or-after
// offset, or -1.
func userMessageIndexAfter(msgs []llm.Message, offset int) int {
	for i := offset; i < len(msgs); i++ {
		if msgs[i].Role == "user" {
			return i
		}
	}

	return -1
}

// Case 2: a message arriving during turn 1's (single) tool call drains at the
// top of turn 2 and lands in request 2 AFTER turn 1's tool results.
func TestInboxTurnTopDrain(t *testing.T) {
	inbox := &scriptedInbox{closeErr: ErrInboxClosed}
	it := &interjectingTool{inbox: inbox, msg: UserMessage{Content: "human steers here", MessageID: "m1"}}
	reg := tools.NewRegistry(it)

	capt := &capturingLLMSeq{responses: []llm.Response{
		// single tool call → mid-batch skip does not apply; drain happens at turn 2 top.
		{ToolCalls: []llm.ToolCall{toolCall("1", "interject", `{}`)}},
		{Content: "done", FinishReason: "stop"},
	}}

	res, err := Run(context.Background(), capt, reg, newEmitter(), "do it", Config{MaxTurns: 10, Inbox: inbox})
	require.NoError(t, err)
	assert.True(t, res.Completed)

	require.Len(t, capt.requests, 2)
	second := capt.requests[1].Messages

	// Turn 1's tool result must be present and precede the injected user message.
	_, ok := findToolResult(second, "1")
	require.True(t, ok, "turn 1 tool result missing from request 2")

	// Locate the tool result index, then the user message after it.
	toolIdx := -1

	for i, m := range second {
		if m.Role == "tool" && m.ToolCallID == "1" {
			toolIdx = i

			break
		}
	}

	require.GreaterOrEqual(t, toolIdx, 0)

	uIdx := userMessageIndexAfter(second, toolIdx)
	require.GreaterOrEqual(t, uIdx, 0, "injected user message not found after tool result")
	assert.Equal(t, "human steers here", second[uIdx].Content)
}

// Case 3: natural stop blocks on Wait, gets one message, continues, then closed.
func TestInboxNaturalStopWaitThenContinue(t *testing.T) {
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()))

	inbox := &scriptedInbox{
		pending:            []UserMessage{{Content: "keep going", MessageID: "m1"}},
		closeErr:           ErrInboxClosed,
		deliverViaWaitOnly: true, // message arrives only via Wait, not the turn-top drain
	}

	// Turn 1: no tool calls. Turn 2: no tool calls. Then inbox is empty → closed.
	f := &fakeLLM{responses: []llm.Response{
		{Content: "first", FinishReason: "stop"},
		{Content: "second", FinishReason: "stop"},
	}}

	var transcript bytes.Buffer

	emit := events.NewEmitter(nil, &transcript)

	res, err := Run(context.Background(), f, reg, emit, "task", Config{MaxTurns: 10, Inbox: inbox})
	require.NoError(t, err)
	assert.True(t, res.Completed)
	assert.Equal(t, "done", res.Reason)
	assert.Equal(t, 2, res.Turns)

	// An awaiting_human state change must precede the wait, carrying the
	// plural "turns" key like every other StateChange payload.
	evs := parseEvents(t, transcript.String())

	var sawAwaiting bool

	for _, ev := range evs {
		if ev.Kind == events.StateChange && ev.Data["state"] == "awaiting_human" {
			sawAwaiting = true

			assert.Contains(t, ev.Data, "turns", "awaiting_human payload must use plural turns key")

			break
		}
	}

	assert.True(t, sawAwaiting, "awaiting_human state change not emitted")
}

// Case 4: closed inbox behaves like autonomous — single turn, done.
func TestInboxClosedIsAutonomous(t *testing.T) {
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()))

	inbox := &scriptedInbox{closeErr: ErrInboxClosed}

	f := &fakeLLM{responses: []llm.Response{
		{Content: "all done", FinishReason: "stop"},
	}}

	res, err := Run(context.Background(), f, reg, newEmitter(), "task", Config{MaxTurns: 10, Inbox: inbox})
	require.NoError(t, err)
	assert.True(t, res.Completed)
	assert.Equal(t, "done", res.Reason)
	assert.Equal(t, 1, res.Turns)
}

// interjectingTool pushes a message into the inbox the first time it executes.
type interjectingTool struct {
	inbox *scriptedInbox
	msg   UserMessage
	calls int
}

func (i *interjectingTool) Name() string { return "interject" }
func (i *interjectingTool) Schema() llm.Tool {
	return llm.Tool{Type: "function", Function: llm.ToolFunction{Name: "interject"}}
}

func (i *interjectingTool) Execute(_ context.Context, _ map[string]any) (tools.Result, error) {
	i.calls++
	if i.calls == 1 {
		i.inbox.push(i.msg)
	}

	return tools.Result{Text: "ok"}, nil
}

// Case 5: mid-batch interrupt. Turn 1 has three tool calls; the first pushes a
// message; calls 2 and 3 are skipped with synthesized results; the user message
// follows after all tool results.
func TestInboxMidBatchInterrupt(t *testing.T) {
	inbox := &scriptedInbox{closeErr: ErrInboxClosed}
	it := &interjectingTool{inbox: inbox, msg: UserMessage{Content: "stop and listen", MessageID: "m1"}}
	reg := tools.NewRegistry(it)

	capt := &capturingLLMSeq{responses: []llm.Response{
		{ToolCalls: []llm.ToolCall{
			toolCall("1", "interject", `{}`),
			toolCall("2", "interject", `{}`),
			toolCall("3", "interject", `{}`),
		}},
		{Content: "done", FinishReason: "stop"},
	}}

	res, err := Run(context.Background(), capt, reg, newEmitter(), "task", Config{MaxTurns: 10, Inbox: inbox})
	require.NoError(t, err)
	assert.True(t, res.Completed)

	// Only the first call executed.
	assert.Equal(t, 1, it.calls, "calls 2 and 3 must not execute")

	require.Len(t, capt.requests, 2)
	second := capt.requests[1].Messages

	// Call 1 executed result.
	c1, ok := findToolResult(second, "1")
	require.True(t, ok)
	assert.Equal(t, "ok", c1)

	// Calls 2 and 3 carry synthesized skip results.
	c2, ok := findToolResult(second, "2")
	require.True(t, ok)
	assert.Equal(t, "skipped: user interjected", c2)

	c3, ok := findToolResult(second, "3")
	require.True(t, ok)
	assert.Equal(t, "skipped: user interjected", c3)

	// The user message must appear AFTER all three tool results.
	lastToolIdx := -1

	for i, m := range second {
		if m.Role == "tool" && (m.ToolCallID == "1" || m.ToolCallID == "2" || m.ToolCallID == "3") {
			lastToolIdx = i
		}
	}

	require.GreaterOrEqual(t, lastToolIdx, 0)

	uIdx := userMessageIndexAfter(second, lastToolIdx)
	require.GreaterOrEqual(t, uIdx, 0, "user message must follow the tool results")
	assert.Equal(t, "stop and listen", second[uIdx].Content)
}

func TestRunDetectsIncapability(t *testing.T) {
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()))

	// Every turn: one tool call whose Arguments is not valid JSON → never executes.
	badCall := toolCall("1", "read", `{ this is not json`)
	responses := make([]llm.Response, 20)

	for i := range responses {
		responses[i] = llm.Response{ToolCalls: []llm.ToolCall{badCall}}
	}

	f := &fakeLLM{responses: responses}

	res, err := Run(context.Background(), f, reg, newEmitter(), "do x",
		Config{Model: "weak/m", MaxTurns: 20, IncapableThreshold: 3})
	require.NoError(t, err)
	assert.Equal(t, "incapable", res.Reason)
	assert.Less(t, res.Turns, 20, "incapable must fire before MaxTurns")
}

// pushOnStreamLLM pushes a human message into the inbox during the turn's stream
// (after the top-of-turn drain), modelling an interjection that arrives while the
// model is producing an all-unparseable, multi-tool-call turn.
type pushOnStreamLLM struct {
	inbox      *scriptedInbox
	msg        UserMessage
	pushOnTurn int
	responses  []llm.Response
	requests   []llm.Request
	i          int
}

func (f *pushOnStreamLLM) Send(context.Context, llm.Request) (llm.Response, error) {
	return llm.Response{FinishReason: "stop"}, nil
}

func (f *pushOnStreamLLM) SendStream(_ context.Context, req llm.Request, _ func(llm.Delta)) (llm.Response, error) {
	f.requests = append(f.requests, req)

	f.i++
	if f.i == f.pushOnTurn {
		f.inbox.push(f.msg)
	}

	if f.i-1 < len(f.responses) {
		return f.responses[f.i-1], nil
	}

	return llm.Response{FinishReason: "stop"}, nil
}

func TestIncapableRecoveryDeliversStashedInterjection(t *testing.T) {
	inbox := &scriptedInbox{closeErr: ErrInboxClosed} // Wait returns "done" if reached
	fake := &pushOnStreamLLM{
		inbox:      inbox,
		msg:        UserMessage{MessageID: "interject-1", Content: "steer here"},
		pushOnTurn: 1,
		responses: []llm.Response{
			{ToolCalls: []llm.ToolCall{ // two calls, both unparseable → turnHadCapableTool=false
				toolCall("t1", "read", "{bad"),
				toolCall("t2", "read", "{also-bad"),
			}},
			{Content: "done", FinishReason: "stop"}, // turn 2 after recovery
		},
	}
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()))

	var transcript bytes.Buffer

	emit := events.NewEmitter(nil, &transcript)

	res, err := Run(context.Background(), fake, reg, emit, "task", Config{
		Interactive:        true,
		Inbox:              inbox,
		IncapableThreshold: 1,
	})
	require.NoError(t, err)

	delivered := false

	for _, ev := range parseEvents(t, transcript.String()) {
		if ev.Kind == events.UserInput && ev.Data["message_id"] == "interject-1" {
			delivered = true
		}
	}

	assert.True(t, delivered, "stashed interjection must be delivered, not dropped")
	assert.Equal(t, "done", res.Reason)

	// Turn 1 is the all-unparseable-tools turn (the interjection arrives mid-stream
	// and is stashed, not sent yet); turn 2 is the recovery turn, which must carry
	// the stashed interjection in its request history.
	require.Len(t, fake.requests, 2, "expected one request for the unparseable turn and one for the recovery turn")

	recoveryReq := fake.requests[1]

	appended := false

	for _, m := range recoveryReq.Messages {
		if m.Role == "user" && m.Content == "steer here" {
			appended = true
		}
	}

	assert.True(t, appended, "recovery turn's request must include the stashed interjection as a user message")
}

func TestRun_SeedsHistoryBeforeTask(t *testing.T) {
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()))
	capt := &capturingLLM{}
	_, err := Run(context.Background(), capt, reg, newEmitter(), "now", Config{
		SystemPrompt: "SYS",
		History: []llm.Message{
			{Role: "user", Content: "prior Q"},
			{Role: "assistant", Content: "prior A"},
		},
	})
	require.NoError(t, err)
	require.Equal(t, []llm.Message{
		{Role: "system", Content: "SYS"},
		{Role: "user", Content: "prior Q"},
		{Role: "assistant", Content: "prior A"},
		{Role: "user", Content: "now"},
	}, capt.last.Messages)
}

func TestRun_EmitsThinkingFromReasoning(t *testing.T) {
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()))
	// Single response with Reasoning set; no tool calls so the loop exits cleanly.
	f := &fakeLLM{responses: []llm.Response{
		{Content: "done", Reasoning: "pondering", FinishReason: "stop"},
	}}

	var transcript bytes.Buffer

	emit := events.NewEmitter(nil, &transcript)

	_, err := Run(context.Background(), f, reg, emit, "task", Config{MaxTurns: 10})
	require.NoError(t, err)

	evs := parseEvents(t, transcript.String())

	thinkingIdx, modelResponseIdx := -1, -1

	for i, ev := range evs {
		if ev.Kind == events.Thinking && thinkingIdx == -1 {
			thinkingIdx = i

			assert.Equal(t, "pondering", ev.Data["content"], "thinking event must carry the reasoning content")
		}

		if ev.Kind == events.ModelResponse && modelResponseIdx == -1 {
			modelResponseIdx = i
		}
	}

	require.GreaterOrEqual(t, thinkingIdx, 0, "thinking event not recorded")
	require.GreaterOrEqual(t, modelResponseIdx, 0, "model_response event not recorded")
	assert.Less(t, thinkingIdx, modelResponseIdx, "thinking event must be emitted before model_response")
}

func TestSeedMessage_TextOnly(t *testing.T) {
	m := seedMessage("do the thing", nil)
	assert.Equal(t, "user", m.Role)
	assert.Equal(t, "do the thing", m.Content)
	assert.Empty(t, m.ContentParts)
}

func TestSeedMessage_WithImages(t *testing.T) {
	m := seedMessage("describe", []llm.ImageURL{{URL: "data:image/png;base64,AAAA"}})
	assert.Equal(t, "user", m.Role)
	assert.Empty(t, m.Content)
	require.Len(t, m.ContentParts, 2)
	assert.Equal(t, llm.ContentPart{Type: "text", Text: "describe"}, m.ContentParts[0])
	assert.Equal(t, "image_url", m.ContentParts[1].Type)
	require.NotNil(t, m.ContentParts[1].ImageURL)
	assert.Equal(t, "data:image/png;base64,AAAA", m.ContentParts[1].ImageURL.URL)
}

// Case 6: ctx cancel during Wait → Run returns ctx.Err() with Reason canceled.
func TestInboxCtxCancelDuringWait(t *testing.T) {
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()))

	// Inbox never closes and has no pending messages → Wait blocks until ctx.
	inbox := &scriptedInbox{}

	f := &fakeLLM{responses: []llm.Response{
		{Content: "first", FinishReason: "stop"},
	}}

	ctx, cancel := context.WithCancel(context.Background())
	// Cancel up front: the fakeLLM ignores ctx, so turn 1 still completes; the
	// loop then reaches the natural-stop Wait, which returns ctx.Err()
	// immediately because Done is already closed.
	cancel()

	res, err := Run(ctx, f, reg, newEmitter(), "task", Config{MaxTurns: 10, Inbox: inbox})
	require.Error(t, err)
	require.ErrorIs(t, err, context.Canceled)
	assert.Equal(t, "canceled", res.Reason)
	assert.False(t, res.Completed)
}

// imageTool returns text plus one inline image (models an MCP get_card result).
type imageTool struct{}

func (imageTool) Name() string { return "img" }
func (imageTool) Schema() llm.Tool {
	return llm.Tool{Type: "function", Function: llm.ToolFunction{Name: "img"}}
}

func (imageTool) Execute(_ context.Context, _ map[string]any) (tools.Result, error) {
	return tools.Result{Text: "card text", Images: []llm.ImageURL{{URL: "data:image/png;base64,AAAA"}}}, nil
}

func TestRunInjectsToolImagesAsUserMessage(t *testing.T) {
	reg := tools.NewRegistry(imageTool{})
	capt := &capturingLLMSeq{responses: []llm.Response{
		{ToolCalls: []llm.ToolCall{toolCall("1", "img", `{}`)}},
		{Content: "done", FinishReason: "stop"},
	}}

	_, err := Run(context.Background(), capt, reg, newEmitter(), "describe the card", Config{MaxTurns: 10})
	require.NoError(t, err)

	require.Len(t, capt.requests, 2)
	second := capt.requests[1].Messages

	text, ok := findToolResult(second, "1")
	require.True(t, ok)
	assert.Equal(t, "card text", text)

	toolIdx := -1

	for i, m := range second {
		if m.Role == "tool" && m.ToolCallID == "1" {
			toolIdx = i

			break
		}
	}

	require.GreaterOrEqual(t, toolIdx, 0)

	uIdx := userMessageIndexAfter(second, toolIdx)
	require.GreaterOrEqual(t, uIdx, 0, "synthetic image user message not found after tool result")

	img := second[uIdx]
	require.Len(t, img.ContentParts, 2)
	assert.Equal(t, "text", img.ContentParts[0].Type)
	assert.Equal(t, toolImagePreamble, img.ContentParts[0].Text)
	assert.Equal(t, "image_url", img.ContentParts[1].Type)
	require.NotNil(t, img.ContentParts[1].ImageURL)
	assert.Equal(t, "data:image/png;base64,AAAA", img.ContentParts[1].ImageURL.URL)
}

func TestRunNoImageMessageWhenToolReturnsNoImages(t *testing.T) {
	root := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(root, "f.txt"), []byte("data"), 0o644))
	reg := tools.NewRegistry(tools.NewReadTool(root))

	capt := &capturingLLMSeq{responses: []llm.Response{
		{ToolCalls: []llm.ToolCall{toolCall("1", "read", `{"path":"f.txt"}`)}},
		{Content: "done", FinishReason: "stop"},
	}}

	_, err := Run(context.Background(), capt, reg, newEmitter(), "task", Config{MaxTurns: 10})
	require.NoError(t, err)

	require.Len(t, capt.requests, 2)
	second := capt.requests[1].Messages

	toolIdx := -1

	for i, m := range second {
		if m.Role == "tool" && m.ToolCallID == "1" {
			toolIdx = i

			break
		}
	}

	require.GreaterOrEqual(t, toolIdx, 0)
	assert.Equal(t, -1, userMessageIndexAfter(second, toolIdx), "text-only tool result must not append a user message")
}

// imageInterjectTool both returns an image and pushes an interjection to the
// inbox on its first call — models the mid-batch-interrupt + image injection
// combination.
type imageInterjectTool struct {
	inbox *scriptedInbox
	msg   UserMessage
	calls int
}

func (t *imageInterjectTool) Name() string { return "imginterject" }
func (t *imageInterjectTool) Schema() llm.Tool {
	return llm.Tool{Type: "function", Function: llm.ToolFunction{Name: "imginterject"}}
}

func (t *imageInterjectTool) Execute(_ context.Context, _ map[string]any) (tools.Result, error) {
	t.calls++
	if t.calls == 1 {
		t.inbox.push(t.msg)
	}

	return tools.Result{Text: "card text", Images: []llm.ImageURL{{URL: "data:image/png;base64,AAAA"}}}, nil
}

// TestImageInterjectOrdering proves that on a mid-batch interrupt where the
// executing tool also returns an image, the next request carries messages in
// this order: all tool results → synthetic image user message → human
// interjection user message.
func TestImageInterjectOrdering(t *testing.T) {
	inbox := &scriptedInbox{closeErr: ErrInboxClosed}
	it := &imageInterjectTool{inbox: inbox, msg: UserMessage{Content: "stop and look", MessageID: "m1"}}
	reg := tools.NewRegistry(it)

	capt := &capturingLLMSeq{responses: []llm.Response{
		{ToolCalls: []llm.ToolCall{
			toolCall("1", "imginterject", `{}`),
			toolCall("2", "imginterject", `{}`),
		}},
		{Content: "done", FinishReason: "stop"},
	}}

	res, err := Run(context.Background(), capt, reg, newEmitter(), "task", Config{MaxTurns: 10, Inbox: inbox})
	require.NoError(t, err)
	assert.True(t, res.Completed)

	// Call 1 executes; call 2 must be skipped.
	assert.Equal(t, 1, it.calls, "only call 1 must execute; call 2 is skipped")

	require.Len(t, capt.requests, 2)
	second := capt.requests[1].Messages

	// Both tool results must be present.
	c1, ok := findToolResult(second, "1")
	require.True(t, ok, "tool result '1' missing")
	assert.Equal(t, "card text", c1)

	c2, ok := findToolResult(second, "2")
	require.True(t, ok, "tool result '2' missing")
	assert.Equal(t, "skipped: user interjected", c2)

	// Locate the last tool-role message.
	lastToolIdx := -1

	for i, m := range second {
		if m.Role == "tool" && (m.ToolCallID == "1" || m.ToolCallID == "2") {
			lastToolIdx = i
		}
	}

	require.GreaterOrEqual(t, lastToolIdx, 0)

	// Synthetic image user message must follow all tool results.
	imgIdx := userMessageIndexAfter(second, lastToolIdx)
	require.GreaterOrEqual(t, imgIdx, 0, "synthetic image user message not found after tool results")

	img := second[imgIdx]
	require.Len(t, img.ContentParts, 2, "expected preamble + 1 image part")
	assert.Equal(t, "text", img.ContentParts[0].Type)
	assert.Equal(t, toolImagePreamble, img.ContentParts[0].Text)
	assert.Equal(t, "image_url", img.ContentParts[1].Type)
	require.NotNil(t, img.ContentParts[1].ImageURL)
	assert.Equal(t, "data:image/png;base64,AAAA", img.ContentParts[1].ImageURL.URL)

	// Human interjection must follow the image message.
	interjIdx := userMessageIndexAfter(second, imgIdx+1)
	require.GreaterOrEqual(t, interjIdx, 0, "interjection user message not found after image message")
	assert.Equal(t, "stop and look", second[interjIdx].Content)
	assert.Empty(t, second[interjIdx].ContentParts, "interjection must be plain text")

	// The crux: image index < interjection index, both after all tool results.
	assert.Less(t, lastToolIdx, imgIdx, "image message must follow all tool results")
	assert.Less(t, imgIdx, interjIdx, "image message must precede the human interjection")
}

// multiImageTool returns a distinct image URL per call.
type multiImageTool struct{ calls int }

func (t *multiImageTool) Name() string { return "multimg" }
func (t *multiImageTool) Schema() llm.Tool {
	return llm.Tool{Type: "function", Function: llm.ToolFunction{Name: "multimg"}}
}

func (t *multiImageTool) Execute(_ context.Context, _ map[string]any) (tools.Result, error) {
	t.calls++

	return tools.Result{
		Text:   fmt.Sprintf("img %d", t.calls),
		Images: []llm.ImageURL{{URL: fmt.Sprintf("data:image/png;base64,IMG%d", t.calls)}},
	}, nil
}

// TestMultiImageAggregation proves that two image-bearing tool calls in one
// turn produce exactly one synthetic user message carrying both images
// (preamble + 2 image parts).
func TestMultiImageAggregation(t *testing.T) {
	mt := &multiImageTool{}
	reg := tools.NewRegistry(mt)

	capt := &capturingLLMSeq{responses: []llm.Response{
		{ToolCalls: []llm.ToolCall{
			toolCall("1", "multimg", `{}`),
			toolCall("2", "multimg", `{}`),
		}},
		{Content: "done", FinishReason: "stop"},
	}}

	_, err := Run(context.Background(), capt, reg, newEmitter(), "task", Config{MaxTurns: 10})
	require.NoError(t, err)

	require.Len(t, capt.requests, 2)
	second := capt.requests[1].Messages

	// Locate the last tool-role message.
	lastToolIdx := -1

	for i, m := range second {
		if m.Role == "tool" && (m.ToolCallID == "1" || m.ToolCallID == "2") {
			lastToolIdx = i
		}
	}

	require.GreaterOrEqual(t, lastToolIdx, 0)

	// Exactly one user message with ContentParts must follow the tool results.
	var imageMsgs []llm.Message

	for i := lastToolIdx + 1; i < len(second); i++ {
		if second[i].Role == "user" && len(second[i].ContentParts) > 0 {
			imageMsgs = append(imageMsgs, second[i])
		}
	}

	require.Len(t, imageMsgs, 1, "expected exactly one synthetic image user message")

	img := imageMsgs[0]
	require.Len(t, img.ContentParts, 3, "preamble + 2 image parts expected")
	assert.Equal(t, "text", img.ContentParts[0].Type)
	assert.Equal(t, toolImagePreamble, img.ContentParts[0].Text)
	assert.Equal(t, "image_url", img.ContentParts[1].Type)
	require.NotNil(t, img.ContentParts[1].ImageURL)
	assert.Equal(t, "data:image/png;base64,IMG1", img.ContentParts[1].ImageURL.URL)
	assert.Equal(t, "image_url", img.ContentParts[2].Type)
	require.NotNil(t, img.ContentParts[2].ImageURL)
	assert.Equal(t, "data:image/png;base64,IMG2", img.ContentParts[2].ImageURL.URL)
}

// TestTransportErrorEmitIsRedactedAndCapped covers the last content-bearing
// emit site: a SendStream error embeds the provider response body (up to
// 8 MiB), so the error event must pass through cfg.RedactToolOutput and the
// ToolOutputMaxBytes cap like every other emit — and in the correct order
// (redact, then truncate), never the reverse.
//
// Two seed secrets are placed to straddle tools.HeadTail's two cut boundaries
// (truncate.go:14-15: head = limit*2/3, tail = limit-head) rather than
// sitting entirely inside the discarded middle. A secret that straddles a cut
// survives as a partial, non-matching fragment on the "kept" side if
// truncation runs before redaction — which the exact-literal redact pass can
// no longer catch, because the fragment isn't the full secret string. If
// redaction runs first (correct order), the whole secret is masked before any
// cut exists, so no fragment of it can ever appear in the output.
func TestTransportErrorEmitIsRedactedAndCapped(t *testing.T) {
	const (
		maxBytes  = 1000
		errPrefix = "llm endpoint status 502: " // must match the fmt.Errorf format below

		secretHead = "sk-or-v1-headstraddle-secretA1"
		secretTail = "sk-or-v1-tailstraddle-secretB2"

		// Bytes of each secret left on the "kept" side of its straddled cut.
		headSurvive = 8
		tailSurvive = 8
	)

	// Mirrors tools.HeadTail's own split so the placement below stays correct
	// even if ToolOutputMaxBytes changes.
	head := maxBytes * 2 / 3
	tail := maxBytes - head

	// Place secretHead so the head-cut (offset `head` in the full error
	// string s = errPrefix+body) lands `headSurvive` bytes into it: a prefix
	// fragment survives in the kept head, the rest falls into the discarded
	// middle.
	bodyHeadCut := head - len(errPrefix)
	require.Greater(t, bodyHeadCut, headSurvive, "test arithmetic: head cut must land inside secretHead")

	headFillerLen := bodyHeadCut - headSurvive

	// Place secretTail so the tail-cut (offset len(s)-tail, equivalently
	// len(body)-tail in body coordinates) lands `tailSurvive` bytes before its
	// end: the discarded middle swallows its prefix, and a suffix fragment
	// survives in the kept tail.
	require.Greater(t, tail, tailSurvive, "test arithmetic: tail cut must land inside secretTail")

	tailFillerLen := tail - tailSurvive

	body := strings.Repeat("E", headFillerLen) + secretHead +
		strings.Repeat("X", 50000) +
		secretTail + strings.Repeat("F", tailFillerLen)

	f := &scriptedLLMWithErrors{results: []scriptedCall{
		{err: fmt.Errorf("llm endpoint status 502: %s", body)},
	}}

	var transcript bytes.Buffer

	emit := events.NewEmitter(nil, &transcript)

	r := redact.New([]string{secretHead, secretTail})

	res, err := Run(context.Background(), f, tools.NewRegistry(), emit, "task", Config{
		MaxTurns:           5,
		ToolOutputMaxBytes: maxBytes,
		RedactToolOutput:   r.Apply,
	})
	require.Error(t, err)
	assert.Equal(t, "error", res.Reason)

	// The error RETURNED by Run must stay raw and uncapped: consumers handle
	// redaction/truncation themselves, and only the emitted event is
	// sanitized.
	wantErr := errPrefix + body
	assert.Equal(t, wantErr, err.Error(), "returned error must be byte-identical to the raw transport error")
	assert.Contains(t, err.Error(), secretHead, "returned error must retain the raw secret, unredacted")
	assert.Contains(t, err.Error(), secretTail, "returned error must retain the raw secret, unredacted")
	assert.Greater(t, len(err.Error()), maxBytes, "returned error must not be size-capped")

	s := transcript.String()
	assert.NotContains(t, s, secretHead, "secret must be redacted on the transport-error emit path")
	assert.NotContains(t, s, secretTail, "secret must be redacted on the transport-error emit path")
	assert.Contains(t, s, "bytes truncated", "oversized transport error must be size-capped")

	// Discriminates redact-then-truncate (correct) from truncate-then-redact
	// (regression): only the buggy order leaves one of these fragments
	// intact, since truncation would run before the exact-literal redact pass
	// ever sees the (now-fragmented) secret.
	headFragment := secretHead[:headSurvive]
	tailFragment := secretTail[len(secretTail)-tailSurvive:]

	assert.NotContains(t, s, headFragment, "head-cut straddling secret fragment leaked: truncation ran before redaction")
	assert.NotContains(t, s, tailFragment, "tail-cut straddling secret fragment leaked: truncation ran before redaction")

	for _, ev := range parseEvents(t, s) {
		if ev.Kind != events.ErrorKind {
			continue
		}

		errStr, ok := ev.Data["error"].(string)
		require.True(t, ok, "error event must carry a string error")
		assert.LessOrEqual(t, len(errStr), maxBytes+80, "error event must be bounded by the cap") // marker allowance
	}
}
