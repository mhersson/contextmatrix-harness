package harness

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/mhersson/contextmatrix-harness/events"
	"github.com/mhersson/contextmatrix-harness/llm"
	"github.com/mhersson/contextmatrix-harness/redact"
	"github.com/mhersson/contextmatrix-harness/tools"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// history20 alternates 20 user and assistant messages, an older prefix long
// enough for compaction to summarize.
func history20() []llm.Message {
	history := make([]llm.Message, 20)
	for i := range history {
		if i%2 == 0 {
			history[i] = llm.Message{Role: "user", Content: fmt.Sprintf("user %d", i)}
		} else {
			history[i] = llm.Message{Role: "assistant", Content: fmt.Sprintf("assistant %d", i)}
		}
	}

	return history
}

// compactingConfig compacts at 900 prompt tokens: a 1000-token window at
// Threshold 0.85 gives an effective threshold of 850.
func compactingConfig(keepRecent int) Config {
	return Config{
		MaxTurns: 10, SystemPrompt: "SYS", ContextWindow: 1000,
		Compaction: &Compaction{Threshold: 0.85, KeepRecentTurns: keepRecent},
		History:    history20(),
	}
}

func roles(msgs []llm.Message) []string {
	out := make([]string, 0, len(msgs))
	for _, m := range msgs {
		out = append(out, m.Role)
	}

	return out
}

// crossingToolTurn is a tool-call reply at 900 prompt tokens, so a run that
// compacts on it still sends a next request.
func crossingToolTurn() llm.Response {
	return llm.Response{ToolCalls: []llm.ToolCall{toolCall("c1", "probe", "{}")}, Usage: llm.Usage{PromptTokens: 900}}
}

// noProgressHistory is one assistant turn with three tool results: compaction
// snaps the whole group into the kept tail, so it never shrinks the history.
func noProgressHistory() []llm.Message {
	return []llm.Message{
		{Role: "user", Content: "old"},
		{Role: "assistant", ToolCalls: []llm.ToolCall{toolCall("h1", "foo", "{}"), toolCall("h2", "foo", "{}"), toolCall("h3", "foo", "{}")}},
		{Role: "tool", ToolCallID: "h1", Content: "r1"},
		{Role: "tool", ToolCallID: "h2", Content: "r2"},
		{Role: "tool", ToolCallID: "h3", Content: "r3"},
	}
}

func TestRunFinalReplySkipsCompaction(t *testing.T) {
	final := llm.Response{Content: "ANSWER", Reasoning: "thought", FinishReason: "stop", Usage: llm.Usage{PromptTokens: 900}}
	seeded := append(history20(),
		llm.Message{Role: "assistant", ToolCalls: []llm.ToolCall{toolCall("c0", "probe", "{}")}},
		llm.Message{Role: "tool", ToolCallID: "c0", Content: "ok"},
	)

	tests := []struct {
		name      string
		history   []llm.Message
		responses []llm.Response
	}{
		{"tool results seeded in history", seeded, []llm.Response{final}},
		{"tool results from this run", history20(), []llm.Response{
			{ToolCalls: []llm.ToolCall{toolCall("c0", "probe", "{}")}, Usage: llm.Usage{PromptTokens: 100}},
			final,
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fake := &capturingLLMSeq{responses: tt.responses}

			cfg := compactingConfig(2)
			cfg.History = tt.history

			var transcript bytes.Buffer

			res, err := Run(context.Background(), fake, tools.NewRegistry(&countingTool{name: "probe"}), events.NewEmitter(nil, &transcript), "go", cfg)
			require.NoError(t, err)
			assert.True(t, res.Completed)
			assert.Equal(t, "done", res.Reason)
			assert.Equal(t, "ANSWER", res.Output)
			require.Len(t, fake.requests, len(tt.responses), "no summarize call follows the final reply")

			compactionEvents := 0

			for _, ev := range parseEvents(t, transcript.String()) {
				switch {
				case ev.Kind == events.StateChange && (ev.Data["event"] == "compaction" || ev.Data["event"] == "compaction_failed"),
					ev.Kind == events.UsageKind && ev.Data["phase"] == "compaction",
					ev.Kind == events.ContextLimit:
					compactionEvents++
				}
			}

			assert.Zero(t, compactionEvents)

			// Consumers read the finished history, so nothing in it is summarized away.
			assert.Equal(t, "user 0", res.Messages[1].Content)

			out, ok := findToolResult(res.Messages, "c0")
			require.True(t, ok, "the tool result stays in Result.Messages")
			assert.Equal(t, "ok", out)
			assert.Equal(t, "ANSWER", res.Messages[len(res.Messages)-1].Content)
		})
	}
}

func TestRunNoProgressCompactionBelowTheHardStopCarriesOn(t *testing.T) {
	// Threshold 0.5 compacts at 500 tokens, under the 850-token hard stop, so a
	// no-progress compaction at 600 falls through without stopping the run.
	final := llm.Response{Content: "ANSWER", FinishReason: "stop", Usage: llm.Usage{PromptTokens: 600}}

	tests := []struct {
		name      string
		inbox     bool
		responses []llm.Response
		wantCalls int
	}{
		{"tool-call reply then a final reply", false, []llm.Response{
			{ToolCalls: []llm.ToolCall{toolCall("c1", "probe", "{}")}, Usage: llm.Usage{PromptTokens: 600}},
			{Content: "SUMMARY"},
			final,
		}, 3},
		{"final reply in an inbox run", true, []llm.Response{final, {Content: "SUMMARY"}}, 2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fake := &capturingLLMSeq{responses: tt.responses}

			cfg := Config{
				MaxTurns: 10, SystemPrompt: "SYS", ContextWindow: 1000,
				Compaction: &Compaction{Threshold: 0.5, KeepRecentTurns: 4},
				History:    noProgressHistory(),
			}
			if tt.inbox {
				cfg.Inbox = &scriptedInbox{closeErr: ErrInboxClosed}
			}

			var transcript bytes.Buffer

			res, err := Run(context.Background(), fake, tools.NewRegistry(&countingTool{name: "probe"}), events.NewEmitter(nil, &transcript), "go", cfg)
			require.NoError(t, err)
			assert.True(t, res.Completed)
			assert.Equal(t, "done", res.Reason)
			assert.Equal(t, "ANSWER", res.Output)
			assert.Len(t, fake.requests, tt.wantCalls)

			failed, limits := 0, 0

			for _, ev := range parseEvents(t, transcript.String()) {
				switch {
				case ev.Kind == events.StateChange && ev.Data["event"] == "compaction_failed":
					failed++
				case ev.Kind == events.ContextLimit:
					limits++
				}
			}

			assert.Equal(t, 1, failed)
			assert.Zero(t, limits)
			assert.Equal(t, "old", res.Messages[1].Content, "a no-progress compaction leaves the history in place")
		})
	}
}

func TestRunCompactionRetriesAnEmptyKeptReply(t *testing.T) {
	fake := &capturingLLMSeq{responses: []llm.Response{
		{Usage: llm.Usage{PromptTokens: 900}},
		{Content: "SUMMARY"},
		{Content: "ANSWER", FinishReason: "stop"},
	}}

	var transcript bytes.Buffer

	res, err := Run(context.Background(), fake, tools.NewRegistry(), events.NewEmitter(nil, &transcript), "go", compactingConfig(2))
	require.NoError(t, err)
	assert.Equal(t, "done", res.Reason)
	assert.Equal(t, "ANSWER", res.Output)
	require.Len(t, fake.requests, 3)

	retries := 0

	for _, ev := range parseEvents(t, transcript.String()) {
		if ev.Kind == events.StateChange && ev.Data["event"] == "empty_response_retry" {
			retries++
		}
	}

	assert.Equal(t, 1, retries, "the kept empty reply goes through the empty-reply retry")

	retry := fake.requests[2].Messages
	assert.Equal(t, []string{"system", "user", "assistant", "user"}, roles(retry), "the retry carries the compacted history")
	assert.Contains(t, retry[1].Content, "SUMMARY")
}

func TestRunCompactionKeepsTheReplyAndAwaitsInput(t *testing.T) {
	tests := []struct {
		keepRecent int
		wantRoles  []string
	}{
		{keepRecent: 0, wantRoles: []string{"system", "user", "assistant", "user"}},
		{keepRecent: 2, wantRoles: []string{"system", "user", "assistant", "user", "assistant", "user"}},
	}

	for _, tt := range tests {
		t.Run(fmt.Sprintf("keepRecent_%d", tt.keepRecent), func(t *testing.T) {
			fake := &capturingLLMSeq{responses: []llm.Response{
				{Content: "FIRST ANSWER", FinishReason: "stop", Usage: llm.Usage{PromptTokens: 900}},
				{Content: "SUMMARY"},
				{Content: "SECOND ANSWER", FinishReason: "stop"},
			}}

			cfg := compactingConfig(tt.keepRecent)
			cfg.Inbox = &scriptedInbox{
				pending:  []UserMessage{{MessageID: "m1", Content: "FOLLOW UP"}},
				closeErr: ErrInboxClosed, deliverViaWaitOnly: true,
			}

			var transcript bytes.Buffer

			res, err := Run(context.Background(), fake, tools.NewRegistry(), events.NewEmitter(nil, &transcript), "go", cfg)
			require.NoError(t, err)
			assert.Equal(t, "done", res.Reason)
			assert.Equal(t, 2, res.Turns)
			require.Len(t, fake.requests, 3, "the turn, the summarize call, and the turn after the user's reply")

			next := fake.requests[2].Messages
			assert.Equal(t, tt.wantRoles, roles(next))
			assert.Contains(t, next[1].Content, "SUMMARY")
			assert.Equal(t, "FIRST ANSWER", next[len(next)-2].Content)
			assert.Equal(t, "FOLLOW UP", next[len(next)-1].Content)

			var seq []string

			for _, ev := range parseEvents(t, transcript.String()) {
				switch {
				case ev.Kind == events.ModelResponse:
					content, _ := ev.Data["content"].(string)
					seq = append(seq, content)
				case ev.Kind == events.StateChange && ev.Data["state"] == "awaiting_human":
					seq = append(seq, "awaiting_human")
				case ev.Kind == events.UserInput:
					seq = append(seq, "user_input")
				}
			}

			assert.Equal(t, []string{"FIRST ANSWER", "awaiting_human", "user_input", "SECOND ANSWER", "awaiting_human"}, seq)
		})
	}
}

func TestRunCompactionRunsTheKeptToolCalls(t *testing.T) {
	tests := []struct {
		name       string
		keepRecent int
		afterTool  bool // the crossing reply answers a tool result, not the task message
		wantRoles  []string
	}{
		{"after user keepRecent 0", 0, false, []string{"system", "user", "assistant", "tool"}},
		{"after user keepRecent 2", 2, false, []string{"system", "user", "assistant", "user", "assistant", "tool"}},
		{"after tool keepRecent 0", 0, true, []string{"system", "user", "assistant", "tool"}},
		{"after tool keepRecent 2", 2, true, []string{"system", "user", "assistant", "tool", "assistant", "tool"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			probe := &countingTool{name: "probe"}

			var responses []llm.Response

			wantCalls := 1

			if tt.afterTool {
				responses = append(responses, llm.Response{ToolCalls: []llm.ToolCall{toolCall("c0", "probe", "{}")}, Usage: llm.Usage{PromptTokens: 100}})
				wantCalls++
			}

			responses = append(responses,
				llm.Response{ToolCalls: []llm.ToolCall{toolCall("c1", "probe", "{}")}, Usage: llm.Usage{PromptTokens: 900}},
				llm.Response{Content: "SUMMARY"},
				llm.Response{Content: "done", FinishReason: "stop"},
			)
			fake := &capturingLLMSeq{responses: responses}

			res, err := Run(context.Background(), fake, tools.NewRegistry(probe), newEmitter(), "go", compactingConfig(tt.keepRecent))
			require.NoError(t, err)
			assert.Equal(t, "done", res.Reason)
			assert.Equal(t, wantCalls, probe.calls, "every tool call runs exactly once")
			require.Len(t, fake.requests, len(responses))

			next := fake.requests[len(fake.requests)-1].Messages
			assert.Equal(t, tt.wantRoles, roles(next))
			assertFirstNonSystemIsUser(t, next)
			assertWellPaired(t, next)
			assert.Contains(t, next[1].Content, "SUMMARY")

			kept := next[len(next)-2]
			require.Len(t, kept.ToolCalls, 1)
			assert.Equal(t, "c1", kept.ToolCalls[0].ID)
			assert.Equal(t, "c1", next[len(next)-1].ToolCallID)
		})
	}
}

func TestRunHardStopLetsAFinalReplyComplete(t *testing.T) {
	final := scriptedCall{resp: llm.Response{Content: "ANSWER", FinishReason: "stop", Usage: llm.Usage{PromptTokens: 900}}}
	toolTurn := scriptedCall{resp: llm.Response{ToolCalls: []llm.ToolCall{toolCall("c1", "probe", "{}")}, Usage: llm.Usage{PromptTokens: 900}}}
	summarizeDown := scriptedCall{err: errors.New("summarize down")}

	tests := []struct {
		name       string
		compaction bool
		inbox      bool
		calls      []scriptedCall
		wantCalls  int
		wantReason string
	}{
		{"final reply without compaction", false, false, []scriptedCall{final}, 1, "done"},
		{"tool-call reply without compaction", false, false, []scriptedCall{toolTurn}, 1, "context_limit"},
		{"final reply with an inbox", false, true, []scriptedCall{final}, 1, "context_limit"},
		{"final reply with compaction configured", true, false, []scriptedCall{final, summarizeDown}, 1, "done"},
		{"tool-call reply after a failed compaction", true, false, []scriptedCall{toolTurn, summarizeDown}, 2, "context_limit"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			probe := &countingTool{name: "probe"}

			cfg := Config{MaxTurns: 10, ContextWindow: 1000}
			if tt.compaction {
				cfg = compactingConfig(2)
			}

			if tt.inbox {
				cfg.Inbox = &scriptedInbox{closeErr: ErrInboxClosed}
			}

			f := &scriptedLLMWithErrors{results: tt.calls}

			var transcript bytes.Buffer

			res, err := Run(context.Background(), f, tools.NewRegistry(probe), events.NewEmitter(nil, &transcript), "go", cfg)
			require.NoError(t, err)
			assert.Equal(t, tt.wantReason, res.Reason)
			assert.Equal(t, tt.wantReason == "done", res.Completed)
			assert.Equal(t, tt.wantCalls, f.i, "no request follows the reply")
			assert.Zero(t, probe.calls, "no tool runs past the hard stop")

			limits := 0

			for _, ev := range parseEvents(t, transcript.String()) {
				if ev.Kind == events.ContextLimit {
					limits++
				}
			}

			if tt.wantReason == "done" {
				assert.Zero(t, limits)
				assert.Equal(t, "ANSWER", res.Output)
			} else {
				assert.Equal(t, 1, limits)
			}
		})
	}
}

// TestRunCompactionFailedErrorIsRedactedAndCapped places one secret across each
// of tools.HeadTail's cuts: capping before redacting would leave a fragment on
// the kept side that the literal redact pass can no longer match.
func TestRunCompactionFailedErrorIsRedactedAndCapped(t *testing.T) {
	const (
		maxBytes = 1000
		// compact's wrap around StatusError's Error text for the reply below.
		errPrefix  = "compaction summarize: llm endpoint status 502: "
		secretHead = "sk-head-straddle-compaction-A1"
		secretTail = "sk-tail-straddle-compaction-B2"
		// Bytes of each secret left on the kept side of its cut.
		survive = 8
	)

	// Mirrors tools.HeadTail's split.
	head := maxBytes * 2 / 3
	tail := maxBytes - head
	body := strings.Repeat("E", head-len(errPrefix)-survive) + secretHead +
		strings.Repeat("X", 5000) +
		secretTail + strings.Repeat("F", tail-survive)

	f := &scriptedLLMWithErrors{results: []scriptedCall{
		{resp: crossingToolTurn()},
		{err: &llm.StatusError{StatusCode: http.StatusBadGateway, Body: body}},
	}}

	cfg := compactingConfig(2)
	cfg.RedactToolOutput = redact.New([]string{secretHead, secretTail}).Apply
	cfg.ToolOutputMaxBytes = maxBytes

	var transcript bytes.Buffer

	_, err := Run(context.Background(), f, tools.NewRegistry(), events.NewEmitter(nil, &transcript), "go", cfg)
	require.NoError(t, err)

	var failed []string

	for _, ev := range parseEvents(t, transcript.String()) {
		if ev.Kind == events.StateChange && ev.Data["event"] == "compaction_failed" {
			msg, ok := ev.Data["error"].(string)
			require.True(t, ok, "compaction_failed must carry a string error")

			failed = append(failed, msg)
		}
	}

	require.Len(t, failed, 1)
	assert.NotContains(t, transcript.String(), secretHead)
	assert.NotContains(t, transcript.String(), secretTail)
	assert.NotContains(t, failed[0], secretHead[:survive], "head-cut fragment leaked: the cap ran before redaction")
	assert.NotContains(t, failed[0], secretTail[len(secretTail)-survive:], "tail-cut fragment leaked: the cap ran before redaction")
	assert.LessOrEqual(t, len(failed[0]), maxBytes+80, "the error is bounded by the cap") // marker allowance
}

func TestCompactKeepsTheStatusErrorReachable(t *testing.T) {
	msgs := append([]llm.Message{{Role: "system", Content: "SYS"}}, history20()...)
	f := &scriptedLLMWithErrors{results: []scriptedCall{{err: &llm.StatusError{StatusCode: http.StatusServiceUnavailable, Body: "busy"}}}}

	_, _, err := compact(context.Background(), f, Config{Model: "m"}, msgs, 2, newEmitter())

	var se *llm.StatusError
	require.ErrorAs(t, err, &se, "the summarize wrap keeps the status error reachable")
	assert.Equal(t, http.StatusServiceUnavailable, se.StatusCode)
	assert.Equal(t, "busy", se.Body)
}

// TestCompact_ForwardsModelsAndProvider verifies that the summarize request
// built inside compact carries Models and Provider from the Config, not just
// Model - a Models[]-only consumer with an empty cfg.Model would otherwise
// send no routing information and cause the summary call to fail silently.
func TestCompact_ForwardsModelsAndProvider(t *testing.T) {
	t.Helper()

	fake := &capturingLLMSeq{responses: []llm.Response{
		crossingToolTurn(),
		{Content: "SUMMARY"},
		{Content: "done", FinishReason: "stop"},
	}}

	var transcript bytes.Buffer

	emit := events.NewEmitter(nil, &transcript)

	cfg := Config{
		MaxTurns:      10,
		SystemPrompt:  "SYS",
		ContextWindow: 1000,
		Compaction:    &Compaction{Threshold: 0.85, KeepRecentTurns: 2},
		History:       history20(),
		Models:        []string{"m1", "m2"},
		Provider:      json.RawMessage(`{"sort":"price"}`),
	}

	_, err := Run(context.Background(), fake, tools.NewRegistry(&countingTool{name: "probe"}), emit, "go", cfg)
	require.NoError(t, err)

	// Three requests: turn-1 SendStream, compact Send, turn-2 SendStream.
	require.Len(t, fake.requests, 3)

	// The summarize request (index 1) must carry Models and Provider from Config.
	summarizeReq := fake.requests[1]
	assert.Equal(t, []string{"m1", "m2"}, summarizeReq.Models, "compact must forward Models to summarize request")
	assert.JSONEq(t, `{"sort":"price"}`, string(summarizeReq.Provider), "compact must forward Provider to summarize request")
}

// TestEffectiveCompactionThreshold covers the compaction threshold helper with
// cases spanning the floor-disabled region (small window), floor-wins region
// (medium window), threshold-wins region (large window), and the boundary at
// exactly reservedHeadroomTokens (floor==0, not applied).
func TestEffectiveCompactionThreshold(t *testing.T) {
	tests := []struct {
		window    int
		threshold float64
		want      int
		desc      string
	}{
		{1000, 0.85, 850, "small window: floor negative, not applied; threshold wins"},
		{20000, 0.85, 11808, "medium window: floor (11808) < threshold (17000); floor wins"},
		{100000, 0.85, 85000, "large window: threshold (85000) < floor (91808); threshold wins"},
		{100000, 0.5, 50000, "large window, low threshold: threshold (50000) < floor (91808); threshold wins"},
		{8192, 0.85, 6963, "window==reservedHeadroomTokens: floor==0, not applied; threshold wins"},
	}

	for _, tt := range tests {
		t.Run(tt.desc, func(t *testing.T) {
			got := effectiveCompactionThreshold(tt.window, tt.threshold)
			assert.Equal(t, tt.want, got)
			assert.Positive(t, got, "result must always be positive for any positive window")
		})
	}
}

func TestCompactForwardsImagePrefixThenDropsIt(t *testing.T) {
	msgs := []llm.Message{
		{Role: "system", Content: "SYS"},
		{Role: "user", Content: "first"},
		{Role: "assistant", Content: "a1"},
		{Role: "user", ContentParts: []llm.ContentPart{
			{Type: "text", Text: "look"},
			{Type: "image_url", ImageURL: &llm.ImageURL{URL: "data:image/png;base64,AAAA"}},
		}},
		{Role: "assistant", Content: "a2"},
		{Role: "user", Content: "recent"},
	}

	capt := &capturingLLM{}

	out, _, err := compact(context.Background(), capt, Config{Model: "m"}, msgs, 1, newEmitter())
	require.NoError(t, err)

	// (a) The summarizer received the image-bearing prefix verbatim (not stripped).
	var sawImage bool

	for _, m := range capt.last.Messages {
		for _, p := range m.ContentParts {
			if p.Type == "image_url" && p.ImageURL != nil && p.ImageURL.URL == "data:image/png;base64,AAAA" {
				sawImage = true
			}
		}
	}

	assert.True(t, sawImage, "summarizer must receive the image-bearing prefix verbatim")

	// (b) The prefix (incl. the image message) is replaced by one text summary;
	// images do not persist past compaction. Result: system + summary + 1 kept.
	require.Len(t, out, 3)
	assert.Equal(t, "SYS", out[0].Content)
	assert.Equal(t, "user", out[1].Role)
	assert.Contains(t, out[1].Content, "[Earlier conversation, summarized]")
	assertFirstNonSystemIsUser(t, out)
	assert.Equal(t, "recent", out[2].Content)

	assert.Empty(t, out[1].ContentParts, "summary message must not carry image parts")
}

// assertFirstNonSystemIsUser skips leading system messages and requires the
// next message's Role to be "user". It reports the full role sequence on
// failure (e.g. [system system assistant tool tool tool]).
func assertFirstNonSystemIsUser(t *testing.T, msgs []llm.Message) {
	t.Helper()

	roles := make([]string, 0, len(msgs))
	first := ""

	for _, m := range msgs {
		roles = append(roles, m.Role)

		if first == "" && m.Role != "system" {
			first = m.Role
		}
	}

	require.Equal(t, "user", first, "first non-system message role must be user; got roles %v", roles)
}

// assertWellPaired verifies that msgs contains no broken tool-call / tool-result
// grouping:
//   - every "tool" message has a preceding "assistant" message whose ToolCalls
//     list contains the matching ToolCallID
//   - every "assistant" message with ToolCalls is immediately followed by
//     contiguous "tool" result messages that cover every ID it issued
func assertWellPaired(t *testing.T, msgs []llm.Message) {
	t.Helper()

	for i, m := range msgs {
		if m.Role == "tool" {
			var owned bool

			for j := range i {
				for _, tc := range msgs[j].ToolCalls {
					if tc.ID == m.ToolCallID {
						owned = true

						break
					}
				}

				if owned {
					break
				}
			}

			assert.True(t, owned,
				"tool message at index %d (ToolCallID=%q) has no preceding assistant that owns it", i, m.ToolCallID)
		}

		if m.Role == "assistant" && len(m.ToolCalls) > 0 {
			needed := make(map[string]bool, len(m.ToolCalls))

			for _, tc := range m.ToolCalls {
				needed[tc.ID] = false
			}

			for j := i + 1; j < len(msgs) && msgs[j].Role == "tool"; j++ {
				needed[msgs[j].ToolCallID] = true
			}

			for id, found := range needed {
				assert.True(t, found,
					"assistant at index %d issued tool call %q but no subsequent tool result found", i, id)
			}
		}
	}
}

// TestCompactPreservesToolGroups verifies that compact() never splits a
// tool-call / tool-result group across the compaction boundary: every
// "tool" message in the kept slice must be owned by an "assistant" that is
// also in the kept slice, and the summarizer must not receive an older prefix
// that ends with an unanswered assistant tool_calls block.
func TestCompactPreservesToolGroups(t *testing.T) {
	t.Helper()

	toolCalls := []llm.ToolCall{
		{ID: "call-X", Type: "function", Function: llm.FunctionCall{Name: "foo", Arguments: "{}"}},
		{ID: "call-Y", Type: "function", Function: llm.FunctionCall{Name: "bar", Arguments: "{}"}},
		{ID: "call-Z", Type: "function", Function: llm.FunctionCall{Name: "baz", Arguments: "{}"}},
	}

	base := []llm.Message{
		{Role: "system", Content: "SYS"},
		{Role: "user", Content: "u1"},
		{Role: "assistant", Content: "a1"},
		{Role: "user", Content: "u2"},
		{Role: "assistant", ToolCalls: toolCalls},
		{Role: "tool", ToolCallID: "call-X", Content: "result-X"},
		{Role: "tool", ToolCallID: "call-Y", Content: "result-Y"},
		{Role: "tool", ToolCallID: "call-Z", Content: "result-Z"},
	}

	withTrailing := append(append([]llm.Message(nil), base...), llm.Message{Role: "user", Content: "u3"})

	tests := []struct {
		name       string
		msgs       []llm.Message
		keepRecent int
	}{
		{
			name:       "keepRecent splits mid-group no trailing message",
			msgs:       base,
			keepRecent: 2,
		},
		{
			name:       "keepRecent splits mid-group with trailing user message",
			msgs:       withTrailing,
			keepRecent: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			capt := &capturingLLM{}

			out, _, err := compact(context.Background(), capt, Config{Model: "m"}, tt.msgs, tt.keepRecent, newEmitter())
			require.NoError(t, err)

			assertFirstNonSystemIsUser(t, out)
			assertWellPaired(t, out)

			// The older slice sent to the summarizer must not end with an assistant
			// that has unanswered tool_calls - that would produce a malformed request
			// and an HTTP 400 from the provider.
			if len(capt.last.Messages) > 1 {
				lastOlder := capt.last.Messages[len(capt.last.Messages)-1]
				assert.False(t, lastOlder.Role == "assistant" && len(lastOlder.ToolCalls) > 0,
					"summarizer received an older prefix ending with unanswered assistant tool_calls")
			}
		})
	}
}

func TestCompactKeepsRecentImageVerbatim(t *testing.T) {
	imgMsg := llm.Message{Role: "user", ContentParts: []llm.ContentPart{
		{Type: "text", Text: "look"},
		{Type: "image_url", ImageURL: &llm.ImageURL{URL: "data:image/png;base64,BBBB"}},
	}}
	msgs := []llm.Message{
		{Role: "system", Content: "SYS"},
		{Role: "user", Content: "old1"},
		{Role: "assistant", Content: "a1"},
		{Role: "user", Content: "old2"},
		{Role: "assistant", Content: "a2"},
		imgMsg,
	}

	out, _, err := compact(context.Background(), &capturingLLM{}, Config{Model: "m"}, msgs, 1, newEmitter())
	require.NoError(t, err)

	assertFirstNonSystemIsUser(t, out)
	assert.Equal(t, "user", out[1].Role)

	last := out[len(out)-1]
	require.Len(t, last.ContentParts, 2)
	require.NotNil(t, last.ContentParts[1].ImageURL)
	assert.Equal(t, "data:image/png;base64,BBBB", last.ContentParts[1].ImageURL.URL)
}

// TestCompactTailStartsWithUser verifies that the first non-system message
// after compaction is a user message, satisfying the Anthropic-shaped endpoint
// requirement that messages[0].role must be user.
func TestCompactTailStartsWithUser(t *testing.T) {
	toolCalls := []llm.ToolCall{
		{ID: "call-X", Type: "function", Function: llm.FunctionCall{Name: "foo", Arguments: "{}"}},
		{ID: "call-Y", Type: "function", Function: llm.FunctionCall{Name: "bar", Arguments: "{}"}},
		{ID: "call-Z", Type: "function", Function: llm.FunctionCall{Name: "baz", Arguments: "{}"}},
	}

	base := []llm.Message{
		{Role: "system", Content: "SYS"},
		{Role: "user", Content: "u1"},
		{Role: "assistant", Content: "a1"},
		{Role: "user", Content: "u2"},
		{Role: "assistant", ToolCalls: toolCalls},
		{Role: "tool", ToolCallID: "call-X", Content: "result-X"},
		{Role: "tool", ToolCallID: "call-Y", Content: "result-Y"},
		{Role: "tool", ToolCallID: "call-Z", Content: "result-Z"},
	}

	withTrailing := append(append([]llm.Message(nil), base...), llm.Message{Role: "user", Content: "u3"})

	// tailStart is the index in msgs where the retained tail begins. Every
	// keepRecent that lands inside the tool group snaps back to the assistant
	// that issued it (index 4); only a trailing user message is kept alone.
	tests := []struct {
		name       string
		msgs       []llm.Message
		keepRecent int
		tailStart  int
	}{
		{name: "base_keepRecent_1", msgs: base, keepRecent: 1, tailStart: 4},
		{name: "base_keepRecent_2", msgs: base, keepRecent: 2, tailStart: 4},
		{name: "base_keepRecent_3", msgs: base, keepRecent: 3, tailStart: 4},
		{name: "base_keepRecent_4", msgs: base, keepRecent: 4, tailStart: 4},
		{name: "withTrailing_keepRecent_1", msgs: withTrailing, keepRecent: 1, tailStart: 8},
		{name: "withTrailing_keepRecent_2", msgs: withTrailing, keepRecent: 2, tailStart: 4},
		{name: "withTrailing_keepRecent_3", msgs: withTrailing, keepRecent: 3, tailStart: 4},
		{name: "withTrailing_keepRecent_4", msgs: withTrailing, keepRecent: 4, tailStart: 4},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			capt := &capturingLLM{}

			out, _, err := compact(context.Background(), capt, Config{Model: "m"}, tt.msgs, tt.keepRecent, newEmitter())
			require.NoError(t, err)

			require.Equal(t, "system", out[0].Role, "first output message must be the system prompt")
			require.Equal(t, "user", out[1].Role, "summary message must have role 'user'")
			require.True(t, strings.HasPrefix(out[1].Content, "[Earlier conversation, summarized]"),
				"summary message content must start with '[Earlier conversation, summarized]'")

			assertFirstNonSystemIsUser(t, out)
			assertWellPaired(t, out)

			assert.Equal(t, tt.msgs[tt.tailStart:], out[2:], "retained tail must be kept verbatim with the tool group intact")
		})
	}
}

func replayHistory20() []llm.Message {
	history := make([]llm.Message, 20)
	for i := range history {
		if i%2 == 0 {
			history[i] = llm.Message{Role: "user", Content: fmt.Sprintf("user %d", i)}
		} else {
			history[i] = llm.Message{
				Role:             "assistant",
				Content:          fmt.Sprintf("assistant %d", i),
				ReasoningDetails: json.RawMessage(fmt.Sprintf(`[{"type":"reasoning.text","text":"t%d","index":0}]`, i)),
			}
		}
	}

	return history
}

func TestCompactStripsReasoningFromSummarizerInputOnly(t *testing.T) {
	fake := &capturingLLMSeq{responses: []llm.Response{
		crossingToolTurn(),
		{Content: "SUMMARY"},
		{Content: "done", FinishReason: "stop"},
	}}

	_, err := Run(context.Background(), fake, tools.NewRegistry(&countingTool{name: "probe"}), newEmitter(), "go", Config{
		MaxTurns: 10, SystemPrompt: "SYS", ContextWindow: 1000,
		Compaction: &Compaction{Threshold: 0.85, KeepRecentTurns: 3},
		History:    replayHistory20(), ReplayReasoning: true,
	})
	require.NoError(t, err)
	require.Len(t, fake.requests, 3)

	for _, m := range fake.requests[1].Messages {
		assert.Nil(t, m.ReasoningDetails, "the summarizer never sees reasoning")
	}

	var tail *llm.Message

	for i, m := range fake.requests[2].Messages {
		if m.Content == "assistant 19" {
			tail = &fake.requests[2].Messages[i]
		}
	}

	require.NotNil(t, tail, "assistant 19 is in the kept tail")
	assert.JSONEq(t, `[{"type":"reasoning.text","text":"t19","index":0}]`, string(tail.ReasoningDetails))
}

func TestCompactFailureKeepsReasoningAndStopsOnProviderCount(t *testing.T) {
	fake := &scriptedLLMWithErrors{results: []scriptedCall{
		{resp: crossingToolTurn()},
		{err: errors.New("summarize down")},
	}}

	res, err := Run(context.Background(), fake, tools.NewRegistry(), newEmitter(), "go", Config{
		MaxTurns: 10, SystemPrompt: "SYS", ContextWindow: 1000,
		Compaction: &Compaction{Threshold: 0.85, KeepRecentTurns: 3},
		History:    replayHistory20(), ReplayReasoning: true,
	})
	require.NoError(t, err)
	assert.Equal(t, "context_limit", res.Reason)

	var withDetails int

	for _, m := range res.Messages {
		if m.Role == "assistant" && len(m.ReasoningDetails) > 0 {
			withDetails++
		}
	}

	assert.Equal(t, 10, withDetails, "a failed compaction leaves every seeded block in place")
}

func TestCompactionRequestIsNeverPinned(t *testing.T) {
	history := replayHistory20()
	for i := range history {
		if history[i].Role == "assistant" {
			history[i].ReasoningProvider = "Wafer"
		}
	}

	fake := &capturingLLMSeq{responses: []llm.Response{
		crossingToolTurn(),
		{Content: "SUMMARY"},
		{Content: "done", FinishReason: "stop"},
	}}

	_, err := Run(context.Background(), fake, tools.NewRegistry(&countingTool{name: "probe"}), newEmitter(), "go", Config{
		MaxTurns: 10, SystemPrompt: "SYS", ContextWindow: 1000, Provider: json.RawMessage(`{"sort":"price"}`),
		Compaction: &Compaction{Threshold: 0.85, KeepRecentTurns: 3},
		History:    history, ReplayReasoning: true,
	})
	require.NoError(t, err)
	require.Len(t, fake.requests, 3)
	assert.JSONEq(t, `{"order":["Wafer"],"sort":"price"}`, string(fake.requests[0].Provider))
	assert.JSONEq(t, `{"sort":"price"}`, string(fake.requests[1].Provider), "the summarizer routes as configured")
	assert.JSONEq(t, `{"order":["Wafer"],"sort":"price"}`, string(fake.requests[2].Provider), "the kept tail pins again")
}
