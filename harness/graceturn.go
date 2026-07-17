package harness

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/mhersson/contextmatrix-harness/events"
	"github.com/mhersson/contextmatrix-harness/llm"
	"github.com/mhersson/contextmatrix-harness/tools"
)

// graceFinish makes the single grace call: terminal-only toolset, one synthetic
// user message, one model call. Returns true only when a Terminal tool executed
// successfully - the caller then returns res as a completed run. Mirrors the
// main loop's usage accounting and emits StateChange events so transcripts show
// the grace call explicitly. res.Turns is never incremented; the grace call is
// evented, not counted.
func graceFinish(ctx context.Context, client llm.LLM, reg *tools.Registry, emit *events.Emitter, cfg Config, msgs []llm.Message, res *Result) bool {
	termSchemas, termNames := terminalSchemas(reg)
	if len(termSchemas) == 0 {
		return false
	}

	emit.Emit(events.StateChange, map[string]any{"event": "grace_turn"})

	msgs = append(msgs, llm.Message{Role: "user", Content: fmt.Sprintf(
		"Turn limit reached. This is a final grace call: the ONLY action available is the %s tool. Call it NOW with your best final arguments; any other response discards the run's completed work.",
		strings.Join(termNames, "/"))})

	resp, err := sendTurn(ctx, client, emit, cfg, msgs, termSchemas, res)
	if err != nil {
		return false
	}

	for _, tc := range resp.ToolCalls {
		tool, ok := reg.Get(tc.Function.Name)
		if !ok {
			continue
		}

		term, isTerminal := tool.(tools.Terminal)
		if !isTerminal || !term.Terminal() {
			continue
		}

		args, perr := parseArgs(tc.Function.Arguments)
		if perr != nil {
			continue
		}

		if _, execErr := tool.Execute(ctx, args); execErr != nil {
			continue
		}

		res.Completed = true
		res.Reason = "done"
		res.CompletionArgs = json.RawMessage(normalizeArgs(tc.Function.Arguments))
		emit.Emit(events.StateChange, map[string]any{"stop": "done", "turns": res.Turns, "via": "grace_turn"})

		return true
	}

	return false
}

// terminalSchemas returns the schemas and names of the registry's Terminal
// tools, in registration order.
func terminalSchemas(reg *tools.Registry) ([]llm.Tool, []string) {
	var (
		schemas []llm.Tool
		names   []string
	)

	for _, tool := range reg.All() {
		if term, ok := tool.(tools.Terminal); ok && term.Terminal() {
			schemas = append(schemas, tool.Schema())
			names = append(names, tool.Name())
		}
	}

	return schemas, names
}

// sendTurn issues one model call with the given tool schemas, mirroring the main
// loop's request construction and usage accounting. res.Turns is read-only here:
// the caller decides whether the call counts (the grace call does not). On a
// transport error the caller declines; the raw error is returned unwrapped.
func sendTurn(ctx context.Context, client llm.LLM, emit *events.Emitter, cfg Config, msgs []llm.Message, toolSchemas []llm.Tool, res *Result) (llm.Response, error) {
	req := llm.Request{
		Model:     cfg.Model,
		Models:    cfg.Models,
		Provider:  cfg.Provider,
		Reasoning: cfg.Reasoning,
		Messages:  msgs,
		Tools:     toolSchemas,
	}
	emit.Emit(events.ModelRequest, map[string]any{"turn": res.Turns, "model": cfg.Model, "messages": len(msgs)})

	resp, err := client.SendStream(ctx, req, nil)
	if err != nil {
		return resp, err
	}

	res.TotalCostUSD += resp.Usage.Cost
	res.PromptTokens += int64(resp.Usage.PromptTokens)
	res.CompletionTokens += int64(resp.Usage.CompletionTokens)

	if resp.Model != "" {
		res.ModelUsed = resp.Model
	}

	res.Output = resp.Content

	if resp.Reasoning != "" {
		emit.Emit(events.Thinking, map[string]any{"turn": res.Turns, "content": redactStr(cfg, resp.Reasoning)})
	}

	emit.Emit(events.ModelResponse, map[string]any{
		"turn": res.Turns, "finish_reason": resp.FinishReason,
		"tool_calls": len(resp.ToolCalls), "content_len": len(resp.Content),
		"content": redactStr(cfg, resp.Content), "model": cfg.Model,
	})
	emit.Emit(events.UsageKind, map[string]any{
		"prompt_tokens": resp.Usage.PromptTokens, "completion_tokens": resp.Usage.CompletionTokens,
		"cost_usd": resp.Usage.Cost, "model": cfg.Model,
	})

	return resp, nil
}
