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

	// Force the terminal call instead of merely offering it: one terminal tool
	// gets a named function choice, several get "required". Instruction-only
	// grace calls measurably fail on weaker models - one production run answered
	// the "ONLY action available" message with yet another exploration call.
	var choice json.RawMessage

	if len(termNames) == 1 {
		name, _ := json.Marshal(termNames[0])
		choice = json.RawMessage(fmt.Sprintf(`{"type":"function","function":{"name":%s}}`, name))
	} else {
		choice = json.RawMessage(`"required"`)
	}

	resp, err := sendTurn(ctx, client, emit, cfg, msgs, termSchemas, choice, res)
	if err != nil && strings.Contains(err.Error(), "llm endpoint status 400") {
		// Some OpenAI-compatible gateways reject tool_choice for some models; the
		// client surfaces a 400 as a terminal error, so retry once without the
		// forcing and fall back to the instruction-only contract.
		resp, err = sendTurn(ctx, client, emit, cfg, msgs, termSchemas, nil, res)
	}

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

		res.ToolCallCount++

		emit.Emit(events.ToolCallKind, map[string]any{"id": tc.ID, "name": tc.Function.Name, "raw_args": redactStr(cfg, tc.Function.Arguments)})

		out, execErr := tool.Execute(ctx, args)
		if execErr != nil {
			// Answer the call even though it failed: an unpaired tool_call is
			// the asymmetry these events exist to remove.
			res.ToolCallFailures++

			em := fmt.Sprintf("tool error: %v", execErr)
			if cfg.RedactToolOutput != nil {
				em = cfg.RedactToolOutput(em)
			}

			em = tools.HeadTail(em, cfg.ToolOutputMaxBytes)
			emit.Emit(events.ToolResult, map[string]any{"id": tc.ID, "error": em})

			continue
		}

		text := out.Text
		if cfg.RedactToolOutput != nil {
			text = cfg.RedactToolOutput(text)
		}

		text = tools.HeadTail(text, cfg.ToolOutputMaxBytes)
		emit.Emit(events.ToolResult, map[string]any{"id": tc.ID, "output_len": len(text)})

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
// choice, when non-nil, is passed through as the request's tool_choice.
func sendTurn(ctx context.Context, client llm.LLM, emit *events.Emitter, cfg Config, msgs []llm.Message, toolSchemas []llm.Tool, choice json.RawMessage, res *Result) (llm.Response, error) {
	req := llm.Request{
		Model:      cfg.Model,
		Models:     cfg.Models,
		Provider:   cfg.Provider,
		Reasoning:  cfg.Reasoning,
		Messages:   msgs,
		Tools:      toolSchemas,
		ToolChoice: choice,
	}
	emit.Emit(events.ModelRequest, map[string]any{"turn": res.Turns, "model": cfg.Model, "messages": len(msgs)})

	resp, err := client.SendStream(ctx, req, nil)
	if err != nil {
		return resp, err
	}

	prompt, cacheRead, cacheCreation := resp.Usage.Buckets()

	res.TotalCostUSD += resp.Usage.Cost
	res.PromptTokens += int64(prompt)
	res.CompletionTokens += int64(resp.Usage.CompletionTokens)
	res.CacheReadTokens += int64(cacheRead)
	res.CacheCreationTokens += int64(cacheCreation)

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
		"prompt_tokens": prompt, "completion_tokens": resp.Usage.CompletionTokens,
		"cost_usd": resp.Usage.Cost, "model": cfg.Model,
		"cache_read_tokens": cacheRead, "cache_creation_tokens": cacheCreation,
		"wire_prompt_tokens": resp.Usage.PromptTokens,
	})

	return resp, nil
}
