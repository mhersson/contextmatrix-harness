package harness

import (
	"context"
	"encoding/json"
	"sync"

	"github.com/mhersson/contextmatrix-harness/events"
	"github.com/mhersson/contextmatrix-harness/llm"
	"github.com/mhersson/contextmatrix-harness/tools"
)

// SubagentSpec describes one child agent.
type SubagentSpec struct {
	Role          string
	Prompt        string
	Model         string // optional per-spec override; falls back to opts.DefaultModel
	MaxTurns      int    // 0 -> harness default
	ContextWindow int    // child model's context window; 0 disables the check
	WrapUpTurns   int    // 0 -> no wrap-up nudge (previous behavior)
	WrapUpMessage string // injected when WrapUpTurns remain
}

// SubagentResult is one child's structured result.
type SubagentResult struct {
	Role   string
	Output string
	Result Result
	Err    error
}

// SubagentOpts configures a fan-out.
type SubagentOpts struct {
	DefaultModel       string              // model for specs that don't override
	ToolOutputMaxBytes int                 // 0 disables the child tool-result size cap
	RedactToolOutput   func(string) string // nil = identity; scrubs child tool output
	ExtraReadOnlyTools []tools.Tool        // appended to tools.ReadOnlyTools; must be side-effect-free (e.g. the Skill tool)
	Provider           json.RawMessage     // OpenRouter provider routing inherited by children; nil = default selection
	Reasoning          json.RawMessage     // reasoning config inherited by children; nil = none
	ReplayReasoning    bool                // reasoning replay inherited by children; see Config.ReplayReasoning
}

// SpawnSubagents runs specs as parallel child Harness.Run calls over the shared
// workspace root. Children get tools.ReadOnlyTools(root) plus
// opts.ExtraReadOnlyTools; the harness does not check the extras, so the caller
// must pass only side-effect-free tools or parallel fan-out can corrupt the
// workspace. Results preserve spec order. Children share emit (thread-safe), so
// their activity interleaves into the same transcript.
func SpawnSubagents(ctx context.Context, client llm.LLM, root string, emit *events.Emitter, specs []SubagentSpec, opts SubagentOpts) ([]SubagentResult, error) {
	if len(specs) == 0 {
		return nil, nil
	}

	reg := tools.NewRegistry(append(tools.ReadOnlyTools(root), opts.ExtraReadOnlyTools...)...)

	emit.Emit(events.StateChange, map[string]any{"spawn_subagents": len(specs)})

	results := make([]SubagentResult, len(specs))

	var wg sync.WaitGroup
	for i, spec := range specs {
		wg.Add(1)

		go func(i int, spec SubagentSpec) {
			defer wg.Done()

			model := spec.Model
			if model == "" {
				model = opts.DefaultModel
			}

			res, err := Run(ctx, client, reg, emit, spec.Prompt,
				Config{
					Model:              model,
					Provider:           opts.Provider,
					Reasoning:          opts.Reasoning,
					ReplayReasoning:    opts.ReplayReasoning,
					MaxTurns:           spec.MaxTurns,
					ContextWindow:      spec.ContextWindow,
					ToolOutputMaxBytes: opts.ToolOutputMaxBytes,
					RedactToolOutput:   opts.RedactToolOutput,
					WrapUpTurns:        spec.WrapUpTurns,
					WrapUpMessage:      spec.WrapUpMessage,
				})
			results[i] = SubagentResult{Role: spec.Role, Output: res.Output, Result: res, Err: err}
		}(i, spec)
	}

	wg.Wait()

	return results, nil
}
