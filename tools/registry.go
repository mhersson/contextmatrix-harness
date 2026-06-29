// Package tools provides the harness tool registry and a small set of
// filesystem/shell tools, each jailed to a workspace root.
package tools

import (
	"context"

	"github.com/mhersson/contextmatrix-harness/llm"
)

// Result is the outcome of a Tool.Execute call: the textual output every tool
// produces, plus optional inline images (OpenAI image_url data URLs). Images is
// nil for the filesystem/shell tools; only MCP-bridged tools that surface
// ImageContent populate it.
type Result struct {
	Text   string
	Images []llm.ImageURL
}

type Tool interface {
	Name() string
	Schema() llm.Tool
	Execute(ctx context.Context, args map[string]any) (Result, error)
}

type Registry struct {
	tools map[string]Tool
	order []string
}

func NewRegistry(ts ...Tool) *Registry {
	r := &Registry{tools: map[string]Tool{}}
	for _, t := range ts {
		if _, dup := r.tools[t.Name()]; !dup {
			r.order = append(r.order, t.Name())
		}

		r.tools[t.Name()] = t
	}

	return r
}

func (r *Registry) Get(name string) (Tool, bool) {
	t, ok := r.tools[name]

	return t, ok
}

func (r *Registry) Schemas() []llm.Tool {
	out := make([]llm.Tool, 0, len(r.order))
	for _, name := range r.order {
		out = append(out, r.tools[name].Schema())
	}

	return out
}
