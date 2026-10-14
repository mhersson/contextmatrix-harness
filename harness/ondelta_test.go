package harness

import (
	"context"
	"fmt"
	"testing"

	"github.com/mhersson/contextmatrix-harness/llm"
	"github.com/mhersson/contextmatrix-harness/tools"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// deltaLLM streams one reasoning and one content fragment per call, then
// plays back the wrapped burnLLM script.
type deltaLLM struct{ *burnLLM }

func (d *deltaLLM) SendStream(ctx context.Context, req llm.Request, onDelta func(llm.Delta)) (llm.Response, error) {
	n := len(d.requests) + 1
	if onDelta != nil {
		onDelta(llm.Delta{Reasoning: fmt.Sprintf("think%d", n)})
		onDelta(llm.Delta{Content: fmt.Sprintf("say%d", n)})
	}

	return d.burnLLM.SendStream(ctx, req, onDelta)
}

func TestConfigOnDeltaReceivesStreamedFragments(t *testing.T) {
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()))
	d := &deltaLLM{&burnLLM{responses: []llm.Response{{Content: "done", FinishReason: "stop"}}}}

	var got []llm.Delta

	res, err := Run(context.Background(), d, reg, newEmitter(), "task", Config{
		MaxTurns: 3,
		OnDelta:  func(dl llm.Delta) { got = append(got, dl) },
	})
	require.NoError(t, err)
	assert.True(t, res.Completed)
	assert.Equal(t, []llm.Delta{{Reasoning: "think1"}, {Content: "say1"}}, got)
}

func TestConfigOnDeltaReceivesGraceTurnFragments(t *testing.T) {
	fin := &graceFinishTool{}
	reg := tools.NewRegistry(tools.NewReadTool(t.TempDir()), fin)
	d := &deltaLLM{&burnLLM{responses: []llm.Response{
		{ToolCalls: []llm.ToolCall{toolCall("1", "read", `{"path":"m1"}`)}},
		{ToolCalls: []llm.ToolCall{toolCall("2", "read", `{"path":"m2"}`)}},
		{ToolCalls: []llm.ToolCall{toolCall("3", "finish", `{}`)}},
	}}}

	var got []llm.Delta

	res, err := Run(context.Background(), d, reg, newEmitter(), "task", Config{
		MaxTurns: 2, GraceTurn: true,
		OnDelta: func(dl llm.Delta) { got = append(got, dl) },
	})
	require.NoError(t, err)
	assert.True(t, res.Completed)
	require.Len(t, got, 6)
	assert.Equal(t, llm.Delta{Content: "say3"}, got[5], "grace call fragments reach OnDelta")
}
