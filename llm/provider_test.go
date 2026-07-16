package llm

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestProviderRaw(t *testing.T) {
	empty, err := Provider{}.Raw()
	require.NoError(t, err)
	assert.Nil(t, empty)

	rp, sort := true, "price"
	raw, err := Provider{RequireParameters: &rp, Order: []string{"openai", "anthropic"}, Sort: &sort}.Raw()
	require.NoError(t, err)
	assert.JSONEq(t, `{"require_parameters":true,"order":["openai","anthropic"],"sort":"price"}`, string(raw))
}

func TestReasoningRaw(t *testing.T) {
	empty, err := Reasoning{}.Raw()
	require.NoError(t, err)
	assert.Nil(t, empty)

	eff := "high"
	raw, err := Reasoning{Effort: &eff}.Raw()
	require.NoError(t, err)
	assert.JSONEq(t, `{"effort":"high"}`, string(raw))
}
