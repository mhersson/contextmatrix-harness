package harness

import (
	"bytes"
	"context"
	"testing"

	"github.com/mhersson/contextmatrix-harness/events"
	"github.com/mhersson/contextmatrix-harness/llm"
	"github.com/mhersson/contextmatrix-harness/tools"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRunEmitsProviderOnlyWhenReported(t *testing.T) {
	for _, provider := range []string{"DeepInfra", ""} {
		t.Run("provider="+provider, func(t *testing.T) {
			f := &fakeLLM{responses: []llm.Response{{Content: "done", Provider: provider}}}

			var transcript bytes.Buffer

			_, err := Run(context.Background(), f, tools.NewRegistry(), events.NewEmitter(nil, &transcript), "task", Config{})
			require.NoError(t, err)

			seen := 0

			for _, ev := range parseEvents(t, transcript.String()) {
				if ev.Kind != events.ModelResponse && ev.Kind != events.UsageKind {
					continue
				}

				seen++
				got, ok := ev.Data["provider"]
				assert.Equal(t, provider != "", ok, "%s provider key", ev.Kind)

				if ok {
					assert.Equal(t, provider, got)
				}
			}

			assert.Equal(t, 2, seen)
		})
	}
}
