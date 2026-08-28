package events

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEmitterSeqAndDualOutput(t *testing.T) {
	var human, transcript bytes.Buffer

	e := NewEmitter(&human, &transcript)
	e.now = func() time.Time { return time.Unix(0, 0).UTC() }

	e.Emit(ModelRequest, map[string]any{"model": "x"})
	e.Emit(ToolCallKind, map[string]any{"name": "read"})

	lines := strings.Split(strings.TrimSpace(transcript.String()), "\n")
	require.Len(t, lines, 2)

	var ev1 Event
	require.NoError(t, json.Unmarshal([]byte(lines[0]), &ev1))
	assert.Equal(t, 1, ev1.Seq)
	assert.Equal(t, ModelRequest, ev1.Kind)

	var ev2 Event
	require.NoError(t, json.Unmarshal([]byte(lines[1]), &ev2))
	assert.Equal(t, 2, ev2.Seq)
	assert.Equal(t, ToolCallKind, ev2.Kind)

	assert.Contains(t, human.String(), "model_request")
	assert.Contains(t, human.String(), "tool_call")
}

func TestThinkingKindValue(t *testing.T) {
	require.Equal(t, "thinking", string(Thinking))
}

// TestEmitterByteIdenticalWithoutOption pins the exact transcript line an
// emitter constructed without WithEnvelopeFields produces. An emitter with
// no static fields must never change this shape.
func TestEmitterByteIdenticalWithoutOption(t *testing.T) {
	var human, transcript bytes.Buffer

	e := NewEmitter(&human, &transcript)
	e.now = func() time.Time { return time.Unix(0, 0).UTC() }

	e.Emit(ModelRequest, map[string]any{"model": "x"})

	got := strings.TrimSpace(transcript.String())
	want := `{"seq":1,"kind":"model_request","time":"1970-01-01T00:00:00Z","data":{"model":"x"}}`
	assert.Equal(t, want, got)
}

func TestEmitterStaticFieldsStampedOnAllKinds(t *testing.T) {
	var human, transcript bytes.Buffer

	e := NewEmitter(&human, &transcript, WithEnvelopeFields(map[string]any{"attempt": 2}))
	e.now = func() time.Time { return time.Unix(0, 0).UTC() }

	e.Emit(ModelRequest, map[string]any{"model": "x"})
	e.Emit(UsageKind, map[string]any{"tokens": 10})
	e.Emit(StateChange, map[string]any{"to": "done"})

	lines := strings.Split(strings.TrimSpace(transcript.String()), "\n")
	require.Len(t, lines, 3)

	for _, line := range lines {
		var m map[string]any
		require.NoError(t, json.Unmarshal([]byte(line), &m))
		assert.Equal(t, 2, int(m["attempt"].(float64))) //nolint:forcetypeassert
	}
}

// TestEmitterStaticFieldsNeverOverrideEnvelopeKeys documents the collision
// decision: caller-provided static fields are refused when they collide
// with an envelope-owned key - the envelope's own value always wins.
func TestEmitterStaticFieldsNeverOverrideEnvelopeKeys(t *testing.T) {
	var human, transcript bytes.Buffer

	e := NewEmitter(&human, &transcript, WithEnvelopeFields(map[string]any{
		"seq":     999,
		"kind":    "bogus",
		"time":    "bogus",
		"data":    "bogus",
		"attempt": 2,
	}))
	e.now = func() time.Time { return time.Unix(0, 0).UTC() }

	e.Emit(ModelRequest, map[string]any{"model": "x"})

	var m map[string]any
	require.NoError(t, json.Unmarshal(transcript.Bytes(), &m))

	assert.Equal(t, 1, int(m["seq"].(float64)), "real seq must win over a colliding static field") //nolint:forcetypeassert
	assert.Equal(t, "model_request", m["kind"], "real kind must win over a colliding static field")
	assert.Equal(t, "1970-01-01T00:00:00Z", m["time"], "real time must win over a colliding static field")
	assert.Equal(t, map[string]any{"model": "x"}, m["data"], "real data must win over a colliding static field")
	assert.Equal(t, 2, int(m["attempt"].(float64)), "non-colliding static field is still stamped") //nolint:forcetypeassert
}

// TestEmitterDataCollisionWithNoRealData covers the case where the event
// itself carries no data (so envelope() never reaches its len(ev.Data) > 0
// branch to overwrite the colliding key): a static field literally named
// "data" must still not survive into the emitted envelope.
func TestEmitterDataCollisionWithNoRealData(t *testing.T) {
	var human, transcript bytes.Buffer

	e := NewEmitter(&human, &transcript, WithEnvelopeFields(map[string]any{"data": "bogus"}))
	e.now = func() time.Time { return time.Unix(0, 0).UTC() }

	e.Emit(StateChange, nil)

	var m map[string]any
	require.NoError(t, json.Unmarshal(transcript.Bytes(), &m))

	_, present := m["data"]
	assert.False(t, present, "a colliding static \"data\" field must not survive an event with no real data")
}

// TestNewEmitterWarnsOnceOnCollidingStaticFields covers the fix-round
// finding: a caller whose static field silently loses to an envelope-owned
// key needs a visible signal, or they debug blind. The warning fires once,
// at construction (the set of static keys never changes afterward), not
// once per Emit call.
func TestNewEmitterWarnsOnceOnCollidingStaticFields(t *testing.T) {
	var human, transcript bytes.Buffer

	e := NewEmitter(&human, &transcript, WithEnvelopeFields(map[string]any{
		"seq":     999,
		"data":    "bogus",
		"attempt": 2,
	}))
	e.now = func() time.Time { return time.Unix(0, 0).UTC() }

	warnings := human.String()
	assert.Contains(t, warnings, "seq", "warning must name the colliding key")
	assert.Contains(t, warnings, "data", "warning must name the colliding key")
	assert.NotContains(t, warnings, "attempt", "a non-colliding field must not be warned about")

	e.Emit(ModelRequest, map[string]any{"model": "x"})
	e.Emit(ModelRequest, map[string]any{"model": "y"})
	e.Emit(ModelRequest, map[string]any{"model": "z"})

	assert.Equal(t, 1, strings.Count(human.String(), "seq"),
		"the seq collision warning must be written once, not once per Emit call")
}
