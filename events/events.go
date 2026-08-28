// Package events defines the harness event stream: a sequence of typed events
// written human-readably to stdout and as JSON lines to a transcript.
package events

import (
	"encoding/json"
	"fmt"
	"io"
	"sync"
	"time"
)

type Kind string

const (
	ModelRequest  Kind = "model_request"
	ModelResponse Kind = "model_response"
	ToolCallKind  Kind = "tool_call"
	ToolResult    Kind = "tool_result"
	UsageKind     Kind = "usage"
	StateChange   Kind = "state_change"
	ContextLimit  Kind = "context_limit"
	Verification  Kind = "verification"
	ErrorKind     Kind = "error"
	ToolRepair    Kind = "tool_repair"
	UserInput     Kind = "user_input"
	Thinking      Kind = "thinking"
)

// Event is one entry in the stream.
type Event struct {
	Seq  int            `json:"seq"`
	Kind Kind           `json:"kind"`
	Time time.Time      `json:"time"`
	Data map[string]any `json:"data,omitempty"`
}

// Emitter serializes events to a human writer and a JSON-lines transcript.
// Either writer may be nil. now is injectable for deterministic tests.
type Emitter struct {
	mu         sync.Mutex
	seq        int
	human      io.Writer
	transcript io.Writer
	now        func() time.Time
	static     map[string]any
}

// Option configures an Emitter at construction.
type Option func(*Emitter)

// WithEnvelopeFields stamps the given key/value pairs on the top level of
// every emitted transcript envelope. The harness treats them as opaque -
// callers use them to carry metadata (e.g. a retry/attempt ordinal) that
// would otherwise require decoding and re-marshalling every event.
//
// A static field never overrides an envelope-owned key (seq, kind, time,
// data): the envelope's own value always wins on collision, so a caller
// mistake can never corrupt the fields the transcript format depends on.
func WithEnvelopeFields(fields map[string]any) Option {
	return func(e *Emitter) {
		if e.static == nil {
			e.static = make(map[string]any, len(fields))
		}

		for k, v := range fields {
			e.static[k] = v
		}
	}
}

// reservedEnvelopeKeys are the Event struct's own JSON keys. A static field
// sharing one of these names is always overwritten by the envelope's own
// value (see envelope) rather than ever reaching the transcript.
var reservedEnvelopeKeys = []string{"seq", "kind", "time", "data"}

func NewEmitter(human, transcript io.Writer, opts ...Option) *Emitter {
	e := &Emitter{human: human, transcript: transcript, now: time.Now}
	for _, opt := range opts {
		opt(e)
	}

	e.warnCollisions()

	return e
}

// warnCollisions writes one line to the human writer for each static field
// whose key collides with an envelope-owned key, naming the field that will
// never appear in an emitted envelope. It runs once, here at construction,
// because the set of static field keys is fixed once NewEmitter returns -
// no per-Emit check or sync.Once is needed. A nil human writer means there
// is nowhere to warn, matching Emit's own nil-writer handling.
func (e *Emitter) warnCollisions() {
	if e.human == nil {
		return
	}

	for _, k := range reservedEnvelopeKeys {
		if _, collides := e.static[k]; collides {
			fmt.Fprintf(e.human, "warning: static envelope field %q collides with an envelope-owned key; the real value always wins and the static one is dropped\n", k) //nolint:errcheck
		}
	}
}

// Emit records an event.
func (e *Emitter) Emit(kind Kind, data map[string]any) {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.seq++

	ev := Event{Seq: e.seq, Kind: kind, Time: e.now(), Data: data}
	if e.transcript != nil {
		if b, err := json.Marshal(e.envelope(ev)); err == nil {
			fmt.Fprintln(e.transcript, string(b)) //nolint:errcheck
		}
	}

	if e.human != nil {
		fmt.Fprintf(e.human, "[%d] %-14s %s\n", ev.Seq, ev.Kind, summarize(data)) //nolint:errcheck
	}
}

// envelope returns the value to marshal for the transcript line. With no
// static fields configured it returns ev unchanged, so the marshalled bytes
// are identical to an emitter built without WithEnvelopeFields. Otherwise it
// returns a map with the static fields merged in beneath the envelope's own
// keys, so seq/kind/time/data always reflect ev, never a caller-supplied
// value of the same name.
func (e *Emitter) envelope(ev Event) any {
	if len(e.static) == 0 {
		return ev
	}

	m := make(map[string]any, len(e.static)+4)
	for k, v := range e.static {
		m[k] = v
	}

	m["seq"] = ev.Seq
	m["kind"] = ev.Kind
	m["time"] = ev.Time

	if len(ev.Data) > 0 {
		m["data"] = ev.Data
	} else {
		delete(m, "data")
	}

	return m
}

func summarize(data map[string]any) string {
	if len(data) == 0 {
		return ""
	}

	b, _ := json.Marshal(data)

	s := string(b)
	if len(s) > 240 {
		s = s[:240] + "…"
	}

	return s
}
