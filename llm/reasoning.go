package llm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"maps"
	"strings"
)

// replayableDetails returns raw (trimmed) when it is a JSON array with at
// least one element, else nil. Anything else is not replayable.
func replayableDetails(raw json.RawMessage) json.RawMessage {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 {
		return nil
	}

	var elems []json.RawMessage
	if err := json.Unmarshal(raw, &elems); err != nil || len(elems) == 0 {
		return nil
	}

	return raw
}

// detailEntry keeps concat-key strings in builders so a long stream is linear;
// fields and cat never hold the same key.
type detailEntry struct {
	fields   map[string]json.RawMessage
	cat      map[string]*strings.Builder
	verbatim json.RawMessage // a non-object fragment, kept as sent
}

func newDetailEntry(fields map[string]json.RawMessage) *detailEntry {
	e := &detailEntry{fields: map[string]json.RawMessage{}, cat: map[string]*strings.Builder{}}

	for k, v := range fields {
		if s, ok := jsonString(v); ok && concatKey(k) {
			e.cat[k] = &strings.Builder{}
			e.cat[k].WriteString(s)

			continue
		}

		e.fields[k] = v
	}

	return e
}

func (e *detailEntry) merge(fields map[string]json.RawMessage) {
	for k, v := range fields {
		if concatKey(k) {
			if s, ok := jsonString(v); ok {
				b := e.cat[k]
				if b == nil {
					b = &strings.Builder{}
					e.cat[k] = b

					delete(e.fields, k)
				}

				b.WriteString(s)

				continue
			}
		}

		if !emptyJSON(v) {
			e.fields[k] = v

			delete(e.cat, k)
		}
	}
}

// detailsAccumulator merges streamed reasoning_details fragments per index.
// The provider-side signature check covers the assembled entries, so unknown
// keys are carried through untouched. Fragments sharing an index merge
// regardless of type, and type takes the latest value: live Gemini streams
// send a reasoning.text and a reasoning.encrypted fragment on one index, and
// the provider accepted the folded entry on replay.
type detailsAccumulator struct {
	entries []*detailEntry
	byIndex map[int]int
}

func (a *detailsAccumulator) add(frag json.RawMessage) {
	if bytes.Equal(bytes.TrimSpace(frag), []byte("null")) {
		return
	}

	var fields map[string]json.RawMessage
	if json.Unmarshal(frag, &fields) != nil {
		a.entries = append(a.entries, &detailEntry{verbatim: frag})

		return
	}

	idx, ok := integerIndex(fields["index"])
	if !ok {
		a.entries = append(a.entries, newDetailEntry(fields))

		return
	}

	pos, seen := a.byIndex[idx]
	if !seen {
		if a.byIndex == nil {
			a.byIndex = map[int]int{}
		}

		a.byIndex[idx] = len(a.entries)
		a.entries = append(a.entries, newDetailEntry(fields))

		return
	}

	a.entries[pos].merge(fields)
}

// integerIndex reports a JSON integer; null, strings and floats are not indexes.
func integerIndex(raw json.RawMessage) (int, bool) {
	if emptyJSON(raw) {
		return 0, false
	}

	var idx int
	if json.Unmarshal(raw, &idx) != nil {
		return 0, false
	}

	return idx, true
}

// result returns the assembled JSON array, or nil when no entries arrived.
func (a *detailsAccumulator) result() (json.RawMessage, error) {
	if len(a.entries) == 0 {
		return nil, nil
	}

	elems := make([]json.RawMessage, 0, len(a.entries))

	for _, e := range a.entries {
		if e.fields == nil {
			elems = append(elems, e.verbatim)

			continue
		}

		out := make(map[string]json.RawMessage, len(e.fields)+len(e.cat))
		maps.Copy(out, e.fields)

		for k, b := range e.cat {
			out[k] = encodeString(b.String())
		}

		b, err := marshalNoEscape(out)
		if err != nil {
			return nil, fmt.Errorf("encode reasoning detail: %w", err)
		}

		elems = append(elems, b)
	}

	out, err := marshalNoEscape(elems)
	if err != nil {
		return nil, fmt.Errorf("encode reasoning details: %w", err)
	}

	return out, nil
}

func concatKey(k string) bool {
	switch k {
	case "text", "summary", "data":
		return true
	default:
		return false
	}
}

// jsonString decodes only a JSON string token; null is not a string.
func jsonString(raw json.RawMessage) (string, bool) {
	if t := bytes.TrimSpace(raw); len(t) == 0 || t[0] != '"' {
		return "", false
	}

	var s string
	if json.Unmarshal(raw, &s) != nil {
		return "", false
	}

	return s, true
}

func emptyJSON(raw json.RawMessage) bool {
	t := bytes.TrimSpace(raw)

	return len(t) == 0 || bytes.Equal(t, []byte("null")) || bytes.Equal(t, []byte(`""`))
}

func encodeString(s string) json.RawMessage {
	b, _ := marshalNoEscape(s) //nolint:errcheck // a string always marshals

	return b
}

// marshalNoEscape keeps "<", ">" and "&" literal in the assembled
// Response.ReasoningDetails bytes; the request encoder may escape them again
// with value-equal JSON.
func marshalNoEscape(v any) (json.RawMessage, error) {
	var buf bytes.Buffer

	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)

	if err := enc.Encode(v); err != nil {
		return nil, err
	}

	return bytes.TrimRight(buf.Bytes(), "\n"), nil
}
