package llm

import (
	"bytes"
	"encoding/json"
	"fmt"
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

type detailEntry struct {
	fields   map[string]json.RawMessage
	verbatim json.RawMessage // a non-object fragment, kept as sent
}

// detailsAccumulator merges streamed reasoning_details fragments per index.
// The provider-side signature check covers the assembled entries, so unknown
// keys are carried through untouched.
type detailsAccumulator struct {
	entries []detailEntry
	byIndex map[int]int
}

func (a *detailsAccumulator) add(frag json.RawMessage) {
	if bytes.Equal(bytes.TrimSpace(frag), []byte("null")) {
		return
	}

	var fields map[string]json.RawMessage
	if json.Unmarshal(frag, &fields) != nil || fields == nil {
		a.entries = append(a.entries, detailEntry{verbatim: frag})

		return
	}

	var idx int

	raw, ok := fields["index"]
	if !ok || json.Unmarshal(raw, &idx) != nil {
		a.entries = append(a.entries, detailEntry{fields: fields})

		return
	}

	pos, seen := a.byIndex[idx]
	if !seen {
		if a.byIndex == nil {
			a.byIndex = map[int]int{}
		}

		a.byIndex[idx] = len(a.entries)
		a.entries = append(a.entries, detailEntry{fields: fields})

		return
	}

	cur := a.entries[pos].fields
	for k, v := range fields {
		if s, isStr := jsonString(v); isStr && concatKey(k) {
			prev, _ := jsonString(cur[k])
			cur[k] = encodeString(prev + s)

			continue
		}

		if !emptyJSON(v) {
			cur[k] = v
		}
	}
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

		b, err := marshalNoEscape(e.fields)
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

func jsonString(raw json.RawMessage) (string, bool) {
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

// marshalNoEscape keeps "<", ">" and "&" literal so opaque payloads are not rewritten.
func marshalNoEscape(v any) (json.RawMessage, error) {
	var buf bytes.Buffer

	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)

	if err := enc.Encode(v); err != nil {
		return nil, err
	}

	return bytes.TrimRight(buf.Bytes(), "\n"), nil
}
