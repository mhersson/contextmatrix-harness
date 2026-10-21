package llm

import (
	"bytes"
	"encoding/json"
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
