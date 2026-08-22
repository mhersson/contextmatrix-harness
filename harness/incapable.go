package harness

import (
	"encoding/json"
	"errors"
	"io"
	"strings"
)

// patternConcatenatedObjects and patternIdenticalPayload are the classifier's
// possible non-empty results, used verbatim as the event's
// suspected_upstream_defect value.
const (
	patternConcatenatedObjects = "concatenated objects"
	patternIdenticalPayload    = "identical payload every turn"
)

// detailConcatenatedObjects and detailIdenticalPayload are the operator-facing
// sentences paired with the patterns above; the matching one becomes
// Result.IncapableDetail.
const (
	detailConcatenatedObjects = "suspected upstream gateway defect: tool-call arguments arrived as concatenated JSON objects"
	detailIdenticalPayload    = "suspected upstream gateway defect: identical tool-call arguments arrived on every turn"
)

// isConcatenatedJSON reports whether raw decodes as two or more complete JSON
// values back to back, tolerating whitespace and differing formatting between
// them - the shape produced by a gateway that sends one pretty-printed copy of
// tool-call arguments immediately followed by a compact copy.
func isConcatenatedJSON(raw string) bool {
	dec := json.NewDecoder(strings.NewReader(raw))
	count := 0

	for {
		var v any

		err := dec.Decode(&v)
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}

			return false
		}

		count++
	}

	return count >= 2
}

// everyPayloadMatches reports whether pred holds for every retained payload in
// every turn of window.
func everyPayloadMatches(window [][]string, pred func(string) bool) bool {
	for _, turn := range window {
		for _, payload := range turn {
			if !pred(payload) {
				return false
			}
		}
	}

	return true
}

// classifyIncapable inspects the failed tool-call argument payloads of an
// unproductive-turn window and reports whether they look like a transport
// defect rather than a model that cannot form arguments. window[i] holds
// every raw arguments string that failed to parse in turn i of the current
// unbroken unproductive streak. Returns ("", "") when the window carries no
// evidence either way. A turn that retained at least one failed payload counts
// as evidence even when other calls in that turn failed for a different reason
// (unknown tool, interjection skip); only a turn with no retained payloads
// makes the whole window unclassifiable.
func classifyIncapable(window [][]string) (pattern, detail string) {
	if len(window) == 0 {
		return "", ""
	}

	for _, turn := range window {
		if len(turn) == 0 {
			return "", ""
		}
	}

	if everyPayloadMatches(window, isConcatenatedJSON) {
		return patternConcatenatedObjects, detailConcatenatedObjects
	}

	if len(window) < 2 {
		return "", ""
	}

	first := window[0][0]
	if everyPayloadMatches(window, func(payload string) bool { return payload == first }) {
		return patternIdenticalPayload, detailIdenticalPayload
	}

	return "", ""
}
