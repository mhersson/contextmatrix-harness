package harness

import (
	"encoding/json"
	"fmt"
	"strings"
)

// normalizeArgs trims a tool-call arguments string and defaults empty input to
// "{}", so callers always parse or surface valid JSON. Empty arguments arrive
// legitimately: no-argument tools, and some OpenAI-compatible proxies (e.g.
// Anthropic/Bedrock behind an OpenAI shim) send "" instead of "{}".
func normalizeArgs(raw string) string {
	s := strings.TrimSpace(raw)
	if s == "" {
		return "{}"
	}

	return s
}

// parseArgs decodes a tool-call arguments string into a map. Weak models emit
// malformed/empty/truncated JSON; a non-nil error signals the caller to ask the
// model to re-emit valid arguments rather than crash the loop.
func parseArgs(raw string) (map[string]any, error) {
	s := normalizeArgs(raw)

	var m map[string]any
	if err := json.Unmarshal([]byte(s), &m); err != nil {
		return nil, fmt.Errorf("invalid JSON arguments: %w", err)
	}

	return m, nil
}

func repairMessage(toolName string, err error) string {
	return fmt.Sprintf(
		"Your call to tool %q had invalid arguments: %v. Re-call the tool with a single valid JSON object matching its parameter schema.",
		toolName, err)
}
