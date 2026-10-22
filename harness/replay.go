package harness

import (
	"bytes"
	"context"
	"encoding/json"
	"slices"

	"github.com/mhersson/contextmatrix-harness/events"
	"github.com/mhersson/contextmatrix-harness/llm"
)

// withoutReasoning returns a copy of msgs with every ReasoningDetails nil.
// It never writes to msgs, which may be the caller's History.
func withoutReasoning(msgs []llm.Message) []llm.Message {
	if len(msgs) == 0 {
		return nil
	}

	out := make([]llm.Message, len(msgs))
	copy(out, msgs)

	for i := range out {
		out[i].ReasoningDetails = nil
	}

	return out
}

// replayPin returns the provider of the newest message still carrying
// reasoning details, or "" when there is none or it has no provenance.
func replayPin(msgs []llm.Message) string {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].HasReasoningDetails() {
			return msgs[i].ReasoningProvider
		}
	}

	return ""
}

// pinProvider merges pin into the routing object raw, because signed
// reasoning validates only on the provider that produced it. It returns raw
// and false when raw is not an object or its only/ignore/order rule out the pin.
func pinProvider(raw json.RawMessage, pin string) (json.RawMessage, bool) {
	obj := map[string]json.RawMessage{}

	if trimmed := bytes.TrimSpace(raw); len(trimmed) > 0 && !bytes.Equal(trimmed, []byte("null")) {
		if err := json.Unmarshal(trimmed, &obj); err != nil || obj == nil {
			return raw, false
		}
	}

	var order []string

	if v, ok := obj["only"]; ok {
		var only []string
		if json.Unmarshal(v, &only) != nil || !slices.Contains(only, pin) {
			return raw, false
		}
	}

	if v, ok := obj["ignore"]; ok {
		var ignore []string
		if json.Unmarshal(v, &ignore) != nil || slices.Contains(ignore, pin) {
			return raw, false
		}
	}

	if v, ok := obj["order"]; ok {
		if json.Unmarshal(v, &order) != nil {
			return raw, false
		}
	}

	merged := []string{pin}
	for _, name := range order {
		if name != pin {
			merged = append(merged, name)
		}
	}

	enc, err := json.Marshal(merged)
	if err != nil {
		return raw, false
	}

	obj["order"] = enc

	out, err := json.Marshal(obj)
	if err != nil {
		return raw, false
	}

	return out, true
}

// withoutForeignReasoning returns a copy of msgs with ReasoningDetails nil on
// every message not produced by pin. It never writes to msgs.
func withoutForeignReasoning(msgs []llm.Message, pin string) []llm.Message {
	if len(msgs) == 0 {
		return nil
	}

	out := make([]llm.Message, len(msgs))
	copy(out, msgs)

	for i := range out {
		if out[i].ReasoningProvider != pin {
			out[i].ReasoningDetails = nil
		}
	}

	return out
}

// replayRequest pins req to the provider that produced the newest replayed
// reasoning. It returns req unchanged and "" when there is no pin to apply.
func replayRequest(req llm.Request) (llm.Request, string) {
	pin := replayPin(req.Messages)
	if pin == "" {
		return req, ""
	}

	provider, ok := pinProvider(req.Provider, pin)
	if !ok {
		return req, ""
	}

	req.Provider = provider
	req.Messages = withoutForeignReasoning(req.Messages, pin)

	return req, pin
}

// sendModel sends a turn or grace request pinned by replayRequest and emits
// its model_request event, adding pinned_provider when a pin applies.
func sendModel(ctx context.Context, client llm.LLM, emit *events.Emitter, cfg Config, req llm.Request, turn int) (llm.Response, error) {
	sent, pin := replayRequest(req)

	data := map[string]any{"turn": turn, "model": cfg.Model, "messages": len(req.Messages)}
	if pin != "" {
		data["pinned_provider"] = pin
	}

	emit.Emit(events.ModelRequest, data)

	return client.SendStream(ctx, sent, cfg.OnDelta)
}
