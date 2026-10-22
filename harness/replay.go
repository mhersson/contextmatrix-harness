package harness

import (
	"bytes"
	"context"
	"encoding/json"
	"slices"
	"strings"

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

// isBadRequest reports whether err is the client's HTTP 400 status error.
func isBadRequest(err error) bool {
	return err != nil && strings.Contains(err.Error(), "llm endpoint status 400")
}

// carriesReasoning reports whether any message in msgs holds reasoning details.
func carriesReasoning(msgs []llm.Message) bool {
	return slices.ContainsFunc(msgs, llm.Message.HasReasoningDetails)
}

// failedRoutingStep extracts the gateway's error.metadata.failed_routing_step
// from a 400 error, or "" when absent or undecodable.
func failedRoutingStep(err error) string {
	_, body, ok := strings.Cut(err.Error(), "llm endpoint status 400: ")
	if !ok {
		return ""
	}

	var env struct {
		Error struct {
			Metadata struct {
				FailedRoutingStep string `json:"failed_routing_step"`
			} `json:"metadata"`
		} `json:"error"`
	}
	if json.Unmarshal([]byte(body), &env) != nil {
		return ""
	}

	return env.Error.Metadata.FailedRoutingStep
}

// sendModel sends a turn or grace request pinned by replayRequest and emits
// its model_request event, adding pinned_provider when a pin applies. A 400 on
// a request that carried a pin or details is retried once without either.
func sendModel(ctx context.Context, client llm.LLM, emit *events.Emitter, cfg Config, req llm.Request, turn int) (llm.Response, error) {
	sent, pin := replayRequest(req)

	data := map[string]any{"turn": turn, "model": cfg.Model, "messages": len(req.Messages)}
	if pin != "" {
		data["pinned_provider"] = pin
	}

	emit.Emit(events.ModelRequest, data)

	resp, err := client.SendStream(ctx, sent, cfg.OnDelta)
	if !isBadRequest(err) || (pin == "" && !carriesReasoning(sent.Messages)) {
		return resp, err
	}

	drop := map[string]any{"event": "replay_dropped", "status": 400, "turn": turn}
	if step := failedRoutingStep(err); step != "" {
		drop["failed_routing_step"] = step
	}

	emit.Emit(events.StateChange, drop)

	req.Messages = withoutReasoning(req.Messages)

	emit.Emit(events.ModelRequest, map[string]any{"turn": turn, "model": cfg.Model, "messages": len(req.Messages)})

	return client.SendStream(ctx, req, cfg.OnDelta)
}
