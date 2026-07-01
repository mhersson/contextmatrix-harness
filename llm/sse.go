package llm

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
)

// maxSSELine bounds a single SSE line (large reasoning/tool-arg frames are real,
// but unbounded growth is not).
const maxSSELine = 10 * 1024 * 1024

type streamChunk struct {
	Model   string         `json:"model"`
	Choices []streamChoice `json:"choices"`
	Usage   *Usage         `json:"usage"`
	Error   *apiError      `json:"error"`
}

type streamChoice struct {
	Delta        streamDelta `json:"delta"`
	FinishReason string      `json:"finish_reason"`
}

type streamDelta struct {
	Content   string           `json:"content"`
	Reasoning string           `json:"reasoning"`
	ToolCalls []streamToolCall `json:"tool_calls"`
}

type streamToolCall struct {
	Index    int                `json:"index"`
	ID       string             `json:"id"`
	Type     string             `json:"type"`
	Function streamFunctionCall `json:"function"`
}

type streamFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type apiError struct {
	Code    any    `json:"code"`
	Message string `json:"message"`
}

// parseStream reads an OpenAI-compatible SSE body and assembles a Response. It skips
// ":" keepalive comments, stops on "[DONE]", accumulates tool_calls by index
// (the name may arrive in a later frame than id/arguments), captures usage from
// the final frame, surfaces mid-stream error frames as an error, and rejects a
// stream that ends without [DONE] or a terminal finish_reason. onDelta
// (nullable) receives incremental content/reasoning.
func parseStream(r io.Reader, onDelta func(Delta)) (Response, error) {
	return parseStreamWithLimit(r, onDelta, maxSSELine)
}

func parseStreamWithLimit(r io.Reader, onDelta func(Delta), maxLine int) (Response, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, min(64*1024, maxLine)), maxLine)

	var resp Response

	var contentBuilder strings.Builder

	var reasoningBuilder strings.Builder

	acc := map[int]*streamToolCall{}
	argBuilders := map[int]*strings.Builder{}

	var order []int

	var sawDone bool

	for sc.Scan() {
		line := sc.Text()
		switch {
		case line == "":
			continue
		case strings.HasPrefix(line, ":"):
			continue // keepalive comment
		case !strings.HasPrefix(line, "data:"):
			continue
		}

		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "[DONE]" {
			sawDone = true

			break
		}

		var chunk streamChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			return resp, fmt.Errorf("decode sse chunk: %w", err)
		}

		if chunk.Error != nil {
			return resp, fmt.Errorf("llm endpoint stream error: %s", chunk.Error.Message)
		}

		if chunk.Model != "" {
			resp.Model = chunk.Model
		}

		if chunk.Usage != nil {
			resp.Usage = *chunk.Usage
		}

		for _, ch := range chunk.Choices {
			if ch.FinishReason != "" {
				resp.FinishReason = ch.FinishReason
			}

			d := ch.Delta
			if d.Content != "" {
				contentBuilder.WriteString(d.Content)
				if onDelta != nil {
					onDelta(Delta{Content: d.Content})
				}
			}

			if d.Reasoning != "" {
				reasoningBuilder.WriteString(d.Reasoning)
				if onDelta != nil {
					onDelta(Delta{Reasoning: d.Reasoning})
				}
			}

			for _, tc := range d.ToolCalls {
				cur, ok := acc[tc.Index]
				if !ok {
					cur = &streamToolCall{Index: tc.Index}
					acc[tc.Index] = cur
					order = append(order, tc.Index)
				}

				if tc.ID != "" {
					cur.ID = tc.ID
				}

				if tc.Type != "" {
					cur.Type = tc.Type
				}

				if tc.Function.Name != "" {
					cur.Function.Name = tc.Function.Name
				}

				if tc.Function.Arguments != "" {
					ab, ok := argBuilders[tc.Index]
					if !ok {
						ab = &strings.Builder{}
						argBuilders[tc.Index] = ab
					}

					ab.WriteString(tc.Function.Arguments)
				}
			}
		}
	}

	resp.Content = contentBuilder.String()
	resp.Reasoning = reasoningBuilder.String()

	if err := sc.Err(); err != nil {
		if errors.Is(err, bufio.ErrTooLong) {
			return resp, fmt.Errorf("sse line exceeds %d bytes: %w", maxLine, err)
		}

		return resp, fmt.Errorf("read sse: %w", err)
	}

	if !sawDone && resp.FinishReason == "" {
		return resp, fmt.Errorf("truncated stream: ended without [DONE] or a terminal finish_reason")
	}

	for _, idx := range order {
		tc := acc[idx]

		typ := tc.Type
		if typ == "" {
			typ = "function"
		}

		args := ""
		if ab, ok := argBuilders[idx]; ok {
			args = ab.String()
		}

		resp.ToolCalls = append(resp.ToolCalls, ToolCall{
			ID:       tc.ID,
			Type:     typ,
			Function: FunctionCall{Name: tc.Function.Name, Arguments: args},
		})
	}

	return normalizeHarmony(resp), nil
}
