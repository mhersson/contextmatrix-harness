package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strconv"
)

type CatalogEntry struct {
	ID                    string
	PromptPricePerTok     float64
	CompletionPricePerTok float64
	ContextLength         int
	SupportedParameters   []string
}

func (e CatalogEntry) SupportsTools() bool {
	return slices.Contains(e.SupportedParameters, "tools")
}

type Catalog []CatalogEntry

func (c Catalog) Find(id string) (CatalogEntry, bool) {
	for _, e := range c {
		if e.ID == id {
			return e, true
		}
	}

	return CatalogEntry{}, false
}

func parseCatalog(r io.Reader) (Catalog, error) {
	var doc struct {
		Data []struct {
			ID            string `json:"id"`
			ContextLength int    `json:"context_length"`
			Pricing       struct {
				Prompt     string `json:"prompt"`
				Completion string `json:"completion"`
			} `json:"pricing"`
			SupportedParameters []string `json:"supported_parameters"`
		} `json:"data"`
	}
	if err := json.NewDecoder(r).Decode(&doc); err != nil {
		return nil, fmt.Errorf("decode catalog: %w", err)
	}

	out := make(Catalog, 0, len(doc.Data))
	for _, d := range doc.Data {
		prompt, _ := strconv.ParseFloat(d.Pricing.Prompt, 64)
		completion, _ := strconv.ParseFloat(d.Pricing.Completion, 64)
		out = append(out, CatalogEntry{
			ID:                    d.ID,
			PromptPricePerTok:     prompt,
			CompletionPricePerTok: completion,
			ContextLength:         d.ContextLength,
			SupportedParameters:   d.SupportedParameters,
		})
	}

	return out, nil
}

// parseCatalogOpenAI decodes an OpenAI-compatible /models response. Tool
// capability is read from capabilities.features (an array containing "tools")
// and mapped onto SupportedParameters so CatalogEntry.SupportsTools() works
// unchanged. Endpoints that omit these fields yield zero-value entries.
func parseCatalogOpenAI(r io.Reader) (Catalog, error) {
	var doc struct {
		Data []struct {
			ID            string `json:"id"`
			ContextLength int    `json:"context_length"`
			Pricing       struct {
				Prompt     string `json:"prompt"`
				Completion string `json:"completion"`
			} `json:"pricing"`
			Capabilities struct {
				Features []string `json:"features"`
			} `json:"capabilities"`
		} `json:"data"`
	}
	if err := json.NewDecoder(r).Decode(&doc); err != nil {
		return nil, fmt.Errorf("decode catalog: %w", err)
	}

	out := make(Catalog, 0, len(doc.Data))
	for _, d := range doc.Data {
		prompt, _ := strconv.ParseFloat(d.Pricing.Prompt, 64)
		completion, _ := strconv.ParseFloat(d.Pricing.Completion, 64)

		var params []string

		if slices.Contains(d.Capabilities.Features, "tools") {
			params = []string{"tools"}
		}

		out = append(out, CatalogEntry{
			ID:                    d.ID,
			PromptPricePerTok:     prompt,
			CompletionPricePerTok: completion,
			ContextLength:         d.ContextLength,
			SupportedParameters:   params,
		})
	}

	return out, nil
}

// FetchCatalog GETs the live model catalog from the configured endpoint and
// parses it according to the client's dialect (OpenRouter or OpenAI-compatible).
func (c *Client) FetchCatalog(ctx context.Context) (Catalog, error) {
	hr, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/models", nil)
	if err != nil {
		return nil, err
	}

	hr.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.http.Do(hr)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close() //nolint:errcheck

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, maxResponseBody+1))
		if len(body) > maxResponseBody {
			return nil, fmt.Errorf("response body exceeds %d bytes", maxResponseBody)
		}

		return nil, fmt.Errorf("catalog status %d: %s", resp.StatusCode, string(body))
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBody+1))
	if err != nil {
		return nil, fmt.Errorf("read catalog body: %w", err)
	}

	if len(body) > maxResponseBody {
		return nil, fmt.Errorf("catalog body exceeds %d bytes", maxResponseBody)
	}

	if c.dialect == DialectOpenAI {
		return parseCatalogOpenAI(bytes.NewReader(body))
	}

	return parseCatalog(bytes.NewReader(body))
}
