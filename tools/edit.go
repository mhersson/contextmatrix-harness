package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/mhersson/contextmatrix-harness/llm"
)

type EditTool struct{ root string }

func NewEditTool(root string) EditTool { return EditTool{root: root} }

func (t EditTool) Name() string { return "edit" }

func (t EditTool) Schema() llm.Tool {
	return llm.Tool{Type: "function", Function: llm.ToolFunction{
		Name:        "edit",
		Description: "Replace an exact substring in a file. Fails if old_string is absent, or appears more than once unless replace_all is true.",
		Parameters: json.RawMessage(`{
			"type":"object",
			"properties":{
				"path":{"type":"string","description":"file path relative to the workspace root"},
				"old_string":{"type":"string","description":"exact text to replace; must be non-empty"},
				"new_string":{"type":"string","description":"replacement text"},
				"replace_all":{"type":"boolean","description":"optional; replace every occurrence"}
			},
			"required":["path","old_string","new_string"]
		}`),
	}}
}

func (t EditTool) Execute(_ context.Context, args map[string]any) (Result, error) {
	rel, err := requireString(args, "path")
	if err != nil {
		return Result{}, err
	}

	oldStr, err := requireString(args, "old_string")
	if err != nil {
		return Result{}, err
	}

	if oldStr == "" {
		return Result{}, errors.New("old_string must be non-empty; use the write tool to create or overwrite file content")
	}

	newStr, err := requireString(args, "new_string")
	if err != nil {
		return Result{}, err
	}

	replaceAll := optBool(args, "replace_all")

	abs, err := resolveInRoot(t.root, rel)
	if err != nil {
		return Result{}, err
	}

	b, err := os.ReadFile(abs)
	if err != nil {
		return Result{}, err
	}

	content := string(b)

	n := strings.Count(content, oldStr)
	if n == 0 {
		return Result{}, fmt.Errorf("old_string not found in %s", rel)
	}

	if n > 1 && !replaceAll {
		return Result{}, fmt.Errorf("old_string appears %d times in %s; set replace_all or provide a unique string", n, rel)
	}

	if replaceAll {
		content = strings.ReplaceAll(content, oldStr, newStr)
	} else {
		content = strings.Replace(content, oldStr, newStr, 1)
	}

	if err := os.WriteFile(abs, []byte(content), 0o644); err != nil { //nolint:gosec // G703: abs is jail-resolved by resolveInRoot above
		return Result{}, err
	}

	return Result{Text: fmt.Sprintf("edited %s (%d replacement(s))", rel, n)}, nil
}
