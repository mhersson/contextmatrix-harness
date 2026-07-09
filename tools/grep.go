package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"

	"github.com/mhersson/contextmatrix-harness/llm"
)

type GrepTool struct{ root string }

func NewGrepTool(root string) GrepTool { return GrepTool{root: root} }

func (t GrepTool) Name() string { return "grep" }

func (t GrepTool) Schema() llm.Tool {
	return llm.Tool{Type: "function", Function: llm.ToolFunction{
		Name:        "grep",
		Description: "Search file contents with ripgrep (regex). Optionally restrict to a path subtree and a glob.",
		Parameters: json.RawMessage(`{
			"type":"object",
			"properties":{
				"pattern":{"type":"string","description":"regular expression to search for"},
				"path":{"type":"string","description":"optional subpath under the workspace root to search"},
				"glob":{"type":"string","description":"optional file glob filter, e.g. *.md"}
			},
			"required":["pattern"]
		}`),
	}}
}

func (t GrepTool) Execute(ctx context.Context, args map[string]any) (Result, error) {
	pattern, err := requireString(args, "pattern")
	if err != nil {
		return Result{}, err
	}

	searchPath := t.root
	if rel := optString(args, "path", ""); rel != "" {
		abs, err := resolveInRoot(t.root, rel)
		if err != nil {
			return Result{}, err
		}

		searchPath = abs
	}

	cmdArgs := []string{"--line-number", "--no-heading", "--color=never"}
	if g := optString(args, "glob", ""); g != "" {
		cmdArgs = append(cmdArgs, "--glob", g)
	}

	cmdArgs = append(cmdArgs, "--", pattern, searchPath)

	cmd := exec.CommandContext(ctx, "rg", cmdArgs...)
	cmd.Dir = t.root
	cmd.Env = ScrubbedEnv(nil)

	out, err := runCombinedCapped(cmd)
	if err != nil {
		// rg exits 1 when there are no matches — not an error for us.
		if ee, ok := err.(*exec.ExitError); ok && ee.ExitCode() == 1 {
			return Result{Text: "no matches"}, nil
		}

		if _, lookErr := exec.LookPath("rg"); lookErr != nil {
			return Result{}, fmt.Errorf("ripgrep (rg) not installed")
		}

		return Result{}, fmt.Errorf("rg failed: %v: %s", err, out)
	}
	// Strip the workspace root prefix for cleaner, portable output.
	return Result{Text: strings.ReplaceAll(out, t.root+"/", "")}, nil
}
