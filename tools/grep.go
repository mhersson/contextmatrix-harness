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
		Description: "Search file contents with ripgrep (regex). Optionally restrict to a path subtree and a glob. Output is capped at 200 matching lines — prefer specific patterns over broad ones.",
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

// grepMaxLines bounds how many matched lines reach the model. Universal
// patterns ("." over "*") otherwise dump tens of KB into context; a model that
// needs more should narrow the pattern, not page.
const grepMaxLines = 200

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
	return Result{Text: capLines(strings.ReplaceAll(out, t.root+"/", ""), grepMaxLines)}, nil
}

// capLines keeps the first maxLines lines and replaces the rest with an
// explicit, corrective note — mirroring the guidance style of read's
// too-large branch.
func capLines(s string, maxLines int) string {
	lines := strings.Split(strings.TrimRight(s, "\n"), "\n")
	if len(lines) <= maxLines {
		return s
	}

	kept := strings.Join(lines[:maxLines], "\n")

	return kept + fmt.Sprintf("\n[... %d more matching lines — narrow the pattern or add a glob/path filter]", len(lines)-maxLines)
}
