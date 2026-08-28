package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"

	"github.com/mhersson/contextmatrix-harness/llm"
)

type GrepTool struct {
	root       string
	extraRoots ReadRoots
}

func NewGrepTool(root string) GrepTool { return GrepTool{root: root} }

func (t GrepTool) Name() string { return "grep" }

func (t GrepTool) Schema() llm.Tool {
	return llm.Tool{Type: "function", Function: llm.ToolFunction{
		Name: "grep",
		Description: "Search file contents with ripgrep (regex). Optionally restrict to a path subtree and a glob. " +
			"Output is capped at 200 matching lines - prefer specific patterns over broad ones." +
			extraRootsSchemaClause(t.extraRoots.Effective),
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
	pattern, err := requireString(args, t.Name(), "pattern", "regular expression to search for")
	if err != nil {
		return Result{}, err
	}

	searchPath := t.root
	if rel := optString(args, "path", ""); rel != "" {
		abs, err := resolveInRoots(t.roots(), rel)
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
		// rg exits 1 when there are no matches - not an error for us.
		if ee, ok := err.(*exec.ExitError); ok && ee.ExitCode() == 1 {
			return Result{Text: grepEmptyMsg(pattern)}, nil
		}

		if _, lookErr := exec.LookPath("rg"); lookErr != nil {
			return Result{}, fmt.Errorf("ripgrep (rg) not installed")
		}

		return Result{}, fmt.Errorf("rg failed: %v: %s", err, out)
	}
	// Strip the workspace root prefix for cleaner, portable output.
	return Result{Text: capLines(stripWorkspacePrefix(out, t.root), grepMaxLines)}, nil
}

// stripWorkspacePrefix removes a leading root+"/" from each line's own path
// field only, never from elsewhere in the line. A whole-output ReplaceAll
// would also rewrite the workspace prefix wherever it happens to appear as an
// INFIX of a secondary-root path (e.g. a mirror tree that reproduces the
// workspace path underneath it), corrupting that line's path into something
// the read tool then rejects. Anchoring to the line start limits the strip to
// genuine path prefixes.
func stripWorkspacePrefix(out, root string) string {
	prefix := root + "/"
	if prefix == "/" {
		return out
	}

	lines := strings.Split(out, "\n")
	for i, line := range lines {
		lines[i] = strings.TrimPrefix(line, prefix)
	}

	return strings.Join(lines, "\n")
}

// grepEmptyMsg returns "no matches" with an optional corrective note when the
// pattern uses GNU BRE alternation syntax that ripgrep interprets differently.
// \( \) \{ \+ are all valid Rust-regex literal escapes (matching a literal
// paren/brace/plus), so they are not flagged here - only \|, the observed
// failure class, actually behaves differently under ripgrep's Rust-regex
// syntax (alternation is unescaped a|b; \| matches a literal pipe).
func grepEmptyMsg(pattern string) string {
	if strings.Contains(pattern, "\\|") {
		return "no matches - note: grep uses ripgrep (Rust regex) syntax where a|b is alternation and \\| matches a literal pipe"
	}

	return "no matches"
}

// capLines keeps the first maxLines lines and replaces the rest with an
// explicit, corrective note - mirroring the guidance style of read's
// too-large branch.
func capLines(s string, maxLines int) string {
	lines := strings.Split(strings.TrimRight(s, "\n"), "\n")
	if len(lines) <= maxLines {
		return s
	}

	kept := strings.Join(lines[:maxLines], "\n")

	return kept + fmt.Sprintf("\n[... %d more matching lines - narrow the pattern or add a glob/path filter]", len(lines)-maxLines)
}

// ReadOnly marks grep as side-effect-free: see tools.ReadOnly.
func (t GrepTool) ReadOnly() bool { return true }

// WithReadRoots returns a copy that may also read inside the given trees, in
// addition to the workspace. Roots are caller-supplied configuration - never a
// tool argument - and are filtered by sanitizeReadRoots, so a root that would
// widen past the workspace is dropped. With no roots the tool behaves exactly
// as before. Only the search path resolves against them: with no path argument
// the tool still works from the workspace root.
func (t GrepTool) WithReadRoots(roots []string) GrepTool {
	t.extraRoots = sanitizeReadRoots(t.root, roots)

	return t
}

// ReadRoots reports the extra read-only roots this tool was configured with:
// which survived sanitizeReadRoots and which were dropped, and why. See
// ReadRoots (jail.go).
func (t GrepTool) ReadRoots() ReadRoots {
	return t.extraRoots
}

// roots is the ordered permitted set: the workspace first, then any extra
// read-only roots.
func (t GrepTool) roots() []string {
	return append([]string{t.root}, t.extraRoots.Effective...)
}
