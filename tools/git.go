package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"

	"github.com/mhersson/contextmatrix-harness/llm"
)

type GitTool struct{ root string }

func NewGitTool(root string) GitTool { return GitTool{root: root} }

func (t GitTool) Name() string { return "git" }

// readonlyGitSubcommands is the allowlist. Mutating git (commit/push/rebase/…)
// is a deterministic orchestrator action, never a model tool.
var readonlyGitSubcommands = map[string]bool{
	"status": true, "diff": true, "log": true, "show": true, "branch": true,
}

func (t GitTool) Schema() llm.Tool {
	return llm.Tool{Type: "function", Function: llm.ToolFunction{
		Name:        "git",
		Description: "Run a read-only git command (status, diff, log, show, branch) in the workspace and return its output.",
		Parameters: json.RawMessage(`{
			"type":"object",
			"properties":{
				"subcommand":{"type":"string","enum":["status","diff","log","show","branch"],"description":"read-only git subcommand"},
				"args":{"type":"array","items":{"type":"string"},"description":"optional extra arguments, e.g. [\"--stat\"]"}
			},
			"required":["subcommand"]
		}`),
	}}
}

func (t GitTool) Execute(ctx context.Context, args map[string]any) (Result, error) {
	sub, err := requireString(args, "subcommand")
	if err != nil {
		return Result{}, err
	}

	if !readonlyGitSubcommands[sub] {
		return Result{}, fmt.Errorf("git subcommand %q is not allowed (read-only: status, diff, log, show, branch)", sub)
	}

	extra := optStringSlice(args, "args")
	if err := validateGitArgs(sub, extra); err != nil {
		return Result{}, err
	}

	cmdArgs := append([]string{sub}, extra...)
	cmd := exec.CommandContext(ctx, "git", cmdArgs...)
	cmd.Dir = t.root
	cmd.Env = ScrubbedEnv(nil)

	out, err := cmd.CombinedOutput()
	if err != nil {
		if _, ok := err.(*exec.ExitError); ok {
			return Result{Text: string(out)}, nil // surface git's own message as output, not a hard failure
		}

		return Result{}, fmt.Errorf("git failed: %v", err)
	}

	return Result{Text: string(out)}, nil
}

// validateGitArgs rejects write/mutation primitives so the read-only git tool
// cannot escape the workspace (--output) or change refs (branch create/delete/
// rename). Keeps the "read-only is structural" contract SpawnSubagents relies on.
func validateGitArgs(sub string, args []string) error {
	for _, a := range args {
		if a == "-o" || a == "--output" || a == "--output-directory" ||
			strings.HasPrefix(a, "--output=") || strings.HasPrefix(a, "--output-directory=") {
			return fmt.Errorf("git arg %q writes to a file and is not allowed (read-only)", a)
		}
	}

	if sub == "branch" {
		listMode := false

		for _, a := range args {
			if a == "--list" || a == "-l" {
				listMode = true
			}
		}

		for _, a := range args {
			if strings.HasPrefix(a, "--set-upstream-to=") || strings.HasPrefix(a, "-u") {
				return fmt.Errorf("git branch %q mutates refs and is not allowed (read-only)", a)
			}

			switch a {
			case "-d", "-D", "--delete", "-m", "-M", "--move", "-c", "-C", "--copy",
				"-f", "--force", "--set-upstream-to", "--unset-upstream", "--edit-description":
				return fmt.Errorf("git branch %q mutates refs and is not allowed (read-only)", a)
			}

			if !strings.HasPrefix(a, "-") && !listMode {
				return fmt.Errorf("git branch %q creates a branch and is not allowed (read-only)", a)
			}
		}
	}

	return nil
}
