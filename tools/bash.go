package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"syscall"
	"time"

	"github.com/mhersson/contextmatrix-harness/llm"
)

const defaultBashMaxTimeout = 600 // seconds - hard server-side ceiling

// bashWaitDelay bounds cmd.Wait after the child exits (or is killed) while a
// descendant that escaped the process-group SIGKILL (e.g. via setsid) still
// holds the output pipe. Without it, Wait blocks until pipe EOF - potentially
// forever - wedging the turn past both the timeout and ctx cancellation.
const bashWaitDelay = 2 * time.Second

type BashTool struct {
	root       string
	maxTimeout int
	extraEnv   []string
}

func NewBashTool(root string) BashTool {
	return BashTool{root: root, maxTimeout: defaultBashMaxTimeout}
}

// WithExtraEnv returns a copy with additional KEY=VALUE entries appended to the
// scrubbed environment (e.g. "GOCACHE=/tmp/gocache").
func (t BashTool) WithExtraEnv(kvs []string) BashTool {
	t.extraEnv = kvs

	return t
}

// WithMaxTimeout returns a copy with a different clamp ceiling (seconds).
// Non-positive input keeps the current ceiling.
func (t BashTool) WithMaxTimeout(s int) BashTool {
	if s > 0 {
		t.maxTimeout = s
	}

	return t
}

func (t BashTool) Name() string { return "bash" }

func (t BashTool) Schema() llm.Tool {
	return llm.Tool{Type: "function", Function: llm.ToolFunction{
		Name: "bash",
		Description: fmt.Sprintf(
			"Run a shell command in the workspace root (%s) and return combined stdout+stderr, plus its exit code. A completed command's exit code is not itself a tool failure - many commands use a non-zero exit as a normal result (no matches, a failed check). Only a timeout is reported as a tool failure.",
			t.root),
		Parameters: json.RawMessage(fmt.Sprintf(`{
			"type":"object",
			"properties":{
				"command":{"type":"string","description":"the shell command to run"},
				"timeout_seconds":{"type":"integer","description":"optional timeout in seconds (default 30, max %d); \"timeout\" is accepted as an alias; numeric strings are coerced"}
			},
			"required":["command"]
		}`, t.maxTimeout)),
	}}
}

func (t BashTool) Execute(ctx context.Context, args map[string]any) (Result, error) {
	command, err := requireString(args, t.Name(), "command", "the shell command to run")
	if err != nil {
		return Result{}, err
	}

	timeout, err := optIntCoerced(args, []string{"timeout_seconds", "timeout"}, 30)
	if err != nil {
		return Result{}, err
	}

	if timeout < 1 {
		timeout = 1
	}

	if timeout > t.maxTimeout {
		timeout = t.maxTimeout
	}

	cmd := exec.Command("bash", "-c", command) //nolint:noctx // ctx cancel is handled below by killing the whole process group; CommandContext would kill only the child
	cmd.Dir = t.root
	cmd.Env = ScrubbedEnv(t.extraEnv)
	// New process group so we can signal the whole tree (the child is the
	// group leader: pgid == child pid). Plain ctx-cancel leaves grandchildren.
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.WaitDelay = bashWaitDelay

	cw := &capWriter{limit: subprocessOutputCap}

	cmd.Stdout = cw
	cmd.Stderr = cw

	if err := cmd.Start(); err != nil {
		return Result{}, fmt.Errorf("start command: %w", err)
	}

	pgid := cmd.Process.Pid

	done := make(chan error, 1)

	go func() { done <- cmd.Wait() }()

	timer := time.NewTimer(time.Duration(timeout) * time.Second)
	defer timer.Stop()

	select {
	case <-timer.C:
		_ = syscall.Kill(-pgid, syscall.SIGKILL) //nolint:errcheck

		<-done

		text := cw.String() + fmt.Sprintf("\n[command timed out after %ds]", timeout)

		// The run loop renders a tool error as "tool error: %v" and discards
		// Result, so the captured output has to ride the error or the model
		// loses the diagnostic. Returning an error is what makes the loop's
		// failure counter see a command that did not succeed.
		return Result{Text: text}, errors.New(text)
	case <-ctx.Done():
		_ = syscall.Kill(-pgid, syscall.SIGKILL) //nolint:errcheck

		<-done

		return Result{Text: cw.String()}, ctx.Err()
	case werr := <-done:
		res := cw.String()
		exitCode := 0

		switch {
		case errors.Is(werr, exec.ErrWaitDelay):
			// The command itself succeeded; a surviving background process still
			// holds the output pipe. Not a command failure.
			res += "\n[stopped reading output: a background process still holds the output pipe]"
		case werr != nil:
			// The command completed; a non-zero exit is its own result, not a
			// tool failure - many commands use it as a boolean (grep, test,
			// diff --quiet). Report the status as data, not as an error.
			var exitErr *exec.ExitError
			if errors.As(werr, &exitErr) {
				exitCode = exitErr.ExitCode()
			}

			res += fmt.Sprintf("\n[command exited with status %d]", exitCode)
		}

		return Result{Text: res, ExitCode: exitCode}, nil
	}
}
