package tools

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBashToolRunsInRoot(t *testing.T) {
	root := t.TempDir()
	out, err := NewBashTool(root).Execute(context.Background(), map[string]any{"command": "pwd"})
	require.NoError(t, err)
	assert.Contains(t, out.Text, root)
}

// A command that exits non-zero with NO output at all reads worst as a tool
// error: many commands (a no-match grep, `test`, `diff --quiet`) use exactly
// this shape - empty output, non-zero status - to mean a normal negative
// result, not a breakage. The completed command must not be a Go error; its
// status must still be visible, since there is nothing else in the output to
// tell the model what happened.
func TestBashNoOutputNonZeroExitReportsSuccess(t *testing.T) {
	t.Parallel()

	tool := NewBashTool(t.TempDir())

	res, err := tool.Execute(t.Context(), map[string]any{"command": "exit 1"})

	require.NoError(t, err, "a completed command's exit code is its own result, not a tool failure")
	assert.Contains(t, res.Text, "exited with status 1",
		"the status must be visible in Result.Text; it is the only diagnostic a silent command leaves")
	assert.Equal(t, 1, res.ExitCode, "the exit code must reach Result for callers that inspect it")
}

// A failing build-shaped command (non-zero exit, with output) is the other
// half of the same contract: the command completed, so it is not a tool
// error, but both its own output and its status must still reach the model
// and the exit code must reach Result.
func TestBashFailingCommandReportsOutputStatusAndExitCode(t *testing.T) {
	t.Parallel()

	tool := NewBashTool(t.TempDir())

	res, err := tool.Execute(t.Context(), map[string]any{"command": "echo marker-out; exit 3"})

	require.NoError(t, err, "a completed command's exit code is its own result, not a tool failure")
	assert.Contains(t, res.Text, "marker-out", "the command's own output must still reach the model")
	assert.Contains(t, res.Text, "exited with status 3", "the status must be visible alongside the output")
	assert.Equal(t, 3, res.ExitCode)
}

func TestBashTimeoutReturnsError(t *testing.T) {
	t.Parallel()

	tool := NewBashTool(t.TempDir())

	_, err := tool.Execute(t.Context(), map[string]any{
		"command":         "echo marker-out; sleep 5",
		"timeout_seconds": 1,
	})

	require.Error(t, err, "a timed-out command must surface as a tool error")
	assert.Contains(t, err.Error(), "marker-out",
		"the error must carry whatever the command printed before the kill")
}

func TestBashZeroExitUnchanged(t *testing.T) {
	t.Parallel()

	tool := NewBashTool(t.TempDir())

	res, err := tool.Execute(t.Context(), map[string]any{"command": "echo ok"})

	require.NoError(t, err)
	assert.Contains(t, res.Text, "ok")
	assert.Equal(t, 0, res.ExitCode)
}

func TestBashTimeoutClamp(t *testing.T) {
	tool := NewBashTool(t.TempDir()).WithMaxTimeout(1)

	start := time.Now()
	out, err := tool.Execute(context.Background(), map[string]any{
		"command": "sleep 30", "timeout_seconds": 9999,
	})
	require.Error(t, err, "a timed-out command must surface as a tool error")
	assert.Less(t, time.Since(start), 5*time.Second)
	assert.Contains(t, out.Text, "timed out after 1s")
}

func TestBashWithMaxTimeout(t *testing.T) {
	root := t.TempDir()

	tool := NewBashTool(root)
	assert.Equal(t, defaultBashMaxTimeout, tool.maxTimeout)

	// Non-positive values are no-ops: the current ceiling is kept.
	assert.Equal(t, defaultBashMaxTimeout, tool.WithMaxTimeout(0).maxTimeout)
	assert.Equal(t, defaultBashMaxTimeout, tool.WithMaxTimeout(-7).maxTimeout)

	// Positive values take effect, without mutating the receiver.
	assert.Equal(t, 5, tool.WithMaxTimeout(5).maxTimeout)
	assert.Equal(t, defaultBashMaxTimeout, tool.maxTimeout)
}

func TestBashSchemaReflectsMaxTimeout(t *testing.T) {
	root := t.TempDir()

	assert.Contains(t, string(NewBashTool(root).Schema().Function.Parameters), "max 600")
	assert.Contains(t, string(NewBashTool(root).WithMaxTimeout(42).Schema().Function.Parameters), "max 42")
}

func TestBashToolKillsProcessGroupOnTimeout(t *testing.T) {
	root := t.TempDir()
	marker := filepath.Join(root, "marker")
	// A backgrounded grandchild touches marker after 2s. With a 1s timeout the
	// whole process group must be killed, so marker must NEVER appear.
	out, err := NewBashTool(root).Execute(context.Background(), map[string]any{
		"command":         "(sleep 2 && touch marker) & echo started",
		"timeout_seconds": 1.0,
	})
	require.Error(t, err, "a timed-out command must surface as a tool error")
	assert.Contains(t, out.Text, "started")
	assert.Contains(t, out.Text, "timed out")

	time.Sleep(3 * time.Second) // outlast the grandchild's sleep

	_, statErr := os.Stat(marker)
	assert.True(t, os.IsNotExist(statErr), "grandchild survived the timeout; process group was not killed")
}

// bashOutcome carries an Execute result across the goroutine boundary used to
// guard against the pre-fix behavior: Execute never returning at all.
type bashOutcome struct {
	out Result
	err error
}

func requireSetsid(t *testing.T) {
	t.Helper()

	if _, err := exec.LookPath("setsid"); err != nil {
		t.Skip("setsid not available on this host")
	}
}

func TestBashToolTimeoutReturnsWhenSetsidGrandchildHoldsPipe(t *testing.T) {
	requireSetsid(t)

	root := t.TempDir()
	ch := make(chan bashOutcome, 1)

	go func() {
		// setsid moves the sleeper into a new session (and process group), so
		// the pgid SIGKILL cannot reach it; it keeps the inherited output pipe
		// open. The foreground sleep keeps bash alive past the 1s timeout.
		out, err := NewBashTool(root).Execute(context.Background(), map[string]any{
			"command":         "setsid sleep 15 & echo started; sleep 5",
			"timeout_seconds": 1.0,
		})
		ch <- bashOutcome{out, err}
	}()

	select {
	case o := <-ch:
		require.Error(t, o.err, "a timed-out command must surface as a tool error")
		assert.Contains(t, o.out.Text, "started", "output captured before the kill must be returned")
		assert.Contains(t, o.out.Text, "timed out after 1s")
	case <-time.After(8 * time.Second):
		t.Fatal("Execute did not return after timeout+grace; pipe-holding grandchild wedged Wait")
	}
}

func TestBashToolCtxCancelReturnsWhenSetsidGrandchildHoldsPipe(t *testing.T) {
	requireSetsid(t)

	root := t.TempDir()
	ctx, cancel := context.WithCancel(context.Background())

	defer cancel()

	ch := make(chan bashOutcome, 1)

	go func() {
		out, err := NewBashTool(root).Execute(ctx, map[string]any{
			"command": "setsid sleep 15 & echo started; sleep 30",
		})
		ch <- bashOutcome{out, err}
	}()

	time.Sleep(500 * time.Millisecond) // let bash start and print
	cancel()

	select {
	case o := <-ch:
		require.ErrorIs(t, o.err, context.Canceled)
		assert.Contains(t, o.out.Text, "started", "output captured before the kill must be returned")
	case <-time.After(8 * time.Second):
		t.Fatal("Execute did not return after ctx cancel; pipe-holding grandchild wedged Wait")
	}
}

func TestBashToolBackgroundDaemonReturnsPromptly(t *testing.T) {
	requireSetsid(t)

	root := t.TempDir()
	ch := make(chan bashOutcome, 1)
	start := time.Now()

	go func() {
		// bash exits immediately; only the setsid'd sleeper holds the pipe.
		// Pre-fix this hangs past even the 30s default timeout.
		out, err := NewBashTool(root).Execute(context.Background(), map[string]any{
			"command": "setsid sleep 15 & echo started",
		})
		ch <- bashOutcome{out, err}
	}()

	select {
	case o := <-ch:
		require.NoError(t, o.err)
		assert.Less(t, time.Since(start), 10*time.Second)
		assert.Contains(t, o.out.Text, "started")
		// ErrWaitDelay is expected plumbing, not a command failure - it must be
		// mapped to a calm note, not the "[command exited with status N]" suffix
		// a completed non-zero exit gets.
		assert.NotContains(t, o.out.Text, "exited with status")
		assert.Equal(t, 0, o.out.ExitCode, "a background process holding the pipe is not a command exit status")
	case <-time.After(15 * time.Second):
		t.Fatal("Execute did not return; background daemon holding the pipe wedged Wait")
	}
}

func TestBashSchemaNamesWorkspaceRoot(t *testing.T) {
	tool := NewBashTool("/work/repo")

	desc := tool.Schema().Function.Description
	assert.Contains(t, desc, "/work/repo",
		"the schema description must name the literal workspace root so models never guess the cwd")
}

func TestOptIntCoerced(t *testing.T) {
	keys := []string{"timeout_seconds", "timeout"}

	tests := []struct {
		name    string
		args    map[string]any
		want    int
		wantErr string
	}{
		{"absent returns default", map[string]any{"command": "x"}, 30, ""},
		{"canonical key float64", map[string]any{"timeout_seconds": 5.0}, 5, ""},
		{"alias key int", map[string]any{"timeout": 7}, 7, ""},
		{"alias numeric string", map[string]any{"timeout": "12"}, 12, ""},
		{"numeric string with whitespace", map[string]any{"timeout": " 9 "}, 9, ""},
		{"non-numeric string", map[string]any{"timeout": "abc"}, 0, `"timeout"`},
		{"unsupported type", map[string]any{"timeout": true}, 0, `"timeout"`},
		{"both keys conflict", map[string]any{"timeout_seconds": 1, "timeout": 2}, 0, "timeout_seconds"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := optIntCoerced(tt.args, keys, 30)
			if tt.wantErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.wantErr)

				return
			}

			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestBashTimeoutAliasDrivesTimeout(t *testing.T) {
	// End to end: the alias spelling, as a numeric STRING, must control the
	// real timeout instead of being silently dropped onto the default.
	out, err := NewBashTool(t.TempDir()).Execute(context.Background(), map[string]any{
		"command": "sleep 5",
		"timeout": "1",
	})
	require.Error(t, err, "a timed-out command must surface as a tool error")
	assert.Contains(t, out.Text, "timed out after 1s")
}

func TestBashTimeoutNonNumericStringError(t *testing.T) {
	_, err := NewBashTool(t.TempDir()).Execute(context.Background(), map[string]any{
		"command": "echo hi",
		"timeout": "abc",
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), `"timeout"`, "the error must name the offending key")
	assert.Contains(t, err.Error(), "numeric string", "the error must name the accepted forms")
}

func TestBashUnknownExtraKeysStillIgnored(t *testing.T) {
	out, err := NewBashTool(t.TempDir()).Execute(context.Background(), map[string]any{
		"command":   "echo ok",
		"unrelated": 42,
	})
	require.NoError(t, err)
	assert.Contains(t, out.Text, "ok")
}

func TestBashSchemaDocumentsTimeoutAlias(t *testing.T) {
	params := string(NewBashTool("/w").Schema().Function.Parameters)
	assert.Contains(t, params, "alias", "schema must document the timeout alias")
	assert.Contains(t, params, "numeric string", "schema must document string coercion")
}
