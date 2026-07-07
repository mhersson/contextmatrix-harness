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

func TestBashToolReturnsFailureAsOutputNotError(t *testing.T) {
	root := t.TempDir()
	// A failing command must NOT return a Go error — the model needs to see it.
	out, err := NewBashTool(root).Execute(context.Background(), map[string]any{"command": "exit 3"})
	require.NoError(t, err)
	assert.Contains(t, out.Text, "exit")
}

func TestBashToolTimeout(t *testing.T) {
	root := t.TempDir()
	out, err := NewBashTool(root).Execute(context.Background(), map[string]any{"command": "sleep 5", "timeout_seconds": 1.0})
	require.NoError(t, err)
	assert.Contains(t, out.Text, "timed out")
}

func TestBashTimeoutClamp(t *testing.T) {
	tool := NewBashTool(t.TempDir()).WithMaxTimeout(1)

	start := time.Now()
	out, err := tool.Execute(context.Background(), map[string]any{
		"command": "sleep 30", "timeout_seconds": 9999,
	})
	require.NoError(t, err)
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
	require.NoError(t, err)
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
		require.NoError(t, o.err)
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
		// ErrWaitDelay is expected plumbing, not a command failure — it must be
		// mapped to a calm note, not "[command exited with error: ...]".
		assert.NotContains(t, o.out.Text, "command exited with error")
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
