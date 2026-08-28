package tools

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestResolveInRootsRefusalNamesPermittedRoots pins the locked decision: a
// model refused by the multi-root jail is told which read-only roots it is
// allowed into, instead of a message that names none of them.
func TestResolveInRootsRefusalNamesPermittedRoots(t *testing.T) {
	base := t.TempDir()
	workspace := filepath.Join(base, "workspace")
	depA := filepath.Join(base, "dep-a")
	depB := filepath.Join(base, "dep-b")
	outside := filepath.Join(base, "elsewhere")

	for _, d := range []string{workspace, depA, depB, outside} {
		require.NoError(t, os.MkdirAll(d, 0o755))
	}

	roots := append([]string{workspace}, sanitizeReadRoots(workspace, []string{depA, depB}).Effective...)
	require.Len(t, roots, 3, "both extra roots survive sanitize")

	_, err := resolveInRoots(roots, filepath.Join(outside, "secret.go"))
	require.Error(t, err)
	assert.Contains(t, err.Error(), depA)
	assert.Contains(t, err.Error(), depB)
}

// TestResolveInRootSingleRootRefusalUnchanged pins that the single-root
// (workspace-only) refusal text - the common case, with no extra read roots
// configured - is untouched by naming roots in the multi-root message.
func TestResolveInRootSingleRootRefusalUnchanged(t *testing.T) {
	base := t.TempDir()
	workspace := filepath.Join(base, "workspace")
	require.NoError(t, os.MkdirAll(workspace, 0o755))

	_, err := resolveInRoot(workspace, filepath.Join(base, "secret.go"))
	require.Error(t, err)
	assert.Equal(t, `path "`+filepath.Join(base, "secret.go")+`" escapes workspace root`, err.Error())
}

// TestSanitizeReadRootsReasonPerCategory pins a reason for every category
// sanitizeReadRoots handles, keyed by DropReason rather than message text -
// the reason is a stable value a caller logs, not wire-ish prose.
func TestSanitizeReadRootsReasonPerCategory(t *testing.T) {
	base := t.TempDir()
	workspace := filepath.Join(base, "workspace")
	require.NoError(t, os.MkdirAll(workspace, 0o755))

	nonexistent := filepath.Join(base, "not-mounted-yet")

	result := sanitizeReadRoots(workspace, []string{"relative/path", "/", filepath.Dir(workspace), nonexistent})

	require.Len(t, result.Dropped, 4)
	assert.Equal(t, DropReasonRelative, result.Dropped[0].Reason)
	assert.Equal(t, DropReasonFilesystemRoot, result.Dropped[1].Reason)
	assert.Equal(t, DropReasonWorkspaceParent, result.Dropped[2].Reason)
	assert.Equal(t, DropReasonNonexistent, result.Dropped[3].Reason)
	assert.Empty(t, result.Effective)
}

// TestExtraRootsSchemaClause pins the discoverability decision: the schema
// clause mentions configured roots and is empty when none are configured.
func TestExtraRootsSchemaClause(t *testing.T) {
	assert.Empty(t, extraRootsSchemaClause(nil))

	clause := extraRootsSchemaClause([]string{"/deps/vendor"})
	assert.Contains(t, clause, "/deps/vendor")
}
