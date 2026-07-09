package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGlobTool(t *testing.T) {
	if fdBinary() == "" {
		if _, err := exec.LookPath("rg"); err != nil {
			t.Skip("neither fd nor rg installed")
		}
	}

	root := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(root, "a.go"), []byte("package x\n"), 0o644))
	require.NoError(t, os.MkdirAll(filepath.Join(root, "sub"), 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(root, "sub", "b.go"), []byte("package y\n"), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(root, "c.txt"), []byte("nope\n"), 0o644))

	out, err := NewGlobTool(root).Execute(context.Background(), map[string]any{"pattern": "*.go"})
	require.NoError(t, err)
	assert.Contains(t, out.Text, "a.go")
	assert.Contains(t, out.Text, filepath.Join("sub", "b.go"))
	assert.NotContains(t, out.Text, "c.txt")

	out, err = NewGlobTool(root).Execute(context.Background(), map[string]any{"pattern": "*.nope"})
	require.NoError(t, err)
	assert.Contains(t, out.Text, "no matches")
}

func TestGlobToolRespectsGitignore(t *testing.T) {
	if fdBinary() == "" {
		if _, err := exec.LookPath("rg"); err != nil {
			t.Skip("neither fd nor rg installed")
		}
	}

	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not installed")
	}

	root := t.TempDir()
	cmd := exec.Command("git", "init")
	cmd.Dir = root
	require.NoError(t, cmd.Run())
	require.NoError(t, os.WriteFile(filepath.Join(root, ".gitignore"), []byte("ignored.go\n"), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(root, "kept.go"), []byte("package x\n"), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(root, "ignored.go"), []byte("package x\n"), 0o644))
	require.NoError(t, os.MkdirAll(filepath.Join(root, "tools"), 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(root, "tools", "kept.go"), []byte("package x\n"), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(root, "tools", "ignored.go"), []byte("package x\n"), 0o644))

	out, err := NewGlobTool(root).Execute(context.Background(), map[string]any{"pattern": "*.go"})
	require.NoError(t, err)
	assert.Contains(t, out.Text, "kept.go")
	assert.NotContains(t, out.Text, "ignored.go")

	// Directory-relative pattern, so the ignore boundary is also exercised on
	// the normalized ("**/tools/*.go") path via whichever backend Execute
	// prefers (fd when installed), not only the rg-direct test below.
	out, err = NewGlobTool(root).Execute(context.Background(), map[string]any{"pattern": "tools/*.go"})
	require.NoError(t, err)
	assert.Contains(t, out.Text, filepath.Join("tools", "kept.go"))
	assert.NotContains(t, out.Text, filepath.Join("tools", "ignored.go"))
}

// writeGlobFixture creates a fixture tree shared by the directory-pattern
// tests: a root-level .go file, a tools/ directory with direct and nested .go
// files, a second same-named tools/ directory at a different depth (proves
// "any depth" rather than "root only"), and a non-.go file to prove pattern
// anchoring.
func writeGlobFixture(t *testing.T, root string) {
	t.Helper()

	require.NoError(t, os.WriteFile(filepath.Join(root, "root.go"), []byte("package x\n"), 0o644))
	require.NoError(t, os.MkdirAll(filepath.Join(root, "tools", "sub", "deep"), 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(root, "tools", "a.go"), []byte("package x\n"), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(root, "tools", "b.go"), []byte("package x\n"), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(root, "tools", "readme.txt"), []byte("readme\n"), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(root, "tools", "sub", "c.go"), []byte("package x\n"), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(root, "tools", "sub", "deep", "d.go"), []byte("package x\n"), 0o644))
	require.NoError(t, os.MkdirAll(filepath.Join(root, "other", "tools"), 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(root, "other", "tools", "e.go"), []byte("package x\n"), 0o644))
}

// TestGlobDirectoryPatterns pins one shared expected-set table for the card
// patterns across BOTH the fd and rg code paths, so results no longer depend
// on which binary is installed -- fd enumerates files and rg enumerates
// files, but both are filtered by the same matchSegments-backed matcher.
func TestGlobDirectoryPatterns(t *testing.T) {
	root := t.TempDir()
	writeGlobFixture(t, root)

	allGo := []string{
		"root.go",
		filepath.Join("tools", "a.go"),
		filepath.Join("tools", "b.go"),
		filepath.Join("tools", "sub", "c.go"),
		filepath.Join("tools", "sub", "deep", "d.go"),
		filepath.Join("other", "tools", "e.go"),
	}

	// allFiles additionally includes the non-.go file, since an
	// empty/match-everything pattern is not anchored to any extension.
	allFiles := append(append([]string{}, allGo...), filepath.Join("tools", "readme.txt"))

	cases := []struct {
		name    string
		pattern string
		want    []string
	}{
		{name: "basename", pattern: "*.go", want: allGo},
		{name: "already anchored", pattern: "**/*.go", want: allGo},
		{
			name:    "directory relative",
			pattern: "tools/*.go",
			want: []string{
				filepath.Join("tools", "a.go"),
				filepath.Join("tools", "b.go"),
				filepath.Join("other", "tools", "e.go"),
			},
		},
		{
			name:    "directory relative recursive",
			pattern: "tools/**/*.go",
			want: []string{
				filepath.Join("tools", "a.go"),
				filepath.Join("tools", "b.go"),
				filepath.Join("tools", "sub", "c.go"),
				filepath.Join("tools", "sub", "deep", "d.go"),
				filepath.Join("other", "tools", "e.go"),
			},
		},
		{
			// Regression for the fd Critical: normalizeGlobPattern implies a
			// leading "**/", so a bare-wildcard-leading pattern matches any file
			// with at least one enclosing directory -- root.go (zero dirs deep)
			// must be excluded. Previously fd matched the OS-absolute path, so
			// "*/" could bind to a real ancestor directory above the search
			// root and incorrectly include root.go; matching root-relative
			// paths makes the result independent of where the search root sits
			// on disk.
			name:    "bare wildcard directory",
			pattern: "*/*.go",
			want:    allGo[1:], // everything except the root-level file
		},
		{name: "empty pattern matches everything", pattern: "", want: allFiles},
	}

	for _, tc := range cases {
		t.Run(tc.name+"/fd", func(t *testing.T) {
			if fdBinary() == "" {
				t.Skip("fd not installed")
			}

			rels, err := NewGlobTool(root).globViaFd(context.Background(), fdBinary(), tc.pattern, root)
			require.NoError(t, err)
			assert.ElementsMatch(t, tc.want, rels)
		})

		t.Run(tc.name+"/rg", func(t *testing.T) {
			if _, err := exec.LookPath("rg"); err != nil {
				t.Skip("rg not installed")
			}

			rels, err := NewGlobTool(root).globViaRg(context.Background(), tc.pattern, root)
			require.NoError(t, err)
			assert.ElementsMatch(t, tc.want, rels)
		})
	}
}

func TestGlobToolPathScoped(t *testing.T) {
	if fdBinary() == "" {
		if _, err := exec.LookPath("rg"); err != nil {
			t.Skip("neither fd nor rg installed")
		}
	}

	root := t.TempDir()
	writeGlobFixture(t, root)

	out, err := NewGlobTool(root).Execute(context.Background(), map[string]any{"pattern": "*.go", "path": "tools"})
	require.NoError(t, err)
	assert.Contains(t, out.Text, filepath.Join("tools", "a.go"))
	assert.Contains(t, out.Text, filepath.Join("tools", "b.go"))
	assert.Contains(t, out.Text, filepath.Join("tools", "sub", "c.go"))
	assert.Contains(t, out.Text, filepath.Join("tools", "sub", "deep", "d.go"))
	assert.NotContains(t, out.Text, "root.go")

	out, err = NewGlobTool(root).Execute(context.Background(), map[string]any{"pattern": "sub/*.go", "path": "tools"})
	require.NoError(t, err)
	assert.Contains(t, out.Text, filepath.Join("tools", "sub", "c.go"))
	assert.NotContains(t, out.Text, filepath.Join("tools", "sub", "deep", "d.go"))
}

// TestGlobToolSchemaDocumentsMatchDialect pins the Round-3 Minor fix: models
// otherwise default to shell-glob assumptions (e.g. brace expansion), but
// filepath.Match treats "{" and "}" as ordinary literal characters, so
// "*.{md,txt}" silently matches nothing -- the same wasted-turn failure mode
// fixed for directory-relative patterns.
func TestGlobToolSchemaDocumentsMatchDialect(t *testing.T) {
	schema := NewGlobTool("/w").Schema().Function
	assert.Contains(t, schema.Description, "filepath.Match")

	var params struct {
		Properties struct {
			Pattern struct {
				Description string `json:"description"`
			} `json:"pattern"`
		} `json:"properties"`
	}

	require.NoError(t, json.Unmarshal(schema.Parameters, &params))
	assert.Contains(t, params.Properties.Pattern.Description, "filepath.Match")
}

func TestNormalizeGlobPattern(t *testing.T) {
	cases := []struct {
		name    string
		pattern string
		want    string
	}{
		{name: "basename gets prefixed", pattern: "*.go", want: "**/*.go"},
		{name: "already anchored is unchanged", pattern: "**/*.go", want: "**/*.go"},
		{name: "already anchored with directory is unchanged", pattern: "**/tools/**/*.go", want: "**/tools/**/*.go"},
		{name: "directory relative gets prefixed", pattern: "tools/*.go", want: "**/tools/*.go"},
		{name: "leading dot-slash is trimmed then prefixed", pattern: "./tools/*.go", want: "**/tools/*.go"},
		{name: "leading slash is trimmed then prefixed", pattern: "/tools/*.go", want: "**/tools/*.go"},
		{name: "doubled leading dot-slash is trimmed", pattern: "././tools/*.go", want: "**/tools/*.go"},
		{name: "doubled leading slash is trimmed", pattern: "//tools/*.go", want: "**/tools/*.go"},
		{name: "empty pattern matches everything", pattern: "", want: "**"},
		{name: "lone slash matches everything", pattern: "/", want: "**"},
		{name: "trailing globstar slash matches everything", pattern: "**/", want: "**"},
		{name: "bare globstar is unchanged", pattern: "**", want: "**"},
		{name: "doubled internal slash collapses", pattern: "tools//sub/*.go", want: "**/tools/sub/*.go"},
		{name: "trailing slash is stripped", pattern: "tools/sub/", want: "**/tools/sub"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, normalizeGlobPattern(tc.pattern))
		})
	}
}

func TestFilterByGlob(t *testing.T) {
	files := []string{"a.go", "sub/b.go", "sub/deep/d.go", "c.txt"}

	cases := []struct {
		name    string
		files   []string
		pattern string
		want    []string
		wantErr bool
	}{
		{
			// No separator → matches only a single path segment; any-depth
			// matching for bare patterns comes from normalizeGlobPattern's **/
			// prefix, not from this matcher.
			name: "basename", files: files, pattern: "*.go", want: []string{"a.go"},
		},
		{
			// Leading ** matches zero or more path segments, so nested files match too.
			name:    "leading globstar",
			files:   files,
			pattern: "**/*.go",
			want:    []string{"a.go", "sub/b.go", "sub/deep/d.go"},
		},
		{name: "other extension", files: files, pattern: "*.txt", want: []string{"c.txt"}},
		{
			// Pattern with a separator → match the relative path; * does not cross '/'.
			name: "directory relative", files: files, pattern: "sub/*.go", want: []string{"sub/b.go"},
		},
		{
			// Mid-pattern ** matches zero or more intervening components
			// (previously a stdlib filepath.Match limitation capped this at
			// exactly one).
			name:    "mid-pattern globstar",
			files:   files,
			pattern: "sub/**/*.go",
			want:    []string{"sub/b.go", "sub/deep/d.go"},
		},
		{name: "no match", files: files, pattern: "*.md", want: nil},
		{
			// ** absorbs zero segments between "tools/" and the match, not just
			// one or more.
			name:    "zero component globstar",
			files:   []string{"tools/a.go", "tools/sub/deep/d.go", "tools/readme.txt"},
			pattern: "tools/**/*.go",
			want:    []string{"tools/a.go", "tools/sub/deep/d.go"},
		},
		{name: "malformed pattern segment: leading bad bracket", files: files, pattern: "[*.go", wantErr: true},
		{
			// filepath.Match(seg, "") fails the leading literal "ab" before ever
			// scanning the unclosed "[" class, so this shape used to silently
			// return "no matches" instead of erroring; validating with
			// filepath.Match(seg, seg) guarantees the malformed tail is reached.
			name: "malformed pattern segment: literal then bad bracket", files: files, pattern: "ab*cd[", wantErr: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := filterByGlob(tc.files, tc.pattern)
			if tc.wantErr {
				require.Error(t, err)
				assert.ErrorIs(t, err, filepath.ErrBadPattern)

				return
			}

			require.NoError(t, err)
			assert.Equal(t, tc.want, got)
		})
	}
}

// TestMatchSegmentsManyGlobstarsIsFast pins the polynomial-time fix for the
// exponential backtracking regression: 30 "**" segments matched against a
// non-matching 21-segment path took 12+s with naive backtracking (reproduced
// against the pre-fix implementation); the iterative two-row DP (see
// matchSegments) is O(len(pattern)*len(pathSegs)) and finishes in
// microseconds. The goroutine + timeout gives a clear failure instead of
// hanging the whole test binary if the regression reappears.
func TestMatchSegmentsManyGlobstarsIsFast(t *testing.T) {
	patternSegs := make([]string, 30)
	for i := range patternSegs {
		patternSegs[i] = "**"
	}

	patternSegs = append(patternSegs, "nomatch.go")

	pathSegs := make([]string, 20)
	for i := range pathSegs {
		pathSegs[i] = "dir"
	}

	pathSegs = append(pathSegs, "other.go")

	done := make(chan bool, 1)

	go func() {
		done <- matchSegments(patternSegs, pathSegs)
	}()

	select {
	case got := <-done:
		assert.False(t, got)
	case <-time.After(3 * time.Second):
		t.Fatal("matchSegments took too long on a **-heavy pattern -- possible exponential backtracking regression")
	}
}

// TestGlobBackendsMalformedPattern proves the malformed-pattern error is
// surfaced identically on both backends, not just from filterByGlob directly.
// Both shapes are validated up front by prepareGlob, before either backend
// spawns its enumeration subprocess.
func TestGlobBackendsMalformedPattern(t *testing.T) {
	root := t.TempDir()
	writeGlobFixture(t, root)

	patterns := []string{
		"tools/[*.go",  // unclosed character class
		"tools/ab*cd[", // literal + star + unclosed class -- only caught by
		// validating filepath.Match(seg, seg) instead of (seg, "")
	}

	for _, pattern := range patterns {
		t.Run(pattern+"/fd", func(t *testing.T) {
			if fdBinary() == "" {
				t.Skip("fd not installed")
			}

			_, err := NewGlobTool(root).globViaFd(context.Background(), fdBinary(), pattern, root)
			require.Error(t, err)
			assert.ErrorIs(t, err, filepath.ErrBadPattern)
		})

		t.Run(pattern+"/rg", func(t *testing.T) {
			if _, err := exec.LookPath("rg"); err != nil {
				t.Skip("rg not installed")
			}

			_, err := NewGlobTool(root).globViaRg(context.Background(), pattern, root)
			require.Error(t, err)
			assert.ErrorIs(t, err, filepath.ErrBadPattern)
		})
	}
}

func TestGlobViaRgRespectsGitignore(t *testing.T) {
	if _, err := exec.LookPath("rg"); err != nil {
		t.Skip("rg not installed")
	}

	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not installed")
	}

	root := t.TempDir()
	cmd := exec.Command("git", "init")
	cmd.Dir = root
	require.NoError(t, cmd.Run())
	require.NoError(t, os.WriteFile(filepath.Join(root, ".gitignore"), []byte("ignored.go\n"), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(root, "kept.go"), []byte("package x\n"), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(root, "ignored.go"), []byte("package x\n"), 0o644))
	require.NoError(t, os.MkdirAll(filepath.Join(root, "tools"), 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(root, "tools", "kept.go"), []byte("package x\n"), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(root, "tools", "ignored.go"), []byte("package x\n"), 0o644))

	// Call the rg path DIRECTLY so it is exercised even on a host where fd exists
	// (fd would otherwise shadow rg inside Execute).
	rels, err := NewGlobTool(root).globViaRg(context.Background(), "*.go", root)
	require.NoError(t, err)
	assert.Contains(t, rels, "kept.go")
	assert.NotContains(t, rels, "ignored.go")

	// Directory-relative pattern, so the ignore boundary is also exercised on
	// the normalized ("**/tools/*.go") path, not just a basename pattern.
	rels, err = NewGlobTool(root).globViaRg(context.Background(), "tools/*.go", root)
	require.NoError(t, err)
	assert.Contains(t, rels, filepath.Join("tools", "kept.go"))
	assert.NotContains(t, rels, filepath.Join("tools", "ignored.go"))
}

// TestFilterGlobStreamNoTruncation pins the Round-2 Critical fix: enumeration
// output used to flow through a 10 MiB capWriter before being filtered, so a
// large tree could be silently truncated mid-list with err == nil, and the
// "[output truncated]" sentinel could be returned as a fake path. This
// synthetic reader exceeds that old cap by a wide margin; filterGlobStream
// must still return every match with no truncation marker, proving matches
// are collected incrementally (memory O(matches)) rather than buffered whole
// then capped.
func TestFilterGlobStreamNoTruncation(t *testing.T) {
	const totalLines = 250_000

	var (
		buf  bytes.Buffer
		want []string
	)

	for i := range totalLines {
		ext := "txt"
		if i%2 == 0 {
			ext = "go"
		}

		line := fmt.Sprintf("some/long/synthetic/directory/tree/dir/file%08d.%s", i, ext)

		buf.WriteString(line)
		buf.WriteByte('\n')

		if ext == "go" {
			want = append(want, line)
		}
	}

	require.Greater(t, buf.Len(), 10<<20, "fixture must exceed the old 10 MiB output cap to prove there is no truncation")

	patternSegs, err := prepareGlob("dir/*.go")
	require.NoError(t, err)

	got, err := filterGlobStream(context.Background(), &buf, "", patternSegs)
	require.NoError(t, err)
	assert.Equal(t, want, got)
	assert.NotContains(t, strings.Join(got, "\n"), "[output truncated]")
}

// TestFilterGlobStreamHonorsCancelledContext proves a cancelled context
// surfaces as an error instead of the filter loop running to completion
// uncancellably once enumeration output starts arriving.
func TestFilterGlobStreamHonorsCancelledContext(t *testing.T) {
	patternSegs, err := prepareGlob("*.go")
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err = filterGlobStream(ctx, strings.NewReader("a.go\nb.go\n"), "", patternSegs)
	require.Error(t, err)
	assert.ErrorIs(t, err, context.Canceled)
}

// TestStreamEnumerationKillsOnScannerError pins the Round-3 Important fix: on
// a scanner read error (bufio.ErrTooLong from a single enumerated line over
// globScanBufSize), filterGlobStream returns early while the enumeration
// child is still alive and blocked writing the rest of its output into the
// now-undrained pipe; calling cmd.Wait() without killing the child first
// blocks forever (mirrors the guard in tools/bash.go). The oversized line
// comes from "head -c <n> /dev/zero" invoked directly (no shell wrapper) so
// cmd.Process is the one and only process holding the pipes -- exactly like
// production, where cmd is always fd or rg directly, never a shell pipeline.
// An earlier version of this test used "sh -c 'head ... | tr ...'", but any
// child sh forks inherits a dup of the stderr fd; killing only sh's PID left
// that grandchild alive and still holding the stderr pipe open, so the
// capWriter io.Copy goroutine Wait() also waits on never saw EOF -- a hang
// caused by the test's shape, not a flaw in the fix. A real subprocess is
// used, not a synthetic reader, so cmd.Wait is genuinely exercised; the
// goroutine + timeout gives a clear failure instead of hanging the whole
// test binary if the kill ever regresses.
func TestStreamEnumerationKillsOnScannerError(t *testing.T) {
	if _, err := exec.LookPath("head"); err != nil {
		t.Skip("head not installed")
	}

	patternSegs, err := prepareGlob("*.go")
	require.NoError(t, err)

	ctx := context.Background()
	// 2,000,000 NUL bytes, no newline anywhere -- one token far over the 1
	// MiB globScanBufSize cap, with output left over for head to block on
	// writing once the scanner stops draining the pipe.
	cmd := exec.CommandContext(ctx, "head", "-c", "2000000", "/dev/zero")

	done := make(chan error, 1)

	go func() {
		_, err := NewGlobTool(t.TempDir()).streamEnumeration(ctx, cmd, patternSegs, "test enumeration failed", nil)
		done <- err
	}()

	select {
	case err := <-done:
		require.Error(t, err)
		assert.Contains(t, err.Error(), "test enumeration failed")
		assert.Contains(t, err.Error(), "token too long")
	case <-time.After(5 * time.Second):
		t.Fatal("streamEnumeration hung on a scanner read error -- cmd.Wait() blocked because the child was never killed")
	}
}

// TestPrepareGlobPatternTooComplex pins the Round-2 Important fix: matching
// cost is O(len(pattern)*len(pathSegs)) per enumerated file, so an unbounded
// pattern length is an unbounded per-file cost multiplier over an entire
// tree. 64 segments is the boundary: at the limit prepareGlob must still
// succeed, one over must fail fast, before any enumeration.
func TestPrepareGlobPatternTooComplex(t *testing.T) {
	makePattern := func(n int) string {
		segs := make([]string, n)
		segs[0] = "**"

		for i := 1; i < n; i++ {
			segs[i] = "a"
		}

		return strings.Join(segs, "/")
	}

	_, err := prepareGlob(makePattern(64))
	require.NoError(t, err)

	_, err = prepareGlob(makePattern(65))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "too complex")
}

// TestPrepareGlobPatternTooComplexOffByOne pins the Round-3 Minor fix: the
// segment count in the "too complex" error is taken after
// normalizeGlobPattern's synthetic leading "**/" (unlike
// TestPrepareGlobPatternTooComplex above, this pattern has no "**" segment of
// its own), so a caller-written globMaxPatternSegments-segment pattern lands
// one over the cap once normalized. The error message says "normalized
// segments" precisely so this isn't a silent, unexplained off-by-one.
func TestPrepareGlobPatternTooComplexOffByOne(t *testing.T) {
	makePattern := func(n int) string {
		segs := make([]string, n)
		for i := range segs {
			segs[i] = "a"
		}

		return strings.Join(segs, "/")
	}

	// globMaxPatternSegments-1 caller segments + 1 synthetic "**/" == the cap
	// exactly -- must still succeed.
	_, err := prepareGlob(makePattern(globMaxPatternSegments - 1))
	require.NoError(t, err)

	// globMaxPatternSegments caller segments + 1 synthetic "**/" == one over
	// the cap.
	_, err = prepareGlob(makePattern(globMaxPatternSegments))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "too complex")
	assert.Contains(t, err.Error(), "normalized")
}

// TestPrepareGlobPatternTooLong pins the Round-3 Important fix: the
// 64-segment cap above bounds segment COUNT but not LENGTH, and
// filepath.Match's cost is O(len(name)*len(chunk)) -- a single long
// literal-bearing segment (well under 64 segments) can still multiply
// per-file matching cost tens to over a hundred times over an entire tree.
// prepareGlob must reject an over-long pattern before any enumeration
// subprocess spawns; at the boundary it must still succeed, one byte over
// must fail fast.
func TestPrepareGlobPatternTooLong(t *testing.T) {
	// One segment, comfortably under the 64-segment cap, sized to land
	// exactly on globMaxPatternLen once normalized ("**/" + literal).
	within := strings.Repeat("a", globMaxPatternLen-len("**/"))

	_, err := prepareGlob(within)
	require.NoError(t, err)

	over := within + "a"

	_, err = prepareGlob(over)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "too long")
}
