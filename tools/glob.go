package tools

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/mhersson/contextmatrix-harness/llm"
)

// globMaxPatternSegments bounds worst-case match cost: matchSegments is
// O(len(pattern)*len(pathSegs)) per candidate, run once per enumerated file,
// so an unbounded pattern length is an unbounded per-file cost multiplier
// over an entire tree. The count checked against this bound is taken after
// normalizeGlobPattern's synthetic leading "**/" (added unless the pattern is
// already anchored), so a caller-written N-segment pattern can validate as
// N+1 -- the "too complex" error says "normalized segments" for exactly this
// reason.
const globMaxPatternSegments = 64

// globMaxPatternLen bounds the total normalized pattern length in bytes: the
// segment cap above bounds segment COUNT, but filepath.Match's cost is also
// O(len(name)*len(chunk)) per segment, so a single long literal-bearing
// segment can multiply per-file matching cost well past what the segment cap
// alone prevents. 1024 bytes is generous for any real glob.
const globMaxPatternLen = 1024

// globScanBufSize is the max token size for the enumeration line scanner --
// comfortably above any real path length, bumped from bufio's 64 KiB default
// as a defensive margin for unusually long lines.
const globScanBufSize = 1 << 20

// globCtxCheckEvery is how often filterGlobStream checks ctx cancellation
// while scanning, so the check itself stays cheap on a large enumeration.
const globCtxCheckEvery = 4096

type GlobTool struct{ root string }

func NewGlobTool(root string) GlobTool { return GlobTool{root: root} }

func (t GlobTool) Name() string { return "glob" }

func (t GlobTool) Schema() llm.Tool {
	return llm.Tool{Type: "function", Function: llm.ToolFunction{
		Name:        "glob",
		Description: "List files matching a glob pattern, honoring .gitignore. Pattern matches the file's path relative to the search root, with a leading **/ implied: docs/*.md matches a docs/ directory at any depth; ** matches zero or more path segments, so */*.md matches any depth of one or more directories, not just one level; an empty pattern matches every file. Uses Go's filepath.Match glob syntax (*, ?, [...]), not shell glob -- e.g. *.{md,txt} brace expansion is literal and matches nothing. Optionally restrict to a subpath.",
		Parameters: json.RawMessage(`{
			"type":"object",
			"properties":{
				"pattern":{"type":"string","description":"glob pattern matched against the path relative to the search root, with a leading **/ implied -- docs/*.md matches a docs/ directory at any depth, ** matches zero or more path segments (so */*.md matches any depth of one or more, not just one level), e.g. *.md, docs/*.md, or docs/**/*.md; an empty pattern matches every file; uses Go's filepath.Match syntax (*, ?, [...]), not shell glob, so *.{md,txt} brace expansion is literal and matches nothing"},
				"path":{"type":"string","description":"optional subpath under the workspace root to search"}
			},
			"required":["pattern"]
		}`),
	}}
}

func (t GlobTool) Execute(ctx context.Context, args map[string]any) (Result, error) {
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

	rels, err := t.list(ctx, pattern, searchPath)
	if err != nil {
		return Result{}, err
	}

	if len(rels) == 0 {
		return Result{Text: "no matches"}, nil
	}

	return Result{Text: strings.Join(rels, "\n")}, nil
}

// list returns the workspace-relative paths matching pattern, preferring fd and
// falling back to rg. Both honor .gitignore natively: fd by default, rg via
// `--files` (see globViaRg for why).
func (t GlobTool) list(ctx context.Context, pattern, searchPath string) ([]string, error) {
	if bin := fdBinary(); bin != "" {
		return t.globViaFd(ctx, bin, pattern, searchPath)
	}

	if _, err := exec.LookPath("rg"); err == nil {
		return t.globViaRg(ctx, pattern, searchPath)
	}

	return nil, fmt.Errorf("glob requires fd or rg on PATH")
}

// globViaFd uses fd only for gitignore-aware file enumeration -- no --glob,
// no --full-path, so there is no glob matching left in fd to disagree with
// the rg path. "." is a regex matching any filename, so this lists every file
// under searchPath honoring .gitignore. fd exits 0 even when nothing matches,
// so any non-zero exit is a real error (not "no matches"). The pattern is
// applied by the same matcher as globViaRg (see streamEnumeration), so the
// two backends can only differ in which binary enumerates files, never in how
// a pattern is interpreted.
func (t GlobTool) globViaFd(ctx context.Context, bin, pattern, searchPath string) ([]string, error) {
	patternSegs, err := prepareGlob(pattern)
	if err != nil {
		return nil, err
	}

	cmd := exec.CommandContext(ctx, bin, "--type", "f", "--", ".", searchPath)
	cmd.Dir = t.root
	cmd.Env = ScrubbedEnv(nil)

	return t.streamEnumeration(ctx, cmd, patternSegs, "fd enumeration failed", nil)
}

// globViaRg lists files with `rg --files` (which honors .gitignore — unlike
// `rg --glob`, which overrides ignore rules) and applies the glob via the
// same streaming filter as globViaFd (see streamEnumeration). rg exits 1 when
// it finds no files, which ignoreExit below treats as "no matches" rather
// than an error; exit >= 2 is a real error.
func (t GlobTool) globViaRg(ctx context.Context, pattern, searchPath string) ([]string, error) {
	patternSegs, err := prepareGlob(pattern)
	if err != nil {
		return nil, err
	}

	cmd := exec.CommandContext(ctx, "rg", "--files", searchPath)
	cmd.Dir = t.root
	cmd.Env = ScrubbedEnv(nil)

	ignoreExit := func(err error) bool {
		ee, ok := err.(*exec.ExitError)

		return ok && ee.ExitCode() == 1
	}

	return t.streamEnumeration(ctx, cmd, patternSegs, "rg --files failed", ignoreExit)
}

// streamEnumeration starts cmd (already configured with Dir/Env and the
// enumeration binary/args) and filters its output against patternSegs. stdout
// is streamed line-by-line and filtered incrementally (see filterGlobStream)
// instead of being buffered through a capped writer, so a large tree can
// never be silently truncated; stdout is read in this goroutine via a pipe,
// while stderr (always small) is written directly into an in-memory
// capWriter by os/exec's own internal copy goroutine, so neither pipe can
// block the other. errPrefix labels any error from starting the command,
// streaming its output, or a non-zero exit (captured stderr is appended to
// the latter). ignoreExit lets a caller treat a specific non-zero exit as
// "no matches" rather than an error (rg's exit 1); pass nil to treat every
// non-zero exit as an error. On a stream error (e.g. bufio.ErrTooLong from a
// single enumerated line over globScanBufSize) the child may still be alive
// and blocked writing into the now-undrained pipe, so it is killed before
// Wait is called -- otherwise Wait blocks forever (mirrors the guard in
// bash.go). ctx cancellation is already safe via CommandContext's own
// kill-on-cancel, so this only matters for a scanner-side error.
func (t GlobTool) streamEnumeration(
	ctx context.Context, cmd *exec.Cmd, patternSegs []string, errPrefix string, ignoreExit func(error) bool,
) ([]string, error) {
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("%s: %w", errPrefix, err)
	}

	stderr := &capWriter{limit: subprocessOutputCap}
	cmd.Stderr = stderr

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("%s: %w", errPrefix, err)
	}

	rels, streamErr := filterGlobStream(ctx, stdout, t.root, patternSegs)
	if streamErr != nil {
		_ = cmd.Process.Kill() // child may be blocked writing to the pipe we stopped reading
	}

	waitErr := cmd.Wait()

	if streamErr != nil {
		return nil, fmt.Errorf("%s: %w", errPrefix, streamErr)
	}

	if waitErr != nil {
		if ignoreExit != nil && ignoreExit(waitErr) {
			return nil, nil
		}

		return nil, fmt.Errorf("%s: %v: %s", errPrefix, waitErr, strings.TrimSpace(stderr.String()))
	}

	return rels, nil
}

// normalizeGlobPattern anchors pattern to the search root: leading "./" or
// "/" segments are loop-trimmed (so doubled prefixes like "//x" or "././x"
// don't leave residue), any empty segment -- from any run of slashes,
// wherever it falls in the pattern, not just doubled-internal or trailing --
// is dropped (so "tools//sub" and "tools/sub/" both normalize to
// "tools/sub" -- no empty segment ever reaches the matcher), and "**/" is
// prepended unless pattern already starts with it or the pattern reduces to
// matching everything (empty, "/", "**", or "**/" all mean "match
// everything", consistently across both backends). globViaFd and globViaRg
// both reach this only via prepareGlob, so the "**/"-anywhere contract can't
// drift between the fd and rg code paths.
func normalizeGlobPattern(pattern string) string {
	for {
		trimmed := strings.TrimPrefix(pattern, "./")
		trimmed = strings.TrimPrefix(trimmed, "/")

		if trimmed == pattern {
			break
		}

		pattern = trimmed
	}

	segs := strings.Split(pattern, "/")

	nonEmpty := make([]string, 0, len(segs))

	for _, s := range segs {
		if s != "" {
			nonEmpty = append(nonEmpty, s)
		}
	}

	pattern = strings.Join(nonEmpty, "/")

	if pattern == "" || pattern == "**" {
		return "**"
	}

	if strings.HasPrefix(pattern, "**/") {
		return pattern
	}

	return "**/" + pattern
}

// validateGlobPatternSegs bounds the total segment count -- matchSegments is
// O(len(pattern)*len(pathSegs)) per candidate, run once per enumerated file,
// so an unbounded pattern length is an unbounded per-file cost multiplier --
// and validates each non-"**" segment with filepath.Match(seg, seg) rather
// than filepath.Match(seg, ""): matching against the segment itself
// guarantees any leading literal is satisfied, so a malformed tail (e.g.
// "ab*cd[") is actually scanned and surfaces filepath.ErrBadPattern, instead
// of failing the literal match first and never reaching the bad syntax.
// original is the pre-split pattern, used only for the error message.
func validateGlobPatternSegs(original string, patternSegs []string) error {
	if n := len(patternSegs); n > globMaxPatternSegments {
		return fmt.Errorf("glob: pattern too complex (%d normalized segments, max %d)", n, globMaxPatternSegments)
	}

	if n := len(strings.Join(patternSegs, "/")); n > globMaxPatternLen {
		return fmt.Errorf("glob: pattern too long (%d normalized bytes, max %d)", n, globMaxPatternLen)
	}

	for _, seg := range patternSegs {
		if seg == "**" {
			continue
		}

		if _, err := filepath.Match(seg, seg); err != nil {
			return fmt.Errorf("glob: invalid pattern %q: %w", original, err)
		}
	}

	return nil
}

// prepareGlob normalizes pattern (see normalizeGlobPattern) and validates it
// before any enumeration subprocess is spawned, so a malformed or oversized
// pattern fails fast without listing the tree. globViaFd and globViaRg both
// call this once up front, so the two backends share one validation gate.
func prepareGlob(pattern string) ([]string, error) {
	patternSegs := strings.Split(normalizeGlobPattern(pattern), "/")

	if err := validateGlobPatternSegs(pattern, patternSegs); err != nil {
		return nil, err
	}

	return patternSegs, nil
}

// filterByGlob keeps the entries in rels matching pattern: pattern is split
// and matched as-is, with no implicit "**/" anchoring -- any-depth matching
// for bare patterns comes from normalizeGlobPattern (see prepareGlob), not
// from this matcher. Kept for direct unit testing of the pure segment-match
// behavior; production callers (globViaFd, globViaRg) go through prepareGlob
// and streamEnumeration instead, since they stream subprocess output rather
// than operate on an in-memory rels slice.
func filterByGlob(rels []string, pattern string) ([]string, error) {
	patternSegs := strings.Split(pattern, "/")

	if err := validateGlobPatternSegs(pattern, patternSegs); err != nil {
		return nil, err
	}

	return filterGlobStream(context.Background(), strings.NewReader(strings.Join(rels, "\n")), "", patternSegs)
}

// filterGlobStream reads newline-separated candidate paths from r, strips a
// root+"/" prefix per line (root == "" is a no-op, used when candidates are
// already relative), and keeps the ones whose "/"-split segments match
// patternSegs (see matchSegments). Matches are collected incrementally as
// they are found -- memory is O(matches), not O(candidates) -- so a large
// enumeration is never buffered whole and then capped. ctx is checked up
// front (covers an empty read or an already-cancelled context) and every
// globCtxCheckEvery lines thereafter, so a cancelled request stops scanning
// instead of running to completion.
func filterGlobStream(ctx context.Context, r io.Reader, root string, patternSegs []string) ([]string, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	rootPrefix := ""
	if root != "" {
		rootPrefix = root + "/"
	}

	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), globScanBufSize)

	var out []string

	for i := 0; scanner.Scan(); i++ {
		if i%globCtxCheckEvery == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}

		ln := strings.TrimSpace(scanner.Text())
		if ln == "" {
			continue
		}

		rel := strings.TrimPrefix(ln, rootPrefix)
		if matchSegments(patternSegs, strings.Split(rel, "/")) {
			out = append(out, rel)
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("glob: read enumeration output: %w", err)
	}

	return out, nil
}

// matchSegments reports whether pathSegs matches pattern: a "**" segment
// matches zero or more path segments, and every other segment is matched with
// stdlib filepath.Match (validateGlobPatternSegs validates segments up front,
// so a malformed pattern surfaces as an error before reaching here). This is
// the standard two-row wildcard DP at segment granularity: prev holds, for
// the pattern segments consumed in the previous row, whether they match the
// first j path segments; curr is this row's result, built left to right (a
// "**" may reuse curr[j-1] -- it can absorb the just-matched segment too) and
// then swapped into prev for the next row. Two alternating []bool rows are
// O(len(pathSegs)) space and O(len(pattern)*len(pathSegs)) time, with no
// per-candidate map allocation or hashing.
func matchSegments(pattern, pathSegs []string) bool {
	n := len(pathSegs)

	prev := make([]bool, n+1)
	curr := make([]bool, n+1)

	prev[0] = true

	for _, seg := range pattern {
		isGlobstar := seg == "**"

		curr[0] = isGlobstar && prev[0]

		for j := 1; j <= n; j++ {
			if isGlobstar {
				curr[j] = prev[j] || curr[j-1]
			} else {
				ok, _ := filepath.Match(seg, pathSegs[j-1])
				curr[j] = ok && prev[j-1]
			}
		}

		prev, curr = curr, prev
	}

	return prev[n]
}

// fdBinary returns the fd executable name on PATH ("fd" or Debian's "fdfind"),
// or "" if neither is present.
func fdBinary() string {
	for _, name := range []string{"fd", "fdfind"} {
		if _, err := exec.LookPath(name); err == nil {
			return name
		}
	}

	return ""
}
