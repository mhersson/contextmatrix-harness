package tools

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// resolveInRoot resolves p (relative to root, or absolute), guaranteeing the
// result stays within root even across symlinks. The returned path is based on
// the (unresolved) clean root so callers see a stable workspace-relative path,
// while containment is checked against the symlink-resolved location. Symlinks
// on the deepest existing ancestor are resolved, so not-yet-created files
// (write/edit targets) still validate.
//
// TOCTOU: containment is checked against the symlink-resolved path, but callers
// open the returned (unresolved) path afterward, so a symlink swapped between
// this check and the open could redirect outside root. This is accepted under
// the single-tenant, non-adversarial-workspace trust model - the agent is the
// sole actor on the workspace. Fully closing it needs openat2(2) with
// RESOLVE_NO_SYMLINKS (Linux 5.6+).
func resolveInRoot(root, p string) (string, error) {
	return resolveInRoots([]string{root}, p)
}

// resolveInRoots is resolveInRoot over an ordered set. roots[0] is the
// workspace: a relative path is joined against it, and it is what the returned
// path is expressed against. Containment is then satisfied by ANY root, so an
// absolute path (or one reaching out with ../) into a caller-permitted
// read-only root resolves instead of being refused.
//
// The extra roots are caller-supplied at construction and never derived from a
// tool argument, so no model-produced string can widen the set. Write tools
// pass a single root and are unaffected.
func resolveInRoots(roots []string, p string) (string, error) {
	if len(roots) == 0 {
		return "", fmt.Errorf("path %q: no permitted root configured", p)
	}

	primary := filepath.Clean(roots[0])

	abs := p
	if !filepath.IsAbs(p) {
		abs = filepath.Join(primary, p)
	}

	abs = filepath.Clean(abs)

	resolved := resolveExisting(abs)

	for _, root := range roots {
		rootResolved := evalOrSelf(filepath.Clean(root))
		if resolved == rootResolved || strings.HasPrefix(resolved, rootResolved+string(os.PathSeparator)) {
			return abs, nil
		}
	}

	if len(roots) == 1 {
		return "", fmt.Errorf("path %q escapes workspace root", p)
	}

	return "", fmt.Errorf("path %q is outside the workspace and every permitted read-only root: %s",
		p, strings.Join(roots[1:], ", "))
}

// DropReason identifies why sanitizeReadRoots refused a caller-supplied extra
// read root - see ReadRoots.
type DropReason string

const (
	// DropReasonRelative covers both a relative path and a blank entry: an
	// empty or whitespace-only string is not absolute either, so it falls out
	// of the same check.
	DropReasonRelative        DropReason = "relative"
	DropReasonNonexistent     DropReason = "nonexistent"
	DropReasonFilesystemRoot  DropReason = "/"
	DropReasonWorkspaceParent DropReason = "workspace-parent"
)

// DroppedReadRoot is one caller-supplied extra read root that sanitizeReadRoots
// refused, paired with why. Root is the entry as given by the caller, not the
// resolved form, so it matches whatever the caller configured.
type DroppedReadRoot struct {
	Root   string
	Reason DropReason
}

// ReadRoots is sanitizeReadRoots' full outcome: the roots that survived
// (symlink-resolved, in the order given - this is what resolveInRoots checks)
// and, for observability, the ones dropped and why. Dropping is silent and
// one-directional - a misconfigured root narrows access, it never widens it -
// so a caller cannot accidentally open the filesystem; ReadRoots exists so the
// caller can still find out what happened. The harness never logs a drop
// itself - a caller that wants an audit trail reads this after construction
// (see e.g. ReadTool.ReadRoots).
type ReadRoots struct {
	Effective []string
	Dropped   []DroppedReadRoot
}

// sanitizeReadRoots drops extra read roots that would widen access rather than
// add a sibling tree: a relative or blank path, one that cannot be resolved,
// the filesystem root, and any root that contains the workspace (which would
// permit the workspace's whole parentage).
func sanitizeReadRoots(workspace string, roots []string) ReadRoots {
	if len(roots) == 0 {
		return ReadRoots{}
	}

	sep := string(os.PathSeparator)

	// Compare RESOLVED paths. resolveInRoots checks containment against the
	// symlink-resolved root, so validating the unresolved one would wave through
	// a root that is a symlink to / or to the workspace's parent - defeating the
	// very widening this function exists to refuse.
	wsResolved := evalOrSelf(filepath.Clean(workspace))

	var result ReadRoots

	for _, r := range roots {
		rc := filepath.Clean(r)

		// A relative root (blank entries included: filepath.Clean("") is "."
		// and neither is absolute) would resolve against the process working
		// directory, which is never what a caller configuring a dependency tree
		// meant.
		if !filepath.IsAbs(rc) {
			result.Dropped = append(result.Dropped, DroppedReadRoot{Root: r, Reason: DropReasonRelative})

			continue
		}

		// EvalSymlinks, not evalOrSelf: a root that cannot be resolved NOW cannot
		// be judged now either, and one that appears later - a dependency tree
		// mounted after the tools were built - can point anywhere. resolveInRoots
		// re-resolves per call, so keeping an unresolvable root would defer this
		// gate to a moment that never runs it.
		resolved, rerr := filepath.EvalSymlinks(rc)
		if rerr != nil {
			result.Dropped = append(result.Dropped, DroppedReadRoot{Root: r, Reason: DropReasonNonexistent})

			continue
		}

		if resolved == sep {
			result.Dropped = append(result.Dropped, DroppedReadRoot{Root: r, Reason: DropReasonFilesystemRoot})

			continue
		}

		// resolved contains (or is) the workspace: permitting it would widen.
		if wsResolved == resolved || strings.HasPrefix(wsResolved, resolved+sep) {
			result.Dropped = append(result.Dropped, DroppedReadRoot{Root: r, Reason: DropReasonWorkspaceParent})

			continue
		}

		result.Effective = append(result.Effective, resolved)
	}

	return result
}

// extraRootsSchemaClause returns a sentence naming a tool's configured extra
// read-only roots, or "" when none survived sanitizeReadRoots. Appended to a
// read-only tool's schema description so a model can discover the capability
// instead of only benefiting from it by guessing absolute paths.
func extraRootsSchemaClause(effective []string) string {
	if len(effective) == 0 {
		return ""
	}

	return " An absolute path under any of these additional read-only roots is also permitted: " +
		strings.Join(effective, ", ") + "."
}

// extraRootsParamClause returns a short clause for a path-like parameter's own
// description, or "" when no extra read-only roots are configured. The roots
// themselves are already named in the tool-level Description
// (extraRootsSchemaClause) - this only keeps the parameter text from
// contradicting it by still reading as workspace-only.
func extraRootsParamClause(effective []string) string {
	if len(effective) == 0 {
		return ""
	}

	return ", or an absolute path under one of this tool's configured additional read-only roots (see the tool description)"
}

// jsonString returns s as a JSON string literal (quoted and escaped), for
// splicing a dynamic value into a hand-written JSON schema template without
// hand-rolling escaping.
func jsonString(s string) string {
	b, err := json.Marshal(s)
	if err != nil {
		// s is always a plain Go string - json.Marshal of a string cannot fail.
		return `""`
	}

	return string(b)
}

// evalOrSelf returns EvalSymlinks(p), or p unchanged if it cannot be resolved.
func evalOrSelf(p string) string {
	if r, err := filepath.EvalSymlinks(p); err == nil {
		return r
	}

	return p
}

// resolveExisting walks up to the deepest existing ancestor of abs, resolves
// its symlinks, then rejoins the non-existent suffix.
func resolveExisting(abs string) string {
	suffix := ""

	cur := abs
	for {
		if resolved, err := filepath.EvalSymlinks(cur); err == nil {
			if suffix == "" {
				return resolved
			}

			return filepath.Join(resolved, suffix)
		}

		parent := filepath.Dir(cur)
		if parent == cur {
			return abs // reached the filesystem root without resolving anything
		}

		suffix = filepath.Join(filepath.Base(cur), suffix)
		cur = parent
	}
}
