package tools

import (
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

	return "", fmt.Errorf("path %q is outside the workspace and every permitted read-only root", p)
}

// sanitizeReadRoots drops extra read roots that would widen access rather than
// add a sibling tree: an empty path, the filesystem root, and any root that
// contains the workspace (which would permit the workspace's whole parentage).
// Dropping is silent and one-directional - a misconfigured root narrows access,
// it never widens it - so a caller cannot accidentally open the filesystem.
func sanitizeReadRoots(workspace string, roots []string) []string {
	if len(roots) == 0 {
		return nil
	}

	sep := string(os.PathSeparator)

	// Compare RESOLVED paths. resolveInRoots checks containment against the
	// symlink-resolved root, so validating the unresolved one would wave through
	// a root that is a symlink to / or to the workspace's parent - defeating the
	// very widening this function exists to refuse.
	wsResolved := evalOrSelf(filepath.Clean(workspace))

	var out []string

	for _, r := range roots {
		if strings.TrimSpace(r) == "" {
			continue
		}

		rc := filepath.Clean(r)

		// A relative root would resolve against the process working directory,
		// which is never what a caller configuring a dependency tree meant.
		if !filepath.IsAbs(rc) {
			continue
		}

		resolved := evalOrSelf(rc)
		if resolved == sep {
			continue
		}

		// resolved contains (or is) the workspace: permitting it would widen.
		if wsResolved == resolved || strings.HasPrefix(wsResolved, resolved+sep) {
			continue
		}

		out = append(out, rc)
	}

	return out
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
