package tools

import (
	"fmt"
	"maps"
	"slices"
	"strconv"
	"strings"
)

// receivedArgs formats the arguments a failing call did carry: key names with
// byte lengths for strings and Go types for anything else, never values. Keys
// are sorted so two identical failures produce the same message.
func receivedArgs(args map[string]any) string {
	if len(args) == 0 {
		return ""
	}

	var b strings.Builder
	b.WriteString("; received: ")

	for i, k := range slices.Sorted(maps.Keys(args)) {
		if i > 0 {
			b.WriteString(", ")
		}

		s, ok := args[k].(string)
		if ok {
			fmt.Fprintf(&b, "%s (%d B)", k, len(s))
		} else {
			fmt.Fprintf(&b, "%s (%T)", k, args[k])
		}
	}

	return b.String()
}

func missingArgError(toolName, key, purpose string, args map[string]any) error {
	return fmt.Errorf("tool %q: missing required argument %q (%s)%s", toolName, key, purpose, receivedArgs(args))
}

func wrongTypeArgError(toolName, key string, args map[string]any) error {
	return fmt.Errorf("tool %q: argument %q must be a string%s", toolName, key, receivedArgs(args))
}

func requireString(args map[string]any, toolName, key, purpose string) (string, error) {
	v, ok := args[key]
	if !ok {
		return "", missingArgError(toolName, key, purpose, args)
	}

	s, ok := v.(string)
	if !ok {
		return "", wrongTypeArgError(toolName, key, args)
	}

	return s, nil
}

func optString(args map[string]any, key, def string) string {
	if v, ok := args[key].(string); ok {
		return v
	}

	return def
}

func optBool(args map[string]any, key string) bool {
	if v, ok := args[key].(bool); ok {
		return v
	}

	return false
}

// optInt accepts JSON numbers (decoded as float64) or ints.
func optInt(args map[string]any, key string, def int) int {
	switch v := args[key].(type) {
	case float64:
		return int(v)
	case int:
		return v
	default:
		return def
	}
}

// optIntCoerced returns the int value of the first key among keys present in
// args, coercing float64, int, or a numeric string (models routinely send
// `"30"` for numeric parameters). Absent keys return def. More than one key
// present, a non-numeric string, or an unsupported type is a corrective error
// the model can act on - silently falling back to the default is exactly the
// failure this helper exists to prevent. keys[0] is the canonical spelling.
func optIntCoerced(args map[string]any, keys []string, def int) (int, error) {
	var found []string

	for _, k := range keys {
		if _, ok := args[k]; ok {
			found = append(found, k)
		}
	}

	if len(found) == 0 {
		return def, nil
	}

	if len(found) > 1 {
		return 0, fmt.Errorf("use exactly one of %v; %q is the canonical spelling", found, keys[0])
	}

	key := found[0]

	switch v := args[key].(type) {
	case float64:
		return int(v), nil
	case int:
		return v, nil
	case string:
		n, err := strconv.Atoi(strings.TrimSpace(v))
		if err != nil {
			return 0, fmt.Errorf("argument %q must be an integer or a numeric string (e.g. 30 or \"30\"), got %q", key, v)
		}

		return n, nil
	default:
		return 0, fmt.Errorf("argument %q must be an integer or a numeric string (e.g. 30 or \"30\"), got %T", key, v)
	}
}

// optStringSlice coerces a JSON array (decoded as []any), a []string, or a
// single space-separated string into []string. Missing key -> nil.
func optStringSlice(args map[string]any, key string) []string {
	switch v := args[key].(type) {
	case []string:
		return v
	case []any:
		out := make([]string, 0, len(v))
		for _, e := range v {
			if s, ok := e.(string); ok {
				out = append(out, s)
			}
		}

		return out
	case string:
		if v == "" {
			return nil
		}

		return strings.Fields(v)
	default:
		return nil
	}
}
