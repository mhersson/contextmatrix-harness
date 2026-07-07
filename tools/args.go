package tools

import (
	"fmt"
	"strconv"
	"strings"
)

func requireString(args map[string]any, key string) (string, error) {
	v, ok := args[key]
	if !ok {
		return "", fmt.Errorf("missing required argument %q", key)
	}

	s, ok := v.(string)
	if !ok {
		return "", fmt.Errorf("argument %q must be a string", key)
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
// the model can act on — silently falling back to the default is exactly the
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
