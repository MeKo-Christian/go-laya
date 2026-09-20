package lang

import (
	"fmt"
	"reflect"
	"slices"
	"strings"

	"github.com/MeKo-Christian/go-laya/jsonx"
)

const (
	// maxDetectChars is state_text's max_chars. Python slices code points, so
	// this counts runes: s[:4000] on a Go string would both truncate a
	// multi-byte state to a fraction of what Python sees and cut a rune in
	// half (lang.py:86-88, invariant #60).
	maxDetectChars = 4000

	// maxDepth is _iter_text's cap. The check is `_depth > 6` on the value, so
	// a leaf nested six objects deep is still collected (lang.py:69).
	maxDepth = 6
)

// StateText flattens a state into the text detection runs on: the string
// leaves only, depth first in value order, joined with a single space and
// truncated to 4000 code points.
//
// Dict keys are deliberately ignored. They are almost always English -- an
// English "subject"/"body" wrapper around Hindi content must still detect as
// Hindi (test_router.py:86).
//
// Python accepts str, dict, list, tuple and None and yields nothing for
// anything else. The Go equivalents are string, [jsonx.Obj], any slice or
// array, and nil; a Go map is also accepted, but see [mapLeaves] for why it
// cannot reproduce a Python dict.
func StateText(state any) string {
	return truncateRunes(strings.Join(textLeaves(state, 0), " "), maxDetectChars)
}

// textLeaves is _iter_text.
func textLeaves(state any, depth int) []string {
	if depth > maxDepth || state == nil {
		return nil
	}

	switch v := state.(type) {
	case string:
		return []string{v}
	case jsonx.Obj:
		out := make([]string, 0, len(v))
		for _, f := range v {
			out = append(out, textLeaves(f.Value, depth+1)...)
		}
		return out
	case []any:
		out := make([]string, 0, len(v))
		for _, e := range v {
			out = append(out, textLeaves(e, depth+1)...)
		}
		return out
	}
	return reflectLeaves(state, depth)
}

// reflectLeaves covers the slice, array and map types the type switch cannot
// enumerate -- []string, [2]any, map[string]string. Everything else yields
// nothing, the same way Python's _iter_text falls through to `return []`.
func reflectLeaves(state any, depth int) []string {
	rv := reflect.ValueOf(state)

	// The kinds not listed have no Python counterpart in a state, or are
	// already handled by the type switch above.
	//exhaustive:ignore
	switch rv.Kind() {
	case reflect.Slice, reflect.Array:
		out := make([]string, 0, rv.Len())
		for i := range rv.Len() {
			out = append(out, textLeaves(rv.Index(i).Interface(), depth+1)...)
		}
		return out
	case reflect.Map:
		return mapLeaves(rv, depth)
	default:
		return nil
	}
}

// mapLeaves walks a Go map in sorted key order.
//
// A Go map cannot express a Python dict: Python preserves insertion order and
// Go randomizes it, which would make both DetectScript's encounter-order
// tie-break and ScriptProfile's key order differ from run to run. Sorting is
// not the Python order either, but it is at least the same order every time,
// and it keeps the content -- silently dropping the map would flip is_english
// on a real state rather than failing.
//
// [jsonx.Obj] is the type that does reproduce a Python dict; prefer it.
func mapLeaves(rv reflect.Value, depth int) []string {
	keys := rv.MapKeys()
	slices.SortFunc(keys, func(a, b reflect.Value) int {
		return strings.Compare(fmt.Sprint(a.Interface()), fmt.Sprint(b.Interface()))
	})

	out := make([]string, 0, len(keys))
	for _, k := range keys {
		out = append(out, textLeaves(rv.MapIndex(k).Interface(), depth+1)...)
	}
	return out
}

// truncateRunes cuts s to at most n code points, which is what Python's
// [:max_chars] does to a str.
func truncateRunes(s string, n int) string {
	count := 0
	for i := range s {
		if count == n {
			return s[:i]
		}
		count++
	}
	return s
}
