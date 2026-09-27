package prompt

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"

	"github.com/MeKo-Christian/go-laya/jsonx"
)

// ErrCriteriaShape reports a question whose criteria Python could not render:
// render_options raises on them (common.py:38-43), or the type is not one
// QTYPES knows (agent.py:264).
var ErrCriteriaShape = errors.New("laya: criteria of the wrong shape")

// CheckCriteria rejects, with ErrCriteriaShape, every Internal that Python
// raises on between _to_internal and QTYPES[q["t"]] (agent.py:260-264), and
// accepts every one it renders.
//
// RenderOptions cannot do this itself: it reproduces render_options' output
// for the shapes Python accepts and, for the ones Python raises on, returns an
// empty or default option list -- a plausible answer to a question nobody
// asked. Its signature is also pinned by the golden tests. So the check is a
// separate guard, called before rendering.
//
// q is the post-_to_internal shape. A choice list has already become a dict
// (agent.py:233-234), so an unconverted []any is rejected here: render_options
// would call .items() on it. Go-only shapes that RenderOptions silently drops
// -- a Go map, a typed slice -- are rejected too; the port has no faithful
// rendering for them.
func CheckCriteria(q Internal) error {
	var ok bool
	switch q.T {
	case TypeChoice:
		// common.py:38, `crit.items()`.
		_, ok = q.Crit.(jsonx.Obj)
	case TypeScore:
		// common.py:40, `enumerate(crit)`. A dict enumerates its keys. A str
		// enumerates its characters, which renderScore cannot express, so only
		// the empty one -- no levels either way -- is accepted.
		switch c := q.Crit.(type) {
		case []any, jsonx.Obj:
			ok = true
		case string:
			ok = c == ""
		}
	case TypeNoul:
		// common.py:42-43, `crit = crit or {}` then `crit.get(...)`. Any falsy
		// value becomes {} and renders the defaults, as renderNoul does for
		// anything that is not a jsonx.Obj.
		_, isObj := q.Crit.(jsonx.Obj)
		ok = isObj || pyFalsy(q.Crit)
	default:
		// render_options renders an unknown type as a noul (common.py:41-46);
		// the KeyError comes one line later, from QTYPES[q["t"]] at agent.py:264.
		return fmt.Errorf("%w: unknown question type %q", ErrCriteriaShape, q.T)
	}
	if !ok {
		return fmt.Errorf("%w: %s question with criteria of type %T", ErrCriteriaShape, q.T, q.Crit)
	}
	return nil
}

// pyFalsy is Python's `not v` for the values a criteria field can hold: None,
// False, a zero number, and an empty str, list or dict. NaN is truthy, as it is
// in Python, because NaN == 0 is false.
func pyFalsy(v any) bool {
	if v == nil {
		return true
	}
	if n, isNum := v.(json.Number); isNum {
		f, err := n.Float64()
		return err == nil && f == 0
	}
	rv := reflect.ValueOf(v)
	switch {
	case rv.Kind() == reflect.Bool:
		return !rv.Bool()
	case rv.CanInt():
		return rv.Int() == 0
	case rv.CanUint():
		return rv.Uint() == 0
	case rv.CanFloat():
		return rv.Float() == 0
	case rv.Kind() == reflect.String, rv.Kind() == reflect.Slice, rv.Kind() == reflect.Map:
		return rv.Len() == 0
	default:
		return false
	}
}
