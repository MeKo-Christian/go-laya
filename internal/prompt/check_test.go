package prompt

import (
	"encoding/json"
	"errors"
	"math"
	"testing"

	"github.com/MeKo-Christian/go-laya/jsonx"
)

// TestCheckCriteria is PLAN.md 7.3.12: every crit shape Python raises on is
// rejected with ErrCriteriaShape, and every shape Python renders is accepted.
// RenderOptions itself stays lenient -- it renders an empty or default option
// list for the raising shapes, which is a silent plausible answer -- so this
// guard is what the Agent calls before it renders.
//
// Crit here is the post-_to_internal shape (agent.py:229-238): a choice list
// has already become a dict by the time render_options sees it.
func TestCheckCriteria(t *testing.T) {
	obj := jsonx.Obj{{Key: "a", Value: nil}}
	cases := []struct {
		name   string
		q      Internal
		accept bool
		why    string // the Python line that decides it
	}{
		// choice: `crit.items()`, common.py:38.
		{"choice/dict", Internal{T: TypeChoice, Crit: obj}, true, "common.py:38 dict.items()"},
		{"choice/empty dict", Internal{T: TypeChoice, Crit: jsonx.Obj{}}, true, "common.py:38 {}.items() is empty, no raise"},
		{"choice/nil", Internal{T: TypeChoice, Crit: nil}, false, "common.py:38 None.items() AttributeError"},
		{"choice/str", Internal{T: TypeChoice, Crit: "a"}, false, "common.py:38 str.items() AttributeError"},
		{"choice/number", Internal{T: TypeChoice, Crit: json.Number("1")}, false, "common.py:38 int.items() AttributeError"},
		{"choice/bool", Internal{T: TypeChoice, Crit: true}, false, "common.py:38 bool.items() AttributeError"},
		{
			"choice/list after _to_internal",
			Internal{T: TypeChoice, Crit: []any{"a"}},
			false,
			"agent.py:233-234 converts a list before common.py:38; an unconverted one would raise there",
		},
		{"choice/go map", Internal{T: TypeChoice, Crit: map[string]any{"a": nil}}, false, "Go-only shape; RenderOptions would drop it"},

		// score: `enumerate(crit)`, common.py:40.
		{"score/list", Internal{T: TypeScore, Crit: []any{"lo", "hi"}}, true, "common.py:40 enumerate(list)"},
		{"score/empty list", Internal{T: TypeScore, Crit: []any{}}, true, "common.py:40 enumerate([]) is empty, no raise"},
		{"score/dict", Internal{T: TypeScore, Crit: obj}, true, "common.py:40 enumerate(dict) walks the keys"},
		{"score/empty str", Internal{T: TypeScore, Crit: ""}, true, `common.py:40 enumerate("") is empty, no raise`},
		{"score/nil", Internal{T: TypeScore, Crit: nil}, false, "common.py:40 enumerate(None) TypeError"},
		{"score/number", Internal{T: TypeScore, Crit: json.Number("3")}, false, "common.py:40 enumerate(int) TypeError"},
		{"score/bool", Internal{T: TypeScore, Crit: false}, false, "common.py:40 enumerate(bool) TypeError"},
		{
			"score/non-empty str",
			Internal{T: TypeScore, Crit: "abc"},
			false,
			"deviation: common.py:40 enumerates the characters; RenderOptions renders none",
		},
		{"score/typed slice", Internal{T: TypeScore, Crit: []string{"lo"}}, false, "Go-only shape; RenderOptions would drop it"},

		// noul: `crit = crit or {}` then `crit.get(...)`, common.py:42-43.
		{"noul/dict", Internal{T: TypeNoul, Crit: jsonx.Obj{{Key: "true", Value: "y"}}}, true, "common.py:43 dict.get"},
		{"noul/empty dict", Internal{T: TypeNoul, Crit: jsonx.Obj{}}, true, "common.py:42 {} or {}"},
		{"noul/nil", Internal{T: TypeNoul, Crit: nil}, true, "common.py:42 None or {}"},
		{"noul/empty str", Internal{T: TypeNoul, Crit: ""}, true, `common.py:42 "" or {}`},
		{"noul/empty list", Internal{T: TypeNoul, Crit: []any{}}, true, "common.py:42 [] or {}"},
		{"noul/false", Internal{T: TypeNoul, Crit: false}, true, "common.py:42 False or {}"},
		{"noul/zero", Internal{T: TypeNoul, Crit: json.Number("0")}, true, "common.py:42 0 or {}"},
		{"noul/zero float", Internal{T: TypeNoul, Crit: 0.0}, true, "common.py:42 0.0 or {}"},
		{"noul/empty go map", Internal{T: TypeNoul, Crit: map[string]any{}}, true, "common.py:42 {} or {}; renders the same defaults"},
		{"noul/list", Internal{T: TypeNoul, Crit: []any{"x"}}, false, "common.py:43 list.get AttributeError"},
		{"noul/str", Internal{T: TypeNoul, Crit: "x"}, false, "common.py:43 str.get AttributeError"},
		{"noul/true", Internal{T: TypeNoul, Crit: true}, false, "common.py:43 bool.get AttributeError"},
		{"noul/number", Internal{T: TypeNoul, Crit: json.Number("1")}, false, "common.py:43 int.get AttributeError"},
		{"noul/nan", Internal{T: TypeNoul, Crit: math.NaN()}, false, "common.py:42 bool(nan) is True, then float.get AttributeError"},
		{"noul/go map", Internal{T: TypeNoul, Crit: map[string]any{"true": "y"}}, false, "Go-only shape; RenderOptions would drop it"},

		// unknown type: render_options falls through to noul (common.py:41-46),
		// then QTYPES[q["t"]] raises KeyError at agent.py:264.
		{"unknown/nil", Internal{T: "yesno", Crit: nil}, false, `agent.py:264 QTYPES["yesno"] KeyError`},
		{"unknown/empty", Internal{T: "", Crit: jsonx.Obj{}}, false, `agent.py:264 QTYPES[""] KeyError`},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := CheckCriteria(tc.q)
			if tc.accept {
				if err != nil {
					t.Errorf("CheckCriteria = %v, want nil (%s)", err, tc.why)
				}
				return
			}
			if !errors.Is(err, ErrCriteriaShape) {
				t.Errorf("CheckCriteria = %v, want ErrCriteriaShape (%s)", err, tc.why)
			}
		})
	}
}
