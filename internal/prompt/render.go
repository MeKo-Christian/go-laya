// Package prompt builds the text a laya question is asked with: the option
// strings that become the head of the sequence, and the serialized state that
// becomes its tail.
//
// It is a port of original/laya/common.py:15-46 and works on Python's internal
// question shape rather than on the public Go types. That is deliberate.
// Agent._to_internal (agent.py:230-238) flattens every question into
// {"t", "ins", "crit"} before anything renders it, and render_options then
// treats `crit` as whatever it happens to be: a dict for choice, a sequence for
// score, a dict for noul -- and `enumerate(crit)` on a score question walks a
// dict's *keys* if a dict is what it was handed. The golden corpus records that
// case (testdata/render.jsonl, options/score/dict-criteria). A renderer written
// against the public types could not express it, so the renderer lives here, on
// the loose shape, and the public types in question/ marshal into it.
package prompt

import (
	"strconv"

	"github.com/MeKo-Christian/go-laya/jsonx"
)

// The "t" values a question carries, as QTYPES spells them at common.py:11.
// They are the single source of truth for the spelling: question.QType.String
// returns these, so the public names and the renderer's switch cannot drift.
const (
	TypeChoice = "choice"
	TypeScore  = "score"
	TypeNoul   = "noul"
)

// The noul defaults, spelled out at common.py:44-45. They are part of the
// prompt the checkpoints were trained against, so they are not configurable.
const (
	noulFalseDefault = "no, the statement does not hold"
	noulTrueDefault  = "yes, the statement holds"
)

// Internal is Python's per-question dict, {"t": ..., "ins": ..., "crit": ...},
// as Agent._to_internal produces it and as testdata/render.jsonl records it.
//
// Crit is untyped because Python's is: a jsonx.Obj for choice and noul, a []any
// for score, nil for a noul question with no criteria -- and, for score, a
// jsonx.Obj too, which enumerates as its keys.
type Internal struct {
	T    string
	Ins  string
	Crit any
}

// SerializeState reproduces common.serialize_state (invariant #18): a string is
// the document itself and passes through untouched, anything else is
// json.dumps(state, ensure_ascii=False).
//
// Unlike RenderCriterion this can fail, and that asymmetry is upstream's:
// serialize_state passes no default=, so a value Python's json module refuses
// raises there and is an error here. Failing is the right outcome -- the state
// is tokenized verbatim, so a stringified stand-in would prompt the model with
// a Go type name and still return a confident answer.
func SerializeState(state any) (string, error) {
	if s, ok := state.(string); ok {
		return s, nil
	}
	b, err := jsonx.Marshal(state)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// RenderCriterion reproduces common.render_criterion (invariant #17): a string
// renders verbatim, anything else becomes compact JSON with Python's
// separators. jsonx.Compact carries Python's default=str, so an unserializable
// criterion is stringified rather than failing -- one bad criterion must not
// take down the whole batch of questions it was submitted with.
func RenderCriterion(value any) string {
	if s, ok := value.(string); ok {
		return s
	}
	return jsonx.Compact(value)
}

// RenderOptions reproduces common.render_options (invariants #14-#16): the
// option texts in label-index order, which is the order the model's output
// logits are indexed by.
//
// Note the fallthrough. Upstream has no `if t == "noul"` guard, so every type
// it does not recognise renders as a noul (common.py:41-46). This port
// reproduces that rather than rejecting the input, because the whole file is
// measured against the Python and an extra error is as much a deviation as a
// missing one.
func RenderOptions(q Internal) []string {
	switch q.T {
	case TypeChoice:
		return renderChoice(q.Crit)
	case TypeScore:
		return renderScore(q.Crit)
	default:
		return renderNoul(q.Crit)
	}
}

// renderChoice is common.py:37-38. A criterion renders as a bare key iff its
// value is nil or "" -- exactly those two (invariant #14). Python writes
// `v is None or v == ""`, which is false for 0, False and 0.0: those are
// legitimate criterion values and render as "zero: 0" and "no: false".
//
// A crit that is not an object yields no options, where Python raises
// AttributeError. Nothing in the public API can build one, and an empty option
// list is already a legal outcome (testdata/render.jsonl, options/choice/empty),
// so there is no error to report that would not be inventing a behaviour.
func renderChoice(crit any) []string {
	obj, _ := crit.(jsonx.Obj)
	out := make([]string, 0, len(obj))
	for _, f := range obj {
		if isBlank(f.Value) {
			out = append(out, f.Key)
			continue
		}
		out = append(out, f.Key+": "+RenderCriterion(f.Value))
	}
	return out
}

// renderScore is common.py:40, "level %d: %s" with i from 0 (invariant #15).
//
// The jsonx.Obj branch is not a convenience: Python's enumerate() over a dict
// walks its keys, so a score question whose criteria were written as a dict
// renders the key names as the level descriptions and drops the values
// entirely. testdata/render.jsonl records it as options/score/dict-criteria.
func renderScore(crit any) []string {
	var levels []any
	switch c := crit.(type) {
	case []any:
		levels = c
	case jsonx.Obj:
		levels = make([]any, len(c))
		for i, f := range c {
			levels[i] = f.Key
		}
	}

	out := make([]string, 0, len(levels))
	for i, level := range levels {
		out = append(out, "level "+strconv.Itoa(i)+": "+RenderCriterion(level))
	}
	return out
}

// renderNoul is common.py:41-46. The two options are always present and always
// in the order false, true -- index 0 is false whatever order the criteria dict
// was written in (invariant #16). A value of nil or "" falls back to the
// default text; every other value renders, including 0 and false.
func renderNoul(crit any) []string {
	obj, _ := crit.(jsonx.Obj)
	falseCrit, _ := obj.Get("false")
	trueCrit, _ := obj.Get("true")

	return []string{
		"false: " + criterionOr(falseCrit, noulFalseDefault),
		"true: " + criterionOr(trueCrit, noulTrueDefault),
	}
}

func criterionOr(value any, fallback string) string {
	if isBlank(value) {
		return fallback
	}
	return RenderCriterion(value)
}

// isBlank is Python's `v is None or v == ""`, written out because a plain Go
// `v == ""` on an `any` is a comparison whose result depends on the dynamic
// type and which panics outright on an uncomparable one.
func isBlank(value any) bool {
	if value == nil {
		return true
	}
	s, ok := value.(string)
	return ok && s == ""
}
