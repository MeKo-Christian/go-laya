package jsonx

import (
	"encoding/json"
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/golden"
)

// renderValue mirrors the two-line branch at common.py:28-31: a string passes
// through byte-identical, anything else becomes compact JSON. The branch itself
// belongs to internal/prompt (task 2.5.2); it is mirrored here so that task
// 2.1.8's "byte-for-byte against render.jsonl" covers every case rather than
// only the ones that happen not to be strings.
func renderValue(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return Compact(v)
}

// TestRenderGolden is task 2.1.8: byte-for-byte against the corpus that was
// dumped from the Python (invariants #17 and #18). These 25 cases are the ones
// that pin the separators, the key order, the absence of HTML escaping and the
// literal non-ASCII -- the four things encoding/json gets wrong.
func TestRenderGolden(t *testing.T) {
	for _, c := range golden.ByFn(t, "render", "render_criterion", "serialize_state") {
		t.Run(c.Name, func(t *testing.T) {
			var rec struct {
				Input  json.RawMessage `json:"input"`
				Output string          `json:"output"`
			}
			c.Unmarshal(t, &rec)

			in, err := Decode(rec.Input)
			if err != nil {
				t.Fatalf("decode the recorded input: %v", err)
			}

			var got string
			switch c.Fn {
			case "render_criterion":
				got = renderValue(in)
			case "serialize_state":
				if s, ok := in.(string); ok {
					// serialize_state returns a str unchanged (common.py:16-17).
					got = s
					break
				}
				b, err := Marshal(in)
				if err != nil {
					t.Fatalf("Marshal: %v", err)
				}
				got = string(b)
			}

			if got != rec.Output {
				t.Errorf("%s\n got: %s\nwant: %s\n"+
					"The bytes the tokenizer sees differ from Python's, so every "+
					"downstream token index differs too (invariant #18).",
					c.Fn, got, rec.Output)
			}
		})
	}
}

// TestRound4Golden is tasks 2.1.5 through 2.1.7 against the 26 chosen tie cases.
// They had to be chosen: no sampled softmax output lands on a tie, so a corpus
// drawn from real data would never exercise the one rounding rule Go gets wrong.
func TestRound4Golden(t *testing.T) {
	for _, c := range golden.Load(t, "round4") {
		t.Run(c.Name, func(t *testing.T) {
			var rec struct {
				Input      float64 `json:"input"`
				InputRepr  string  `json:"input_repr"`
				Output     float64 `json:"output"`
				OutputRepr string  `json:"output_repr"`
			}
			c.Unmarshal(t, &rec)

			got := Round4(rec.Input)
			if got != rec.Output {
				t.Errorf("Round4(%s) = %v, want %v (Python rounds half to even on the "+
					"binary double; math.Round rounds half away from zero)",
					rec.InputRepr, got, rec.Output)
			}
			// == is true for -0.0 against 0.0, so the sign needs its own check:
			// round4/08 is -2.5e-05 -> -0.0 and Python emits "-0.0".
			if math.Signbit(got) != math.Signbit(rec.Output) {
				t.Errorf("Round4(%s) = %s, want %s: the sign of zero is observable in the JSON",
					rec.InputRepr, Repr(got), rec.OutputRepr)
			}
			if r := Repr(rec.Input); r != rec.InputRepr {
				t.Errorf("Repr(input) = %s, want %s", r, rec.InputRepr)
			}
			if r := Repr(rec.Output); r != rec.OutputRepr {
				t.Errorf("Repr(output) = %s, want %s", r, rec.OutputRepr)
			}
		})
	}
}

// TestReprMatchesPythonRepr pins the float presentation beyond the corpus's
// range. Go's strconv and encoding/json each disagree with Python here, and the
// disagreement is not at the edges: 1.0 and 1e15 are ordinary values.
func TestReprMatchesPythonRepr(t *testing.T) {
	for _, tc := range []struct {
		in   float64
		want string
	}{
		{1, "1.0"},                     // encoding/json writes 1
		{0, "0.0"},                     // encoding/json writes 0
		{math.Copysign(0, -1), "-0.0"}, // encoding/json writes 0
		{3.5, "3.5"},
		{1e15, "1000000000000000.0"},             // strconv 'g' writes 1e+15
		{1234567890123456, "1234567890123456.0"}, // 16 digits: still fixed
		{1e16, "1e+16"},                          // 17: switches to exponent
		{1e-4, "0.0001"},
		{1e-5, "1e-05"}, // encoding/json writes 0.00001
		{1e100, "1e+100"},
		{math.Inf(1), "Infinity"}, // Python's non-standard literals
		{math.Inf(-1), "-Infinity"},
	} {
		if got := Repr(tc.in); got != tc.want {
			t.Errorf("Repr(%v) = %s, want %s", tc.in, got, tc.want)
		}
	}
	if got := Repr(math.NaN()); got != "NaN" {
		t.Errorf("Repr(NaN) = %s, want NaN", got)
	}
}

// TestEncoderDiffersFromEncodingJSON is the regression guard for the reason this
// package exists. Each case is a value where encoding/json produces different
// bytes, so if someone "simplifies" jsonx to wrap the stdlib, this fails.
func TestEncoderDiffersFromEncodingJSON(t *testing.T) {
	for _, tc := range []struct {
		name string
		in   any
		want string
	}{
		{"separators", Obj{{"a", 1}, {"b", 2}}, `{"a": 1, "b": 2}`},
		{"key order is not sorted", Obj{{"z", 1}, {"a", 2}, {"m", 3}}, `{"z": 1, "a": 2, "m": 3}`},
		{"no HTML escaping", Obj{{"html", "<b>a & b</b>"}}, `{"html": "<b>a & b</b>"}`},
		{"non-ASCII is literal", Obj{{"city", "München"}}, `{"city": "München"}`},
		{"U+2028 is literal", "a b", "\"a b\""},
		{"DEL is literal", "a\x7fb", "\"a\x7fb\""},
		{"C0 shorthands", "\b\f\n\r\t", `"\b\f\n\r\t"`},
		{"other C0 is \\u00xx", "\x00\x1f", `"\u0000\u001f"`},
		{"nested spacing", Obj{{"u", Obj{{"id", 7}}}}, `{"u": {"id": 7}}`},
		{"list spacing", []any{1, 2, 3}, `[1, 2, 3]`},
		{"float keeps .0", Obj{{"x", 1.0}}, `{"x": 1.0}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b, err := Marshal(tc.in)
			if err != nil {
				t.Fatalf("Marshal: %v", err)
			}
			if string(b) != tc.want {
				t.Errorf("Marshal = %s, want %s", b, tc.want)
			}
			if std, err := json.Marshal(tc.in); err == nil && string(std) == tc.want {
				t.Logf("note: encoding/json now agrees on %q; the case no longer discriminates", tc.name)
			}
		})
	}
}

// TestObjUnmarshalPreservesOrder is task 2.1.9. Without it the fixture loader
// would decode render.jsonl's inputs through map[string]any and destroy the key
// order that those very cases assert.
func TestObjUnmarshalPreservesOrder(t *testing.T) {
	var o Obj
	if err := json.Unmarshal([]byte(`{"z": 1, "a": 2, "m": 3}`), &o); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	keys := make([]string, 0, len(o))
	for _, f := range o {
		keys = append(keys, f.Key)
	}
	if strings.Join(keys, ",") != "z,a,m" {
		t.Errorf("key order = %v, want [z a m]", keys)
	}

	b, err := Marshal(o)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if string(b) != `{"z": 1, "a": 2, "m": 3}` {
		t.Errorf("round trip = %s", b)
	}
}

// TestDecodeKeepsIntegersIntegral guards the other half of the round trip:
// Python writes 3 for an int and 3.0 for a float, so a decoder that funnels
// every number through float64 changes the bytes on the way back out.
func TestDecodeKeepsIntegersIntegral(t *testing.T) {
	v, err := Decode([]byte(`{"i": 3, "f": 3.0}`))
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	b, err := Marshal(v)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if string(b) != `{"i": 3, "f": 3.0}` {
		t.Errorf("round trip = %s, want {\"i\": 3, \"f\": 3.0}", b)
	}
}

// TestMarshalRefusesMap: a Go map cannot express Python's insertion order, so
// encoding one would produce bytes that look right and are not. Refusing is the
// loud form of the rule that Obj exists to enforce.
func TestMarshalRefusesMap(t *testing.T) {
	if _, err := Marshal(map[string]any{"a": 1}); !errors.Is(err, ErrUnorderedMap) {
		t.Errorf("Marshal(map) error = %v, want ErrUnorderedMap", err)
	}
	// Compact follows Python's default=str instead of failing (invariant #17
	// says an unserializable value must not raise).
	if got := Compact(map[string]any{"a": 1}); got == "" {
		t.Error("Compact(map) returned empty; default=str must produce something")
	}
}

// TestCompactFallsBackToString is Python's default=str, per value.
func TestCompactFallsBackToString(t *testing.T) {
	ch := make(chan int)
	if got := Compact(Obj{{"c", ch}}); !strings.HasPrefix(got, `{"c": "0x`) {
		t.Errorf("Compact = %s, want the channel stringified inside the object", got)
	}
	if _, err := Marshal(Obj{{"c", ch}}); !errors.Is(err, ErrUnsupportedType) {
		t.Error("Marshal must refuse what Compact stringifies: serialize_state has no default=")
	}
}
