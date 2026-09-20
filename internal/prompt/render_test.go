package prompt

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/golden"
	"github.com/MeKo-Christian/go-laya/jsonx"
)

// num builds the json.Number a decoded fixture would carry, so a hand-written
// table and a fixture-driven one exercise the same code path.
func num(s string) json.Number { return json.Number(s) }

// internalFromGolden rebuilds the {"t", "ins", "crit"} dict the fixture records
// -- the same shape Agent._to_internal produces at agent.py:230-238.
//
// The input is decoded with jsonx.Decode and not encoding/json: several cases
// assert key order (options/choice/mixed renders a, b, c in that order), and a
// map[string]any would sort it away before the renderer ever saw it.
func internalFromGolden(t *testing.T, raw json.RawMessage) Internal {
	t.Helper()

	v, err := jsonx.Decode(raw)
	if err != nil {
		t.Fatalf("decode the recorded input: %v", err)
	}
	obj, ok := v.(jsonx.Obj)
	if !ok {
		t.Fatalf("the recorded input is %T, not an object; the fixture records the "+
			"internal question dict, so anything else means the corpus changed shape", v)
	}

	var q Internal
	if tv, found := obj.Get("t"); found {
		q.T, _ = tv.(string)
	}
	if iv, found := obj.Get("ins"); found {
		q.Ins, _ = iv.(string)
	}
	q.Crit, _ = obj.Get("crit")
	return q
}

// TestRenderOptionsGolden is invariants #14 through #16 against the 16 recorded
// render_options cases. These strings are the prompt the model reads: a
// difference here shifts every marker position and therefore every probability,
// and it does so without raising anything.
func TestRenderOptionsGolden(t *testing.T) {
	for _, c := range golden.ByFn(t, "render", "render_options") {
		t.Run(c.Name, func(t *testing.T) {
			var rec struct {
				Input  json.RawMessage `json:"input"`
				Output []string        `json:"output"`
			}
			c.Unmarshal(t, &rec)

			got := RenderOptions(internalFromGolden(t, rec.Input))
			if !slices.Equal(got, rec.Output) {
				t.Errorf("RenderOptions\n got: %q\nwant: %q\n"+
					"The rendered options are the head of the prompt, so a mismatch "+
					"moves every [MASK] marker and silently returns a plausible "+
					"answer rather than an error (invariants #14-#16).", got, rec.Output)
			}
		})
	}
}

// TestRenderCriterionGolden is invariant #17 at the renderer level. jsonx's own
// suite asserts the encoder; what is asserted here and nowhere else is the
// string-passthrough branch (common.py:28-31), which must not JSON-quote.
func TestRenderCriterionGolden(t *testing.T) {
	for _, c := range golden.ByFn(t, "render", "render_criterion") {
		t.Run(c.Name, func(t *testing.T) {
			var rec struct {
				Input  json.RawMessage `json:"input"`
				Output string          `json:"output"`
			}
			c.Unmarshal(t, &rec)

			in, err := jsonx.Decode(rec.Input)
			if err != nil {
				t.Fatalf("decode the recorded input: %v", err)
			}

			if got := RenderCriterion(in); got != rec.Output {
				t.Errorf("RenderCriterion\n got: %q\nwant: %q\n"+
					"A criterion renders into the option text, so the model is "+
					"prompted with different bytes than the Python version (#17).",
					got, rec.Output)
			}
		})
	}
}

// TestSerializeStateGolden is invariant #18 at the renderer level: a str passes
// through, anything else is json.dumps(state, ensure_ascii=False).
func TestSerializeStateGolden(t *testing.T) {
	for _, c := range golden.ByFn(t, "render", "serialize_state") {
		t.Run(c.Name, func(t *testing.T) {
			var rec struct {
				Input  json.RawMessage `json:"input"`
				Output string          `json:"output"`
			}
			c.Unmarshal(t, &rec)

			in, err := jsonx.Decode(rec.Input)
			if err != nil {
				t.Fatalf("decode the recorded input: %v", err)
			}

			got, err := SerializeState(in)
			if err != nil {
				t.Fatalf("SerializeState: %v", err)
			}
			if got != rec.Output {
				t.Errorf("SerializeState\n got: %q\nwant: %q\n"+
					"The serialized state is tokenized verbatim, so different bytes "+
					"mean a different token sequence (#18).", got, rec.Output)
			}
		})
	}
}

// TestCriteriaPort is original/tests/test_criteria.py:33-100 as a table (task
// 2.5.3). It overlaps the golden corpus on purpose: the upstream file is the
// regression test for the reported crash (a noul question whose criteria were
// dicts raised TypeError), and the port keeps that history addressable.
//
// The upstream block at test_criteria.py:103-116 is deliberately NOT ported.
// It uses inspect.getsource to assert five string literals appear in
// Agent.__init__ -- a test of the source text rather than of behaviour, with no
// Go counterpart. PLAN.md task 2.5.4 replaces it with a behavioural test on the
// device-fallback policy, which cannot exist before the Agent does (M7).
func TestCriteriaPort(t *testing.T) {
	criteria := []struct {
		name string
		in   any
		want string
	}{
		{"criterion/str passes through", "phishing or scam", "phishing or scam"},
		{"criterion/dict -> json", jsonx.Obj{{Key: "desc", Value: "phishing"}}, `{"desc": "phishing"}`},
		{"criterion/list -> json", []any{"a", "b"}, `["a", "b"]`},
		{"criterion/int -> json", num("3"), "3"},
		{"criterion/bool -> json", false, "false"},
		{"criterion/non-ascii kept", jsonx.Obj{{Key: "d", Value: "münchen"}}, `{"d": "münchen"}`},
	}
	for _, tc := range criteria {
		t.Run(tc.name, func(t *testing.T) {
			if got := RenderCriterion(tc.in); got != tc.want {
				t.Errorf("RenderCriterion(%#v) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}

	options := []struct {
		name string
		q    Internal
		want []string
	}{
		{
			// The reported crash: dict criteria on noul (test_criteria.py:46-58).
			"noul/dict criteria does not crash",
			Internal{T: "noul", Ins: "Is this phishing?", Crit: jsonx.Obj{
				{Key: "true", Value: jsonx.Obj{{Key: "desc", Value: "phishing, scam or fraud"}}},
				{Key: "false", Value: jsonx.Obj{{Key: "desc", Value: "legitimate"}}},
			}},
			[]string{`false: {"desc": "legitimate"}`, `true: {"desc": "phishing, scam or fraud"}`},
		},
		{
			"choice/dict, None and empty string",
			Internal{T: "choice", Ins: "x", Crit: jsonx.Obj{
				{Key: "billing", Value: jsonx.Obj{{Key: "desc", Value: "payments"}}},
				{Key: "tech", Value: nil},
				{Key: "sales", Value: ""},
			}},
			[]string{`billing: {"desc": "payments"}`, "tech", "sales"},
		},
		{
			// 0 and False are real criterion values, not "missing" (#14).
			"choice/zero and false are kept",
			Internal{T: "choice", Ins: "x", Crit: jsonx.Obj{
				{Key: "zero", Value: num("0")},
				{Key: "no", Value: false},
			}},
			[]string{"zero: 0", "no: false"},
		},
		{
			"score/mixed levels",
			Internal{T: "score", Ins: "x", Crit: []any{
				jsonx.Obj{{Key: "d", Value: "low"}}, "high", num("2"),
			}},
			[]string{`level 0: {"d": "low"}`, "level 1: high", "level 2: 2"},
		},
		{
			"noul/defaults",
			Internal{T: "noul", Ins: "x", Crit: nil},
			[]string{"false: no, the statement does not hold", "true: yes, the statement holds"},
		},
		{
			"noul/string criteria still work",
			Internal{T: "noul", Ins: "x", Crit: jsonx.Obj{
				{Key: "true", Value: "yes it is"},
				{Key: "false", Value: "no"},
			}},
			[]string{"false: no", "true: yes it is"},
		},
		{
			"choice/string criteria still work",
			Internal{T: "choice", Ins: "x", Crit: jsonx.Obj{
				{Key: "a", Value: "first"},
				{Key: "b", Value: nil},
			}},
			[]string{"a: first", "b"},
		},
		{
			"score/string criteria still work",
			Internal{T: "score", Ins: "x", Crit: []any{"low", "high"}},
			[]string{"level 0: low", "level 1: high"},
		},
	}
	for _, tc := range options {
		t.Run(tc.name, func(t *testing.T) {
			got := RenderOptions(tc.q)
			if !slices.Equal(got, tc.want) {
				t.Errorf("RenderOptions\n got: %q\nwant: %q", got, tc.want)
			}
			// test_criteria.py:57 and :69 -- no Python repr may leak into a prompt.
			if joined := strings.Join(got, ""); strings.Contains(joined, "{'") {
				t.Errorf("a Python-style repr leaked into the prompt: %q", joined)
			}
		})
	}
}

// TestRenderedOptionsAreEverJSON is test_criteria.py:96-100: what a criterion
// renders to must parse back as JSON, which is the difference between JSON and
// the Python repr the bug leaked.
func TestRenderedOptionsAreEverJSON(t *testing.T) {
	got := RenderOptions(Internal{T: "noul", Ins: "x", Crit: jsonx.Obj{
		{Key: "true", Value: jsonx.Obj{{Key: "a", Value: num("1")}}},
		{Key: "false", Value: jsonx.Obj{{Key: "b", Value: num("2")}}},
	}})
	const prefix = "true: "
	body, ok := strings.CutPrefix(got[1], prefix)
	if !ok {
		t.Fatalf("option 1 is %q, want it to start with %q", got[1], prefix)
	}

	var back map[string]int
	if err := json.Unmarshal([]byte(body), &back); err != nil {
		t.Fatalf("the emitted option is not JSON (%v); a Python repr in the prompt "+
			"is the exact bug test_criteria.py was written for: %q", err, body)
	}
	if back["a"] != 1 {
		t.Errorf("round-tripped %v, want map[a:1]", back)
	}
}

// TestRenderCriterionNeverFails is the second half of invariant #17: Python
// passes default=str, so an unserializable criterion is stringified rather than
// raising. A criterion that raised would take down a whole batch of questions.
func TestRenderCriterionNeverFails(t *testing.T) {
	for _, v := range []any{make(chan int), func() {}, map[string]int{"a": 1}} {
		if got := RenderCriterion(v); got == "" {
			t.Errorf("RenderCriterion(%T) = %q; #17 requires a string form, not an "+
				"empty prompt slot", v, got)
		}
	}
}

// TestSerializeStateFailsLoudly is the contrast invariant #18 draws with #17:
// serialize_state has no default=, so a value Python's json module refuses is
// an error here rather than a stringified stand-in.
func TestSerializeStateFailsLoudly(t *testing.T) {
	if got, err := SerializeState(make(chan int)); err == nil {
		t.Errorf("SerializeState(chan) = %q, nil; #18 has no default= fallback, so "+
			"silently prompting the model with a Go type name is worse than failing",
			got)
	}
	if got, err := SerializeState(map[string]int{"a": 1}); err == nil {
		t.Errorf("SerializeState(map) = %q, nil; a Go map has no observable key "+
			"order, so encoding one would produce bytes that look right and are not",
			got)
	}
}

// TestRenderOptionsUnknownTypeIsNoul pins a quirk rather than a design: Python's
// render_options has no `if t == "noul"` guard (common.py:41-46), so the noul
// branch is the fallthrough for every type it does not recognise. Reproducing
// the quirk is the parity-port default; deviating would need a line in PLAN.md.
func TestRenderOptionsUnknownTypeIsNoul(t *testing.T) {
	got := RenderOptions(Internal{T: "not-a-type", Ins: "x", Crit: nil})
	want := []string{"false: no, the statement does not hold", "true: yes, the statement holds"}
	if !slices.Equal(got, want) {
		t.Errorf("RenderOptions(unknown type)\n got: %q\nwant: %q", got, want)
	}
}
