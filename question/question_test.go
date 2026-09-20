package question

import (
	"encoding/json"
	"errors"
	"slices"
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/golden"
	"github.com/MeKo-Christian/go-laya/internal/prompt"
	"github.com/MeKo-Christian/go-laya/jsonx"
)

// fromGolden rebuilds a public question from the internal {"t", "ins", "crit"}
// dict a fixture records. It reports false for a shape the public types cannot
// express, which is exactly one case in the corpus -- see the skip below.
//
// The input is decoded with jsonx.Decode rather than encoding/json because the
// criteria key order is the option order, and a map[string]any would sort it
// away before anything under test saw it.
func fromGolden(t *testing.T, raw json.RawMessage) (Question, bool) {
	t.Helper()

	v, err := jsonx.Decode(raw)
	if err != nil {
		t.Fatalf("decode the recorded input: %v", err)
	}
	obj, ok := v.(jsonx.Obj)
	if !ok {
		t.Fatalf("the recorded input is %T, not the internal question object", v)
	}

	qt, _ := obj.Get("t")
	ins, _ := obj.Get("ins")
	crit, _ := obj.Get("crit")
	insStr, _ := ins.(string)

	switch qt {
	case "choice":
		critObj, isObj := crit.(jsonx.Obj)
		if !isObj {
			return nil, false
		}
		opts := make([]ChoiceOption, 0, len(critObj))
		for _, f := range critObj {
			opts = append(opts, ChoiceOption{Key: f.Key, Desc: f.Value})
		}
		return ChoiceQuestion{Ins: insStr, Opts: opts}, true
	case "score":
		levels, isList := crit.([]any)
		if !isList {
			return nil, false
		}
		out := make([]Criterion, 0, len(levels))
		for _, l := range levels {
			out = append(out, l)
		}
		return ScoreQuestion{Ins: insStr, Levels: out}, true
	case "noul":
		critObj, _ := crit.(jsonx.Obj)
		falseCrit, _ := critObj.Get("false")
		trueCrit, _ := critObj.Get("true")
		return NoulQuestion{Ins: insStr, False: falseCrit, True: trueCrit}, true
	default:
		return nil, false
	}
}

// TestRenderOptionsGolden asserts the public types render byte-identically to
// the recorded Python output. The renderer itself is asserted in
// internal/prompt; what this covers is the marshalling from the Go type design
// into Python's internal shape, which is where an option can silently change
// order or lose its "no description" flag.
func TestRenderOptionsGolden(t *testing.T) {
	for _, c := range golden.ByFn(t, "render", "render_options") {
		t.Run(c.Name, func(t *testing.T) {
			var rec struct {
				Input  json.RawMessage `json:"input"`
				Output []string        `json:"output"`
			}
			c.Unmarshal(t, &rec)

			q, ok := fromGolden(t, rec.Input)
			if !ok {
				// options/score/dict-criteria: a score question whose criteria are
				// a dict. Python's enumerate() then walks the keys, which
				// ScoreQuestion.Levels has nowhere to hold. Deliberately not
				// representable; internal/prompt asserts the case instead.
				t.Skip("not expressible in the public types; asserted in internal/prompt")
			}

			if got := q.RenderOptions(); !slices.Equal(got, rec.Output) {
				t.Errorf("RenderOptions\n got: %q\nwant: %q\n"+
					"The option strings are the head of the prompt, so a mismatch "+
					"moves every marker and returns a plausible wrong answer rather "+
					"than an error (invariants #14-#16).", got, rec.Output)
			}
		})
	}
}

// TestQTypeValues pins the integer values: they index the checkpoint's type
// embedding (common.py:11), so renumbering them would feed the model a
// different question type without changing a single test that only reads names.
func TestQTypeValues(t *testing.T) {
	for _, tc := range []struct {
		qt   QType
		n    uint8
		name string
	}{
		{Choice, 0, "choice"},
		{Score, 1, "score"},
		{Noul, 2, "noul"},
	} {
		if uint8(tc.qt) != tc.n {
			t.Errorf("%s = %d, want %d; the value indexes the checkpoint's type "+
				"embedding and cannot be renumbered", tc.name, uint8(tc.qt), tc.n)
		}
		if got := tc.qt.String(); got != tc.name {
			t.Errorf("QType(%d).String() = %q, want %q", tc.n, got, tc.name)
		}
	}

	if got := QType(9).String(); got == "" {
		t.Error("QType(9).String() is empty; an unknown type must still be nameable " +
			"in an error message")
	}
}

// TestQuestionTypes checks each question reports the type whose branch it must
// take in render_options and whose index it must carry into the model.
func TestQuestionTypes(t *testing.T) {
	cases := []struct {
		q    Question
		want QType
	}{
		{ChoiceQuestion{Ins: "a"}, Choice},
		{ScoreQuestion{Ins: "b"}, Score},
		{NoulQuestion{Ins: "c"}, Noul},
	}
	for _, tc := range cases {
		if got := tc.q.Type(); got != tc.want {
			t.Errorf("%T.Type() = %v, want %v", tc.q, got, tc.want)
		}
	}

	if got := (ChoiceQuestion{Ins: "hello"}).Instructions(); got != "hello" {
		t.Errorf("Instructions() = %q, want %q", got, "hello")
	}
}

// TestLabels covers the list form of Python's `criteria`: a bare list of names
// becomes options with no description, exactly what Agent._to_internal does at
// agent.py:233-234.
func TestLabels(t *testing.T) {
	q := ChoiceQuestion{Ins: "x", Opts: Labels("tech", "sales")}

	want := []string{"tech", "sales"}
	if got := q.RenderOptions(); !slices.Equal(got, want) {
		t.Errorf("RenderOptions\n got: %q\nwant: %q\n"+
			"A labels-only choice must render bare keys; a description that "+
			"appeared from nowhere would change the prompt.", got, want)
	}
}

// TestNoulDefaultsFromZeroValue asserts the useful zero value: a noul question
// with no criteria at all gets Python's two default texts, in the fixed order
// false, true (invariant #16).
func TestNoulDefaultsFromZeroValue(t *testing.T) {
	want := []string{
		"false: no, the statement does not hold",
		"true: yes, the statement holds",
	}
	if got := (NoulQuestion{Ins: "x"}).RenderOptions(); !slices.Equal(got, want) {
		t.Errorf("RenderOptions\n got: %q\nwant: %q", got, want)
	}
}

// TestChoiceRejectsDuplicateKeys covers a hazard the Python cannot have. Its
// criteria are a dict, so two options with the same name collapse into one;
// []ChoiceOption happily holds both, which would emit two option slots where
// Python emits one and shift every later logit index by one.
func TestChoiceRejectsDuplicateKeys(t *testing.T) {
	q := ChoiceQuestion{Ins: "x", Opts: []ChoiceOption{
		{Key: "a", Desc: "first"},
		{Key: "a", Desc: "second"},
	}}

	err := q.validate()
	if !errors.Is(err, ErrDuplicateOption) {
		t.Errorf("validate() = %v, want ErrDuplicateOption; a duplicate key renders "+
			"two option slots where a Python dict renders one, shifting every "+
			"later answer index", err)
	}

	ok := ChoiceQuestion{Ins: "x", Opts: Labels("a", "b")}
	if err := ok.validate(); err != nil {
		t.Errorf("validate() on distinct keys = %v, want nil", err)
	}
}

// TestQuestionsValidate covers the collection: duplicate ids are impossible in
// Python's dict of questions, so a Questions slice that holds two must be an
// error rather than last-wins, which would drop an answer the caller asked for.
func TestQuestionsValidate(t *testing.T) {
	dup := Questions{
		{ID: "intent", Q: ChoiceQuestion{Ins: "x", Opts: Labels("a")}},
		{ID: "intent", Q: NoulQuestion{Ins: "y"}},
	}
	if err := dup.Validate(); !errors.Is(err, ErrDuplicateQuestionID) {
		t.Errorf("Validate() = %v, want ErrDuplicateQuestionID; last-wins would "+
			"silently return fewer answers than questions asked", err)
	}

	bad := Questions{{ID: "intent", Q: ChoiceQuestion{Opts: []ChoiceOption{{Key: "a"}, {Key: "a"}}}}}
	if err := bad.Validate(); !errors.Is(err, ErrDuplicateOption) {
		t.Errorf("Validate() = %v, want it to report the per-question failure", err)
	}

	if err := (Questions{{ID: "intent"}}).Validate(); err == nil {
		t.Error("Validate() accepted a nil question; it would panic at render time " +
			"instead, well after the caller could do anything about it")
	}

	good := Questions{
		{ID: "intent", Q: ChoiceQuestion{Ins: "x", Opts: Labels("a", "b")}},
		{ID: "urgency", Q: ScoreQuestion{Ins: "y", Levels: []Criterion{"low", "high"}}},
	}
	if err := good.Validate(); err != nil {
		t.Errorf("Validate() = %v, want nil", err)
	}
}

// TestQuestionsOrder asserts the collection keeps definition order: it is the
// order the answers come back in and the order the batch rows are built in.
func TestQuestionsOrder(t *testing.T) {
	qs := Questions{
		{ID: "z", Q: NoulQuestion{Ins: "1"}},
		{ID: "a", Q: NoulQuestion{Ins: "2"}},
		{ID: "m", Q: NoulQuestion{Ins: "3"}},
	}

	want := []string{"z", "a", "m"}
	if got := qs.IDs(); !slices.Equal(got, want) {
		t.Errorf("IDs() = %q, want %q; sorted ids would reorder the answers against "+
			"the questions the caller wrote", got, want)
	}

	q, ok := qs.Get("a")
	if !ok {
		t.Fatal(`Get("a") reported missing; the id is right there in the slice`)
	}
	if got := q.Instructions(); got != "2" {
		t.Errorf(`Get("a").Instructions() = %q, want "2"`, got)
	}
	if _, ok := qs.Get("nope"); ok {
		t.Error(`Get("nope") reported found`)
	}
}

// TestInternalShape asserts the bridge into internal/prompt carries exactly the
// three fields Agent._to_internal produces, with the criteria in definition
// order. The renderer is only correct if what reaches it has the right shape.
func TestInternalShape(t *testing.T) {
	got := ChoiceQuestion{Ins: "pick", Opts: []ChoiceOption{
		{Key: "b", Desc: "second"},
		{Key: "a"},
	}}.internal()

	if got.T != "choice" || got.Ins != "pick" {
		t.Errorf("internal() = {T:%q, Ins:%q}, want {choice, pick}", got.T, got.Ins)
	}
	crit, ok := got.Crit.(jsonx.Obj)
	if !ok {
		t.Fatalf("Crit is %T, want jsonx.Obj; Python's choice criteria are a dict "+
			"and render_options calls .items() on them", got.Crit)
	}
	if len(crit) != 2 || crit[0].Key != "b" || crit[1].Key != "a" {
		t.Errorf("Crit = %v, want b before a in definition order", crit)
	}
	if crit[1].Value != nil {
		t.Errorf("Crit[1].Value = %v, want nil so it renders as a bare key", crit[1].Value)
	}

	score := ScoreQuestion{Ins: "rate", Levels: []Criterion{"low", "high"}}.internal()
	if _, ok := score.Crit.([]any); !ok {
		t.Errorf("score Crit is %T, want []any; Python enumerates a sequence here "+
			"and enumerating anything else changes what the levels say", score.Crit)
	}

	noul := NoulQuestion{Ins: "is it?", True: "yes"}.internal()
	nc, ok := noul.Crit.(jsonx.Obj)
	if !ok {
		t.Fatalf("noul Crit is %T, want jsonx.Obj", noul.Crit)
	}
	if v, found := nc.Get("true"); !found || v != "yes" {
		t.Errorf(`noul Crit["true"] = %v, %v; want "yes", true`, v, found)
	}
}

// TestRenderOptionsGoesThroughPrompt is the layering claim: the public type is
// a marshaller, not a second implementation. Two renderers that agree today
// drift apart on the next fixture regeneration.
func TestRenderOptionsGoesThroughPrompt(t *testing.T) {
	q := ScoreQuestion{Ins: "x", Levels: []Criterion{
		jsonx.Obj{{Key: "d", Value: "low"}}, "high", json.Number("2"),
	}}

	want := prompt.RenderOptions(q.internal())
	if got := q.RenderOptions(); !slices.Equal(got, want) {
		t.Errorf("RenderOptions\n got: %q\nwant: %q", got, want)
	}
	if !slices.Equal(want, []string{`level 0: {"d": "low"}`, "level 1: high", "level 2: 2"}) {
		t.Errorf("the shared renderer produced %q", want)
	}
}
