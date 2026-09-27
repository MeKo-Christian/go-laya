package laya

import (
	"bytes"
	"encoding/json"
	"slices"
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/calib"
	"github.com/MeKo-Christian/go-laya/internal/golden"
	"github.com/MeKo-Christian/go-laya/jsonx"
	"github.com/MeKo-Christian/go-laya/question"
)

// answerKeys is the key order agent.py:316-337 builds each answer dict in
// (docs/API.md). Invariant #30: the key set is decided by the type alone.
var answerKeys = map[string][]string{
	"choice": {"type", "choice", "probabilities", "confidence", "action"},
	"score":  {"type", "score", "legend", "probabilities", "confidence", "action"},
	"noul":   {"type", "noul", "confidence", "action"},
}

// answerFixture is one record of testdata/answers.jsonl. Criteria stays raw so
// it can be decoded order-preserving with jsonx.
type answerFixture struct {
	QType                string             `json:"qtype"`
	K                    int                `json:"k"`
	Criteria             json.RawMessage    `json:"criteria"`
	Logits               []float32          `json:"logits"`
	ActLogits            []float32          `json:"act_logits"`
	Temperature          [3]float64         `json:"temperature"`
	TemperatureByOptions map[string]float64 `json:"temperature_by_options"`
	AnswerJSON           string             `json:"answer_json"`
}

// fixtureQuestion rebuilds the public question the fixture's raw criteria
// describe, keeping the criterion values exactly as jsonx decodes them.
func fixtureQuestion(t *testing.T, rec answerFixture) question.Question {
	t.Helper()

	crit, err := jsonx.Decode(rec.Criteria)
	if err != nil {
		t.Fatalf("decode criteria: %v", err)
	}
	switch rec.QType {
	case "choice":
		obj, ok := crit.(jsonx.Obj)
		if !ok {
			t.Fatalf("choice criteria are %T, want an object", crit)
		}
		opts := make([]question.ChoiceOption, len(obj))
		for i, f := range obj {
			opts[i] = question.ChoiceOption{Key: f.Key, Desc: f.Value}
		}
		return question.ChoiceQuestion{Opts: opts}
	case "score":
		levels, ok := crit.([]any)
		if !ok {
			t.Fatalf("score criteria are %T, want a list", crit)
		}
		out := make([]question.Criterion, len(levels))
		for i, l := range levels {
			out[i] = l
		}
		return question.ScoreQuestion{Levels: out}
	case "noul":
		obj, ok := crit.(jsonx.Obj)
		if !ok {
			t.Fatalf("noul criteria are %T, want an object", crit)
		}
		f, _ := obj.Get("false")
		tr, _ := obj.Get("true")
		return question.NoulQuestion{False: f, True: tr}
	}
	t.Fatalf("unknown qtype %q", rec.QType)
	return nil
}

func objKeys(o Obj) []string {
	keys := make([]string, len(o))
	for i, f := range o {
		keys[i] = f.Key
	}
	return keys
}

// TestAnswerGolden is Tasks 7.3.1-7.3.4, 7.3.7 and 7.3.8 against the whole
// corpus: every answer, formatted from the recorded logits, must be
// byte-equal to Python's json.dumps and carry exactly its type's keys.
func TestAnswerGolden(t *testing.T) {
	cases := golden.Load(t, "answers")
	if len(cases) != 33 {
		t.Fatalf("answers.jsonl has %d cases, want 33", len(cases))
	}
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			var rec answerFixture
			c.Unmarshal(t, &rec)

			temps := calib.Temperatures{ByQType: rec.Temperature, ByOptions: rec.TemperatureByOptions}
			a := formatAnswer(fixtureQuestion(t, rec), rec.Logits, rec.ActLogits, rec.K, temps)

			got, err := jsonx.Marshal(a.Map())
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			if string(got) != rec.AnswerJSON {
				t.Errorf("answer bytes\n got %s\nwant %s", got, rec.AnswerJSON)
			}

			want := answerKeys[rec.QType]
			if keys := objKeys(a.Map()); !slices.Equal(keys, want) {
				t.Errorf("keys = %q, want %q", keys, want)
			}
			// The fixture itself must agree with the key table, or the table
			// is asserting an order Python does not produce.
			dec, err := jsonx.Decode([]byte(rec.AnswerJSON))
			if err != nil {
				t.Fatalf("decode answer_json: %v", err)
			}
			if keys := objKeys(dec.(jsonx.Obj)); !slices.Equal(keys, want) {
				t.Errorf("fixture keys = %q, want %q", keys, want)
			}

			// encoding/json re-compacts a MarshalJSON's output (D12), so
			// json.Marshal can only match the compacted Python bytes.
			viaStd, err := json.Marshal(a)
			if err != nil {
				t.Fatalf("json.Marshal: %v", err)
			}
			var compact bytes.Buffer
			if err := json.Compact(&compact, []byte(rec.AnswerJSON)); err != nil {
				t.Fatalf("compact: %v", err)
			}
			if !bytes.Equal(viaStd, compact.Bytes()) {
				t.Errorf("json.Marshal\n got %s\nwant %s", viaStd, compact.Bytes())
			}
		})
	}
}

// TestAnswerTieTakesFirstKey is 7.3.7's tie-break: numpy's argmax returns the
// first maximum, so two exactly equal logits choose the earlier key, whatever
// the keys sort to.
func TestAnswerTieTakesFirstKey(t *testing.T) {
	q := question.ChoiceQuestion{Opts: question.Labels("b", "a")}
	a := formatAnswer(q, []float32{0, 0}, []float32{0, 0}, 2, calib.DefaultTemperatures())
	if a.Choice != "b" {
		t.Errorf("choice = %q, want %q (first maximum)", a.Choice, "b")
	}
	got, err := jsonx.Marshal(a.Map())
	if err != nil {
		t.Fatal(err)
	}
	want := `{"type": "choice", "choice": "b", "probabilities": {"b": 0.5, "a": 0.5}, ` +
		`"confidence": 0.0, "action": {"act_probability": 0.5}}`
	if string(got) != want {
		t.Errorf("got  %s\nwant %s", got, want)
	}
}

// TestAnswerEmptyKeyIsEmitted is 7.3.8's injected "" key: "" is a legal dict
// key in Python, and an omitempty port would drop "choice" entirely (#30).
func TestAnswerEmptyKeyIsEmitted(t *testing.T) {
	q := question.ChoiceQuestion{Opts: question.Labels("", "x")}
	a := formatAnswer(q, []float32{3, -3}, []float32{1, 0}, 2, calib.DefaultTemperatures())
	if a.Choice != "" {
		t.Fatalf("choice = %q, want the empty key", a.Choice)
	}
	got, err := jsonx.Marshal(a.Map())
	if err != nil {
		t.Fatal(err)
	}
	dec, err := jsonx.Decode(got)
	if err != nil {
		t.Fatal(err)
	}
	obj := dec.(jsonx.Obj)
	if keys := objKeys(obj); !slices.Equal(keys, answerKeys["choice"]) {
		t.Errorf("keys = %q, want %q", keys, answerKeys["choice"])
	}
	if v, _ := obj.Get("choice"); v != "" {
		t.Errorf(`"choice" = %#v, want ""`, v)
	}
	probs, _ := obj.Get("probabilities")
	if keys := objKeys(probs.(jsonx.Obj)); !slices.Equal(keys, []string{"", "x"}) {
		t.Errorf("probability keys = %q", keys)
	}
}

// TestAnswerEmptyCollectionsAreEmitted: an answer with an empty legend or
// empty probabilities still carries the key, as {} (#30).
func TestAnswerEmptyCollectionsAreEmitted(t *testing.T) {
	zero := 0.0
	a := Answer{Type: "score", Score: &zero}
	got, err := jsonx.Marshal(a.Map())
	if err != nil {
		t.Fatal(err)
	}
	want := `{"type": "score", "score": 0.0, "legend": {}, "probabilities": {}, ` +
		`"confidence": 0.0, "action": {"act_probability": 0.0}}`
	if string(got) != want {
		t.Errorf("got  %s\nwant %s", got, want)
	}
}

// TestAnswerScoreIsExpectation is 7.3.2: the score is Σ i·p[i], so a
// distribution split between levels 0 and 2 scores 1.0, a level no argmax
// would pick.
func TestAnswerScoreIsExpectation(t *testing.T) {
	q := question.ScoreQuestion{Levels: []question.Criterion{"lo", "mid", "hi"}}
	a := formatAnswer(q, []float32{5, -50, 5}, []float32{0}, 3, calib.DefaultTemperatures())
	if a.Score == nil || *a.Score != 1.0 {
		t.Fatalf("score = %v, want 1.0", a.Score)
	}
}

// TestAnswerNoulConfidence is 7.3.3: a uniform noul is 0.5 confident, where
// the entropy formula would say 0 (#26), and carries no probabilities/legend.
func TestAnswerNoulConfidence(t *testing.T) {
	q := question.NoulQuestion{}
	a := formatAnswer(q, []float32{1, 1}, []float32{0}, 2, calib.DefaultTemperatures())
	if a.Confidence != 0.5 {
		t.Errorf("confidence = %v, want 0.5", a.Confidence)
	}
	if a.Noul == nil || *a.Noul != 0.5 {
		t.Errorf("noul = %v, want 0.5", a.Noul)
	}
	if keys := objKeys(a.Map()); !slices.Equal(keys, answerKeys["noul"]) {
		t.Errorf("keys = %q", keys)
	}
}

// TestAnswerLegendIsRaw is 7.3.4: the legend holds the criterion value as the
// caller gave it, not the rendered option text, and covers every level.
func TestAnswerLegendIsRaw(t *testing.T) {
	levels := []question.Criterion{Obj{{Key: "d", Value: "low"}}, "", nil, json.Number("2")}
	q := question.ScoreQuestion{Levels: levels}
	a := formatAnswer(q, []float32{0, 0}, []float32{0}, 2, calib.DefaultTemperatures())
	got, err := jsonx.Marshal(a.Legend)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"0": {"d": "low"}, "1": "", "2": null, "3": 2}`
	if string(got) != want {
		t.Errorf("legend = %s, want %s", got, want)
	}
	if keys := objKeys(a.Probabilities.obj()); !slices.Equal(keys, []string{"0", "1"}) {
		t.Errorf("probability keys = %q, want k of them", keys)
	}
}

func TestProbsGetAndMaxTie(t *testing.T) {
	p := Probs{{Key: "x", P: 0.2}, {Key: "y", P: 0.4}, {Key: "z", P: 0.4}}
	if k, v := p.Max(); k != "y" || v != 0.4 {
		t.Errorf("Max() = %q, %v, want y, 0.4 (first maximum)", k, v)
	}
	if got := p.Get("z"); got != 0.4 {
		t.Errorf("Get(z) = %v", got)
	}
	if got := p.Get("missing"); got != 0 {
		t.Errorf("Get(missing) = %v, want 0", got)
	}
	if k, v := (Probs{}).Max(); k != "" || v != 0 {
		t.Errorf("empty Max() = %q, %v", k, v)
	}
	b, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	if string(b) != `{"x":0.2,"y":0.4,"z":0.4}` {
		t.Errorf("json.Marshal = %s", b)
	}
}

func TestResultAnswerSetOrderAndGet(t *testing.T) {
	n := 0.75
	as := AnswerSet{
		{ID: "zeta", A: Answer{Type: "noul", Noul: &n, Confidence: 0.75}},
		{ID: "alpha", A: Answer{Type: "choice", Choice: "a", Probabilities: Probs{{Key: "a", P: 1}}, Confidence: 1}},
	}
	if a, ok := as.Get("alpha"); !ok || a.Choice != "a" {
		t.Errorf("Get(alpha) = %+v, %v", a, ok)
	}
	if _, ok := as.Get("nope"); ok {
		t.Error("Get(nope) found an answer")
	}
	b, err := json.Marshal(as)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"zeta":{"type":"noul","noul":0.75,"confidence":0.75,"action":{"act_probability":0.0}},` +
		`"alpha":{"type":"choice","choice":"a","probabilities":{"a":1.0},"confidence":1.0,"action":{"act_probability":0.0}}}`
	if string(b) != want {
		t.Errorf("got  %s\nwant %s", b, want)
	}
}

func TestResultMarshal(t *testing.T) {
	n := 0.5
	r := Result{
		Model:   "laya-rl-agent",
		Answers: AnswerSet{{ID: "q", A: Answer{Type: "noul", Noul: &n, Confidence: 0.5}}},
		Usage:   Usage{InputTokens: 12},
	}
	got, err := jsonx.Marshal(r.Map())
	if err != nil {
		t.Fatal(err)
	}
	base := `{"model": "laya-rl-agent", "answers": {"q": {"type": "noul", "noul": 0.5, ` +
		`"confidence": 0.5, "action": {"act_probability": 0.0}}}, ` +
		`"usage": {"input_tokens": 12, "output_tokens": 0}`
	if string(got) != base+"}" {
		t.Errorf("without routing\n got %s\nwant %s}", got, base)
	}

	r.Routing = &RouteDecision{Model: "english", Repo: "r/e", Reason: "default"}
	got, err = jsonx.Marshal(r.Map())
	if err != nil {
		t.Fatal(err)
	}
	want := base + `, "routing": {"model": "english", "repo": "r/e", "reason": "default", ` +
		`"detection": null, "workflow": null}}`
	if string(got) != want {
		t.Errorf("with routing\n got %s\nwant %s", got, want)
	}

	viaStd, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	var compact bytes.Buffer
	if err := json.Compact(&compact, []byte(want)); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(viaStd, compact.Bytes()) {
		t.Errorf("json.Marshal\n got %s\nwant %s", viaStd, compact.Bytes())
	}
}
