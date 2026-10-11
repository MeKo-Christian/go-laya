package evalset

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/MeKo-Christian/go-laya"
)

// fakePredictor answers every noul with P(true) = p, so each side of a run is
// recognisable by its number.
type fakePredictor struct {
	p      float64
	closed *bool
}

func (f fakePredictor) SystemOne(_ context.Context, _ any, _ laya.Questions) (*laya.Result, error) {
	p := f.p
	return &laya.Result{Answers: laya.AnswerSet{{ID: QuestionID, A: laya.Answer{Type: "noul", Noul: &p}}}}, nil
}

func (f fakePredictor) Close() error { *f.closed = true; return nil }

type openCall struct {
	ck, dir   string
	quantized bool
}

// TestEvaluateUsesBothDirectories: the fp32 side runs on the fp32 exports and
// the int8 side on the int8 ones, each closed after its run. Comparing fp32
// against itself would report a perfect int8 without measuring one.
func TestEvaluateUsesBothDirectories(t *testing.T) {
	answers := map[string]float64{"/fp32": 0.9, "/int8": 0.4}
	var calls []openCall
	var closed []*bool
	open := func(_ context.Context, ck, dir string, quantized bool) (laya.Predictor, func() error, error) {
		calls = append(calls, openCall{ck, dir, quantized})
		c := new(bool)
		closed = append(closed, c)
		f := fakePredictor{p: answers[dir], closed: c}
		return f, f.Close, nil
	}
	spec, _ := SpecByName("sst2")
	rows := []Row{{Text: "a", Label: 1}, {Text: "b", Label: 0}}
	cases := make([]Case, 0, len(rows))
	for _, r := range rows {
		c, err := spec.Case(r)
		if err != nil {
			t.Fatal(err)
		}
		cases = append(cases, c)
	}

	pairs, err := evaluate(context.Background(), "english", cases, "/fp32", "/int8", open)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	want := []openCall{{"english", "/fp32", false}, {"english", "/int8", true}}
	if !slices.Equal(calls, want) {
		t.Errorf("opened %+v, want %+v", calls, want)
	}
	for i, c := range closed {
		if !*c {
			t.Errorf("predictor %d was not closed", i)
		}
	}
	for i, p := range pairs {
		if p.FP32.Probs[1] != 0.9 || p.Int8.Probs[1] != 0.4 || p.Label != cases[i].Label || p.Type != cases[i].Type {
			t.Errorf("pair %d = %+v, want fp32 P(true) 0.9, int8 0.4, label %d", i, p, cases[i].Label)
		}
	}
}

func TestOutcomeOf(t *testing.T) {
	ag, _ := SpecByName("ag_news")
	choice, _ := ag.Case(Row{Text: "x", Label: 1})
	a := laya.Answer{
		Type: "choice", Choice: "business",
		Probabilities: laya.Probs{{Key: "world", P: 0.1}, {Key: "sports", P: 0.2}, {Key: "business", P: 0.6}, {Key: "sci_tech", P: 0.1}},
	}
	o, err := outcomeOf(choice, a)
	if err != nil || o.Pred != 2 || o.Conf != 0.6 {
		t.Errorf("choice outcome = %+v, %v; want pred 2, conf 0.6", o, err)
	}
	// Looked up by key: the same answer with its entries reversed scores the
	// same.
	rev := slices.Clone(a.Probabilities)
	slices.Reverse(rev)
	if o2, err := outcomeOf(choice, laya.Answer{Type: "choice", Choice: "business", Probabilities: rev}); err != nil ||
		!slices.Equal(o2.Probs, o.Probs) || o2.Pred != 2 {
		t.Errorf("reversed entries: %+v, %v; want %+v", o2, err, o)
	}
	a.Probabilities = a.Probabilities[:3]
	if _, err := outcomeOf(choice, a); err == nil {
		t.Error("three probabilities for four options were accepted")
	}
	p0 := 0.5
	if _, err := outcomeOf(choice, laya.Answer{Type: "noul", Noul: &p0}); err == nil {
		t.Error("a noul answer to a choice question was accepted")
	}

	yelp, _ := SpecByName("yelp")
	score, _ := yelp.Case(Row{Text: "x", Label: 3})
	sv := 3.1
	o, err = outcomeOf(score, laya.Answer{
		Type: "score", Score: &sv,
		Probabilities: laya.Probs{{Key: "0", P: 0}, {Key: "1", P: 0.1}, {Key: "2", P: 0.1}, {Key: "3", P: 0.4}, {Key: "4", P: 0.4}},
	})
	if err != nil || o.Pred != 3 || o.Score != 3.1 {
		t.Errorf("score outcome = %+v, %v; want pred 3 (the first of two maxima), score 3.1", o, err)
	}

	sst, _ := SpecByName("sst2")
	noul, _ := sst.Case(Row{Text: "x", Label: 1})
	p := 0.7
	if o, err := outcomeOf(noul, laya.Answer{Type: "noul", Noul: &p}); err != nil || o.Pred != 1 || o.Conf != 0.7 {
		t.Errorf("noul outcome = %+v, %v", o, err)
	}
}

func TestCheckPrecision(t *testing.T) {
	dir := t.TempDir()
	fp32 := filepath.Join(dir, "fp32.onnx")
	quant := filepath.Join(dir, "int8.onnx")
	if err := os.WriteFile(fp32, []byte("\x08\x0aMatMul\x12MatMul"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(quant, []byte("\x08\x0aDynamicQuantizeLinear\x12MatMulInteger"), 0o600); err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		path      string
		quantized bool
		ok        bool
	}{{fp32, false, true}, {fp32, true, false}, {quant, true, true}, {quant, false, false}} {
		if err := checkPrecision(tt.path, tt.quantized); (err == nil) != tt.ok {
			t.Errorf("checkPrecision(%s, quantized=%v) = %v, want ok=%v", filepath.Base(tt.path), tt.quantized, err, tt.ok)
		}
	}
}
