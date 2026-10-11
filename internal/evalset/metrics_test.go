package evalset

import (
	"math"
	"slices"
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/calib"
	"github.com/MeKo-Christian/go-laya/question"
)

func near(a, b float64) bool { return math.Abs(a-b) < 1e-12 }

func TestOutcomes(t *testing.T) {
	// 4-decimal rounding leaves a choice summing to 0.9999; the notebook
	// renormalizes before scoring, and so does ChoiceOutcome.
	c := ChoiceOutcome([]float64{0.2, 0.2, 0.5999}, 2)
	if c.Pred != 2 || !near(c.Conf, 0.5999/0.9999) || !near(c.Probs[0], 0.2/0.9999) {
		t.Errorf("ChoiceOutcome = %+v", c)
	}
	// A score is predicted by its argmax level, not by rounding the expectation.
	s := ScoreOutcome([]float64{0.1, 0.2, 0.3, 0.4}, 1.9)
	if s.Pred != 3 || !near(s.Conf, 0.4) || s.Score != 1.9 {
		t.Errorf("ScoreOutcome = %+v", s)
	}
	for _, tt := range []struct {
		p    float64
		pred int
		conf float64
	}{{0.8, 1, 0.8}, {0.5, 1, 0.5}, {0.3, 0, 0.7}} {
		n := NoulOutcome(tt.p)
		if n.Pred != tt.pred || !near(n.Conf, tt.conf) || !slices.Equal(n.Probs, []float64{1 - tt.p, tt.p}) {
			t.Errorf("NoulOutcome(%v) = %+v, want pred %d conf %v", tt.p, n, tt.pred, tt.conf)
		}
	}
}

// choicePairs are four 3-option answers. fp32 gets 0 and 2 right; int8 also
// gets 1 right, which is the one flip. The largest per-option difference is
// pair 1's 0.2.
var choicePairs = []Pair{
	{
		Type: question.Choice, Label: 0,
		FP32: ChoiceOutcome([]float64{0.7, 0.2, 0.1}, 0), Int8: ChoiceOutcome([]float64{0.6, 0.3, 0.1}, 0),
	},
	{
		Type: question.Choice, Label: 1,
		FP32: ChoiceOutcome([]float64{0.5, 0.4, 0.1}, 0), Int8: ChoiceOutcome([]float64{0.3, 0.6, 0.1}, 1),
	},
	{
		Type: question.Choice, Label: 2,
		FP32: ChoiceOutcome([]float64{0.1, 0.1, 0.8}, 2), Int8: ChoiceOutcome([]float64{0.1, 0.2, 0.7}, 2),
	},
	{
		Type: question.Choice, Label: 0,
		FP32: ChoiceOutcome([]float64{0.2, 0.5, 0.3}, 1), Int8: ChoiceOutcome([]float64{0.2, 0.5, 0.3}, 1),
	},
}

func TestSummarizeChoice(t *testing.T) {
	s := Summarize(choicePairs)
	if s.Type != question.Choice || s.N != 4 {
		t.Fatalf("Summarize = %+v", s)
	}
	if s.FP32.Accuracy != 0.5 || s.Int8.Accuracy != 0.75 {
		t.Errorf("accuracy = %v / %v, want 0.5 / 0.75", s.FP32.Accuracy, s.Int8.Accuracy)
	}
	if s.Flips != 1 || s.FlipRate() != 0.25 {
		t.Errorf("flips = %d (%v), want 1 (0.25)", s.Flips, s.FlipRate())
	}
	if math.Abs(s.MaxDiff-0.2) > 1e-9 {
		t.Errorf("max diff = %v, want 0.2", s.MaxDiff)
	}
	// Brier by hand: fp32 (0.14+0.62+0.06+0.98)/4, int8 (0.26+0.26+0.14+0.98)/4.
	if math.Abs(s.FP32.Brier-0.45) > 1e-9 || math.Abs(s.Int8.Brier-0.41) > 1e-9 {
		t.Errorf("Brier = %v / %v, want 0.45 / 0.41", s.FP32.Brier, s.Int8.Brier)
	}
	// ECE over the max-probability confidences, as the notebook's
	// ece_score(all_confs, all_corrects) is: 0.25 + 0.075 + 0.05 for fp32.
	wantFP32 := calib.ECE([]float64{0.7, 0.5, 0.8, 0.5}, []bool{true, false, true, false}, calib.DefaultECEBins)
	wantInt8 := calib.ECE([]float64{0.6, 0.6, 0.7, 0.5}, []bool{true, true, true, false}, calib.DefaultECEBins)
	if !near(s.FP32.ECE, wantFP32) || math.Abs(wantFP32-0.375) > 1e-9 || !near(s.Int8.ECE, wantInt8) {
		t.Errorf("ECE = %v / %v, want %v / %v", s.FP32.ECE, s.Int8.ECE, wantFP32, wantInt8)
	}
	if !math.IsNaN(s.FP32.MAE) {
		t.Errorf("a choice has no MAE, got %v", s.FP32.MAE)
	}
}

func TestSummarizeScoreMAE(t *testing.T) {
	pairs := []Pair{
		{
			Type: question.Score, Label: 2,
			FP32: ScoreOutcome([]float64{0.1, 0.3, 0.6}, 1.5), Int8: ScoreOutcome([]float64{0.1, 0.4, 0.5}, 1.4),
		},
		{
			Type: question.Score, Label: 0,
			FP32: ScoreOutcome([]float64{0.8, 0.15, 0.05}, 0.25), Int8: ScoreOutcome([]float64{0.4, 0.5, 0.1}, 0.7),
		},
	}
	s := Summarize(pairs)
	if math.Abs(s.FP32.MAE-0.375) > 1e-9 || math.Abs(s.Int8.MAE-0.65) > 1e-9 {
		t.Errorf("MAE = %v / %v, want 0.375 / 0.65", s.FP32.MAE, s.Int8.MAE)
	}
	if s.FP32.Accuracy != 1 || s.Int8.Accuracy != 0.5 || s.Flips != 1 {
		t.Errorf("accuracy %v / %v, flips %d", s.FP32.Accuracy, s.Int8.Accuracy, s.Flips)
	}
}

func TestSummarizeNoul(t *testing.T) {
	pairs := []Pair{
		{Type: question.Noul, Label: 1, FP32: NoulOutcome(0.9), Int8: NoulOutcome(0.4)},
		{Type: question.Noul, Label: 0, FP32: NoulOutcome(0.2), Int8: NoulOutcome(0.1)},
	}
	s := Summarize(pairs)
	// Brier over [1-p, p]: fp32 (0.02 + 0.08)/2, int8 (0.72 + 0.02)/2.
	if math.Abs(s.FP32.Brier-0.05) > 1e-9 || math.Abs(s.Int8.Brier-0.37) > 1e-9 {
		t.Errorf("Brier = %v / %v, want 0.05 / 0.37", s.FP32.Brier, s.Int8.Brier)
	}
	if s.Flips != 1 || math.Abs(s.MaxDiff-0.5) > 1e-9 {
		t.Errorf("flips %d, max diff %v", s.Flips, s.MaxDiff)
	}
}

func TestByType(t *testing.T) {
	mixed := []Pair{
		{Type: question.Noul, Label: 1, FP32: NoulOutcome(0.9), Int8: NoulOutcome(0.9)},
		choicePairs[0],
		{Type: question.Score, Label: 0, FP32: ScoreOutcome([]float64{1, 0}, 0), Int8: ScoreOutcome([]float64{1, 0}, 0)},
		choicePairs[1],
	}
	got := ByType(mixed)
	types := make([]question.QType, 0, len(got))
	ns := make([]int, 0, len(got))
	for _, s := range got {
		types = append(types, s.Type)
		ns = append(ns, s.N)
	}
	if !slices.Equal(types, []question.QType{question.Choice, question.Score, question.Noul}) ||
		!slices.Equal(ns, []int{2, 1, 1}) {
		t.Errorf("ByType = types %v, sizes %v", types, ns)
	}
}

func TestSummarizeRejectsMixedTypes(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("Summarize accepted two question types")
		}
	}()
	Summarize([]Pair{choicePairs[0], {Type: question.Noul, FP32: NoulOutcome(1), Int8: NoulOutcome(1)}})
}

// TestVerdict pins the thresholds B.5.4 committed to before B.5.3 ran: int8
// may lose at most 0.02 ECE and 0.02 Brier against fp32, and flip at most 5%
// of the argmaxes. Being better is never a failure.
func TestVerdict(t *testing.T) {
	base := Summary{N: 100, FP32: Side{ECE: 0.10, Brier: 0.30}, Int8: Side{ECE: 0.10, Brier: 0.30}}
	tests := []struct {
		name string
		edit func(*Summary)
		ok   bool
	}{
		{"identical", func(*Summary) {}, true},
		{"ECE at the margin", func(s *Summary) { s.Int8.ECE = 0.12 }, true},
		{"ECE past the margin", func(s *Summary) { s.Int8.ECE = 0.1201 }, false},
		{"ECE better", func(s *Summary) { s.Int8.ECE = 0.01 }, true},
		{"Brier at the margin", func(s *Summary) { s.Int8.Brier = 0.32 }, true},
		{"Brier past the margin", func(s *Summary) { s.Int8.Brier = 0.3201 }, false},
		{"5% flips", func(s *Summary) { s.Flips = 5 }, true},
		{"6% flips", func(s *Summary) { s.Flips = 6 }, false},
		{"NaN ECE", func(s *Summary) { s.Int8.ECE = math.NaN() }, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := base
			tt.edit(&s)
			ok, why := s.Verdict()
			if ok != tt.ok || ok != (len(why) == 0) {
				t.Errorf("Verdict = %v %q, want %v", ok, why, tt.ok)
			}
		})
	}
}
