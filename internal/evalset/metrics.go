package evalset

import (
	"fmt"
	"math"
	"slices"

	"github.com/MeKo-Christian/go-laya/internal/calib"
	"github.com/MeKo-Christian/go-laya/question"
)

// The thresholds int8 is held to against fp32, per checkpoint and question
// type (PLAN.md B.5.4). They were fixed before B.5.3 produced a number. int8
// may be worse by at most these margins; being better never fails.
const (
	// MaxECEIncrease bounds ECE(int8) - ECE(fp32), absolute.
	MaxECEIncrease = 0.02
	// MaxBrierIncrease bounds Brier(int8) - Brier(fp32), absolute.
	MaxBrierIncrease = 0.02
	// MaxFlipRate bounds the share of cases whose argmax int8 changes.
	MaxFlipRate = 0.05
)

// Outcome is one answer at one precision, scored the way upstream's
// evaluation notebook scores it
// (original/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb, the
// cell computing all_confs): the probabilities a caller reads off the answer,
// renormalized because they are rounded to four decimals; the predicted index;
// and the confidence ECE bins it by.
type Outcome struct {
	Probs []float64
	Pred  int
	Conf  float64
	Score float64 // the expected level, score only
}

// ChoiceOutcome is a choice answer: its Probabilities in option order, and
// the index of its Choice. The confidence is the largest probability, as the
// notebook's all_confs.append(p_probs.max()).
func ChoiceOutcome(probs []float64, choice int) Outcome {
	p := normalized(probs)
	return Outcome{Probs: p, Pred: choice, Conf: slices.Max(p)}
}

// ScoreOutcome is a score answer: its Probabilities by level and its Score.
// The prediction is the most probable level, not the rounded expectation, as
// the notebook's np.argmax(p_probs_score).
func ScoreOutcome(probs []float64, score float64) Outcome {
	p := normalized(probs)
	return Outcome{Probs: p, Pred: argmax(p), Conf: slices.Max(p), Score: score}
}

// NoulOutcome is a noul answer, P(true) = p, as the two classes [1-p, p].
// True is predicted from p >= 0.5, and the confidence is max(p, 1-p).
func NoulOutcome(p float64) Outcome {
	pred := 0
	if p >= 0.5 {
		pred = 1
	}
	return Outcome{Probs: []float64{1 - p, p}, Pred: pred, Conf: max(p, 1-p)}
}

// normalized divides probs by their sum. All zeros, which four-decimal
// rounding cannot produce from a softmax, stays all zeros rather than NaN.
func normalized(probs []float64) []float64 {
	var sum float64
	for _, v := range probs {
		sum += v
	}
	if sum == 0 {
		return slices.Clone(probs)
	}
	p := make([]float64, len(probs))
	for i, v := range probs {
		p[i] = v / sum
	}
	return p
}

// argmax is the first index of the largest value, as numpy's.
func argmax(p []float64) int {
	best := 0
	for i := range p {
		if p[i] > p[best] {
			best = i
		}
	}
	return best
}

// Pair is one case answered by both precisions, on the same input.
type Pair struct {
	Type  question.QType
	Label int
	FP32  Outcome
	Int8  Outcome
}

// Side is one precision's scores over a set of cases.
type Side struct {
	ECE      float64
	Brier    float64
	Accuracy float64
	MAE      float64 // |score - label| for a score question, NaN otherwise
}

// Summary compares the two precisions over cases of one question type.
type Summary struct {
	Type    question.QType
	N       int
	FP32    Side
	Int8    Side
	Flips   int     // cases whose prediction differs between the two
	MaxDiff float64 // the largest |p_fp32 - p_int8| over every case and class
}

// FlipRate is Flips as a share of N.
func (s Summary) FlipRate() float64 { return float64(s.Flips) / float64(s.N) }

// slack keeps a difference that is the margin, such as 0.32 - 0.30, from
// failing on its last float64 bit.
const slack = 1e-9

// Verdict holds s to the thresholds, returning why it fails if it does. A NaN
// fails.
func (s Summary) Verdict() (bool, []string) {
	var why []string
	if d := s.Int8.ECE - s.FP32.ECE; !(d <= MaxECEIncrease+slack) {
		why = append(why, fmt.Sprintf("ECE %+.4f > %.2f", d, MaxECEIncrease))
	}
	if d := s.Int8.Brier - s.FP32.Brier; !(d <= MaxBrierIncrease+slack) {
		why = append(why, fmt.Sprintf("Brier %+.4f > %.2f", d, MaxBrierIncrease))
	}
	if r := s.FlipRate(); !(r <= MaxFlipRate+slack) {
		why = append(why, fmt.Sprintf("flip rate %.3f > %.2f", r, MaxFlipRate))
	}
	return len(why) == 0, why
}

// Summarize compares the precisions over pairs, which must all be of one
// question type.
func Summarize(pairs []Pair) Summary {
	if len(pairs) == 0 {
		return Summary{}
	}
	s := Summary{Type: pairs[0].Type, N: len(pairs)}
	for _, p := range pairs {
		if p.Type != s.Type {
			panic(fmt.Sprintf("evalset: Summarize over %v and %v", s.Type, p.Type))
		}
		if p.FP32.Pred != p.Int8.Pred {
			s.Flips++
		}
		for i := range p.FP32.Probs {
			s.MaxDiff = max(s.MaxDiff, math.Abs(p.FP32.Probs[i]-p.Int8.Probs[i]))
		}
	}
	s.FP32 = side(pairs, s.Type, func(p Pair) Outcome { return p.FP32 })
	s.Int8 = side(pairs, s.Type, func(p Pair) Outcome { return p.Int8 })
	return s
}

func side(pairs []Pair, qt question.QType, of func(Pair) Outcome) Side {
	n := len(pairs)
	conf := make([]float64, n)
	correct := make([]bool, n)
	probs := make([][]float64, n)
	labels := make([]int, n)
	var right int
	var absErr float64
	for i, p := range pairs {
		o := of(p)
		conf[i], probs[i], labels[i] = o.Conf, o.Probs, p.Label
		correct[i] = o.Pred == p.Label
		if correct[i] {
			right++
		}
		absErr += math.Abs(o.Score - float64(p.Label))
	}
	s := Side{
		ECE:      calib.ECE(conf, correct, calib.DefaultECEBins),
		Brier:    calib.Brier(probs, labels),
		Accuracy: float64(right) / float64(n),
		MAE:      math.NaN(),
	}
	if qt == question.Score {
		s.MAE = absErr / float64(n)
	}
	return s
}

// ByType summarizes pairs per question type, in the order choice, score,
// noul, leaving out the types that have no pairs.
func ByType(pairs []Pair) []Summary {
	var out []Summary
	for _, qt := range []question.QType{question.Choice, question.Score, question.Noul} {
		var of []Pair
		for _, p := range pairs {
			if p.Type == qt {
				of = append(of, p)
			}
		}
		if len(of) > 0 {
			out = append(out, Summarize(of))
		}
	}
	return out
}
