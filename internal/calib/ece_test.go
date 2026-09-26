package calib

import (
	"math"
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/golden"
)

// eceCase is one "ece_score" record of testdata/ece.jsonl. JSON has no NaN, so
// the generator writes null and sets the flag.
type eceCase struct {
	Conf     []float64 `json:"conf"`
	Correct  []bool    `json:"correct"`
	Bins     int       `json:"bins"`
	ECE      float64   `json:"ece"`
	ECEIsNaN bool      `json:"ece_is_nan"`
}

// TestECEFixture replays every ece_score case (common.py:187-197) and requires
// the result to be equal to Python's, not close: the bins, the edge rule and
// numpy's summation order all show in the last bit (Task 7.1.6).
func TestECEFixture(t *testing.T) {
	for _, c := range golden.ByFn(t, "ece", "ece_score") {
		t.Run(c.Name, func(t *testing.T) {
			var rec eceCase
			c.Unmarshal(t, &rec)
			got := ECE(rec.Conf, rec.Correct, rec.Bins)
			if rec.ECEIsNaN {
				if !math.IsNaN(got) {
					t.Errorf("ECE = %v, want NaN", got)
				}
				return
			}
			if got != rec.ECE {
				t.Errorf("ECE = %v, want %v", got, rec.ECE)
			}
		})
	}
}

// TestECEZeroInNoBin pins the (lo, hi] bins: a confidence of exactly 0 is in
// none of them, so it counts towards n but never towards a bin's gap. With
// [0, 0.5] and [true, false] only the 0.5 contributes: 1/2 · |0.5 − 0|.
func TestECEZeroInNoBin(t *testing.T) {
	if got := ECE([]float64{0, 0.5}, []bool{true, false}, DefaultECEBins); got != 0.25 {
		t.Errorf("ECE = %v, want 0.25", got)
	}
	if got := ECE(nil, nil, DefaultECEBins); !math.IsNaN(got) {
		t.Errorf("ECE(empty) = %v, want NaN", got)
	}
}

// TestBrierRejectsLabelOutOfRange pins that a label outside the row's k is an
// error, not an all-zero target: silently scoring Σp² would make a malformed
// evaluation look valid (PR #22 review). -1 included, which numpy's
// np.eye(k)[y] would quietly read as the last class.
func TestBrierRejectsLabelOutOfRange(t *testing.T) {
	for _, y := range []int{-1, 2, 3} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("Brier with label %d on k=2 did not panic", y)
				}
			}()
			Brier([][]float64{{0.5, 0.5}}, []int{y})
		}()
	}
}

// brierCase is one "brier" record of testdata/ece.jsonl.
type brierCase struct {
	Probs      [][]float64 `json:"probs"`
	Labels     []int       `json:"labels"`
	Brier      float64     `json:"brier"`
	BrierIsNaN bool        `json:"brier_is_nan"`
}

// TestBrierFixture replays the plan's Brier definition against the numpy
// computation the generator records, ragged k included.
func TestBrierFixture(t *testing.T) {
	for _, c := range golden.ByFn(t, "ece", "brier") {
		t.Run(c.Name, func(t *testing.T) {
			var rec brierCase
			c.Unmarshal(t, &rec)
			got := Brier(rec.Probs, rec.Labels)
			if rec.BrierIsNaN {
				if !math.IsNaN(got) {
					t.Errorf("Brier = %v, want NaN", got)
				}
				return
			}
			if got != rec.Brier {
				t.Errorf("Brier = %v, want %v", got, rec.Brier)
			}
		})
	}
}
