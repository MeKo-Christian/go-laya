package calib

import (
	"fmt"
	"math"
)

// DefaultECEBins is ece_score's default bin count (common.py:187).
const DefaultECEBins = 15

// ECE is ece_score (common.py:187-197), the expected calibration error: over
// bins equal-width confidence bins, the fraction of answers in each bin times
// the gap between its mean confidence and its accuracy. The bins are (lo, hi],
// so a confidence of exactly 0 -- or NaN -- falls in none of them and only
// counts towards n. An empty input is NaN, as upstream returns.
//
// It reproduces numpy's float64 arithmetic: the edges are linspace's
// i·(1/bins) with the last forced to 1, and a bin's mean confidence is
// numpy's pairwise sum over the selected values in order. conf and correct
// must have the same length.
func ECE(conf []float64, correct []bool, bins int) float64 {
	if len(correct) != len(conf) {
		panic("calib: ECE needs one correct flag per confidence")
	}
	if len(conf) == 0 {
		return math.NaN()
	}
	n := float64(len(conf))
	step := 1 / float64(bins)
	sel := make([]float64, 0, len(conf))
	var e float64
	for b := range bins {
		lo, hi := float64(b)*step, float64(b+1)*step
		if b == bins-1 {
			hi = 1
		}
		sel = sel[:0]
		var right int
		for i, c := range conf {
			if c > lo && c <= hi {
				sel = append(sel, c)
				if correct[i] {
					right++
				}
			}
		}
		if len(sel) == 0 {
			continue
		}
		m := float64(len(sel))
		e += m / n * math.Abs(pairwiseSum(sel)/m-float64(right)/m)
	}
	return e
}

// Brier is the multi-class Brier score, the mean over answers of
// Σ_j (p_j − [j = label])². Upstream has none: this is PLAN.md's definition
// for Task 6.9.3, which compares int8 against fp32 calibration. Rows may
// differ in length, as a choice question's k does; a float32 softmax output
// is widened by the caller. Both sums are numpy's pairwise order, matching
// the generator's reference. An empty input is NaN, like ECE. A label outside
// [0, len(row)) panics: scoring it as an all-zero target would make a
// malformed evaluation look valid.
func Brier(probs [][]float64, labels []int) float64 {
	if len(labels) != len(probs) {
		panic("calib: Brier needs one label per row")
	}
	if len(probs) == 0 {
		return math.NaN()
	}
	rows := make([]float64, len(probs))
	for r, p := range probs {
		y := labels[r]
		if y < 0 || y >= len(p) {
			panic(fmt.Sprintf("calib: Brier label %d outside the row's %d classes", y, len(p)))
		}
		sq := make([]float64, len(p))
		for j, v := range p {
			if j == y {
				v--
			}
			sq[j] = v * v
		}
		rows[r] = pairwiseSum(sq)
	}
	return pairwiseSum(rows) / float64(len(rows))
}
