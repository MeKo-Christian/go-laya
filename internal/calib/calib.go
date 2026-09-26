// Package calib turns a question's logits into laya's calibrated numbers: the
// temperature lookup, the softmax, and the entropy confidence
// (original/laya/common.py:200-211, agent.py:294-335; invariants #22-#29).
//
// The arithmetic follows numpy's precision rather than Go's default. The
// logits arrive as float32, and under NumPy 2's weak-scalar promotion dividing
// them by a Python float keeps them float32, so the softmax and the entropy
// are float32 arithmetic (invariant #24a). Only the score expectation and the
// noul confidence widen to float64, where Python calls float() or multiplies
// by an int64 arange. Doing it all in float64 moves the fourth decimal after
// rounding.
//
// Not bit-exact: exp and log. They are the float64 functions narrowed to
// float32, whereas numpy runs its own float32 kernel, chosen per CPU (SIMD
// with FMA where available, libm otherwise). Against numpy 2.5.3's AVX2
// kernel, exp differs by up to 2 ULP and log by up to 4, and 32 of 200 000
// random softmaxes changed a rounded probability. PLAN.md Task 7.1.8 tracks
// porting the kernel.
//
// Nothing here rounds. Python applies round(x, 4) in the agent, after these
// functions return, and a caller does the same with jsonx.Round4.
package calib

import (
	"math"

	"github.com/MeKo-Christian/go-laya/question"
)

// minTemperature is the floor under a temperature scale: agent.py:305 divides
// by max(1e-3, t_scale), so a zero or negative fitted temperature cannot
// divide by zero or flip the distribution.
const minTemperature = 1e-3

// TempBucket is temp_bucket (common.py:209-211): the question type's name and
// a size class of k, "2" up to 2, "3-5", "6-10", then "11+". It is the key
// into a checkpoint's temperature_by_options.
func TempBucket(qt question.QType, k int) string {
	var size string
	switch {
	case k <= 2:
		size = "2"
	case k <= 5:
		size = "3-5"
	case k <= 10:
		size = "6-10"
	default:
		size = "11+"
	}
	return qt.String() + ":" + size
}

// Temperatures are a checkpoint's fitted temperature scales, the
// "temperature" and "temperature_by_options" keys of rl_agent_config.json.
type Temperatures struct {
	// ByQType is indexed by question.QType.
	ByQType [3]float64
	// ByOptions is keyed by TempBucket. It may be empty: laya-multilingual
	// ships no fitted buckets.
	ByOptions map[string]float64
}

// DefaultTemperatures is what agent.py:194-195 uses when the config carries
// neither key: 1.0 for every type and no buckets.
func DefaultTemperatures() Temperatures {
	return Temperatures{ByQType: [3]float64{1, 1, 1}}
}

// Scale is the temperature a question of type qt with k options is divided
// by (agent.py:304): the fitted bucket when there is one, otherwise the
// per-type temperature. The 1e-3 floor is Softmax's, not applied here.
func (t Temperatures) Scale(qt question.QType, k int) float64 {
	if s, ok := t.ByOptions[TempBucket(qt, k)]; ok {
		return s
	}
	return t.ByQType[qt]
}

// Softmax is agent.py:305-307: a max-subtracted softmax over exactly the first
// k logits, each divided by max(1e-3, temp). The tail past k is the head's
// padding and never takes part. It computes in float32, as numpy does, apart
// from exp (see the package comment).
func Softmax(logits []float32, k int, temp float64) []float32 {
	div := float32(max(minTemperature, temp))
	z := make([]float32, k)
	for i := range z {
		z[i] = logits[i] / div
	}
	top := z[0]
	for _, v := range z[1:] {
		top = max(top, v)
	}
	for i, v := range z {
		z[i] = float32(math.Exp(float64(v - top)))
	}
	sum := numpySum(z)
	for i := range z {
		z[i] /= sum
	}
	return z
}

// logClip is the lower clip confidence_from_probs puts under p before the log.
const logClip = 1e-12

// Confidence is confidence_from_probs (common.py:200-206), the confidence of a
// choice or score answer: 1 - H(p)/ln k clipped to [0, 1], with the entropy
// over the first k entries and p clipped to 1e-12 inside the log. k < 2 is
// fully confident. The entropy is float32 arithmetic, apart from log (see
// the package comment), and the result is widened as Python's float() widens
// it.
//
// A noul answer does not use this; see NoulConfidence.
func Confidence(p []float32, k int) float64 {
	if k < 2 {
		return 1
	}
	terms := make([]float32, k)
	for i, v := range p[:k] {
		c := min(max(v, float32(logClip)), 1)
		terms[i] = v * float32(math.Log(float64(c)))
	}
	ent := -numpySum(terms)
	conf := 1 - ent/float32(math.Log(float64(k)))
	return float64(min(max(conf, 0), 1))
}

// NoulConfidence is a noul answer's confidence, max(p1, 1-p1) in float64 from
// the float32 P(true) (agent.py:335). It is deliberately not Confidence: for a
// uniform k=2 distribution that gives 0 and this gives 0.5 (invariant #26).
func NoulConfidence(p1 float32) float64 {
	v := float64(p1)
	return max(v, 1-v)
}

// Expectation is a score answer's value, Σ i·p[i] over the levels
// (agent.py:322): an expectation, not an argmax (invariant #28). numpy
// multiplies an int64 arange by the float32 p, so the sum is float64.
func Expectation(p []float32) float64 {
	terms := make([]float64, len(p))
	for i, v := range p {
		terms[i] = float64(i) * float64(v)
	}
	return numpySum(terms)
}

// numpySum adds in the order numpy's add.reduce does over a contiguous 1-D
// array: pairwise_sum over the whole array, which is left to right below eight
// elements. In float32 the order is visible in the last bit.
func numpySum[F float32 | float64](a []F) F {
	return pairwiseSum(a)
}

// pairwiseSum is numpy's pairwise_sum (loops_utils.h.src): a plain loop below
// 8 elements, eight interleaved accumulators up to 128, and halving beyond.
func pairwiseSum[F float32 | float64](a []F) F {
	n := len(a)
	switch {
	case n < 8:
		var res F
		for _, v := range a {
			res += v
		}
		return res
	case n <= 128:
		var r [8]F
		copy(r[:], a[:8])
		i := 8
		for ; i < n-n%8; i += 8 {
			for j := range r {
				r[j] += a[i+j]
			}
		}
		res := ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
		for ; i < n; i++ {
			res += a[i]
		}
		return res
	default:
		n2 := n / 2
		n2 -= n2 % 8
		return pairwiseSum(a[:n2]) + pairwiseSum(a[n2:])
	}
}
