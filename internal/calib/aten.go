package calib

import "math"

// torch's float32 CPU softmax over a short last dimension, bit for bit
// (PLAN.md Task 7.3.10).
//
// agent.py:295 turns the act head's [batch, 2] logits into act_probability
// with torch.softmax(act.float(), -1), not with numpy, so calib.Softmax (numpy's
// exp32 and a divide) is the wrong reference for it: it misses torch in the
// last bits on about half of all rows and after Round4 near a tie.
//
// The reference is torch 2.14.0+cpu on the AVX2 kernel (the capability
// testdata/act_softmax.jsonl records). For a contiguous tensor and dim = -1,
// ATen's softmax_lastdim_kernel runs _vec_softmax_lastdim
// (aten/src/ATen/native/cpu/SoftMaxKernel.cpp), which per row does
//
//	max = vec::reduce_all(vec::maximum, input, n)
//	out = vec::map((x - max).exp(), input, n)
//	sum = vec::reduce_all(+, out, n)
//	out = vec::map(x * (1 / sum), out, n)
//
// SoftMaxKernel.cpp is not in the wheel. The reduction and exp come from the
// headers it includes, ATen/cpu/vec/functional_base.h and
// ATen/cpu/vec/vec256/vec256_float.h. The two details that set the bits are
// the multiply by 1/sum, not a divide, and Vectorized<float>::exp, which is
// Sleef_expf8_u10 and not the exp_u20 polynomial next to it. Both were picked
// by testing against the fixture: numpy's exp, libm's exp and exp_u20, each
// with a divide or a reciprocal, all miss on hundreds of its 4529 rows. This
// port matches every one.

// atenVecWidth is Vectorized<float>::size() under AVX2. Below it, reduce_all
// takes vec_reduce_all's lane-by-lane path (functional_base.h:30-45 and 190-191),
// which is the only path ActSoftmax ports.
const atenVecWidth = 8

// sleefExpf's constants, from SLEEF's sleefsimdsp.c xexpf, which
// Sleef_expf8_u10 is under AVX2 with FMA: R_LN2f, the split ln 2 (L2Uf, L2Lf)
// and the minimax coefficients.
const (
	sleefRLn2 = float32(1.442695040888963407359924681001892137426645954152985934135449)
	sleefL2U  = float32(0.693145751953125)
	sleefL2L  = float32(1.428606765330187045e-06)
	sleefC5   = float32(0.000198527617612853646278381)
	sleefC4   = float32(0.00139304355252534151077271)
	sleefC3   = float32(0.00833336077630519866943359)
	sleefC2   = float32(0.0416664853692054748535156)
	sleefC1   = float32(0.166666671633720397949219)
	sleefC0   = float32(0.5)
	sleefMin  = float32(-104) // below this xexpf returns 0
	sleefMax  = float32(100)  // above this xexpf returns +Inf; ActSoftmax only feeds it x - max <= 0
)

// ActSoftmax is torch.softmax(row, -1) for one float32 row on the CPU, as
// agent.py:295 computes the act head's probabilities: a NaN-propagating max,
// exp(x - max) by SLEEF's xexpf, a left-to-right sum, and a multiply by the
// float32 reciprocal of that sum. p[0] of a two-wide row is what agent.py:311
// rounds into act_probability.
//
// It reproduces ATen's AVX2 kernel (see the comment above atenVecWidth) and
// panics on a row of eight or more, where ATen switches to a tree reduction
// that is not ported. The act head is two wide. A NaN anywhere makes every
// output NaN, as vec::maximum does; the NaN's payload is not reproduced.
func ActSoftmax(row []float32) []float32 {
	n := len(row)
	if n == 0 || n >= atenVecWidth {
		panic("calib: ActSoftmax ports ATen's short-row path only (1 <= len < 8)")
	}
	out := make([]float32, n)

	// vec_reduce_all folds lane i into lane 0 in order: acc = op(acc, x[i]).
	top := row[0]
	for _, v := range row[1:] {
		top = atenMaximum(top, v)
	}
	if top != top {
		for i := range out {
			out[i] = float32(math.NaN())
		}
		return out
	}

	for i, v := range row {
		out[i] = sleefExpf(v - top)
	}
	sum := out[0]
	for _, v := range out[1:] {
		sum += v
	}
	inv := 1 / sum
	for i := range out {
		out[i] *= inv
	}
	return out
}

// atenMaximum is vec::maximum on one lane (vec256_float.h:585-591):
// _mm256_max_ps(a, b), which is a > b ? a : b and so returns b on equal
// operands (−0 against +0 included), or'ed with an all-ones NaN when either
// is NaN.
func atenMaximum(a, b float32) float32 {
	if a != a || b != b {
		return math.Float32frombits(0xffffffff)
	}
	if a > b {
		return a
	}
	return b
}

// sleefExpf is SLEEF's xexpf for one lane, as Sleef_expf8_u10 computes it
// with FMA: every vmla is fma32, every other operation one float32 rounding.
//
//	q = rint(d · R_LN2f)
//	s = d − q·L2Uf − q·L2Lf
//	u = 1 + (s·s·poly(s) + s)
//	u = ldexp2k(u, q), 0 below −104, +Inf above 100
func sleefExpf(d float32) float32 {
	switch {
	case d != d:
		return d
	case d < sleefMin:
		return 0 // checked first: Go leaves int32(±Inf) implementation-defined
	case d > sleefMax:
		return float32(math.Inf(1))
	}
	q := int32(math.RoundToEven(float64(float32(d * sleefRLn2)))) // vrint_vi2_vf
	qf := float32(q)
	s := fma32(qf, -sleefL2U, d)
	s = fma32(qf, -sleefL2L, s)

	u := fma32(sleefC5, s, sleefC4)
	u = fma32(u, s, sleefC3)
	u = fma32(u, s, sleefC2)
	u = fma32(u, s, sleefC1)
	u = fma32(u, s, sleefC0)
	u = float32(1 + fma32(float32(s*s), u, s))

	// vldexp2_vf_vf_vi2: u · 2^(q>>1) · 2^(q − q>>1), each factor built from
	// its exponent bits, so that a subnormal result rounds only once more.
	h := q >> 1
	u = float32(u * pow2i(h))
	return float32(u * pow2i(q-h))
}

// pow2i is SLEEF's vpow2i_vf_vi2: 2^q from the exponent bits, for q in the
// normal range.
func pow2i(q int32) float32 {
	return math.Float32frombits(uint32(q+127) << 23)
}
