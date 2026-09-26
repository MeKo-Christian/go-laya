package calib

import "math"

// numpy's float32 exp and log, bit for bit (PLAN.md Task 7.1.8).
//
// numpy does not call libm for float32 exp/log. On an AVX2+FMA3 host (the
// X86_V3 dispatch target, which the golden vectors came from) it runs
// simd_exp_FLOAT and simd_log_FLOAT from
// numpy/_core/src/umath/loops_exponent_log.dispatch.c.src (numpy 2.5.3),
// scalar calls included. What follows is one lane of that kernel: every
// _mm256_fmadd_ps is fma32, every other vector op is a single float32
// operation. Separate products carry an explicit float32 conversion, because
// Go may otherwise fuse x*y + z into one FMA on arm64, ppc64 and s390x, and
// the kernel rounds them apart.
//
// The constants are numpy's npy_simd_data.h and npy_math.h literals. Go
// rounds an untyped constant to float32 exactly as C rounds an f-suffixed
// literal, so the bits match.
const (
	codyWaiteLn2Hi = float32(-6.93145752e-1) // NPY_CODY_WAITE_LOGE_2_HIGHf
	codyWaiteLn2Lo = float32(-1.42860677e-6) // NPY_CODY_WAITE_LOGE_2_LOWf

	expP0 = float32(9.999999999980870924916e-01)
	expP1 = float32(7.257664613233124478488e-01)
	expP2 = float32(2.473615434895520810817e-01)
	expP3 = float32(5.114512081637298353406e-02)
	expP4 = float32(6.757896990527504603057e-03)
	expP5 = float32(5.082762527590693718096e-04)
	expQ0 = float32(1.000000000000000000000e+00)
	expQ1 = float32(-2.742335390411667452936e-01)
	expQ2 = float32(2.159509375685829852307e-02)

	logP0 = float32(0.000000000000000000000e+00)
	logP1 = float32(9.999999999999998702752e-01)
	logP2 = float32(2.112677543073053063722e+00)
	logP3 = float32(1.480000633576506585156e+00)
	logP4 = float32(3.808837741388407920751e-01)
	logP5 = float32(2.589979117907922693523e-02)
	logQ0 = float32(1.000000000000000000000e+00)
	logQ1 = float32(2.612677543073109236779e+00)
	logQ2 = float32(2.453006071784736363091e+00)
	logQ3 = float32(9.864942958519418960339e-01)
	logQ4 = float32(1.546476374983906719538e-01)
	logQ5 = float32(5.875095403124574342950e-03)

	rintMagic = float32(0x1.8p23)                                    // NPY_RINT_CVT_MAGICf
	log2e     = float32(1.442695040888963407359924681001892137)      // NPY_LOG2Ef
	ln2       = float32(0.693147180559945309417232121458176568)      // NPY_LOGE2f
	sqrtHalf  = float32(0.707106781186547524400844362104849039)      // NPY_SQRT1_2f
	expXMax   = float32(88.72283935546875)                           // simd_exp_FLOAT's xmax
	expXMin   = float32(-103.97208404541015625)                      // and xmin
	twoTo100  = float32(0x1p100)                                     // 0x71800000
	minNormal = float32(1.1754943508222875079687365372222456778e-38) // FLT_MIN
)

// The kernel's special results: NPY_NANF is a positive quiet NaN, and
// log(x < 0) returns its negation.
var (
	npyNaN    = math.Float32frombits(0x7fc00000)
	npyNegNaN = math.Float32frombits(0xffc00000)
)

// fma32 is a·b + c rounded once to float32, as _mm256_fmadd_ps computes it.
// Narrowing math.FMA's float64 result would round twice and miss by one ULP
// whenever the float64 result lands on a float32 tie.
//
// The product of two float32 values is exact in float64 (24 + 24 bits < 53).
// The sum is rounded to float64 with round-to-odd, recovering the error with
// TwoSum and forcing the last bit to 1 when anything was lost. Rounding to odd
// at 53 bits and then to nearest at 24 is a single correct rounding, because
// 53 ≥ 24 + 2.
func fma32(a, b, c float32) float32 {
	p := float64(float64(a) * float64(b)) // explicit: must not fuse with the add
	cc := float64(c)
	s := p + cc
	if math.IsInf(s, 0) || math.IsNaN(s) {
		return float32(s)
	}
	bb := s - p
	e := (p - (s - bb)) + (cc - bb)
	if e != 0 && math.Float64bits(s)&1 == 0 {
		s = math.Nextafter(s, math.Copysign(math.Inf(1), e))
	}
	return float32(s)
}

// exp32 is simd_exp_FLOAT for one lane.
func exp32(x float32) float32 {
	switch {
	case x != x: // NaN
		return npyNaN
	case x >= expXMax:
		return float32(math.Inf(1))
	case x <= expXMin:
		return 0
	}

	quadrant := float32(x * log2e)
	quadrant = float32(quadrant + rintMagic) // round to nearest via the magic add
	quadrant = float32(quadrant - rintMagic)

	// Cody-Waite range reduction; c3 is 0 in simd_exp_FLOAT.
	x = fma32(quadrant, codyWaiteLn2Hi, x)
	x = fma32(quadrant, codyWaiteLn2Lo, x)
	x = fma32(quadrant, 0, x)

	num := fma32(expP5, x, expP4)
	num = fma32(num, x, expP3)
	num = fma32(num, x, expP2)
	num = fma32(num, x, expP1)
	num = fma32(num, x, expP0)
	den := fma32(expQ2, x, expQ1)
	den = fma32(den, x, expQ0)
	return scalef(num/den, quadrant)
}

// scalef is fma_scalef_ps: poly·2^quadrant by adding quadrant to poly's
// exponent bits. At quadrant ≤ −125 the result is subnormal and adding
// exponents breaks down, so the kernel scales by 2^-125 that way and divides
// by 2^(−125−quadrant) to produce the subnormal.
func scalef(poly, quadrant float32) float32 {
	const minQuadrant = float32(-125)
	if quadrant <= minQuadrant {
		diff := float32(0 - float32(quadrant-minQuadrant))
		pow := int32(1) << uint32(int32(diff))
		poly = addExponent(poly, minQuadrant)
		return poly / float32(pow)
	}
	return addExponent(poly, quadrant)
}

func addExponent(poly, quadrant float32) float32 {
	return math.Float32frombits(uint32(int32(math.Float32bits(poly)) + int32(quadrant)<<23))
}

// log32 is simd_log_FLOAT for one lane.
func log32(x float32) float32 {
	switch {
	case x != x: // NaN
		return npyNaN
	case x < 0:
		return npyNegNaN
	case x == 0:
		return float32(math.Inf(-1))
	case math.IsInf(float64(x), 1):
		return float32(math.Inf(1))
	}

	exponent, m := frexp32(x)
	if m <= sqrtHalf {
		m = float32(m + m)
		exponent = float32(exponent - 1)
	}
	m = float32(m - 1)

	num := fma32(logP5, m, logP4)
	num = fma32(num, m, logP3)
	num = fma32(num, m, logP2)
	num = fma32(num, m, logP1)
	num = fma32(num, m, logP0)
	den := fma32(logQ5, m, logQ4)
	den = fma32(den, m, logQ3)
	den = fma32(den, m, logQ2)
	den = fma32(den, m, logQ1)
	den = fma32(den, m, logQ0)
	return fma32(exponent, ln2, num/den)
}

// frexp32 is fma_get_exponent and fma_get_mantissa together: x = m·2^e with
// m in [0.5, 1), both as float32. A subnormal x is first scaled by 2^100 and
// the 100 taken back off the exponent.
func frexp32(x float32) (exponent, mantissa float32) {
	sub := x < minNormal
	if sub {
		x = float32(x * twoTo100)
	}
	bits := math.Float32bits(x)
	exponent = float32(int32(bits>>23) - 0x7e)
	if sub {
		exponent = float32(exponent - 100)
	}
	mantissa = math.Float32frombits(bits&0x7fffff | 126<<23)
	return exponent, mantissa
}
