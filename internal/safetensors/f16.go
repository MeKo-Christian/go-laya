package safetensors

import "math"

// f16ToF32 decodes an IEEE 754 binary16 bit pattern. Every binary16 value is
// exactly representable in float32, so the result is exact: the sign moves to
// bit 31, a normal's exponent is rebiased from 15 to 127, a subnormal is
// normalised into a float32 normal, and a NaN keeps its sign and its 10-bit
// payload (quiet bit included) in the top of the float32 mantissa.
func f16ToF32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h & 0x3ff)

	switch {
	case exp == 0x1f: // Inf (mant == 0) or NaN
		return math.Float32frombits(sign | 0xff<<23 | mant<<13)
	case exp != 0: // normal: 2^(exp-15) = 2^((exp+112)-127)
		return math.Float32frombits(sign | (exp+112)<<23 | mant<<13)
	case mant == 0: // ±0
		return math.Float32frombits(sign)
	}

	// Subnormal, 2^-14 · mant/1024: shift the leading 1 up to the implicit
	// bit's place (bit 10), lowering the float32 exponent, 127-14, once per
	// shift.
	exp = 127 - 14
	for mant&0x400 == 0 {
		mant <<= 1
		exp--
	}
	return math.Float32frombits(sign | exp<<23 | (mant&0x3ff)<<13)
}
