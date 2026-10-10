package safetensors

import (
	"math"
	"testing"
)

// refF16 decodes an IEEE 754 binary16 bit pattern straight from the format's
// definition, in float64, independently of f16ToF32's bit manipulation:
// (-1)^s · 2^(e-15) · (1 + m/1024) for a normal, (-1)^s · 2^-14 · (m/1024) for
// a subnormal or zero, and Inf (m = 0) or NaN (m ≠ 0) for e = 31.
func refF16(h uint16) float64 {
	s, e, m := h>>15, int(h>>10)&0x1f, float64(h&0x3ff)
	sign := 1.0
	if s == 1 {
		sign = -1
	}
	switch {
	case e == 31 && m == 0:
		return math.Inf(int(sign))
	case e == 31:
		return math.Copysign(math.NaN(), sign)
	case e == 0:
		return math.Copysign(math.Ldexp(m/1024, -14), sign)
	default:
		return math.Copysign(math.Ldexp(1+m/1024, e-15), sign)
	}
}

// Every binary16 value is exactly representable in float32, so the decode is
// checked bit for bit over all 65536 patterns: normals, subnormals, ±0 and ±Inf
// by their float32 bits, NaN by NaN-ness, sign and payload.
func TestF16ToF32Exhaustive(t *testing.T) {
	var nans, subnormals int
	for i := range 1 << 16 {
		h := uint16(i)
		got, want := f16ToF32(h), refF16(h)
		gotBits := math.Float32bits(got)

		if math.IsNaN(want) {
			nans++
			if !math.IsNaN(float64(got)) {
				t.Errorf("f16ToF32(%#04x) = %v, want NaN", h, got)
				continue
			}
			if gotNeg := gotBits>>31 == 1; gotNeg != math.Signbit(want) {
				t.Errorf("f16ToF32(%#04x): NaN sign bit %v, want %v", h, gotNeg, math.Signbit(want))
			}
			// The 10-bit payload, quiet bit included, lands in the top of the
			// 23-bit float32 mantissa.
			if payload := gotBits & 0x7fffff; payload != uint32(h&0x3ff)<<13 {
				t.Errorf("f16ToF32(%#04x): NaN payload %#06x, want %#06x", h, payload, uint32(h&0x3ff)<<13)
			}
			continue
		}
		if h&0x7c00 == 0 && h&0x3ff != 0 {
			subnormals++
		}
		if wantBits := math.Float32bits(float32(want)); gotBits != wantBits {
			t.Errorf("f16ToF32(%#04x) = %v (%#08x), want %v (%#08x)", h, got, gotBits, float32(want), wantBits)
		}
	}
	// Both signs of every non-zero mantissa: the loop covered each class.
	if nans != 2046 || subnormals != 2046 {
		t.Errorf("saw %d NaNs and %d subnormals, want 2046 of each", nans, subnormals)
	}
}

// A few values pinned by hand anchor the reference decoder itself, so it and
// f16ToF32 cannot agree on the same mistake.
func TestF16ToF32Known(t *testing.T) {
	tests := []struct {
		h    uint16
		want float32
	}{
		{0x3c00, 1},
		{0xc000, -2},
		{0x3800, 0.5},
		{0x3e00, 1.5},
		{0x7bff, 65504},          // largest normal
		{0x0400, 0x1p-14},        // smallest normal
		{0x03ff, 1023 * 0x1p-24}, // largest subnormal
		{0x0001, 0x1p-24},        // smallest subnormal
		{0x8001, -0x1p-24},
		{0x3555, 0.333251953125},
		{0x7c00, float32(math.Inf(1))},
		{0xfc00, float32(math.Inf(-1))},
		{0x0000, 0},
	}
	for _, tt := range tests {
		if got := f16ToF32(tt.h); math.Float32bits(got) != math.Float32bits(tt.want) {
			t.Errorf("f16ToF32(%#04x) = %v, want %v", tt.h, got, tt.want)
		}
		if ref := float32(refF16(tt.h)); math.Float32bits(ref) != math.Float32bits(tt.want) {
			t.Errorf("refF16(%#04x) = %v, want %v", tt.h, ref, tt.want)
		}
	}
	if got := f16ToF32(0x8000); math.Float32bits(got) != 0x80000000 {
		t.Errorf("f16ToF32(0x8000) = %#08x, want -0 (0x80000000)", math.Float32bits(got))
	}
}
