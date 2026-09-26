package calib

import (
	"math"
	"math/big"
	"math/rand/v2"
	"strconv"
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/golden"
)

// fmaOracle is a·b + c computed exactly and rounded once to float32 by
// math/big, which rounds to nearest even and handles subnormal results.
func fmaOracle(a, b, c float32) float32 {
	const prec = 1024 // wide enough to hold any float32 a·b + c exactly
	x := new(big.Float).SetPrec(prec).SetFloat64(float64(a))
	y := new(big.Float).SetPrec(prec).SetFloat64(float64(b))
	z := new(big.Float).SetPrec(prec).SetFloat64(float64(c))
	x.Mul(x, y).Add(x, z)
	f, _ := x.Float32()
	if f == 0 && x.Sign() == 0 {
		// big.Float drops the sign of an exact zero sum; IEEE gives +0 for
		// x + (−x) under round-to-nearest, and −0 only when both are −0.
		if math.Signbit(float64(a*b)) && math.Signbit(float64(c)) {
			return float32(math.Copysign(0, -1))
		}
		return 0
	}
	return f
}

// TestFMA32 checks the single-rounding float32 FMA against math/big. Narrowing
// math.FMA's float64 result rounds twice and fails this on the tie cases,
// which is exactly the error Task 7.1.8 warns about.
func TestFMA32(t *testing.T) {
	check := func(a, b, c float32) {
		t.Helper()
		got, want := fma32(a, b, c), fmaOracle(a, b, c)
		if math.Float32bits(got) != math.Float32bits(want) {
			t.Fatalf("fma32(%x, %x, %x) = %x, want %x",
				math.Float32bits(a), math.Float32bits(b), math.Float32bits(c),
				math.Float32bits(got), math.Float32bits(want))
		}
	}

	// Double-rounding traps: (1 + 2^-12)² = 1 + 2^-11 + 2^-24 is exactly
	// halfway between two float32 values. A c of ±2^-80 moves the exact sum a
	// hair off the tie, which rounding to float64 first erases, leaving the
	// tie to round to even: wrong for +2^-80.
	a := float32(1 + 0x1p-12)
	check(a, a, 0x1p-80)
	check(a, a, -0x1p-80)
	check(a, a, 0)
	check(-a, a, 0x1p-80)

	rng := rand.New(rand.NewPCG(7, 18))
	finite := func(f float32) bool {
		return !math.IsNaN(float64(f)) && !math.IsInf(float64(f), 0)
	}
	randF := func() float32 {
		for {
			f := math.Float32frombits(rng.Uint32())
			if finite(f) {
				return f
			}
		}
	}
	nearby := func(x float32, n int) float32 {
		return math.Float32frombits(math.Float32bits(x) + uint32(rng.IntN(2*n+1)-n))
	}
	const n = 200_000
	found := 0
	for range n {
		a, b := randF(), randF()
		p := a * b
		if math.IsInf(float64(p), 0) || p == 0 {
			continue
		}
		// c near −a·b forces cancellation, c tiny against a·b forces a tie
		// hunt in the low bits, and an unrelated c covers the rest.
		for _, c := range []float32{-nearby(p, 3), nearby(p, 3) * 0x1p-24, nearby(p, 3) * 0x1p-25, randF()} {
			if !finite(c) || math.IsInf(float64(fmaOracle(a, b, c)), 0) {
				continue
			}
			check(a, b, c)
			found++
		}
	}
	// Mantissas of up to 12 bits, so products are short and exact ties are
	// common rather than vanishingly rare.
	short := func() float32 {
		m := float32(rng.IntN(1<<12) + 1<<12)
		return float32(math.Ldexp(float64(m), rng.IntN(40)-40))
	}
	for range n {
		a, b := short(), short()
		c := short() * float32(math.Ldexp(1, -rng.IntN(30)))
		if rng.IntN(2) == 0 {
			c = -c
		}
		check(a, b, c)
		found++
	}
	// Subnormal results.
	for range n / 10 {
		a := math.Float32frombits(rng.Uint32N(0x0C000000) | 0x00800000)
		b := math.Float32frombits(rng.Uint32N(0x0C000000) | 0x00800000)
		c := math.Float32frombits(rng.Uint32N(0x00800000))
		if rng.IntN(2) == 0 {
			c = -c
		}
		check(a, b, c)
		found++
	}
	if found < n {
		t.Fatalf("only %d triples checked", found)
	}
}

// f32mathCase is one record of testdata/f32math.jsonl: float32 bit patterns in
// hex, so no float printer or JSON reader can round them.
type f32mathCase struct {
	In  []string `json:"in"`
	Out []string `json:"out"`
}

func hexBits(t *testing.T, s string) uint32 {
	t.Helper()
	v, err := strconv.ParseUint(s, 0, 32)
	if err != nil {
		t.Fatalf("bad bit pattern %q: %v", s, err)
	}
	return uint32(v)
}

// sameFloat32 is bit equality, except that NaN payloads are not compared: the
// kernel emits one canonical NaN per sign, and the sign is what log(x < 0)
// pins, so it is compared.
func sameFloat32(got, want uint32) bool {
	gf, wf := math.Float32frombits(got), math.Float32frombits(want)
	if gf != gf || wf != wf { // either is NaN
		return gf != gf && wf != wf && got>>31 == want>>31
	}
	return got == want
}

func replayF32Math(t *testing.T, fn string, f func(float32) float32) {
	t.Helper()
	cases := golden.ByFn(t, "f32math", fn)
	if len(cases) == 0 {
		t.Fatalf("f32math.jsonl has no %s cases", fn)
	}
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			var rec f32mathCase
			c.Unmarshal(t, &rec)
			if len(rec.In) != len(rec.Out) {
				t.Fatalf("%d inputs, %d outputs", len(rec.In), len(rec.Out))
			}
			bad := 0
			for i, s := range rec.In {
				x := hexBits(t, s)
				want := hexBits(t, rec.Out[i])
				got := math.Float32bits(f(math.Float32frombits(x)))
				if !sameFloat32(got, want) {
					if bad < 5 {
						t.Errorf("%s(%#08x = %v) = %#08x, want %#08x", fn, x,
							math.Float32frombits(x), got, want)
					}
					bad++
				}
			}
			if bad > 0 {
				t.Errorf("%d of %d differ", bad, len(rec.In))
			}
		})
	}
}

// TestExp32Fixture requires exp32 to equal numpy's float32 exp bit for bit on
// every f32math.jsonl input (Task 7.1.8).
func TestExp32Fixture(t *testing.T) { replayF32Math(t, "exp", exp32) }

// TestLog32Fixture requires log32 to equal numpy's float32 log bit for bit on
// every f32math.jsonl input (Task 7.1.8).
func TestLog32Fixture(t *testing.T) { replayF32Math(t, "log", log32) }

// f32softmaxCase is a "softmax" record of testdata/f32math.jsonl: one
// agent.py:305-307 softmax and its confidence_from_probs, in bits.
type f32softmaxCase struct {
	K           int      `json:"k"`
	Temperature float64  `json:"temperature"`
	Logits      []string `json:"logits"`
	Probs       []string `json:"probs"`
	Confidence  float64  `json:"confidence"`
}

// TestSoftmaxBits requires Softmax and Confidence to equal numpy bit for bit,
// not after Round4. At four decimals answers.jsonl cannot tell libm's exp and
// log from numpy's kernel; in bits almost every row can.
func TestSoftmaxBits(t *testing.T) {
	cases := golden.ByFn(t, "f32math", "softmax")
	if len(cases) == 0 {
		t.Fatal("f32math.jsonl has no softmax cases")
	}
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			var rec f32softmaxCase
			c.Unmarshal(t, &rec)
			logits := make([]float32, len(rec.Logits))
			for i, s := range rec.Logits {
				logits[i] = math.Float32frombits(hexBits(t, s))
			}
			p := Softmax(logits, rec.K, rec.Temperature)
			for i, s := range rec.Probs {
				if got, want := math.Float32bits(p[i]), hexBits(t, s); got != want {
					t.Errorf("p[%d] = %#08x, want %#08x", i, got, want)
				}
			}
			if got := Confidence(p, rec.K); got != rec.Confidence {
				t.Errorf("Confidence = %v, want %v", got, rec.Confidence)
			}
		})
	}
}
