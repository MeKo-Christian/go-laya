package calib

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/golden"
	"github.com/MeKo-Christian/go-laya/jsonx"
)

// actSoftmaxCase is one group of testdata/act_softmax.jsonl: [n, 2] act-head
// logit rows, torch's float32 softmax of them, and round(p[0], 4), with the
// float32 values as hex bit patterns.
type actSoftmaxCase struct {
	ActLogits      [][]string `json:"act_logits"`
	P              [][]string `json:"p"`
	ActProbability []float64  `json:"act_probability"`
}

// softmaxLibm is a float32 softmax over one row with exp taken from math.Exp
// and narrowed: the alternative to exp32 that Task 7.3.10 measures.
func softmaxLibm(row []float32) []float32 {
	top := row[0]
	for _, v := range row[1:] {
		top = max(top, v)
	}
	z := make([]float32, len(row))
	for i, v := range row {
		z[i] = float32(math.Exp(float64(v - top)))
	}
	sum := numpySum(z)
	for i := range z {
		z[i] /= sum
	}
	return z
}

// actTally counts one candidate's disagreements with torch.
type actTally struct {
	bits, round4 int
	maxULP       int64
	first        string
	perGroup     map[string]int
}

func (a *actTally) add(group string, got []float32, want []uint32, r4 float64,
	desc func(uint32, uint32) string,
) {
	bitDiff := false
	for i := range want {
		g := math.Float32bits(got[i])
		if !sameFloat32(g, want[i]) {
			bitDiff = true
			a.maxULP = max(a.maxULP, abs64(int64(g)-int64(want[i])))
		}
	}
	if bitDiff {
		a.bits++
	}
	if jsonx.Round4(float64(got[0])) != r4 {
		a.round4++
		if a.perGroup == nil {
			a.perGroup = map[string]int{}
		}
		a.perGroup[group]++
		if a.first == "" {
			a.first = desc(math.Float32bits(got[0]), want[0])
		}
	}
}

// TestActSoftmax requires ActSoftmax to equal the act head's torch float32
// softmax (agent.py:295) bit for bit, every column, on every row of every
// head width in testdata/act_softmax.jsonl (Task 7.3.10). Widths of eight and
// more take ATen's chunked reduction, tail and shuffle tree, so the fixture
// covers 1 to 20, 32 and 33. answers.jsonl cannot check this:
// its act_probability values come from the dumper's numpy copy, not torch.
//
// It also logs, as measurement, how far the two alternatives fall short:
// calib.Softmax (numpy's exp32, divide by the sum) and a softmax on narrowed
// math.Exp. Both miss in the last bits, and after Round4 near a tie, which is
// why act_probability has its own function.
func TestActSoftmax(t *testing.T) {
	cases := golden.ByFn(t, "act_softmax", "act_softmax")
	var rows int
	var aten, numpy, libm actTally
	perWidth := map[int][2]int{} // width -> {rows, ActSoftmax bit mismatches}
	for _, c := range cases {
		var rec actSoftmaxCase
		c.Unmarshal(t, &rec)
		if len(rec.ActLogits) != len(rec.P) || len(rec.P) != len(rec.ActProbability) {
			t.Fatalf("%s: %d logit rows, %d p rows, %d act_probability", c.Name,
				len(rec.ActLogits), len(rec.P), len(rec.ActProbability))
		}
		for r, lr := range rec.ActLogits {
			row := make([]float32, len(lr))
			for i, s := range lr {
				row[i] = math.Float32frombits(hexBits(t, s))
			}
			want := make([]uint32, len(rec.P[r]))
			for i, s := range rec.P[r] {
				want[i] = hexBits(t, s)
			}
			r4 := rec.ActProbability[r]
			// The fixture must be self-consistent before it can judge anything.
			if got := jsonx.Round4(float64(math.Float32frombits(want[0]))); got != r4 {
				t.Fatalf("%s row %d: Round4(torch p0) = %v, fixture act_probability %v",
					c.Name, r, got, r4)
			}
			desc := func(got, want uint32) string {
				return fmt.Sprintf("%s row %d: logits %v p0 %s vs torch %s",
					c.Name, r, lr, hex32(got), hex32(want))
			}
			numpy.add(c.Name, Softmax(row, len(row), 1.0), want, r4, desc)
			libm.add(c.Name, softmaxLibm(row), want, r4, desc)
			got := ActSoftmax(row)
			aten.add(c.Name, got, want, r4, desc)
			w := perWidth[len(row)]
			w[0]++
			if !sameRow(got, want) {
				w[1]++
				if aten.bits <= 5 { // name the first few rows
					gotHex := make([]string, len(got))
					for i, v := range got {
						gotHex[i] = hex32(math.Float32bits(v))
					}
					t.Errorf("%s row %d: ActSoftmax(%v) = %v; torch %v", c.Name, r, lr, gotHex, rec.P[r])
				}
			}
			perWidth[len(row)] = w
			rows++
		}
	}

	t.Logf("%d rows in %d groups", rows, len(cases))
	for _, c := range []struct {
		name string
		a    *actTally
	}{
		{"ActSoftmax (SLEEF exp, * 1/sum)", &aten},
		{"calib.Softmax (exp32, / sum)", &numpy},
		{"narrowed math.Exp, / sum", &libm},
	} {
		t.Logf("%-32s %4d bit mismatches (max %d ulp), %3d Round4 mismatches %v",
			c.name, c.a.bits, c.a.maxULP, c.a.round4, c.a.perGroup)
	}
	widths := make([]int, 0, len(perWidth))
	for w := range perWidth {
		widths = append(widths, w)
	}
	slices.Sort(widths)
	for _, w := range widths {
		t.Logf("width %2d: %4d rows, %d ActSoftmax bit mismatches", w, perWidth[w][0], perWidth[w][1])
	}
	if numpy.first != "" {
		t.Logf("first calib.Softmax Round4 mismatch: %s", numpy.first)
	}
	if aten.bits != 0 || aten.round4 != 0 {
		t.Errorf("ActSoftmax differs from torch in bits on %d of %d rows (%d after Round4)",
			aten.bits, rows, aten.round4)
	}
}

func sameRow(got []float32, want []uint32) bool {
	for i := range want {
		if !sameFloat32(math.Float32bits(got[i]), want[i]) {
			return false
		}
	}
	return true
}

// TestActSoftmaxEdges pins what the fixture does not reach: a NaN spreads to
// every output as vec::maximum makes it, in the lane-by-lane path, in a full
// chunk and in the blended tail alike, and an empty row gives an empty
// result, as torch.softmax does for a zero-width last dimension.
func TestActSoftmaxEdges(t *testing.T) {
	nan := float32(math.NaN())
	for _, tc := range []struct{ n, at int }{{2, 0}, {2, 1}, {9, 3}, {9, 8}, {17, 16}} {
		row := make([]float32, tc.n)
		row[tc.at] = nan
		for i, v := range ActSoftmax(row) {
			if v == v {
				t.Errorf("width %d, NaN at %d: p[%d] = %v, want NaN", tc.n, tc.at, i, v)
			}
		}
	}
	if p := ActSoftmax([]float32{3}); p[0] != 1 {
		t.Errorf("ActSoftmax([3]) = %v, want [1]", p)
	}
	if p := ActSoftmax(nil); p == nil || len(p) != 0 {
		t.Errorf("ActSoftmax(nil) = %#v, want an empty, non-nil slice", p)
	}
}

func abs64(x int64) int64 {
	if x < 0 {
		return -x
	}
	return x
}

func hex32(b uint32) string {
	const digits = "0123456789abcdef"
	out := []byte("0x00000000")
	for i := 9; i >= 2; i-- {
		out[i] = digits[b&0xf]
		b >>= 4
	}
	return string(out)
}
