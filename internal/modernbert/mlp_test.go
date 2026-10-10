package modernbert

import (
	"encoding/json"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/runtime/tensor"
)

// mlpCase is one "mlp" record of testdata/ops.json: the real ModernBertMLP of
// the tiny config, with the seeded random Wi [2I, H] and Wo [H, I] it ran with.
type mlpCase struct {
	Name             string    `json:"name"`
	Module           string    `json:"module"`
	ModuleClass      string    `json:"module_class"`
	HiddenActivation string    `json:"hidden_activation"`
	MLPBias          *bool     `json:"mlp_bias"`
	Input            tensorRec `json:"input"`
	Wi               tensorRec `json:"wi"`
	Wo               tensorRec `json:"wo"`
	Output           tensorRec `json:"output"`
}

// mlpVariant is ModernBertMLP read correctly or in one of the plausible wrong
// ways. The wrong ones exist so that the test can check the fixture tells each
// apart from the right one; a fixture that cannot would let that bug through.
type mlpVariant int

const (
	mlpExact        mlpVariant = iota
	mlpTanhGELU                // ACT2FN["gelu_new"] instead of ["gelu"]
	mlpErfUnscaled             // erf(x) where GELU has erf(x/sqrt 2)
	mlpHalvesSwap              // the gate activated, the input multiplying
	mlpWoTransposed            // Wo's buffer read as [I, H]
)

var mlpVariantNames = map[mlpVariant]string{
	mlpTanhGELU:     "tanh GELU",
	mlpErfUnscaled:  "erf(x) for erf(x/sqrt 2)",
	mlpHalvesSwap:   "halves swapped",
	mlpWoTransposed: "Wo read as [I, H]",
}

// mlpRef computes ModernBertMLP over rows of x in float64, read as v. It also
// returns, per output element, the magnitude mlpTolerance scales with:
//
//	M_o = sum_j |Wo_oj| * (1.13*|g_j|*A_j + |gelu(a_j)|*G_j + 2*|h_j|)
//
// where a = x*Wi[0:I]^T, g = x*Wi[I:2I]^T, h = gelu(a)*g, and A_j, G_j are
// the sums of |x_i*Wi_ji| behind a_j and g_j. Each term is how far a float32
// rounding in one stage can move y_o: an error in a_j through gelu' (at most
// 1.13 in magnitude), one in g_j through gelu(a_j), and the rounding of h_j
// itself and of the sum over j.
func mlpRef(x, wi, wo []float32, hidden, inter int, v mlpVariant) (y, mag []float64) {
	rows := len(x) / hidden
	y = make([]float64, rows*hidden)
	mag = make([]float64, rows*hidden)
	h := make([]float64, inter)
	hMag := make([]float64, inter)
	for r := range rows {
		xr := x[r*hidden : (r+1)*hidden]
		for j := range inter {
			var a, g, aAbs, gAbs float64
			for i, xv := range xr {
				wa, wg := float64(wi[j*hidden+i]), float64(wi[(inter+j)*hidden+i])
				a += float64(xv) * wa
				g += float64(xv) * wg
				aAbs += math.Abs(float64(xv) * wa)
				gAbs += math.Abs(float64(xv) * wg)
			}
			if v == mlpHalvesSwap {
				a, g = g, a
			}
			var act float64
			switch v {
			case mlpTanhGELU:
				act = 0.5 * a * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(a+0.044715*a*a*a)))
			case mlpErfUnscaled:
				act = 0.5 * a * (1 + math.Erf(a))
			case mlpExact, mlpHalvesSwap, mlpWoTransposed:
				act = 0.5 * a * (1 + math.Erf(a/math.Sqrt2))
			}
			h[j] = act * g
			hMag[j] = 1.13*math.Abs(g)*aAbs + math.Abs(act)*gAbs + 2*math.Abs(h[j])
		}
		for o := range hidden {
			var sum, m float64
			for j := range inter {
				w := float64(wo[o*inter+j])
				if v == mlpWoTransposed {
					w = float64(wo[j*hidden+o])
				}
				sum += w * h[j]
				m += math.Abs(w) * hMag[j]
			}
			y[r*hidden+o], mag[r*hidden+o] = sum, m
		}
	}
	return y, mag
}

// mlpTolerance bounds |go - torch| for an output element of magnitude m (see
// mlpRef).
//
// Both sides run in float32 and round in different places: torch's GEMM
// blocks its sums its own way, tensor.Linear uses dotF32's lanes, and gelu
// here goes through float64 math.Erf where torch uses a float32 erf. m is what
// one float32 rounding per stage can move y_o by; sums of n terms can in the
// worst case lose n roundings, so the bound is empirical rather than proven.
// The sums are only H = 8 and I = 6 long here, and |go - torch| stays a small
// multiple of float32Eps*m: the worst case seen uses 6 % of the bound (the
// test logs it). The factor 4 leaves room for another CPU's kernels, and the
// 1e-6 for outputs that cancel to ~0. The bound runs from 3e-6 on the
// unit-scale cases to 8e-4 on the large one.
//
// The failure modes the test is for sit far above it, and TestMLPMatchesTorch
// asserts that they do. On this fixture the tanh GELU misses torch by up to
// 61 times the bound, erf(x) for erf(x/sqrt 2) by 27000 times, the halves
// swapped and Wo read transposed by 700000 times and more.
func mlpTolerance(m float64) float64 {
	return 1e-6 + 4*float32Eps*m
}

// TestMLPMatchesTorch runs each recorded ModernBertMLP against what the pinned
// transformers module returned, and checks that the fixture is strong enough
// to catch the wrong readings of the module: each must miss the recorded
// output by ten times the tolerance somewhere. The case list is fixed, as in
// TestNormMatchesTorch, so a regeneration cannot drop the case that guards one.
func TestMLPMatchesTorch(t *testing.T) {
	f := loadOps(t)
	hidden, inter := f.Header.Config.HiddenSize, f.Header.Config.IntermediateSize
	if inter <= 0 {
		t.Fatalf("fixture config intermediate_size = %d", inter)
	}
	wantCases := []string{"mlp_large_negative", "mlp_layer0", "mlp_rank3"}

	// The worst |variant - torch| / tol over all cases, per wrong reading.
	separation := map[mlpVariant]float64{}

	raws := casesOf(t, f, "mlp")
	names := make([]string, 0, len(raws))
	for _, raw := range raws {
		var c mlpCase
		if err := json.Unmarshal(raw, &c); err != nil {
			t.Fatalf("decode case: %v", err)
		}
		names = append(names, c.Name)
		t.Run(c.Name, func(t *testing.T) {
			if c.ModuleClass != "ModernBertMLP" || c.HiddenActivation != "gelu" ||
				c.MLPBias == nil || *c.MLPBias {
				t.Fatalf("%s is a %s with hidden_activation %q, mlp_bias %v; want ModernBertMLP, \"gelu\", false",
					c.Module, c.ModuleClass, c.HiddenActivation, c.MLPBias)
			}
			if !slices.Equal(c.Wi.Shape, []int64{2 * inter, hidden}) ||
				!slices.Equal(c.Wo.Shape, []int64{hidden, inter}) {
				t.Fatalf("Wi %v, Wo %v; want [%d %d], [%d %d]",
					c.Wi.Shape, c.Wo.Shape, 2*inter, hidden, hidden, inter)
			}
			x := c.Input.tensor(t, "input")
			want := c.Output.tensor(t, "output")

			m, err := NewMLP(c.Wi.tensor(t, "wi"), c.Wo.tensor(t, "wo"))
			if err != nil {
				t.Fatalf("NewMLP: %v", err)
			}
			got, err := m.Forward(x)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
			if !slices.Equal(got.Shape(), want.Shape()) {
				t.Fatalf("shape %v, want %v", got.Shape(), want.Shape())
			}

			h, in := int(hidden), int(inter)
			_, mag := mlpRef(c.Input.Data, c.Wi.Data, c.Wo.Data, h, in, mlpExact)
			var maxDiff, worst, minTol, maxTol float64
			minTol = math.Inf(1)
			g, e := got.Data(), want.Data()
			for i := range e {
				diff := math.Abs(float64(g[i]) - float64(e[i]))
				tol := mlpTolerance(mag[i])
				minTol, maxTol = math.Min(minTol, tol), math.Max(maxTol, tol)
				maxDiff, worst = math.Max(maxDiff, diff), math.Max(worst, diff/tol)
				if diff > tol {
					t.Errorf("[%d] = %.9g, want %.9g (|diff| %.3g > tol %.3g)", i, g[i], e[i], diff, tol)
				}
			}
			t.Logf("%s: max |diff| %.2e, worst |diff|/tol %.3f, tol %.1e to %.1e",
				c.Module, maxDiff, worst, minTol, maxTol)

			for v := range mlpVariantNames {
				ref, _ := mlpRef(c.Input.Data, c.Wi.Data, c.Wo.Data, h, in, v)
				for i := range e {
					sep := math.Abs(ref[i]-float64(e[i])) / mlpTolerance(mag[i])
					separation[v] = math.Max(separation[v], sep)
				}
			}
		})
	}

	slices.Sort(names)
	if !slices.Equal(names, wantCases) {
		t.Errorf("mlp cases = %v, want %v", names, wantCases)
	}
	for v, name := range mlpVariantNames {
		t.Logf("%s misses torch by up to %.0f x tol", name, separation[v])
		if separation[v] < 10 {
			t.Errorf("the fixture cannot tell %s apart: it misses torch by at most %.1f x tol, want >= 10",
				name, separation[v])
		}
	}
}

// TestMLPRealDimensions runs the MLP at both checkpoints' sizes on synthetic
// weights, which no fixture covers: ModernBERT-large's mlp.Wi [5248, 1024] and
// mlp.Wo [1024, 2624] give I = 2624 and an output of [n, 1024], and mmBERT-base
// has hidden 768, intermediate 1152 (docs/ARCHITECTURE.md §1.2). Every output
// element is checked against the float64 reference, so an indexing slip that
// only shows when I is not a small multiple of H fails here.
func TestMLPRealDimensions(t *testing.T) {
	for _, tc := range []struct {
		name          string
		hidden, inter int
	}{
		{"modernbert-large", 1024, 2624},
		{"mmbert-base", 768, 1152},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const rows = 3
			rng := rand.New(rand.NewPCG(83, uint64(tc.hidden)))
			normal := func(n int, scale float64) []float32 {
				out := make([]float32, n)
				for i := range out {
					out[i] = float32(rng.NormFloat64() * scale)
				}
				return out
			}
			wiData := normal(2*tc.inter*tc.hidden, 1/math.Sqrt(float64(tc.hidden)))
			woData := normal(tc.hidden*tc.inter, 1/math.Sqrt(float64(tc.inter)))
			xData := normal(rows*tc.hidden, 2)

			wi := mustTensor(t, wiData, []int64{int64(2 * tc.inter), int64(tc.hidden)})
			wo := mustTensor(t, woData, []int64{int64(tc.hidden), int64(tc.inter)})
			m, err := NewMLP(wi, wo)
			if err != nil {
				t.Fatalf("NewMLP: %v", err)
			}
			got, err := m.Forward(mustTensor(t, xData, []int64{rows, int64(tc.hidden)}))
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
			if want := []int64{rows, int64(tc.hidden)}; !slices.Equal(got.Shape(), want) {
				t.Fatalf("shape %v, want %v", got.Shape(), want)
			}

			// The reference is float64, so only go's rounding counts. The
			// bound is mlpTolerance's, empirical as there: sums of 1024 and
			// 2624 terms could in the worst case lose far more than 4
			// roundings' worth, but terms of random sign keep the partial
			// sums well below the magnitudes m adds up, and the worst case
			// seen uses 0.2 % of it. An indexing slip moves outputs by O(1).
			ref, mag := mlpRef(xData, wiData, woData, tc.hidden, tc.inter, mlpExact)
			var worst float64
			for i, g := range got.Data() {
				diff := math.Abs(float64(g) - ref[i])
				tol := mlpTolerance(mag[i])
				worst = math.Max(worst, diff/tol)
				if diff > tol {
					t.Fatalf("[%d] = %.9g, want %.9g (|diff| %.3g > tol %.3g)", i, g, ref[i], diff, tol)
				}
			}
			t.Logf("worst |diff|/tol %.3f", worst)
		})
	}
}

func TestMLPErrors(t *testing.T) {
	// H = 3, I = 2: Wi [4, 3], Wo [3, 2].
	wi := mustTensor(t, make([]float32, 12), []int64{4, 3})
	wo := mustTensor(t, make([]float32, 6), []int64{3, 2})
	for _, tc := range []struct {
		name   string
		wi, wo *tensor.Tensor
	}{
		{"nil Wi", nil, wo},
		{"nil Wo", wi, nil},
		{"Wi rank 1", mustTensor(t, make([]float32, 12), []int64{12}), wo},
		{"Wi odd rows", mustTensor(t, make([]float32, 9), []int64{3, 3}), wo},
		{"Wo rank 3", wi, mustTensor(t, make([]float32, 6), []int64{1, 3, 2})},
		// The transposed layout, Wo as [I, H]: same element count, wrong shape.
		{"Wo as [I, H]", wi, mustTensor(t, make([]float32, 6), []int64{2, 3})},
		{"Wo hidden mismatch", wi, mustTensor(t, make([]float32, 8), []int64{4, 2})},
		// Consistent shapes, but nothing to multiply: a truncated header
		// must fail here, not divide by zero in Forward.
		{"intermediate 0", mustTensor(t, nil, []int64{0, 3}), mustTensor(t, nil, []int64{3, 0})},
		{"hidden 0", mustTensor(t, nil, []int64{4, 0}), mustTensor(t, nil, []int64{0, 2})},
	} {
		if _, err := NewMLP(tc.wi, tc.wo); err == nil {
			t.Errorf("NewMLP(%s): no error", tc.name)
		}
	}

	m, err := NewMLP(wi, wo)
	if err != nil {
		t.Fatalf("NewMLP: %v", err)
	}
	if _, err := m.Forward(nil); err == nil {
		t.Error("Forward(nil): no error")
	}
	if _, err := m.Forward(mustTensor(t, make([]float32, 8), []int64{2, 4})); err == nil {
		t.Error("input last dimension 4 against H = 3: no error")
	}
	if y, err := m.Forward(mustTensor(t, make([]float32, 3), []int64{3})); err != nil {
		t.Errorf("rank-1 input [H]: %v", err)
	} else if !slices.Equal(y.Shape(), []int64{3}) {
		t.Errorf("rank-1 input [H]: output shape %v, want [3]", y.Shape())
	}

	var zero MLP
	if _, err := zero.Forward(mustTensor(t, make([]float32, 3), []int64{1, 3})); err == nil ||
		!strings.Contains(err.Error(), "uninitialized") {
		t.Errorf("zero MLP Forward: err = %v, want an uninitialized-MLP error", err)
	}
}

// TestGELU pins the exact erf GELU at points where the tanh approximation and
// erf(x) without the 1/sqrt 2 differ from it. The wants are x*Phi(x) in
// float64 at float32(x), through erfc so that the negative tail does not
// cancel; at -4 the tanh form gives -7.0e-5 and erf(x) -3.1e-8.
func TestGELU(t *testing.T) {
	for _, tc := range []struct{ x, want float64 }{
		{0, 0},
		{1, 0.8413447460685429},
		{-1, -0.15865525393145707},
		{-2.7, -0.009360828091874055},
		{2.7, 2.690639219591842},
		{-4, -0.00012668496733247986},
		{10, 10},
	} {
		got := float64(gelu(float32(tc.x)))
		if diff := math.Abs(got - tc.want); diff > float32Eps*math.Abs(tc.want) {
			t.Errorf("gelu(%g) = %.9g, want %.9g", tc.x, got, tc.want)
		}
	}
}
