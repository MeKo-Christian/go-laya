package modernbert

import (
	"encoding/json"
	"maps"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/runtime/tensor"
)

// normCase is one "layernorm" record of testdata/ops.json. Weight stays raw so
// that a missing key (a broken fixture) and null (nn.Identity) stay apart.
type normCase struct {
	Name        string          `json:"name"`
	Module      string          `json:"module"`
	ModuleClass string          `json:"module_class"`
	Input       tensorRec       `json:"input"`
	Weight      json.RawMessage `json:"weight"`
	Output      tensorRec       `json:"output"`
}

// weight decodes the recorded weight, nil where upstream has nn.Identity. A
// missing key is a broken fixture, not the identity.
func (c normCase) weight(t *testing.T) *tensorRec {
	t.Helper()

	switch {
	case len(c.Weight) == 0:
		t.Fatal("case has no \"weight\" key; null is the identity, absent is a broken fixture")
	case string(c.Weight) == "null":
		if c.ModuleClass != "Identity" {
			t.Fatalf("weight is null but %s is a %s", c.Module, c.ModuleClass)
		}
		return nil
	}

	if c.ModuleClass != "LayerNorm" {
		t.Fatalf("%s is a %s with a weight", c.Module, c.ModuleClass)
	}
	var w tensorRec
	if err := json.Unmarshal(c.Weight, &w); err != nil {
		t.Fatalf("decode weight: %v", err)
	}
	return &w
}

// norm builds the Go side of a case the way the layer loop will: a real norm
// from the recorded weight, the identity where upstream has nn.Identity.
func norm(t *testing.T, w *tensorRec) Norm {
	t.Helper()

	if w == nil {
		return IdentityNorm()
	}
	n, err := NewNorm(w.tensor(t, "weight"))
	if err != nil {
		t.Fatalf("NewNorm: %v", err)
	}
	return n
}

// float32Eps is the float32 machine epsilon, 2^-23.
const float32Eps = 1.0 / (1 << 23)

// normTolerance bounds |go - torch| for one output element.
//
// torch computes the mean, the variance and the product in float32.
// tensor.LayerNorm accumulates the mean and variance in float64 but still
// rounds the mean to float32 before subtracting it, so both sides carry a
// float32-rounded mean. That is the dominant term: an error of a few ulp of the
// row's magnitude, divided by the row's standard deviation. So the bound scales
// with kappa = max|x| / sqrt(var+eps): at kappa ~3, the unit-scale cases, it is
// ~2e-6 and Go lands within 4e-7 of torch; at the +1000 offset case it is
// ~5e-4, and Go lands 3e-5 from torch, which is itself 5e-5 from the float64
// result (dump_modernbert_ops.py prints that per case). The 1e-6 absorbs the
// rounding of the product. The worst case seen uses 13 % of the bound; the
// factor 4 keeps a regeneration on another CPU's kernels from tripping it.
//
// None of the failure modes this test is for sits near that bound: a dropped
// weight moves outputs by up to 50 %, a bias by 0.1, eps 1e-6 for 1e-5 moves
// the tiny-variance rows by 50-65 %, and a float32 single-pass variance at
// +1000 is noise.
func normTolerance(kappa float64, w float32) float64 {
	return 1e-6 + 4*float32Eps*kappa*math.Abs(float64(w))
}

// rowKappa returns max|x| / sqrt(var + eps) for each row of the last dimension.
func rowKappa(x []float32, d int) []float64 {
	out := make([]float64, len(x)/d)
	for r := range out {
		row := x[r*d : (r+1)*d]
		var mean, maxAbs float64
		for _, v := range row {
			mean += float64(v)
			maxAbs = math.Max(maxAbs, math.Abs(float64(v)))
		}
		mean /= float64(d)
		var variance float64
		for _, v := range row {
			variance += (float64(v) - mean) * (float64(v) - mean)
		}
		variance /= float64(d)
		out[r] = maxAbs / math.Sqrt(variance+float64(NormEps))
	}
	return out
}

// TestNormMatchesTorch runs every recorded norm -- the embeddings norm, an
// encoder layer's attn_norm and mlp_norm, layer 0's identity attn_norm and the
// final norm -- against what the pinned transformers modules returned.
//
// The case list is fixed: a regenerated fixture that lost the identity, the
// +1000 offset (the only variance-algorithm guard) or the tiny variance (the
// only eps guard) would otherwise still pass. So is each case's module class:
// only layer 0's attn_norm is nn.Identity upstream, and a fixture that turned
// any other norm into Identity with a null weight would otherwise pass as an
// exact identity check.
func TestNormMatchesTorch(t *testing.T) {
	f := loadOps(t)
	hidden := f.Header.Config.HiddenSize
	wantClass := map[string]string{
		"attn_norm_layer0_identity": "Identity",
		"attn_norm_offset":          "LayerNorm",
		"embeddings_norm":           "LayerNorm",
		"final_norm_wide":           "LayerNorm",
		"mlp_norm_rank3":            "LayerNorm",
		"mlp_norm_tiny_variance":    "LayerNorm",
	}

	raws := casesOf(t, f, "layernorm")
	names := make([]string, 0, len(raws))
	for _, raw := range raws {
		var c normCase
		if err := json.Unmarshal(raw, &c); err != nil {
			t.Fatalf("decode case: %v", err)
		}
		names = append(names, c.Name)
		t.Run(c.Name, func(t *testing.T) {
			if want, ok := wantClass[c.Name]; ok && c.ModuleClass != want {
				t.Fatalf("%s is a %s, want %s", c.Module, c.ModuleClass, want)
			}
			w := c.weight(t)
			x := c.Input.tensor(t, "input")
			want := c.Output.tensor(t, "output")
			if shape := x.Shape(); shape[len(shape)-1] != hidden {
				t.Fatalf("input shape %v, want a last dimension of hidden_size %d", shape, hidden)
			}
			if w != nil && !slices.Equal(w.Shape, []int64{hidden}) {
				t.Fatalf("weight shape %v, want [%d]", w.Shape, hidden)
			}

			got, err := norm(t, w).Forward(x)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
			if !slices.Equal(got.Shape(), want.Shape()) {
				t.Fatalf("shape %v, want %v", got.Shape(), want.Shape())
			}

			shape := x.Shape()
			d := int(shape[len(shape)-1])
			kappa := rowKappa(x.Data(), d)

			var maxDiff, worst float64
			g, e := got.Data(), want.Data()
			for i := range e {
				diff := math.Abs(float64(g[i]) - float64(e[i]))
				if w == nil {
					// nn.Identity hands back its input: no tolerance applies.
					if diff != 0 {
						t.Errorf("[%d] = %g, want %g exactly (identity)", i, g[i], e[i])
					}
					continue
				}
				tol := normTolerance(kappa[i/d], w.Data[i%d])
				maxDiff, worst = math.Max(maxDiff, diff), math.Max(worst, diff/tol)
				if diff > tol {
					t.Errorf("[%d] = %.9g, want %.9g (|diff| %.3g > tol %.3g)", i, g[i], e[i], diff, tol)
				}
			}
			t.Logf("%s (%s): max |diff| %.2e, worst |diff|/tol %.3f", c.Module, c.ModuleClass, maxDiff, worst)
		})
	}

	slices.Sort(names)
	if wantNames := slices.Sorted(maps.Keys(wantClass)); !slices.Equal(names, wantNames) {
		t.Errorf("layernorm cases = %v, want %v", names, wantNames)
	}
}

// TestIdentityIsNotOnesNorm pins down the distinction layer 0 depends on:
// upstream's missing attn_norm is nn.Identity, and a LayerNorm whose weight is
// all ones still centres and scales. Mistaking one for the other is a
// plausible wrong answer, not an error, so it gets its own test.
func TestIdentityIsNotOnesNorm(t *testing.T) {
	x := mustTensor(t, []float32{1, 2, 3, 6, -4, 0, 0, 0}, []int64{2, 4})
	ones := mustTensor(t, []float32{1, 1, 1, 1}, []int64{4})

	id, err := IdentityNorm().Forward(x)
	if err != nil {
		t.Fatalf("identity Forward: %v", err)
	}
	if !slices.Equal(id.Shape(), x.Shape()) || !slices.Equal(id.Data(), x.Data()) {
		t.Fatalf("identity = %v %v, want its input %v %v", id.Shape(), id.Data(), x.Shape(), x.Data())
	}

	n, err := NewNorm(ones)
	if err != nil {
		t.Fatalf("NewNorm(ones): %v", err)
	}
	normed, err := n.Forward(x)
	if err != nil {
		t.Fatalf("ones Forward: %v", err)
	}
	if slices.Equal(normed.Data(), x.Data()) {
		t.Fatalf("a ones-weight norm returned its input %v unchanged; it must centre and scale", x.Data())
	}
	// Row 0 is [1 2 3 6]: mean 3, variance 3.5.
	if got, want := normed.Data()[0], float32(-2/math.Sqrt(3.5+1e-5)); math.Abs(float64(got-want)) > 1e-6 {
		t.Errorf("ones norm [0] = %g, want %g", got, want)
	}
}

// TestIdentityOwnsItsOutput guards the identity's contract that Forward always
// returns a tensor the caller owns, as the LayerNorm path does: writing into
// the result must not reach the residual stream it came from.
func TestIdentityOwnsItsOutput(t *testing.T) {
	x := mustTensor(t, []float32{1, 2, 3, 4}, []int64{1, 4})
	out, err := IdentityNorm().Forward(x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	out.RawData()[0] = 99
	if x.Data()[0] != 1 {
		t.Fatal("writing the identity's output changed its input")
	}
}

func TestNormErrors(t *testing.T) {
	x := mustTensor(t, []float32{1, 2, 3, 4, 5, 6}, []int64{2, 3})
	four := mustTensor(t, []float32{1, 1, 1, 1}, []int64{4})
	matrix := mustTensor(t, []float32{1, 1, 1, 1, 1, 1}, []int64{2, 3})

	if _, err := NewNorm(nil); err == nil {
		t.Error("NewNorm(nil): no error; a norm without a weight is a missing tensor, not the identity")
	}
	if _, err := NewNorm(matrix); err == nil {
		t.Error("NewNorm(rank 2): no error")
	}

	short, err := NewNorm(four)
	if err != nil {
		t.Fatalf("NewNorm: %v", err)
	}
	if _, err := short.Forward(x); err == nil {
		t.Error("weight of 4 over a last dimension of 3: no error")
	}

	var zero Norm
	if _, err := zero.Forward(x); err == nil || !strings.Contains(err.Error(), "uninitialized") {
		t.Errorf("zero Norm Forward: err = %v, want an uninitialized-norm error", err)
	}

	for _, n := range []Norm{IdentityNorm(), short} {
		if _, err := n.Forward(nil); err == nil {
			t.Errorf("%+v Forward(nil): no error", n)
		}
	}
}

func mustTensor(t *testing.T, data []float32, shape []int64) *tensor.Tensor {
	t.Helper()

	x, err := tensor.New(data, shape)
	if err != nil {
		t.Fatal(err)
	}
	return x
}
