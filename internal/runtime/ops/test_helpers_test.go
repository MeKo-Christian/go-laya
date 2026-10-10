package ops

import (
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/runtime/tensor"
)

func seqDataT(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32((i%17)-8) / 17
	}

	return out
}

func randDataT(rng *rand.Rand, n int64) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = rng.Float32()*2 - 1
	}

	return out
}

func equalApprox(got, want []float32, tol float64) bool {
	if len(got) != len(want) {
		return false
	}

	for i := range got {
		// A NaN delta compares false against tol, so reject it explicitly.
		delta := math.Abs(float64(got[i] - want[i]))
		if math.IsNaN(delta) || delta > tol {
			return false
		}
	}

	return true
}

func mustTensorT(t *testing.T, data []float32, shape []int64) *tensor.Tensor {
	t.Helper()

	tt, err := tensor.New(data, shape)
	if err != nil {
		t.Fatalf("tensor.New(%v, %v): %v", data, shape, err)
	}

	return tt
}

func assertErrContains(t *testing.T, err error, substr string) {
	t.Helper()

	if err == nil {
		t.Fatalf("expected error containing %q, got nil", substr)
	}

	if !strings.Contains(err.Error(), substr) {
		t.Fatalf("error %q does not contain %q", err.Error(), substr)
	}
}
