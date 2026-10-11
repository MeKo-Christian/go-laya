package native

import (
	"math"
	"math/rand/v2"
	"runtime"
	"testing"

	"github.com/cwbudde/algo-tensor/tensor"
)

// The kernels come from github.com/cwbudde/algo-tensor (D35). These are the
// tests of go-laya's former internal/runtime/tensor that the native backend
// relies on and algo-tensor does not make in this form: they live here,
// beside the only caller, so that a new algo-tensor tag must pass them
// before go-laya takes it.

// TestSetWorkersOneDisables pins that SetWorkers(1) and below still mean
// single-threaded although the default is GOMAXPROCS.
func TestSetWorkersOneDisables(t *testing.T) {
	prev := tensor.Workers()
	t.Cleanup(func() { tensor.SetWorkers(prev) })

	for _, n := range []int{1, 0, -3} {
		tensor.SetWorkers(n)
		if got := tensor.Workers(); got != 1 {
			t.Errorf("SetWorkers(%d): Workers() = %d, want 1", n, got)
		}
	}
}

// TestKernelsSameBitsAnyWorkers pins that the worker count changes how
// Linear, MatMul and MatMulTransB split their work, never their result: every
// output of Linear and MatMul is one dot product over the same values in the
// same order wherever a piece starts, and MatMulTransB splits only between
// its 4×4 blocks. The parity goldens hold at one worker count only because
// of this, and the default follows the machine (D34). Each case is past its
// kernel's parallel threshold: 2^18 multiply-adds for Linear, 2^19 for
// MatMul, 2^20 for MatMulTransB.
func TestKernelsSameBitsAnyWorkers(t *testing.T) {
	prev := tensor.Workers()
	t.Cleanup(func() { tensor.SetWorkers(prev) })

	rng := rand.New(rand.NewPCG(7, 0))
	w, bias := randTensor(t, rng, 384, 768), randTensor(t, rng, 384)
	rows, row := randTensor(t, rng, 5, 768), randTensor(t, rng, 1, 768) // batch > 1, and one row split over outputs
	a, b := randTensor(t, rng, 2, 96, 64), randTensor(t, rng, 2, 64, 80)

	// MatMulTransB as the attention of internal/modernbert and internal/head
	// calls it, (m, n, k): scores over 512 tokens with 64-wide heads, the
	// output over them, a ragged size that leaves row, column and k tails,
	// and fewer than 8 rows, which splits over the rows of b instead of a.
	transB := [][3]int{{512, 512, 64}, {512, 64, 512}, {161, 157, 65}, {5, 1027, 300}}
	transBIn := make([][2][]float32, len(transB))
	for i, s := range transB {
		if s[0]*s[1]*s[2] < 1<<20 {
			t.Fatalf("MatMulTransB %v is below its parallel threshold", s)
		}
		transBIn[i] = [2][]float32{randData(rng, s[0]*s[2]), randData(rng, s[1]*s[2])}
	}

	run := func(n int) [][]float32 {
		tensor.SetWorkers(n)
		outs := make([][]float32, 0, 3+len(transB))
		for _, x := range []*tensor.Tensor{rows, row} {
			y, err := tensor.Linear(x, w, bias)
			if err != nil {
				t.Fatal(err)
			}
			outs = append(outs, y.RawData())
		}
		y, err := tensor.MatMul(a, b)
		if err != nil {
			t.Fatal(err)
		}
		outs = append(outs, y.RawData())
		for i, s := range transB {
			c := make([]float32, s[0]*s[1])
			tensor.MatMulTransB(c, s[1], transBIn[i][0], transBIn[i][1], s[0], s[1], s[2])
			outs = append(outs, c)
		}
		return outs
	}

	want := run(1)
	for _, n := range []int{2, 3, 7, max(1, runtime.GOMAXPROCS(0))} {
		for i, got := range run(n) {
			for j := range got {
				if math.Float32bits(got[j]) != math.Float32bits(want[i][j]) {
					t.Fatalf("workers=%d: kernel %d output %d = %v, want %v as at one worker", n, i, j, got[j], want[i][j])
				}
			}
		}
	}
}

// randTensor is a tensor of the given shape with values in [-1, 1).
func randTensor(t *testing.T, rng *rand.Rand, shape ...int64) *tensor.Tensor {
	t.Helper()
	n := int64(1)
	for _, d := range shape {
		n *= d
	}
	x, err := tensor.New(randData(rng, int(n)), shape)
	if err != nil {
		t.Fatal(err)
	}
	return x
}

// randData is n values in [-1, 1).
func randData(rng *rand.Rand, n int) []float32 {
	data := make([]float32, n)
	for i := range data {
		data[i] = rng.Float32()*2 - 1
	}
	return data
}
