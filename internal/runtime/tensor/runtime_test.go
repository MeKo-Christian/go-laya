package tensor

import (
	"math"
	"math/rand/v2"
	"runtime"
	"testing"
)

// TestWorkersDefault pins the default a caller of laya.Open gets: one kernel
// worker per GOMAXPROCS, not one in all. This package is internal, so no
// caller outside the module can raise the count itself.
func TestWorkersDefault(t *testing.T) {
	prev := workers.Load()
	t.Cleanup(func() { workers.Store(prev) })
	workers.Store(0) // as at start-up, before any SetWorkers

	if got, want := Workers(), max(1, runtime.GOMAXPROCS(0)); got != want {
		t.Fatalf("Workers() = %d by default, want GOMAXPROCS %d", got, want)
	}
}

// TestSetWorkersOneDisables pins that SetWorkers(1) and below still mean
// single-threaded once the default is no longer 1.
func TestSetWorkersOneDisables(t *testing.T) {
	prev := workers.Load()
	t.Cleanup(func() { workers.Store(prev) })

	for _, n := range []int{1, 0, -3} {
		SetWorkers(n)
		if got := Workers(); got != 1 {
			t.Errorf("SetWorkers(%d): Workers() = %d, want 1", n, got)
		}
	}
}

// TestKernelsSameBitsAnyWorkers pins that the worker count changes how
// Linear and MatMul split their work, never their result: every output is
// one dot product over the same values in the same order wherever a piece
// starts. The parity goldens hold at one worker count only because of this,
// and the default now follows the machine. Each shape is past its kernel's
// parallel threshold (2^18 multiply-adds for Linear, 2^19 for MatMul).
func TestKernelsSameBitsAnyWorkers(t *testing.T) {
	prev := workers.Load()
	t.Cleanup(func() { workers.Store(prev) })

	rng := rand.New(rand.NewPCG(7, 0))
	w, bias := randTensor(rng, 384, 768), randTensor(rng, 384)
	rows, row := randTensor(rng, 5, 768), randTensor(rng, 1, 768) // batch > 1, and one row split over outputs
	a, b := randTensor(rng, 2, 96, 64), randTensor(rng, 2, 64, 80)

	run := func(n int) [][]float32 {
		SetWorkers(n)
		var outs [][]float32
		for _, x := range []*Tensor{rows, row} {
			y, err := Linear(x, w, bias)
			if err != nil {
				t.Fatal(err)
			}
			outs = append(outs, y.data)
		}
		y, err := MatMul(a, b)
		if err != nil {
			t.Fatal(err)
		}
		return append(outs, y.data)
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
