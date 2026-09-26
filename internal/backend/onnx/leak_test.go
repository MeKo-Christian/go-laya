//go:build linux && (amd64 || arm64 || loong64) && !android

package onnx

import (
	"context"
	"os"
	"os/exec"
	"testing"

	"github.com/MeKo-Christian/go-laya/backend"
	"github.com/MeKo-Christian/go-laya/internal/golden"
)

// mmapThresholdEnv pins glibc's mmap threshold. Left dynamic, glibc raises it
// after the first large free, and from then on freed weights stay in the heap
// for reuse instead of going back to the OS. Measured on linux/amd64 with ORT
// 1.23.0 (2026-09-26), identical runs of this test then ended their warm-up at
// 42 MiB or at 1482 MiB, and one swung 1317 → 1083 → 41 MiB across three
// cycles: resident memory said nothing about the Backend. With the threshold
// pinned, four runs stayed within 40–53 MiB on every cycle.
const mmapThresholdEnv = "MALLOC_MMAP_THRESHOLD_"

// Cycle counts for TestCloseReleasesMemory.
const (
	leakWarmup   = 2
	leakMeasured = 3
)

// TestCloseReleasesMemory is Task 6.3.3's assertion behind 3.2.4: the Router
// closes what it evicts, and this checks that Close actually gives the memory
// back. It loads the english export, runs one forward pass and closes it, over
// and over, and requires resident memory not to grow once warm. A Backend that
// kept its session would grow by roughly the weights (1.6 GiB) on every cycle.
//
// Linux only: it reads VmRSS from /proc, and it re-runs itself with glibc's
// mmap threshold pinned (see mmapThresholdEnv).
func TestCloseReleasesMemory(t *testing.T) {
	if testing.Short() {
		t.Skip("-short: needs an ONNX Runtime library and the S1 exports")
	}
	lib := requireORTLibrary(t)
	model, err := findModel("laya-" + golden.English + "-dynamo.onnx")
	if err != nil {
		t.Skipf("no export: %v", err)
	}

	if os.Getenv(mmapThresholdEnv) == "" {
		// #nosec G204 -- re-runs this test binary with its own arguments.
		cmd := exec.CommandContext(t.Context(), os.Args[0], "-test.run=^TestCloseReleasesMemory$", "-test.v")
		cmd.Env = append(os.Environ(), mmapThresholdEnv+"=65536")
		out, err := cmd.CombinedOutput()
		t.Logf("re-run with %s=65536:\n%s", mmapThresholdEnv, out)
		if err != nil {
			t.Fatalf("re-run: %v", err)
		}
		return
	}

	st, err := os.Stat(model + ".data")
	if err != nil {
		t.Fatalf("external data: %v", err)
	}
	weightsKB := st.Size() / 1024

	cycle := func() {
		b, err := Open(model, Options{Library: lib, Device: deviceCPU})
		if err != nil {
			t.Fatalf("Open: %v", err)
		}
		_, _, err = b.Forward(context.Background(), backend.Batch{
			InputIDs: [][]int64{{1, 2, 3, 4}}, AttentionMask: [][]int64{{1, 1, 1, 1}},
			MarkerPos: [][]int64{{1, 2}}, MarkerMask: [][]bool{{true, true}}, QType: []int64{0},
		})
		if err != nil {
			t.Fatalf("Forward: %v", err)
		}
		if err := b.Close(); err != nil {
			t.Fatalf("Close: %v", err)
		}
	}

	for range leakWarmup {
		cycle()
	}
	base := residentKB()
	for i := range leakMeasured {
		cycle()
		t.Logf("after cycle %d: %d MiB (warm %d MiB, weights %d MiB)", i+1, residentKB()/1024, base/1024, weightsKB/1024)
	}

	// One leaked session is roughly the weights, and leakMeasured of them
	// would be several GiB; a quarter of one is far above the 13 MiB of drift
	// measured with the threshold pinned.
	if growth := residentKB() - base; growth > weightsKB/4 {
		t.Fatalf("resident memory grew %d MiB over %d warm cycles: Close does not release the session",
			growth/1024, leakMeasured)
	}
}
