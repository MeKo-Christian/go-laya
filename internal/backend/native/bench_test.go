// How long does the native backend take for one question? (Task 8.9)
//
// PLAN.md 8.9 holds the native backend to "within 10x of ORT-CPU at 512 tokens
// on the same hardware". The ORT side is Spike S3's BenchmarkForward in
// internal/backend/onnx, so this is that harness with the session swapped for
// a Backend: the same reference cell, the same worker count, the same input,
// the same metrics. Anything that differs would make the ratio in
// BENCHMARKS.md compare two different questions. Run it through
// `just bench-native`, which supplies the -benchtime=Nx it assumes and the
// memory cap.

package native

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"runtime"
	"runtime/debug"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/MeKo-Christian/go-laya/backend"
	"github.com/MeKo-Christian/go-laya/internal/golden"
	"github.com/MeKo-Christian/go-laya/internal/runtime/tensor"
)

// benchCheckpoints are the three checkpoints S3 measured, in its order.
var benchCheckpoints = []string{golden.English, golden.Multilingual, golden.TypedDecisions}

// benchShape is one (batch, seq, k): questions in the call, tokens per
// question, options per question. It prints as S3's does, so the two
// transcripts name their cells alike.
type benchShape struct {
	batch, seq, k int
}

func (s benchShape) String() string {
	return fmt.Sprintf("b%d_s%d_k%d", s.batch, s.seq, s.k)
}

// benchRefShape and benchRefWorkers are S3's reference cell: one question at
// agent.py:256's default max_len with four options, at 8 threads, the setting
// S3 found fastest on this hardware. tensor.SetWorkers is the native
// backend's counterpart of IntraOpNumThreads. TestBenchHarness pins both to
// internal/backend/onnx's values.
//
// There is no sweep: 8.9's budget is stated at this one cell, and an
// iteration here costs tens of seconds.
var (
	benchRefShape   = benchShape{batch: 1, seq: 512, k: 4}
	benchRefWorkers = 8
)

// benchIDSpan bounds the token ids, as S3's benchIDs does: 1..997 is inside
// every checkpoint's vocabulary. The latency does not depend on the values,
// since the encoder is dense with no data-dependent control flow, but the
// ratio is cleaner when both sides ran the very same batch.
const benchIDSpan = 997

// newBenchBatch builds S3's inputs for one shape (newBenchInputs in
// internal/backend/onnx), with ids 1 + (i*7919) % span. attention_mask is all
// ones: a fully attended sequence is the worst case, and the one a caller
// hitting max_len sees. The markers sit at 1+3i, the spacing
// scripts/export_onnx.py:158 uses, and qtype is 0 (choice).
func newBenchBatch(sh benchShape, span int) backend.Batch {
	var in backend.Batch
	n := 0
	for range sh.batch {
		ids, mask := make([]int64, sh.seq), make([]int64, sh.seq)
		for i := range ids {
			ids[i] = int64(1 + (n*7919)%span)
			mask[i] = 1
			n++
		}
		pos, live := make([]int64, sh.k), make([]bool, sh.k)
		for i := range pos {
			pos[i] = int64(1 + i*3)
			live[i] = true
		}
		in.InputIDs = append(in.InputIDs, ids)
		in.AttentionMask = append(in.AttentionMask, mask)
		in.MarkerPos = append(in.MarkerPos, pos)
		in.MarkerMask = append(in.MarkerMask, live)
		in.QType = append(in.QType, 0)
	}
	return in
}

// benchOpen opens the checkpoint in dir for the rest of the benchmark. Closing
// returns the weights to the OS, so the next checkpoint in the same process
// does not start on top of this one's 1-2 GB.
func benchOpen(b *testing.B, dir string) *Backend {
	b.Helper()
	be, err := Open(context.Background(), dir, Options{})
	if err != nil {
		b.Fatalf("Open(%s): %v", dir, err)
	}
	b.Cleanup(func() {
		if err := be.Close(); err != nil {
			b.Errorf("Close: %v", err)
		}
		runtime.GC()
		debug.FreeOSMemory()
	})
	return be
}

// BenchmarkForward is Task 8.9: one Forward at S3's reference cell, per
// checkpoint. It needs the real checkpoints (LAYA_MODELS) and skips without
// them.
func BenchmarkForward(b *testing.B) {
	root := golden.SkipWithoutModels(b)
	for _, ck := range benchCheckpoints {
		b.Run(ck, func(b *testing.B) {
			be := benchOpen(b, golden.CheckpointDir(root, ck))
			b.Run(fmt.Sprintf("workers=%d", benchRefWorkers), func(b *testing.B) {
				b.Run(benchRefShape.String(), func(b *testing.B) {
					benchForward(b, be, benchRefShape, benchIDSpan)
				})
			})
		})
	}
}

// benchForward times Forward on sh's batch at benchRefWorkers kernel workers
// and reports p50 and p90, as S3's benchShapeRun does: the machine is a 15 W
// part that throttles, so the spread is itself a result, and upstream quotes
// p50.
//
// One untimed Forward goes first, as S3's does. The native backend has no
// arena or graph optimisation to warm, but the first pass still faults in
// the pages its intermediates land on. b.Loop rather than b.N: a b.N
// benchmark is called once with N=1 before the real run, which here is an
// extra two passes of tens of seconds each, while b.Loop runs the function
// once.
func benchForward(b *testing.B, be *Backend, sh benchShape, span int) {
	b.Helper()
	prev := tensor.Workers()
	tensor.SetWorkers(benchRefWorkers)
	b.Cleanup(func() { tensor.SetWorkers(prev) })
	b.Logf("GOMAXPROCS=%d, %d kernel workers", runtime.GOMAXPROCS(0), tensor.Workers())

	in := newBenchBatch(sh, span)
	forwardOnce(b, be, in)

	var latencies []time.Duration
	for b.Loop() {
		start := time.Now()
		forwardOnce(b, be, in)
		latencies = append(latencies, time.Since(start))
	}

	slices.Sort(latencies)
	b.ReportMetric(millis(quantile(latencies, 0.5)), "p50_ms")
	b.ReportMetric(millis(quantile(latencies, 0.9)), "p90_ms")
}

// forwardOnce is one call as a caller makes it.
func forwardOnce(b *testing.B, be *Backend, in backend.Batch) {
	b.Helper()
	if _, _, err := be.Forward(context.Background(), in); err != nil {
		b.Fatalf("Forward: %v", err)
	}
}

// quantile is nearest-rank on an already-sorted slice, as S3's is. With the
// handful of samples a multi-second benchmark affords, interpolating would
// invent precision.
func quantile(sorted []time.Duration, q float64) time.Duration {
	if len(sorted) == 0 {
		return 0
	}
	i := int(q * float64(len(sorted)))
	return sorted[min(i, len(sorted)-1)]
}

func millis(d time.Duration) float64 {
	return float64(d) / float64(time.Millisecond)
}

// BenchmarkLoad measures Open, S3's BenchmarkSessionLoad for the native
// backend: ns/op is the load time, and rss_MiB the resident growth across the
// first load only. Later loads land in heap the Go runtime has not returned
// to the OS yet and would report about zero, which is why `just
// bench-native` runs one process per checkpoint. It needs the real
// checkpoints and skips without them.
func BenchmarkLoad(b *testing.B) {
	root := golden.SkipWithoutModels(b)
	for _, ck := range benchCheckpoints {
		b.Run(ck, func(b *testing.B) {
			benchLoad(b, golden.CheckpointDir(root, ck))
		})
	}
}

// benchLoad opens and closes the checkpoint in dir once per iteration.
func benchLoad(b *testing.B, dir string) {
	b.Helper()
	var growth int64
	first := true
	for b.Loop() {
		before := residentKB()
		be, err := Open(context.Background(), dir, Options{})
		if err != nil {
			b.Fatalf("Open(%s): %v", dir, err)
		}
		if first {
			growth, first = residentKB()-before, false
		}
		if err := be.Close(); err != nil {
			b.Fatalf("Close: %v", err)
		}
	}
	if growth > 0 {
		b.ReportMetric(float64(growth)/1024, "rss_MiB")
	}
}

// residentKB reads VmRSS from /proc/self/status, or 0 where that does not
// exist; S3's helper of the same name. It is the whole process, so it means
// something only as a delta around one allocation.
func residentKB() int64 {
	f, err := os.Open("/proc/self/status")
	if err != nil {
		return 0
	}
	defer func() { _ = f.Close() }()
	scan := bufio.NewScanner(f)
	for scan.Scan() {
		rest, ok := strings.CutPrefix(scan.Text(), "VmRSS:")
		if !ok {
			continue
		}
		kb, err := strconv.ParseInt(strings.TrimSuffix(strings.TrimSpace(rest), " kB"), 10, 64)
		if err != nil {
			return 0
		}
		return kb
	}
	return 0
}

// TestBenchHarness is the part of Task 8.9 CI can run: the benchmarks need
// the real checkpoints, so this pins their reference cell to S3's and runs
// both harnesses on the synthetic checkpoint at a tiny shape. A broken
// harness then fails here, not an hour into `just bench-native`.
func TestBenchHarness(t *testing.T) {
	// BENCHMARKS.md divides the native p50 by the ORT one. That is a ratio
	// only while both ran the same cell: internal/backend/onnx's
	// benchRefShape and benchRefThreads. Test code of another package cannot
	// be imported, so these are literals, and a change on the ORT side must
	// be made here too.
	if want := (benchShape{batch: 1, seq: 512, k: 4}); benchRefShape != want {
		t.Errorf("benchRefShape = %v, want %v, the S3 reference cell (internal/backend/onnx/bench_test.go)",
			benchRefShape, want)
	}
	if benchRefWorkers != 8 {
		t.Errorf("benchRefWorkers = %d, want 8, S3's benchRefThreads", benchRefWorkers)
	}

	// The batch is newBenchInputs' in internal/backend/onnx, value for value.
	in := newBenchBatch(benchRefShape, benchIDSpan)
	if len(in.InputIDs) != 1 || len(in.InputIDs[0]) != 512 || len(in.MarkerPos[0]) != 4 {
		t.Fatalf("batch %dx%d with %d markers, want 1x512 with 4",
			len(in.InputIDs), len(in.InputIDs[0]), len(in.MarkerPos[0]))
	}
	// 1 + (i*7919) % 997 for i = 0, 1, 2.
	if got := in.InputIDs[0][:3]; !slices.Equal(got, []int64{1, 941, 884}) {
		t.Errorf("input_ids start %v, want [1 941 884]", got)
	}
	if hi := slices.Max(in.InputIDs[0]); hi > 997 {
		t.Errorf("largest id %d, want at most 997", hi)
	}
	if slices.ContainsFunc(in.AttentionMask[0], func(v int64) bool { return v != 1 }) {
		t.Errorf("attention_mask %v, want all ones", in.AttentionMask[0])
	}
	if !slices.Equal(in.MarkerPos[0], []int64{1, 4, 7, 10}) ||
		!slices.Equal(in.MarkerMask[0], []bool{true, true, true, true}) ||
		!slices.Equal(in.QType, []int64{0}) {
		t.Errorf("marker_pos %v, marker_mask %v, qtype %v, want [1 4 7 10], all true, [0]",
			in.MarkerPos[0], in.MarkerMask[0], in.QType)
	}

	// testing.Benchmark would otherwise calibrate b.N against a second.
	setBenchtime(t, "3x")
	dir := newSynth(11).write(t)
	smoke := benchShape{batch: 1, seq: 16, k: 4}
	prev := tensor.Workers()

	t.Run("forward", func(t *testing.T) {
		r := testing.Benchmark(func(b *testing.B) {
			benchForward(b, benchOpen(b, dir), smoke, tinyVocab-1)
		})
		if r.N != 3 {
			t.Fatalf("%d timed iterations, want 3: the harness failed", r.N)
		}
		p50, p90 := r.Extra["p50_ms"], r.Extra["p90_ms"]
		if p50 <= 0 || p90 < p50 {
			t.Errorf("p50_ms %v, p90_ms %v, want 0 < p50 <= p90", p50, p90)
		}
		if got := tensor.Workers(); got != prev {
			t.Errorf("tensor.Workers() = %d after the benchmark, want %d restored", got, prev)
		}
	})
	t.Run("load", func(t *testing.T) {
		r := testing.Benchmark(func(b *testing.B) { benchLoad(b, dir) })
		if r.N != 3 {
			t.Fatalf("%d timed loads, want 3: the harness failed", r.N)
		}
	})
}

// setBenchtime sets -test.benchtime, which testing.Benchmark reads, for the
// rest of the test.
func setBenchtime(t *testing.T, v string) {
	t.Helper()
	f := flag.Lookup("test.benchtime")
	if f == nil {
		t.Fatal("no -test.benchtime flag")
	}
	old := f.Value.String()
	if err := f.Value.Set(v); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = f.Value.Set(old) })
}
