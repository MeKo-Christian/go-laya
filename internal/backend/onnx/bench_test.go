//go:build !windows && !js && !wasm

// How long does one forward pass actually take on a CPU? (Spike S3)
//
// PLAN.md §3 S3.1 asks for ORT-CPU latency "per checkpoint at realistic sequence lengths
// and thread counts, on the hardware we actually ship numbers for". The numbers have to
// come through the binding go-laya ships, not through Python, because that is the thing
// whose latency a caller will experience -- hence a Go benchmark next to the S2 spike
// rather than a script. It moved here unchanged from internal/onnxspike (PLAN.md Task
// 6.10): Task 6.9.2 re-runs it against quantized graphs and needs the same harness.

package onnx

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	ort "github.com/shota3506/onnxruntime-purego/onnxruntime"
)

// benchCheckpoint is one S1 export. maxSeq is the context the checkpoint was *trained*
// for (router.py:6-8), not a limit of the graph: every axis is dynamic, so a longer run
// would happily produce a number, and that number would mean nothing.
type benchCheckpoint struct {
	name   string
	onnx   string
	maxSeq int
}

var benchCheckpoints = []benchCheckpoint{
	{name: "english", onnx: "laya-english-dynamo.onnx", maxSeq: 512},
	{name: "multilingual", onnx: "laya-multilingual-dynamo.onnx", maxSeq: 1024},
	{name: "typed-decisions", onnx: "laya-typed-decisions-dynamo.onnx", maxSeq: 1024},
}

// benchShape is one (batch, seq, k) the sweep measures. batch is the number of questions
// in the call, which is what upstream's latency table varies; k is the option count.
type benchShape struct {
	batch, seq, k int
}

func (s benchShape) String() string {
	return fmt.Sprintf("b%d_s%d_k%d", s.batch, s.seq, s.k)
}

// The sweep is two one-dimensional sweeps sharing a corner, not a cross product: a full
// product of every thread count against every shape runs for hours and the extra cells
// say nothing the two axes do not.
//
// benchRefShape is the corner -- one question, the default max_len of agent.py:256, four
// options -- held fixed while threads vary. benchRefThreads is held fixed while the shape
// varies.
//
// 8, not "every hardware thread": the thread sweep measured throughput peaking at 6-8 and
// then *regressing*, on a part with 10 physical cores of which only 2 are SMT. Handing ORT
// every sibling thread costs 25-35% against the optimum, so pinning the shape sweep to all
// 12 would measure a setting no one should ship.
var (
	benchRefShape   = benchShape{batch: 1, seq: 512, k: 4}
	benchRefThreads = 8
	benchThreads    = []int{1, 2, 4, 6, 8, 12}
	benchShapes     = []benchShape{
		{batch: 1, seq: 128, k: 4},
		{batch: 1, seq: 256, k: 4},
		{batch: 1, seq: 512, k: 4},
		{batch: 1, seq: 1024, k: 4},
		{batch: 8, seq: 128, k: 4},
		{batch: 8, seq: 512, k: 4},
		// A spot check, not an axis: the head runs over k markers while the encoder runs
		// over seq tokens, so k should be lost in the noise. Should is not measured.
		{batch: 1, seq: 512, k: 20},
	}
)

// shapesFor returns the cells of the two sweeps that apply to one (checkpoint, threads)
// pair. Shapes past the checkpoint's trained context are dropped, and batch 8 is not run
// at 1024: a single iteration there costs the better part of a minute and duplicates what
// batch 8 at 512 and batch 1 at 1024 already show.
func shapesFor(ck benchCheckpoint, threads int) []benchShape {
	var out []benchShape

	for _, sh := range benchShapes {
		if sh.seq > ck.maxSeq || (sh.batch > 1 && sh.seq > 512) {
			continue
		}

		if threads != benchRefThreads && sh != benchRefShape {
			continue
		}

		out = append(out, sh)
	}

	return out
}

// newBenchEnv opens the runtime and an environment once for the whole sweep. Both are
// cheap; the session is the expensive part and is created per (checkpoint, threads).
func newBenchEnv(tb testing.TB) (*ort.Runtime, *ort.Env) {
	tb.Helper()

	lib := requireORTLibrary(tb)

	rt, err := ort.NewRuntime(lib, APIVersion)
	if err != nil {
		tb.Fatalf("NewRuntime(%q, %d): %v", lib, APIVersion, err)
	}

	tb.Cleanup(func() {
		if err := rt.Close(); err != nil {
			tb.Errorf("runtime close: %v", err)
		}
	})

	env, err := rt.NewEnv("laya-onnxbench", ort.LoggingLevelWarning)
	if err != nil {
		tb.Fatalf("NewEnv: %v", err)
	}

	tb.Cleanup(env.Close)
	tb.Logf("ONNX Runtime %s (C API %d) from %s, GOMAXPROCS=%d",
		rt.GetVersionString(), rt.GetAPIVersion(), lib, runtime.GOMAXPROCS(0))

	return rt, env
}

// benchIDs fills a token-id buffer deterministically.
//
// Latency is data-independent here: the graph is a dense encoder with no early exit, no
// unpadding and no data-dependent control flow, so at a fixed shape the ids only have to
// be in range. 1..997 is inside every checkpoint's vocabulary (50368 and 256000), and a
// fixed multiply keeps the buffer reproducible without pulling in a PRNG.
func benchIDs(n int) []int64 {
	ids := make([]int64, n)
	for i := range ids {
		ids[i] = int64(1 + (i*7919)%997)
	}

	return ids
}

// newBenchInputs builds the five graph inputs for one shape. attention_mask is all ones
// on purpose: nothing in the exported graph skips padded positions, so a fully attended
// sequence is both the worst case and the one a caller hitting max_len will see.
func newBenchInputs(tb testing.TB, rt *ort.Runtime, sh benchShape) map[string]*ort.Value {
	tb.Helper()

	mask := make([]int64, sh.batch*sh.seq)
	for i := range mask {
		mask[i] = 1
	}

	markerPos := make([]int64, sh.batch*sh.k)
	markerMask := make([]bool, sh.batch*sh.k)

	for b := range sh.batch {
		for i := range sh.k {
			// The same 1+3i spacing scripts/export_onnx.py:158 uses, and it must stay
			// inside seq: the head gathers encoder states at these indices.
			markerPos[b*sh.k+i] = int64(1 + i*3)
			markerMask[b*sh.k+i] = true
		}
	}

	pair := [2]int64{int64(sh.batch), int64(sh.seq)}
	kPair := [2]int64{int64(sh.batch), int64(sh.k)}

	inputs := map[string]*ort.Value{
		"input_ids":      newValue(tb, rt, "input_ids", benchIDs(sh.batch*sh.seq), pair[:]),
		"attention_mask": newValue(tb, rt, "attention_mask", mask, pair[:]),
		"marker_pos":     newValue(tb, rt, "marker_pos", markerPos, kPair[:]),
		"marker_mask":    newValue(tb, rt, "marker_mask", markerMask, kPair[:]),
		"qtype":          newValue(tb, rt, "qtype", make([]int64, sh.batch), []int64{int64(sh.batch)}),
	}

	return inputs
}

// newValue wraps NewTensorValue so the caller reads as a list of inputs rather than a
// list of error checks. Closing is the caller's job -- see D5: a *Value left to the GC
// races Runtime.Close.
func newValue[T int64 | bool | float32](tb testing.TB, rt *ort.Runtime, name string, data []T, shape []int64) *ort.Value {
	tb.Helper()

	v, err := ort.NewTensorValue(rt, data, shape)
	if err != nil {
		tb.Fatalf("%s: NewTensorValue: %v", name, err)
	}

	return v
}

func closeValues(vs map[string]*ort.Value) {
	for _, v := range vs {
		v.Close()
	}
}

// runOnce is one call as a caller would make it, outputs released included. Releasing two
// small tensors costs microseconds against a forward pass measured in hundreds of
// milliseconds, and leaving them to the GC would walk straight into the D5 race.
func runOnce(tb testing.TB, sess *ort.Session, inputs map[string]*ort.Value) {
	out, err := sess.Run(context.Background(), inputs)
	if err != nil {
		tb.Fatalf("Run: %v", err)
	}

	closeValues(out)
}

// BenchmarkForward is S3.1. Run it through `just bench-onnx`, which supplies the
// -benchtime=Nx the timing below assumes.
//
// Report p50 and p90 rather than trusting the mean ns/op: upstream quotes p50, and the
// machine these numbers come from is a 15 W mobile part that throttles, so the spread
// between the two is itself a result.
func BenchmarkForward(b *testing.B) {
	if testing.Short() {
		b.Skip("-short: needs an ONNX Runtime library and the S1 exports")
	}

	rt, env := newBenchEnv(b)

	for _, ck := range benchCheckpoints {
		model, modelErr := findModel(ck.onnx)

		b.Run(ck.name, func(b *testing.B) {
			if modelErr != nil {
				b.Skipf("no export: %v (run scripts/export_onnx.py --all --dynamo)", modelErr)
			}

			for _, threads := range benchThreads {
				shapes := shapesFor(ck, threads)
				if len(shapes) == 0 {
					continue
				}

				b.Run(fmt.Sprintf("threads=%d", threads), func(b *testing.B) {
					benchThreadGroup(b, rt, env, model, threads, shapes)
				})
			}
		})
	}
}

// benchThreadGroup holds one session open across every shape at a given thread count.
// IntraOpNumThreads is a session option, so the session cannot be shared across the
// thread sweep -- but it can be shared across the shapes, which saves 1.7 GB of loading
// per cell.
func benchThreadGroup(b *testing.B, rt *ort.Runtime, env *ort.Env, model string, threads int, shapes []benchShape) {
	b.Helper()

	sess, err := rt.NewSession(env, model, &ort.SessionOptions{IntraOpNumThreads: threads})
	if err != nil {
		b.Fatalf("NewSession(%q, threads=%d): %v", model, threads, err)
	}

	defer sess.Close()

	for _, sh := range shapes {
		b.Run(sh.String(), func(b *testing.B) {
			benchShapeRun(b, rt, sess, sh)
		})
	}
}

func benchShapeRun(b *testing.B, rt *ort.Runtime, sess *ort.Session, sh benchShape) {
	inputs := newBenchInputs(b, rt, sh)
	defer closeValues(inputs)

	// ORT pays arena allocation and graph optimisation on the first Run of a session at a
	// given shape. No steady-state number should carry that.
	runOnce(b, sess, inputs)

	latencies := make([]time.Duration, 0, b.N)

	b.ResetTimer()

	for range b.N {
		start := time.Now()

		runOnce(b, sess, inputs)
		latencies = append(latencies, time.Since(start))
	}

	b.StopTimer()
	slices.Sort(latencies)
	b.ReportMetric(millis(quantile(latencies, 0.5)), "p50_ms")
	b.ReportMetric(millis(quantile(latencies, 0.9)), "p90_ms")
}

// quantile is nearest-rank on an already-sorted slice. With the handful of samples a
// multi-second benchmark can afford, interpolating would invent precision.
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

// BenchmarkSessionLoad measures what it costs to bring a checkpoint up, which is the
// number that decides whether the Router's max_loaded=1 (README: "7.4 s median reload on
// CPU") is survivable in Go. ns/op is the load time; the reported RSS is the growth
// across the first load only, because ORT does not return the arena to the OS when the
// session closes and every later iteration would report roughly zero.
func BenchmarkSessionLoad(b *testing.B) {
	if testing.Short() {
		b.Skip("-short: needs an ONNX Runtime library and the S1 exports")
	}

	rt, env := newBenchEnv(b)

	for _, ck := range benchCheckpoints {
		model, modelErr := findModel(ck.onnx)

		b.Run(ck.name, func(b *testing.B) {
			if modelErr != nil {
				b.Skipf("no export: %v", modelErr)
			}

			var growth int64

			for i := range b.N {
				before := residentKB()

				sess, err := rt.NewSession(env, model, &ort.SessionOptions{IntraOpNumThreads: benchRefThreads})
				if err != nil {
					b.Fatalf("NewSession(%q): %v", model, err)
				}

				if i == 0 {
					growth = residentKB() - before
				}

				sess.Close()
			}

			if growth > 0 {
				b.ReportMetric(float64(growth)/1024, "rss_MiB")
			}
		})
	}
}

// residentKB reads VmRSS from /proc/self/status, or 0 where that does not exist. It is
// the whole process, so it is only meaningful as a delta around a single allocation --
// which is exactly how BenchmarkSessionLoad uses it.
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
