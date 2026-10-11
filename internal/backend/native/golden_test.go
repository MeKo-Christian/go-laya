package native

import (
	"encoding/json"
	"math"
	"runtime"
	"runtime/debug"
	"testing"
	"time"

	"github.com/MeKo-Christian/go-laya/internal/golden"
	"github.com/MeKo-Christian/go-laya/internal/runtime/tensor"
)

// goldenTol is the largest maxScaledDiff TestForwardGolden accepts against the
// PyTorch logits recorded in testdata/logits.jsonl (torch cpu, fp32, sdpa).
//
// Set from the measured worst case over all ten batches of each checkpoint
// (2026-10-10, amd64 with AVX2), all columns of logits and act_logits:
// english 8.2e-06, multilingual 1.1e-05, typed-decisions 3.8e-06, every
// argmax the same. The checkpoints store fp16 weights, which both sides
// upcast to the same float32 values, so what remains is float32 rounding in
// another order of operations through 28 or 22 encoder layers. That is the
// gap ORT has too (9.4e-06, 9.8e-06 and 5.8e-06, internal/backend/onnx), and
// this is ORT's bound: 4x the worst case here. A wrong mapping misses by
// orders of magnitude, not by a factor of 4.
var goldenTol = map[string]float64{
	golden.English:        5e-5,
	golden.Multilingual:   5e-5,
	golden.TypedDecisions: 5e-5,
}

// TestForwardGolden runs every batch recorded in testdata/logits.jsonl through
// the native backend on the real checkpoint and compares the raw logits and
// act_logits with what PyTorch recorded for the same collated tensors. It is
// the whole-model check the per-block fixtures of internal/modernbert and
// internal/head cannot make: the name-to-weight mapping on real weights, with
// both layer types and real sequence lengths. It is evidence toward Task 8.8,
// which gates the calibrated probabilities through SystemOne.
//
// One checkpoint is loaded at a time and closed before the next, so the test
// needs about 2 GB rather than 5.
func TestForwardGolden(t *testing.T) {
	root := golden.SkipWithoutModels(t)

	// The kernels' worker count is process-wide (tensor.SetWorkers); the
	// test sets it only to finish sooner, and restores it.
	prev := tensor.Workers()
	tensor.SetWorkers(runtime.NumCPU())
	t.Cleanup(func() { tensor.SetWorkers(prev) })

	cases := golden.Load(t, "logits")
	for _, ck := range []string{golden.English, golden.Multilingual, golden.TypedDecisions} {
		t.Run(ck, func(t *testing.T) {
			start := time.Now()
			b, err := Open(t.Context(), golden.CheckpointDir(root, ck), Options{})
			if err != nil {
				t.Fatalf("Open: %v", err)
			}
			loaded := time.Since(start)
			defer func() {
				if err := b.Close(); err != nil {
					t.Errorf("Close: %v", err)
				}
				runtime.GC()
				debug.FreeOSMemory()
			}()

			worst := map[string]float64{}
			var n, questions, agree int
			var forward time.Duration
			for _, c := range cases {
				var r struct {
					Checkpoint string                     `json:"checkpoint"`
					Collated   map[string]json.RawMessage `json:"collated"`
					Logits     json.RawMessage            `json:"logits"`
					ActLogits  json.RawMessage            `json:"act_logits"`
				}
				c.Unmarshal(t, &r)
				if r.Checkpoint != ck {
					continue
				}
				n++

				batch := golden.CollatedBatch(t, r.Collated)
				begin := time.Now()
				logits, act, err := b.Forward(t.Context(), batch)
				forward += time.Since(begin)
				if err != nil {
					t.Fatalf("%s: Forward: %v", c.Name, err)
				}
				wantLogits := golden.Matrix[float32](t, r.Logits, c.Name+" logits")
				for name, pair := range map[string][2][][]float32{
					"logits":     {logits, wantLogits},
					"act_logits": {act, golden.Matrix[float32](t, r.ActLogits, c.Name+" act_logits")},
				} {
					got, want := pair[0], pair[1]
					if len(got) != len(want) {
						t.Fatalf("%s %s: %d rows, recorded %d", c.Name, name, len(got), len(want))
					}
					for i := range got {
						d := maxScaledDiff(got[i], want[i])
						if d < 0 {
							t.Fatalf("%s %s[%d]: width %d, recorded %d", c.Name, name, i, len(got[i]), len(want[i]))
						}
						worst[name] = max(worst[name], d)
						if d > goldenTol[ck] {
							t.Errorf("%s %s[%d]: max scaled diff %.3g exceeds %g\n got %v\nwant %v",
								c.Name, name, i, d, goldenTol[ck], got[i], want[i])
						}
					}
				}
				for i := range logits {
					questions++
					if g, w := argmax(logits[i], batch.MarkerMask[i]), argmax(wantLogits[i], batch.MarkerMask[i]); g == w {
						agree++
					} else {
						t.Errorf("%s question %d: argmax %d, PyTorch %d", c.Name, i, g, w)
					}
				}
			}
			if n == 0 {
				t.Fatalf("logits.jsonl has no recordings for %s", ck)
			}
			t.Logf("%d batches; worst max scaled diff vs PyTorch: logits %.3g, act_logits %.3g; argmax %d/%d questions agree",
				n, worst["logits"], worst["act_logits"], agree, questions)
			t.Logf("load %v, Forward %v in total (%v per batch, %d kernel workers)",
				loaded.Round(time.Millisecond), forward.Round(time.Millisecond),
				(forward / time.Duration(n)).Round(time.Millisecond), tensor.Workers())
		})
	}
}

// maxScaledDiff is the largest |got - want| / max(1, |want|) over a row, the
// metric internal/backend/onnx's golden test uses, or -1 when the widths
// differ. A NaN or Inf anywhere is +Inf, so it fails every tolerance.
func maxScaledDiff(got, want []float32) float64 {
	if len(got) != len(want) {
		return -1
	}
	worst := 0.0
	for i := range got {
		d := math.Abs(float64(got[i])-float64(want[i])) / max(1, math.Abs(float64(want[i])))
		if math.IsNaN(d) || math.IsInf(d, 0) {
			return math.Inf(1)
		}
		worst = max(worst, d)
	}
	return worst
}

// argmax is the index of the largest logit among a row's real markers.
func argmax(row []float32, mask []bool) int {
	best := -1
	for i, v := range row {
		if mask[i] && (best < 0 || v > row[best]) {
			best = i
		}
	}
	return best
}
