//go:build !android && !ios && ((darwin && (amd64 || arm64)) || (linux && (amd64 || arm64 || loong64)) || (netbsd && (amd64 || arm64)))

package onnx

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/golden"
)

// TestInt8Exports is PLAN.md B.5.1's proof that the int8 graphs
// `scripts/export_onnx.py --int8` writes load and run in the ORT this module
// pins, not only in the Python environment that wrote them: Open's header
// check (IR, opset, external data), the session, and a Forward over every
// batch recorded in testdata/logits.jsonl, with finite outputs of the right
// shape. How far int8 lands from the recorded fp32 logits is logged, not
// asserted; whether that distance matters is B.5.3's question, asked of the
// answers and not of raw logits.
//
// Gated on LAYA_ONNX_INT8_DIR, the --out of the --int8 run, which holds
// laya-<checkpoint>.onnx.
func TestInt8Exports(t *testing.T) {
	if testing.Short() {
		t.Skip("-short: needs an ONNX Runtime library and the int8 exports")
	}
	dir := os.Getenv("LAYA_ONNX_INT8_DIR")
	if dir == "" {
		t.Skip("set LAYA_ONNX_INT8_DIR to the --out of scripts/export_onnx.py --int8")
	}
	lib := requireORTLibrary(t)
	newRuntime(t, lib) // logs which runtime answered

	for _, ck := range []string{golden.English, golden.Multilingual, golden.TypedDecisions} {
		t.Run(ck, func(t *testing.T) {
			model := filepath.Join(dir, "laya-"+ck+".onnx")
			b, err := Open(model, Options{Library: lib})
			if err != nil {
				t.Fatalf("Open(%q): %v", model, err)
			}
			t.Cleanup(func() {
				if err := b.Close(); err != nil {
					t.Errorf("Close: %v", err)
				}
			})

			var batches, rows, agree int
			worst := map[string]float64{}
			for _, c := range golden.Load(t, "logits") {
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
				batches++
				logits, act, err := b.Forward(context.Background(), golden.CollatedBatch(t, r.Collated))
				if err != nil {
					t.Fatalf("%s: Forward: %v", c.Name, err)
				}
				for name, pair := range map[string][2][][]float32{
					"logits":     {logits, golden.Matrix[float32](t, r.Logits, c.Name+" logits")},
					"act_logits": {act, golden.Matrix[float32](t, r.ActLogits, c.Name+" act_logits")},
				} {
					got, want := pair[0], pair[1]
					if len(got) != len(want) {
						t.Fatalf("%s %s: %d rows, recorded %d", c.Name, name, len(got), len(want))
					}
					for i := range got {
						if slices.ContainsFunc(got[i], func(v float32) bool { return math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) }) {
							t.Fatalf("%s %s[%d]: not finite: %v", c.Name, name, i, got[i])
						}
						d := maxScaledDiff(got[i], want[i])
						if d < 0 {
							t.Fatalf("%s %s[%d]: width %d, recorded %d", c.Name, name, i, len(got[i]), len(want[i]))
						}
						worst[name] = max(worst[name], d)
						if name == "logits" {
							rows++
							if argmax32(got[i]) == argmax32(want[i]) {
								agree++
							}
						}
					}
				}
			}
			if batches == 0 {
				t.Fatalf("logits.jsonl has no recordings for %s", ck)
			}
			t.Logf("%s: %d batches, %d rows; argmax agrees with fp32 PyTorch on %d of %d; "+
				"worst max scaled diff: logits %.3g, act_logits %.3g",
				model, batches, rows, agree, rows, worst["logits"], worst["act_logits"])
		})
	}
}

func argmax32(row []float32) int {
	best := 0
	for i, v := range row {
		if v > row[best] {
			best = i
		}
	}
	return best
}
