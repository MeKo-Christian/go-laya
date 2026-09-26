//go:build !android && !ios && ((darwin && (amd64 || arm64)) || (linux && (amd64 || arm64 || loong64)) || (netbsd && (amd64 || arm64)))

package onnx

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/MeKo-Christian/go-laya/backend"
	"github.com/MeKo-Christian/go-laya/internal/golden"
)

// goldenTol is the largest maxScaledDiff TestForwardGolden accepts against the
// PyTorch logits recorded in testdata/logits.jsonl, per checkpoint.
//
// Set from the measured worst case over all ten batches of each checkpoint
// (2026-09-26, ORT 1.23.0 on the dynamo exports): english 9.4e-06, multilingual
// 9.8e-06, typed-decisions 5.8e-06, all columns of logits and act_logits. Each
// gets roughly 5x headroom. multilingual measures no looser than the others
// here, although S1 recorded it an order of magnitude looser at its own shapes.
// It keeps its own entry so Task 6.8.2 can move it without touching the rest.
//
// These numbers already include the eager-vs-sdpa gap (Task 6.5): the exports use
// eager attention while logits.jsonl was recorded under upstream's sdpa. An sdpa
// export measured english 1.2e-05, multilingual 7.4e-06, typed-decisions 7.3e-06
// here -- no closer overall, because at opset 18 both lower attention to the same
// MatMul/Softmax -- so the exports stay eager and the tolerance needs no extra room.
var goldenTol = map[string]float64{
	golden.English:        5e-5,
	golden.Multilingual:   5e-5,
	golden.TypedDecisions: 5e-5,
}

// TestForwardGolden runs every batch recorded in testdata/logits.jsonl through
// the real backend and compares it with the logits Python recorded for the same
// collated tensors. It is the discriminating test for Task 6.3.1: a transposed
// flattening, a bool mask packed as the wrong width, or outputs split by the
// wrong width each land far outside the tolerance.
func TestForwardGolden(t *testing.T) {
	if testing.Short() {
		t.Skip("-short: needs an ONNX Runtime library and the S1 exports")
	}
	lib := requireORTLibrary(t)

	for _, ck := range []string{golden.English, golden.Multilingual, golden.TypedDecisions} {
		t.Run(ck, func(t *testing.T) {
			model, err := findModel("laya-" + ck + "-dynamo.onnx")
			if err != nil {
				t.Skipf("no export: %v (run scripts/export_onnx.py --all --dynamo and set LAYA_ONNX_DIR)", err)
			}

			b, err := Open(model, Options{Library: lib})
			if err != nil {
				t.Fatalf("Open(%q): %v", model, err)
			}
			t.Cleanup(func() {
				if err := b.Close(); err != nil {
					t.Errorf("Close: %v", err)
				}
			})

			worst := map[string]float64{}
			n := 0
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
				n++

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
			}
			if n == 0 {
				t.Fatalf("logits.jsonl has no recordings for %s", ck)
			}
			t.Logf("%d batches; worst max scaled diff vs PyTorch: logits %.3g, act_logits %.3g",
				n, worst["logits"], worst["act_logits"])
		})
	}
}

// TestForwardChecksContextFirst is Task 6.3.5's first half. The binding's
// Session.Run ignores its ctx, so a Forward that has already been cancelled must
// return before touching the session. A zero Backend has no session, so the
// order of the two checks is the only thing that decides which error comes back.
func TestForwardChecksContextFirst(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	var b Backend
	if _, _, err := b.Forward(ctx, backend.Batch{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("Forward(cancelled) = %v, want context.Canceled", err)
	}
	if _, _, err := b.Forward(context.Background(), backend.Batch{}); !errors.Is(err, ErrClosed) {
		t.Fatalf("Forward on an unopened Backend = %v, want ErrClosed", err)
	}
}

// TestCloseIsIdempotent: the Router closes what it evicts, and a double Close
// must not release the session twice.
func TestCloseIsIdempotent(t *testing.T) {
	var b Backend
	for i := range 2 {
		if err := b.Close(); err != nil {
			t.Fatalf("Close #%d: %v", i+1, err)
		}
	}
}
