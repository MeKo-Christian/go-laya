//go:build !android && !ios && ((darwin && (amd64 || arm64)) || (linux && (amd64 || arm64 || loong64)) || (netbsd && (amd64 || arm64)))

package onnx

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/MeKo-Christian/go-laya/backend"
	"github.com/MeKo-Christian/go-laya/internal/golden"
)

// matrixShapes are S1's four validation shapes, the ones compare_with_ort in
// scripts/export_onnx.py runs: the traced one, then a different seq, k and
// batch in turn. A fixture file missing one of them fails rather than
// shrinking the matrix.
var matrixShapes = []string{"traced", "other_seq", "other_k", "other_batch"}

// matrixTol is the largest maxScaledDiff TestForwardMatrix accepts, per
// checkpoint, against the Python ORT outputs and against PyTorch (Task 6.8.2).
//
// Set from the measured worst case over all four shapes and both outputs
// (2026-09-26, Go on ORT 1.23.0 against fixtures from ORT 1.30.0, identical
// over three runs), rounded up from roughly 5x:
//
//	                 vs python-ORT   vs PyTorch
//	english          7.0e-05         3.1e-05
//	multilingual     6.4e-05         4.7e-04
//	typed-decisions  2.6e-05         3.2e-05
//
// Against PyTorch multilingual is some 15x looser than the other two, as S1
// recorded, so one global constant would either fail it or wave through a
// tenfold regression on the others. Against Python ORT the three agree; the
// worst there is english at seq 61, the shape S1 also measured loosest.
var matrixTol = map[string]struct{ ORT, Torch float64 }{
	golden.English:        {ORT: 4e-4, Torch: 2e-4},
	golden.Multilingual:   {ORT: 4e-4, Torch: 2.5e-3},
	golden.TypedDecisions: {ORT: 1.5e-4, Torch: 2e-4},
}

// matrixCase is one shape of a scripts/export_onnx.py --fixture-matrix file.
type matrixCase struct {
	Shape  string                `json:"shape"`
	Seq    int                   `json:"seq"`
	K      int                   `json:"k"`
	Batch  int                   `json:"batch"`
	Inputs map[string]tensorJSON `json:"inputs"`
	ORT    map[string]tensorJSON `json:"ort"`
	Torch  map[string]tensorJSON `json:"torch"`
}

type matrixFixture struct {
	Checkpoint  string            `json:"checkpoint"`
	ONNX        string            `json:"onnx"`
	Attn        string            `json:"attn"`
	Versions    map[string]string `json:"versions"`
	OutputNames []string          `json:"output_names"`
	Cases       []matrixCase      `json:"cases"`
}

func loadMatrixFixture(t *testing.T, ck string) matrixFixture {
	t.Helper()

	raw, err := os.ReadFile(filepath.Join("testdata", "matrix", "forward-"+ck+".json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}

	var fx matrixFixture
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatalf("decode fixture: %v", err)
	}

	if fx.Checkpoint != ck {
		t.Fatalf("fixture is for %q, want %q", fx.Checkpoint, ck)
	}

	got := make([]string, 0, len(fx.Cases))
	for _, c := range fx.Cases {
		got = append(got, c.Shape)
	}

	if !slices.Equal(got, matrixShapes) {
		t.Fatalf("fixture shapes = %v, want %v", got, matrixShapes)
	}

	return fx
}

// fixtureRows reshapes one flat [rows, cols] fixture tensor into the row
// slices backend.Batch holds.
func fixtureRows[T any](t *testing.T, name string, tj tensorJSON) [][]T {
	t.Helper()

	if len(tj.Shape) != 2 {
		t.Fatalf("%s: shape %v, want two dimensions", name, tj.Shape)
	}

	var flat []T
	mustUnmarshal(t, name, tj.Data, &flat)

	rows, cols := int(tj.Shape[0]), int(tj.Shape[1])
	if len(flat) != rows*cols {
		t.Fatalf("%s: %d values for shape %v", name, len(flat), tj.Shape)
	}

	out := make([][]T, rows)
	for i := range out {
		out[i] = flat[i*cols : (i+1)*cols]
	}

	return out
}

// matrixBatch turns a fixture's input tensors back into the batch Forward
// takes, so the comparison runs through the product path, flattening included.
func matrixBatch(t *testing.T, in map[string]tensorJSON) backend.Batch {
	t.Helper()

	var qtype []int64
	mustUnmarshal(t, "qtype", in["qtype"].Data, &qtype)

	return backend.Batch{
		InputIDs:      fixtureRows[int64](t, "input_ids", in["input_ids"]),
		AttentionMask: fixtureRows[int64](t, "attention_mask", in["attention_mask"]),
		MarkerPos:     fixtureRows[int64](t, "marker_pos", in["marker_pos"]),
		MarkerMask:    fixtureRows[bool](t, "marker_mask", in["marker_mask"]),
		QType:         qtype,
	}
}

// TestForwardMatrix is Task 6.8.1: every checkpoint at every one of S1's four
// shapes, through Open and Forward under the Go-side runtime, against the
// Python ORT and PyTorch outputs --fixture-matrix recorded for the same inputs.
// It is what shows the dynamic axes hold from Go: before it, only Python ORT
// had run the other seq, k and batch.
func TestForwardMatrix(t *testing.T) {
	if testing.Short() {
		t.Skip("-short: needs an ONNX Runtime library and the S1 exports")
	}

	lib := requireORTLibrary(t)

	for _, ck := range []string{golden.English, golden.Multilingual, golden.TypedDecisions} {
		t.Run(ck, func(t *testing.T) {
			fx := loadMatrixFixture(t, ck)

			model, err := findModel(fx.ONNX)
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
			t.Logf("fixture built with onnxruntime %s, attn %s", fx.Versions["onnxruntime"], fx.Attn)

			tol := matrixTol[ck]
			for _, c := range fx.Cases {
				t.Run(c.Shape, func(t *testing.T) {
					logits, act, err := b.Forward(context.Background(), matrixBatch(t, c.Inputs))
					if err != nil {
						t.Fatalf("Forward(seq %d, k %d, batch %d): %v", c.Seq, c.K, c.Batch, err)
					}

					for _, name := range fx.OutputNames {
						got := map[string][][]float32{"logits": logits, "act_logits": act}[name]
						if shape := []int64{int64(len(got)), int64(len(got[0]))}; !slices.Equal(shape, c.ORT[name].Shape) {
							t.Fatalf("%s: shape %v, want %v", name, shape, c.ORT[name].Shape)
						}

						flat := slices.Concat(got...)
						vsORT := maxScaledDiff(flat, wantFloats(t, name, c.ORT[name]))
						vsTorch := maxScaledDiff(flat, wantFloats(t, name, c.Torch[name]))
						t.Logf("seq %d k %d batch %d %s: max scaled diff vs python-ORT %.3g, vs PyTorch %.3g",
							c.Seq, c.K, c.Batch, name, vsORT, vsTorch)

						if vsORT < 0 || vsORT > tol.ORT {
							t.Errorf("%s: %.3g vs python-ORT exceeds %g", name, vsORT, tol.ORT)
						}

						if vsTorch < 0 || vsTorch > tol.Torch {
							t.Errorf("%s: %.3g vs PyTorch exceeds %g", name, vsTorch, tol.Torch)
						}
					}
				})
			}
		})
	}
}
