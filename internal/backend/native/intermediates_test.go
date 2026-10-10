package native

import (
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"runtime"
	"runtime/debug"
	"testing"

	"github.com/MeKo-Christian/go-laya/backend"
	"github.com/MeKo-Christian/go-laya/internal/golden"
	"github.com/MeKo-Christian/go-laya/internal/head"
	"github.com/MeKo-Christian/go-laya/internal/runtime/tensor"
)

// intermediatesTol is the largest difference TestHeadIntermediates accepts per
// stage against what PyTorch recorded in testdata/head_intermediates.jsonl:
// rowScaledDiff for the hidden states, maxScaledDiff for the rest.
//
// Measured on all three checkpoints (2026-10-10, amd64 with AVX2), worst of
// english, multilingual and typed-decisions. What differs is float32
// rounding in another order of operations through the encoder, which the
// head inherits; a wrong stage misses by orders of magnitude, as the
// mutations in the Task 8.8 PR show.
var intermediatesTol = map[string]float64{
	// h after type_emb, the encoder's output plus a type_emb row: 1.8e-6,
	// 1.6e-6, 1.4e-6. Leaving the marker rows without type_emb (#35) misses
	// by the embedding itself.
	"h_typed": 1e-5,
	// h after each head layer: 1.0e-6, 7.0e-7, 9.7e-7, below h_typed because
	// the layers widen each row's scale. The same 1e-5 as h_typed.
	"h_layers": 1e-5,
	// The gathered rows are the last layer's output at the markers and at
	// [CLS] for the fill, measured together with h_layers.
	"gathered": 1e-5,
	// The filled logits: 8.2e-6, 1.1e-5, 3.8e-6, TestForwardGolden's figures
	// for the same batches. goldenTol's 5e-5 for the same reason. The fill
	// itself is held to exactly -1e4 (#37).
	"logits": 5e-5,
	// The act features, probabilities and k/255 in [0, 1]: 5.5e-6, 7.0e-6,
	// 2.7e-6, as close as the logits they come from, so the same 5e-5. An
	// unclamped ln(k) is NaN on the single-option row (#38).
	"feats": 5e-5,
	// The act logits, relative as they sit in the thousands: 1.4e-6, 4.7e-7,
	// 4.1e-7. goldenTol's 5e-5, as TestForwardGolden holds them.
	"act_logits": 5e-5,
}

// intermediatesCase is one record of testdata/head_intermediates.jsonl, which
// scripts/dump_python_parity.py's dump_head_intermediates writes (Task 8.8).
type intermediatesCase struct {
	Checkpoint    string                     `json:"checkpoint"`
	Collated      map[string]json.RawMessage `json:"collated"`
	HeadLayers    int                        `json:"head_layers"`
	FastPathCalls int                        `json:"fast_path_calls"`
	Logits        json.RawMessage            `json:"logits"`
	Feats         json.RawMessage            `json:"feats"`
	ActLogits     json.RawMessage            `json:"act_logits"`
	// Trace is on one case per checkpoint: the hidden states at [CLS] and
	// at each row's real markers, and the whole gathered tensor.
	Trace *struct {
		Positions [][2]int          `json:"positions"`
		HTyped    json.RawMessage   `json:"h_typed"`
		HLayers   []json.RawMessage `json:"h_layers"`
		Gathered  json.RawMessage   `json:"gathered"`
	} `json:"trace"`
}

// tracedCase is a case and what the native backend computed for it.
type tracedCase struct {
	name  string
	rec   intermediatesCase
	batch backend.Batch
	tr    *head.Trace
}

// TestHeadIntermediates asserts invariants #35-38 on the real checkpoints:
// every case of testdata/head_intermediates.jsonl runs through the native
// backend, and the head's stages are held to the ones upstream's
// DecisionModel produced for the same batch in PyTorch. The file holds the
// logits.jsonl batches and one single-option batch per checkpoint, the only
// rows with fewer than two markers, where #38's clamp decides the entropy's
// denominator.
//
// One checkpoint is loaded at a time and closed before the next.
func TestHeadIntermediates(t *testing.T) {
	root := golden.SkipWithoutModels(t)

	prev := tensor.Workers()
	tensor.SetWorkers(runtime.NumCPU())
	t.Cleanup(func() { tensor.SetWorkers(prev) })

	cases := golden.Load(t, "head_intermediates")
	recorded := map[string]golden.Case{}
	for _, c := range golden.Load(t, "logits") {
		recorded[c.Name] = c
	}

	for _, ck := range []string{golden.English, golden.Multilingual, golden.TypedDecisions} {
		t.Run(ck, func(t *testing.T) {
			traced := traceCheckpoint(t, golden.CheckpointDir(root, ck), ck, cases, recorded)
			t.Run("35_type_emb_every_position", func(t *testing.T) { checkTypeEmb(t, onlyTrace(t, traced)) })
			t.Run("36_head_layers_manual_loop", func(t *testing.T) { checkHeadLayers(t, onlyTrace(t, traced)) })
			t.Run("37_mask_fill", func(t *testing.T) { checkMaskFill(t, traced) })
			t.Run("38_act_features", func(t *testing.T) { checkActFeatures(t, traced) })
		})
	}
}

// traceCheckpoint opens the checkpoint in dir, runs every case of ck through
// forwardTrace and closes it again, so the next checkpoint loads into freed
// memory. The traces own their tensors and outlive the backend.
func traceCheckpoint(t *testing.T, dir, ck string, cases []golden.Case, recorded map[string]golden.Case) []tracedCase {
	t.Helper()
	b, err := Open(t.Context(), dir, Options{})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer func() {
		if err := b.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
		runtime.GC()
		debug.FreeOSMemory()
	}()

	var traced []tracedCase
	shared, singles := 0, 0
	for _, c := range cases {
		var rec intermediatesCase
		c.Unmarshal(t, &rec)
		if rec.Checkpoint != ck {
			continue
		}
		batch := golden.CollatedBatch(t, rec.Collated)
		if lc, ok := recorded[c.Name]; ok {
			shared++
			sameAsLogits(t, c.Name, lc, batch, rec)
		} else if fewestMarkers(batch) < 2 {
			singles++
		}
		// The dumper counted torch's fused-kernel calls: one per head layer
		// is the fast path, the one the checkpoints' 16 and 12 heads run.
		if rec.FastPathCalls != rec.HeadLayers {
			t.Errorf("%s: %d fast-path calls for %d head layers; the recording is not the path the checkpoints run",
				c.Name, rec.FastPathCalls, rec.HeadLayers)
		}

		tr, err := b.forwardTrace(t.Context(), batch)
		if err != nil {
			t.Fatalf("%s: forwardTrace: %v", c.Name, err)
		}
		// The trace is Forward's pass, nothing else. Once per checkpoint:
		// a pass of the large encoder takes seconds.
		if rec.Trace != nil {
			logits, act, err := b.Forward(t.Context(), batch)
			if err != nil {
				t.Fatalf("%s: Forward: %v", c.Name, err)
			}
			if !reflect.DeepEqual(logits, tr.Logits) || !reflect.DeepEqual(act, tr.Act) {
				t.Fatalf("%s: forwardTrace's outputs differ from Forward's", c.Name)
			}
		}
		if len(tr.Layers) != rec.HeadLayers {
			t.Fatalf("%s: %d head layers, PyTorch ran %d", c.Name, len(tr.Layers), rec.HeadLayers)
		}
		traced = append(traced, tracedCase{name: c.Name, rec: rec, batch: batch, tr: tr})
	}
	// The file is the logits.jsonl batches, and the clamp needs a row with
	// fewer than two markers, which only the extra batch has.
	if shared != 10 || singles != 1 || len(traced) != shared+singles {
		t.Fatalf("%d cases: %d shared with logits.jsonl (want 10), %d with a row of fewer than two markers (want 1)",
			len(traced), shared, singles)
	}
	return traced
}

// checkTypeEmb is #35: h after the type embedding, at [CLS] and at every
// marker, where an embedding added at [CLS] only would be missing.
func checkTypeEmb(t *testing.T, tc tracedCase) {
	want := golden.Matrix[float32](t, tc.rec.Trace.HTyped, tc.name+" h_typed")
	worst := compareRows(t, tc.name+" h_typed", tc.tr.Typed, tc.rec.Trace.Positions, want, intermediatesTol["h_typed"])
	t.Logf("worst h_typed %.3g (tolerance %g)", worst, intermediatesTol["h_typed"])
}

// checkHeadLayers is #36: h after each head layer of the manual loop, and the
// rows the gather reads from the last one.
func checkHeadLayers(t *testing.T, tc tracedCase) {
	if len(tc.rec.Trace.HLayers) != len(tc.tr.Layers) {
		t.Fatalf("%s: %d recorded layer outputs, %d computed", tc.name, len(tc.rec.Trace.HLayers), len(tc.tr.Layers))
	}
	worst := 0.0
	for i, raw := range tc.rec.Trace.HLayers {
		name := fmt.Sprintf("%s h_layers[%d]", tc.name, i)
		want := golden.Matrix[float32](t, raw, name)
		worst = max(worst, compareRows(t, name, tc.tr.Layers[i], tc.rec.Trace.Positions, want, intermediatesTol["h_layers"]))
	}
	// The gather reads the last layer's output: the marker rows, and [CLS]
	// for every fill column (marker_pos 0, clamp(min=0)).
	got, shape := tc.tr.Gathered.Data(), tc.tr.Gathered.Shape()
	dim := int(shape[2])
	want := decodeTensor(t, tc.rec.Trace.Gathered, tc.name+" gathered")
	if !reflect.DeepEqual(want.Shape, []int{int(shape[0]), int(shape[1]), dim}) {
		t.Fatalf("%s: gathered is %v, PyTorch %v", tc.name, shape, want.Shape)
	}
	d := 0.0
	for r := range len(want.Data) / dim {
		d = max(d, rowScaledDiff(got[r*dim:(r+1)*dim], want.Data[r*dim:(r+1)*dim]))
	}
	worst = max(worst, d)
	if d > intermediatesTol["gathered"] {
		t.Errorf("%s: gathered max scaled diff %.3g exceeds %g", tc.name, d, intermediatesTol["gathered"])
	}
	t.Logf("worst h_layers and gathered %.3g (tolerances %g, %g)",
		worst, intermediatesTol["h_layers"], intermediatesTol["gathered"])
}

// checkMaskFill is #37: the marker logits match torch's, and every fill
// column is exactly -1e4 on both sides, the literal rather than
// head.MaskFill, so a drifted constant cannot follow itself.
func checkMaskFill(t *testing.T, traced []tracedCase) {
	worst := 0.0
	for _, tc := range traced {
		want := golden.Matrix[float32](t, tc.rec.Logits, tc.name+" logits")
		worst = max(worst, compareMatrix(t, tc.name+" logits", tc.tr.Logits, want, intermediatesTol["logits"]))
		for i, row := range tc.batch.MarkerMask {
			if len(row) != len(want[i]) || len(row) != len(tc.tr.Logits[i]) {
				t.Fatalf("%s: marker_mask[%d] has %d columns, the logits %d and %d",
					tc.name, i, len(row), len(tc.tr.Logits[i]), len(want[i]))
			}
			for j, isMarker := range row {
				if !isMarker && (tc.tr.Logits[i][j] != -1e4 || want[i][j] != -1e4) {
					t.Errorf("%s: logits[%d][%d] = %g over the fill, PyTorch %g; want exactly -1e4",
						tc.name, i, j, tc.tr.Logits[i][j], want[i][j])
				}
			}
		}
	}
	t.Logf("worst logits %.3g (tolerance %g)", worst, intermediatesTol["logits"])
}

// checkActFeatures is #38: the four act features and the act logits they
// feed, including the single-option rows where the clamp decides ln(k).
func checkActFeatures(t *testing.T, traced []tracedCase) {
	worstFeats, worstAct := 0.0, 0.0
	for _, tc := range traced {
		want := golden.Matrix[float32](t, tc.rec.Feats, tc.name+" feats")
		worstFeats = max(worstFeats, compareMatrix(t, tc.name+" feats", tc.tr.Feats, want, intermediatesTol["feats"]))
		wantAct := golden.Matrix[float32](t, tc.rec.ActLogits, tc.name+" act_logits")
		worstAct = max(worstAct, compareMatrix(t, tc.name+" act_logits", tc.tr.Act, wantAct, intermediatesTol["act_logits"]))
	}
	t.Logf("worst feats %.3g, act_logits %.3g (tolerances %g, %g)",
		worstFeats, worstAct, intermediatesTol["feats"], intermediatesTol["act_logits"])
}

// sameAsLogits holds a head_intermediates.jsonl case to the logits.jsonl case
// of the same name: the same collated batch and bit for bit the same logits
// and act logits, so the two files describe the same forward passes.
func sameAsLogits(t *testing.T, name string, lc golden.Case, batch backend.Batch, rec intermediatesCase) {
	t.Helper()
	var r struct {
		Collated  map[string]json.RawMessage `json:"collated"`
		Logits    json.RawMessage            `json:"logits"`
		ActLogits json.RawMessage            `json:"act_logits"`
	}
	lc.Unmarshal(t, &r)
	if !reflect.DeepEqual(golden.CollatedBatch(t, r.Collated), batch) {
		t.Errorf("%s: the batch differs from logits.jsonl's", name)
	}
	for _, m := range []struct {
		key         string
		them, these json.RawMessage
	}{
		{"logits", r.Logits, rec.Logits},
		{"act_logits", r.ActLogits, rec.ActLogits},
	} {
		if !reflect.DeepEqual(golden.Matrix[float32](t, m.them, name+" logits.jsonl "+m.key),
			golden.Matrix[float32](t, m.these, name+" "+m.key)) {
			t.Errorf("%s: the %s differ from logits.jsonl's", name, m.key)
		}
	}
}

// fewestMarkers is the smallest number of real markers in any row.
func fewestMarkers(b backend.Batch) int {
	fewest := -1
	for _, row := range b.MarkerMask {
		n := 0
		for _, m := range row {
			if m {
				n++
			}
		}
		if fewest < 0 || n < fewest {
			fewest = n
		}
	}
	return fewest
}

// onlyTrace returns the one case of a checkpoint that carries hidden states,
// requiring its batch to have both padding and marker fill: the traced
// positions then include rows the key-padding mask shortens and the fill
// columns read [CLS].
func onlyTrace(t *testing.T, traced []tracedCase) tracedCase {
	t.Helper()
	var found []tracedCase
	for _, tc := range traced {
		if tc.rec.Trace != nil {
			found = append(found, tc)
		}
	}
	if len(found) != 1 {
		t.Fatalf("%d cases carry a trace, want 1", len(found))
	}
	tc := found[0]
	padded, filled := false, false
	for i, row := range tc.batch.AttentionMask {
		for _, v := range row {
			padded = padded || v == 0
		}
		for _, m := range tc.batch.MarkerMask[i] {
			filled = filled || !m
		}
	}
	if !padded || !filled {
		t.Fatalf("%s: the traced batch has padding %v and fill %v, want both", tc.name, padded, filled)
	}
	return tc
}

// compareRows holds the rows of h [B, S, d] at positions to want [N, d] and
// returns the worst rowScaledDiff.
func compareRows(t *testing.T, name string, h *tensor.Tensor, positions [][2]int, want [][]float32, tol float64) float64 {
	t.Helper()
	shape := h.Shape()
	batch, seq, d := int(shape[0]), int(shape[1]), int(shape[2])
	if len(positions) != len(want) {
		t.Fatalf("%s: %d positions, %d recorded rows", name, len(positions), len(want))
	}
	worst := 0.0
	for r, p := range positions {
		if p[0] < 0 || p[0] >= batch || p[1] < 0 || p[1] >= seq {
			t.Fatalf("%s: position %v is outside the hidden state's [%d, %d]", name, p, batch, seq)
		}
		row := h.Data()[(p[0]*seq+p[1])*d : (p[0]*seq+p[1]+1)*d]
		diff := rowScaledDiff(row, want[r])
		if diff < 0 {
			t.Fatalf("%s at %v: width %d, recorded %d", name, p, len(row), len(want[r]))
		}
		worst = max(worst, diff)
		if diff > tol {
			t.Errorf("%s at row %d, position %d: max scaled diff %.3g exceeds %g", name, p[0], p[1], diff, tol)
		}
	}
	return worst
}

// compareMatrix holds got to want row by row and returns the worst
// maxScaledDiff.
func compareMatrix(t *testing.T, name string, got, want [][]float32, tol float64) float64 {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d rows, recorded %d", name, len(got), len(want))
	}
	worst := 0.0
	for i := range got {
		d := maxScaledDiff(got[i], want[i])
		if d < 0 {
			t.Fatalf("%s[%d]: width %d, recorded %d", name, i, len(got[i]), len(want[i]))
		}
		worst = max(worst, d)
		if d > tol {
			t.Errorf("%s[%d]: max scaled diff %.3g exceeds %g\n got %v\nwant %v", name, i, d, tol, got[i], want[i])
		}
	}
	return worst
}

// rowScaledDiff is the largest |got - want| over a hidden-state row, divided
// by max(1, the row's largest |want|), or -1 when the widths differ. A NaN
// or Inf anywhere is +Inf.
//
// Hidden states are held to their row's scale rather than each element's,
// as maxScaledDiff holds logits: these rows carry a few dimensions in the
// hundreds beside many near 0, and the float32 rounding of the attention and
// the LayerNorms that mix them is relative to the row, so an element near 0
// inherits an absolute error of about 1e-4 that maxScaledDiff would read as
// a relative 1e-4.
func rowScaledDiff(got, want []float32) float64 {
	if len(got) != len(want) {
		return -1
	}
	worst, scale := 0.0, 1.0
	for i := range got {
		d := math.Abs(float64(got[i]) - float64(want[i]))
		if math.IsNaN(d) || math.IsInf(d, 0) {
			return math.Inf(1)
		}
		worst = max(worst, d)
		scale = max(scale, math.Abs(float64(want[i])))
	}
	return worst / scale
}

// recordedTensor is a recorded tensor of any rank.
type recordedTensor struct {
	Shape []int     `json:"shape"`
	Data  []float32 `json:"data"`
}

// decodeTensor decodes a recorded float32 tensor and checks that its data
// fills its shape.
func decodeTensor(t *testing.T, raw json.RawMessage, name string) recordedTensor {
	t.Helper()
	var rec recordedTensor
	if err := json.Unmarshal(raw, &rec); err != nil {
		t.Fatalf("decode %s: %v", name, err)
	}
	n := 1
	for _, d := range rec.Shape {
		n *= d
	}
	if n != len(rec.Data) {
		t.Fatalf("%s: shape %v holds %d values, data has %d", name, rec.Shape, n, len(rec.Data))
	}
	return rec
}
