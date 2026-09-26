package onnx

import (
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/MeKo-Christian/go-laya/backend"
)

func validBatch() backend.Batch {
	return backend.Batch{
		InputIDs:      [][]int64{{1, 2, 3}, {4, 5, 0}},
		AttentionMask: [][]int64{{1, 1, 1}, {1, 1, 0}},
		MarkerPos:     [][]int64{{1, 2}, {1, 0}},
		MarkerMask:    [][]bool{{true, true}, {true, false}},
		QType:         []int64{0, 2},
	}
}

func TestFlattenRowMajor(t *testing.T) {
	f, err := flatten(validBatch())
	if err != nil {
		t.Fatalf("flatten: %v", err)
	}
	if f.rows != 2 || f.seq != 3 || f.k != 2 {
		t.Fatalf("dims = %d×%d, k=%d; want 2×3, k=2", f.rows, f.seq, f.k)
	}
	if want := []int64{1, 2, 3, 4, 5, 0}; !slices.Equal(f.inputIDs, want) {
		t.Errorf("input_ids = %v, want %v", f.inputIDs, want)
	}
	if want := []int64{1, 1, 1, 1, 1, 0}; !slices.Equal(f.attention, want) {
		t.Errorf("attention_mask = %v, want %v", f.attention, want)
	}
	if want := []int64{1, 2, 1, 0}; !slices.Equal(f.markerPos, want) {
		t.Errorf("marker_pos = %v, want %v", f.markerPos, want)
	}
	if want := []bool{true, true, true, false}; !slices.Equal(f.markerMask, want) {
		t.Errorf("marker_mask = %v, want %v", f.markerMask, want)
	}
	if want := []int64{0, 2}; !slices.Equal(f.qtype, want) {
		t.Errorf("qtype = %v, want %v", f.qtype, want)
	}
}

// TestFlattenRejectsMalformed: the graph would happily run on a mis-shaped
// flat buffer and return logits for the wrong cells, so every shape mismatch has
// to fail here, before a tensor exists.
func TestFlattenRejectsMalformed(t *testing.T) {
	cases := map[string]func(*backend.Batch){
		"empty":                 func(b *backend.Batch) { *b = backend.Batch{} },
		"ragged input_ids":      func(b *backend.Batch) { b.InputIDs[1] = b.InputIDs[1][:2] },
		"attention_mask rows":   func(b *backend.Batch) { b.AttentionMask = b.AttentionMask[:1] },
		"attention_mask width":  func(b *backend.Batch) { b.AttentionMask[0] = append(b.AttentionMask[0], 1) },
		"marker_pos rows":       func(b *backend.Batch) { b.MarkerPos = b.MarkerPos[:1] },
		"ragged marker_pos":     func(b *backend.Batch) { b.MarkerPos[1] = b.MarkerPos[1][:1] },
		"marker_mask width":     func(b *backend.Batch) { b.MarkerMask[0] = b.MarkerMask[0][:1] },
		"marker_mask rows":      func(b *backend.Batch) { b.MarkerMask = b.MarkerMask[:1] },
		"qtype length":          func(b *backend.Batch) { b.QType = b.QType[:1] },
		"zero-width input_ids":  func(b *backend.Batch) { b.InputIDs, b.AttentionMask = [][]int64{{}, {}}, [][]int64{{}, {}} },
		"zero-width marker_pos": func(b *backend.Batch) { b.MarkerPos, b.MarkerMask = [][]int64{{}, {}}, [][]bool{{}, {}} },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			b := validBatch()
			mutate(&b)
			if _, err := flatten(b); !errors.Is(err, ErrBadBatch) {
				t.Fatalf("flatten = %v, want ErrBadBatch", err)
			}
		})
	}
}

func TestUnflatten(t *testing.T) {
	got, err := unflatten("logits", []float32{1, 2, 3, 4, 5, 6}, []int64{2, 3}, 2, 3)
	if err != nil {
		t.Fatalf("unflatten: %v", err)
	}
	if len(got) != 2 || !slices.Equal(got[0], []float32{1, 2, 3}) || !slices.Equal(got[1], []float32{4, 5, 6}) {
		t.Fatalf("unflatten = %v", got)
	}
	// The rows are independent slices: a caller writing into one must not
	// clobber its neighbour through a shared backing array.
	got[0] = append(got[0], 99)
	if got[1][0] != 4 {
		t.Fatalf("rows share a backing array: %v", got)
	}

	for name, tc := range map[string]struct {
		data  []float32
		shape []int64
	}{
		"not 2-D":        {[]float32{1, 2}, []int64{2}},
		"wrong rows":     {[]float32{1, 2, 3}, []int64{3, 1}},
		"short buffer":   {[]float32{1, 2, 3}, []int64{2, 2}},
		"negative width": {nil, []int64{2, -1}},
	} {
		if _, err := unflatten("logits", tc.data, tc.shape, 2, anyWidth); err == nil {
			t.Errorf("%s: unflatten accepted shape %v over %d values", name, tc.shape, len(tc.data))
		}
	}
}

// A graph whose output is not as wide as the batch and the config say is not
// a malformed run but the wrong checkpoint: logits must be kmax wide and
// act_logits len(act_costs)+1 wide (PLAN.md 6.4.5). anyWidth skips the
// check for a width nothing pinned.
func TestUnflattenWidth(t *testing.T) {
	data := []float32{1, 2, 3, 4, 5, 6}
	if _, err := unflatten("act_logits", data, []int64{2, 3}, 2, anyWidth); err != nil {
		t.Fatalf("anyWidth: %v", err)
	}
	for _, name := range []string{"logits", "act_logits"} {
		_, err := unflatten(name, data, []int64{2, 3}, 2, 2)
		if !errors.Is(err, backend.ErrIncompatibleCheckpoint) {
			t.Fatalf("%s 3 wide, want 2: %v, want ErrIncompatibleCheckpoint", name, err)
		}
		for _, w := range []string{name, "[2 3]", "2 wide"} {
			if !strings.Contains(err.Error(), w) {
				t.Errorf("error %q does not mention %q", err, w)
			}
		}
	}
}
