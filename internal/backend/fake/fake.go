// Package fake is a backend.Backend that replays testdata/logits.jsonl instead
// of running a model (PLAN.md Task 6.1), so every test above the backend runs
// without ONNX Runtime, weights or Python.
//
// It answers only batches it has a recording for, matched exactly on all five
// collated tensors. Anything else is ErrUnknownBatch, never zeros: a fake that
// returns a plausible default would let a prompt regression pass every test
// built on it (Task 6.1.3).
package fake

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"sync/atomic"
	"testing"

	"github.com/MeKo-Christian/go-laya/backend"
	"github.com/MeKo-Christian/go-laya/internal/golden"
)

var (
	// ErrUnknownBatch is Forward's error for a batch that is not one of the
	// recorded inputs of the fake's checkpoint.
	ErrUnknownBatch = errors.New("fake backend: batch is not a recorded logits.jsonl input")

	// ErrClosed is Forward's error after Close.
	ErrClosed = errors.New("fake backend: closed")
)

type record struct {
	name        string
	batch       backend.Batch
	logits, act [][]float32
}

// Backend replays one checkpoint's recordings. It is per checkpoint because
// the fixture requires it: english and typed-decisions share a tokenizer and
// budgets, so they record identical batches with different logits.
type Backend struct {
	checkpoint string
	recs       []record
	closed     atomic.Bool
}

var _ backend.Backend = (*Backend)(nil)

// New loads the recordings of checkpoint ("english", "multilingual" or
// "typed-decisions"). A checkpoint with no recordings fails tb: a fake that
// rejects every batch would fail its test for the wrong reason.
func New(tb testing.TB, checkpoint string) *Backend {
	tb.Helper()

	b := &Backend{checkpoint: checkpoint}
	for _, c := range golden.Load(tb, "logits") {
		var r struct {
			Checkpoint string                     `json:"checkpoint"`
			Collated   map[string]json.RawMessage `json:"collated"`
			Logits     json.RawMessage            `json:"logits"`
			ActLogits  json.RawMessage            `json:"act_logits"`
		}
		c.Unmarshal(tb, &r)
		if r.Checkpoint != checkpoint {
			continue
		}
		b.recs = append(b.recs, record{
			name:   c.Name,
			batch:  golden.CollatedBatch(tb, r.Collated),
			logits: golden.Matrix[float32](tb, r.Logits, c.Name+" logits"),
			act:    golden.Matrix[float32](tb, r.ActLogits, c.Name+" act_logits"),
		})
	}
	if len(b.recs) == 0 {
		tb.Fatalf("logits.jsonl has no recordings for checkpoint %q", checkpoint)
	}
	return b
}

// Forward returns copies of the recorded logits and act_logits for a batch
// equal to a recording, and ErrUnknownBatch with no outputs for any other.
// The error names the nearest recording and where it first differs, which is
// the token a prompt regression moved.
func (b *Backend) Forward(ctx context.Context, in backend.Batch) (logits, act [][]float32, err error) {
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	if b.closed.Load() {
		return nil, nil, ErrClosed
	}

	for _, r := range b.recs {
		if reflect.DeepEqual(in, r.batch) {
			return clone2(r.logits), clone2(r.act), nil
		}
	}
	return nil, nil, fmt.Errorf("%w: %s, input_ids %s; %s",
		ErrUnknownBatch, b.checkpoint, shape(in.InputIDs), b.nearest(in))
}

// Close marks the fake closed; Forward fails afterwards, as a real session's
// would.
func (b *Backend) Close() error {
	b.closed.Store(true)
	return nil
}

// nearest describes the same-shape recording with the fewest differing cells
// and the first cell where it differs.
func (b *Backend) nearest(in backend.Batch) string {
	best, bestN, first := "", -1, ""
	for _, r := range b.recs {
		n, at, ok := diff(in, r.batch)
		if ok && (bestN < 0 || n < bestN) {
			best, bestN, first = r.name, n, at
		}
	}
	if bestN < 0 {
		return "no recording has that shape"
	}
	return fmt.Sprintf("nearest recording %s differs in %d cells, first at %s", best, bestN, first)
}

// diff counts the cells in which two same-shape batches differ and describes
// the first; ok is false when the shapes differ.
func diff(got, want backend.Batch) (n int, first string, ok bool) {
	note := func(tensor string, idx string, g, w any) {
		if n == 0 {
			first = fmt.Sprintf("%s%s: got %v, recorded %v", tensor, idx, g, w)
		}
		n++
	}
	if !sameShape(got.InputIDs, want.InputIDs) || !sameShape(got.AttentionMask, want.AttentionMask) ||
		!sameShape(got.MarkerPos, want.MarkerPos) || !sameShape(got.MarkerMask, want.MarkerMask) ||
		len(got.QType) != len(want.QType) {
		return 0, "", false
	}
	diff2(got.InputIDs, want.InputIDs, "input_ids", note)
	diff2(got.AttentionMask, want.AttentionMask, "attention_mask", note)
	diff2(got.MarkerPos, want.MarkerPos, "marker_pos", note)
	diff2(got.MarkerMask, want.MarkerMask, "marker_mask", note)
	for i := range got.QType {
		if got.QType[i] != want.QType[i] {
			note("qtype", fmt.Sprintf("[%d]", i), got.QType[i], want.QType[i])
		}
	}
	return n, first, true
}

func diff2[T comparable](got, want [][]T, tensor string, note func(string, string, any, any)) {
	for i := range got {
		for j := range got[i] {
			if got[i][j] != want[i][j] {
				note(tensor, fmt.Sprintf("[%d][%d]", i, j), got[i][j], want[i][j])
			}
		}
	}
}

func sameShape[T any](a, b [][]T) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return false
		}
	}
	return true
}

func shape[T any](m [][]T) string {
	if len(m) == 0 {
		return "[0 0]"
	}
	return fmt.Sprintf("[%d %d]", len(m), len(m[0]))
}

func clone2[T any](m [][]T) [][]T {
	out := make([][]T, len(m))
	for i, row := range m {
		out[i] = append(make([]T, 0, len(row)), row...)
	}
	return out
}
