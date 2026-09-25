package golden

import (
	"encoding/json"
	"testing"

	"github.com/MeKo-Christian/go-laya/backend"
)

// tensorRec is dump_python_parity.py's tensor_rec: a row-major flat buffer
// and the shape to fold it back into.
type tensorRec[T any] struct {
	DType string `json:"dtype"`
	Shape []int  `json:"shape"`
	Data  []T    `json:"data"`
}

func loadTensor[T any](tb testing.TB, raw json.RawMessage, name string) tensorRec[T] {
	tb.Helper()

	var rec tensorRec[T]
	if err := json.Unmarshal(raw, &rec); err != nil {
		tb.Fatalf("decode %s: %v", name, err)
	}
	n := 1
	for _, d := range rec.Shape {
		n *= d
	}
	if n != len(rec.Data) {
		tb.Fatalf("%s: shape %v holds %d values, data has %d", name, rec.Shape, n, len(rec.Data))
	}
	return rec
}

// Matrix folds a recorded 2-D tensor back into rows; name only labels the
// failure. A zero-width tensor still has its rows, as torch.zeros((n, 0)) does.
func Matrix[T any](tb testing.TB, raw json.RawMessage, name string) [][]T {
	tb.Helper()

	rec := loadTensor[T](tb, raw, name)
	if len(rec.Shape) != 2 {
		tb.Fatalf("%s: shape %v is not 2-D", name, rec.Shape)
	}
	rows, cols := rec.Shape[0], rec.Shape[1]
	out := make([][]T, rows)
	for i := range out {
		out[i] = append(make([]T, 0, cols), rec.Data[i*cols:(i+1)*cols]...)
	}
	return out
}

// Vector decodes a recorded 1-D tensor.
func Vector[T any](tb testing.TB, raw json.RawMessage, name string) []T {
	tb.Helper()

	rec := loadTensor[T](tb, raw, name)
	if len(rec.Shape) != 1 {
		tb.Fatalf("%s: shape %v is not 1-D", name, rec.Shape)
	}
	return rec.Data
}

// CollatedBatch rebuilds the backend.Batch recorded under a logits.jsonl case's
// "collated" key: the five tensors collate_items returned (Task 1.7.1).
func CollatedBatch(tb testing.TB, collated map[string]json.RawMessage) backend.Batch {
	tb.Helper()

	field := func(name string) json.RawMessage {
		raw, ok := collated[name]
		if !ok {
			tb.Fatalf("collated has no %q", name)
		}
		return raw
	}
	return backend.Batch{
		InputIDs:      Matrix[int64](tb, field("input_ids"), "input_ids"),
		AttentionMask: Matrix[int64](tb, field("attention_mask"), "attention_mask"),
		MarkerPos:     Matrix[int64](tb, field("marker_pos"), "marker_pos"),
		MarkerMask:    Matrix[bool](tb, field("marker_mask"), "marker_mask"),
		QType:         Vector[int64](tb, field("qtype"), "qtype"),
	}
}
