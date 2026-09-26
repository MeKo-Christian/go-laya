package onnx

import (
	"errors"
	"fmt"
	"log/slog"

	"github.com/MeKo-Christian/go-laya/backend"
)

// APIVersion is the only ONNX Runtime C API version the binding implements
// (onnxruntime/runtime.go: `supportedAPIVersions = []uint32{23}`), i.e. ONNX
// Runtime 1.23.x. Checking the loaded library's own version is Task 6.6.2.
const APIVersion = 23

var (
	// ErrClosed is Forward's error on a Backend that was closed or never
	// opened.
	ErrClosed = errors.New("onnx backend: closed")

	// ErrBadBatch is Forward's error for a batch whose tensors are not the
	// rectangular, mutually consistent shapes the graph takes.
	ErrBadBatch = errors.New("onnx backend: malformed batch")

	// ErrUnsupportedPlatform is Open's error on targets the binding does not
	// compile for; backend.go's build constraint lists the ones it does
	// (PLAN.md Task 6.7).
	ErrUnsupportedPlatform = errors.New("onnx backend: not supported on this platform")
)

// Options configures Open.
type Options struct {
	// Library is the ONNX Runtime shared library to load. Empty resolves it
	// through LAYA_ORT_LIB, ORT_LIBRARY_PATH and the platform's default
	// install locations, in that order.
	Library string

	// IntraOpThreads is ORT's intra-op thread count; 0 leaves it to ORT.
	// Spike S3 measured the physical core count as the right setting (D6),
	// but picking it is the caller's job.
	IntraOpThreads int

	// Device is "cpu", "cuda", "cuda:N", "coreml", or "" / "auto" for the
	// best one the library can use. A requested device that cannot be used,
	// or whose session fails to build, falls back to the CPU with one
	// warning on Logger; "auto" and "cpu" never warn. Anything else is
	// ErrUnknownDevice. The binding cannot enable CUDA yet (PLAN.md Task
	// 6.3.6), so "cuda" currently always falls back.
	Device string

	// Logger receives the fallback warnings. Nil means slog.Default().
	Logger *slog.Logger
}

// The graph's declared inputs and outputs, as scripts/export_onnx.py names
// them.
var (
	graphInputs  = []string{"input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"}
	graphOutputs = []string{"logits", "act_logits"}
)

// flat is a Batch as five row-major buffers plus the dimensions they fold
// back into: rows questions, seq tokens, k marker columns.
type flat struct {
	rows, seq, k int

	inputIDs, attention, markerPos []int64
	markerMask                     []bool
	qtype                          []int64
}

// flatten checks that b is n×L token tensors, n×K marker tensors and n qtypes,
// with n, L and K all positive, and lays them out row-major. The graph accepts
// any buffer whose length matches the shape it is given, so a ragged row would
// otherwise shift every later cell silently rather than fail.
func flatten(b backend.Batch) (flat, error) {
	n := len(b.InputIDs)
	if n == 0 {
		return flat{}, fmt.Errorf("%w: no rows", ErrBadBatch)
	}
	f := flat{rows: n, seq: len(b.InputIDs[0])}
	if len(b.MarkerPos) > 0 {
		f.k = len(b.MarkerPos[0])
	}
	if f.seq == 0 || f.k == 0 {
		return flat{}, fmt.Errorf("%w: input_ids %d wide, marker_pos %d wide", ErrBadBatch, f.seq, f.k)
	}
	if len(b.QType) != n {
		return flat{}, fmt.Errorf("%w: qtype has %d rows, input_ids %d", ErrBadBatch, len(b.QType), n)
	}

	var err error
	if f.inputIDs, err = rowMajor("input_ids", b.InputIDs, n, f.seq); err != nil {
		return flat{}, err
	}
	if f.attention, err = rowMajor("attention_mask", b.AttentionMask, n, f.seq); err != nil {
		return flat{}, err
	}
	if f.markerPos, err = rowMajor("marker_pos", b.MarkerPos, n, f.k); err != nil {
		return flat{}, err
	}
	if f.markerMask, err = rowMajor("marker_mask", b.MarkerMask, n, f.k); err != nil {
		return flat{}, err
	}
	f.qtype = b.QType
	return f, nil
}

func rowMajor[T any](name string, m [][]T, rows, cols int) ([]T, error) {
	if len(m) != rows {
		return nil, fmt.Errorf("%w: %s has %d rows, want %d", ErrBadBatch, name, len(m), rows)
	}
	out := make([]T, 0, rows*cols)
	for i, row := range m {
		if len(row) != cols {
			return nil, fmt.Errorf("%w: %s[%d] is %d wide, want %d", ErrBadBatch, name, i, len(row), cols)
		}
		out = append(out, row...)
	}
	return out, nil
}

// unflatten folds one of the graph's float outputs back into rows, checking
// that it really is rows×w for some w and that data holds exactly that many
// values. Each row gets its own backing array.
func unflatten(name string, data []float32, shape []int64, rows int) ([][]float32, error) {
	if len(shape) != 2 || shape[0] != int64(rows) || shape[1] < 0 {
		return nil, fmt.Errorf("onnx backend: %s has shape %v, want [%d w]", name, shape, rows)
	}
	w := int(shape[1])
	if len(data) != rows*w {
		return nil, fmt.Errorf("onnx backend: %s shape %v holds %d values, got %d", name, shape, rows*w, len(data))
	}
	out := make([][]float32, rows)
	for i := range out {
		out[i] = append(make([]float32, 0, w), data[i*w:(i+1)*w]...)
	}
	return out, nil
}
