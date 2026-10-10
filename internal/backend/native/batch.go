package native

import (
	"fmt"

	"github.com/MeKo-Christian/go-laya/backend"
)

// checkBatch requires b to be n×L token tensors, n×K marker tensors and n
// qtypes, with n, L and K all positive: the checks internal/backend/onnx's
// flatten (batch.go) makes, so both backends refuse the same batches.
// Encoder.Forward and Head.Forward check their own inputs too, but a ragged
// batch is the caller's error, and this names it as such.
func checkBatch(b backend.Batch) error {
	n := len(b.InputIDs)
	if n == 0 {
		return fmt.Errorf("%w: no rows", ErrBadBatch)
	}
	seq, k := len(b.InputIDs[0]), 0
	if len(b.MarkerPos) > 0 {
		k = len(b.MarkerPos[0])
	}
	if seq == 0 || k == 0 {
		return fmt.Errorf("%w: input_ids %d wide, marker_pos %d wide", ErrBadBatch, seq, k)
	}
	if len(b.QType) != n {
		return fmt.Errorf("%w: qtype has %d rows, input_ids %d", ErrBadBatch, len(b.QType), n)
	}
	if err := rectangular("input_ids", b.InputIDs, n, seq); err != nil {
		return err
	}
	if err := rectangular("attention_mask", b.AttentionMask, n, seq); err != nil {
		return err
	}
	if err := rectangular("marker_pos", b.MarkerPos, n, k); err != nil {
		return err
	}
	return rectangular("marker_mask", b.MarkerMask, n, k)
}

// rectangular requires m to be rows×cols.
func rectangular[T any](name string, m [][]T, rows, cols int) error {
	if len(m) != rows {
		return fmt.Errorf("%w: %s has %d rows, want %d", ErrBadBatch, name, len(m), rows)
	}
	for i, row := range m {
		if len(row) != cols {
			return fmt.Errorf("%w: %s[%d] is %d wide, want %d", ErrBadBatch, name, i, len(row), cols)
		}
	}
	return nil
}
