// Package backend is the seam between laya and whatever runs the model: the
// Backend interface and the Batch it consumes.
//
// It is a public leaf with no dependency beyond the standard library (PLAN.md
// D9), so a caller can supply their own runtime through it without importing
// the ONNX binding the default implementation uses.
package backend

import (
	"context"
	"errors"
)

// ErrIncompatibleCheckpoint reports a model that is not a laya decision model
// of the shape this package drives: a missing input or output, a missing
// config key, a head of the wrong width. Implementations wrap it with what
// was wrong. It lives here rather than in the root package so that a Backend,
// which the root package imports, can return it; laya.ErrIncompatibleCheckpoint
// is the same value.
var ErrIncompatibleCheckpoint = errors.New("laya: incompatible checkpoint")

// Backend runs the decision model's forward pass.
type Backend interface {
	// Forward returns, per row of in, the option logits (one per marker
	// column, kmax wide) and the act logits the calibrator reads.
	Forward(ctx context.Context, in Batch) (logits [][]float32, act [][]float32, err error)

	// Close releases the runtime session. The Backend is unusable afterwards.
	Close() error
}

// Batch is the collated input of one forward pass: the five tensors
// common.collate_items builds (common.py:218-251) and the exported graph
// declares as inputs. Every row describes one question; the rows are padded
// to a common width, so the masks say which cells are real.
type Batch struct {
	// InputIDs is the token id matrix, n rows by L columns, right-padded
	// with the tokenizer's pad id. AttentionMask is 1 over real tokens and 0
	// over padding, with the same shape.
	InputIDs, AttentionMask [][]int64

	// MarkerPos holds, per row, the token index of each option's mask token,
	// zero-filled to kmax columns.
	MarkerPos [][]int64

	// MarkerMask is true over a row's real markers and false over the fill.
	MarkerMask [][]bool

	// QType is each row's question type: 0 choice, 1 score, 2 noul.
	QType []int64
}
