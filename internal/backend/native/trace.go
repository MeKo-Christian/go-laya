package native

import (
	"context"
	"fmt"

	"github.com/MeKo-Christian/go-laya/backend"
	"github.com/MeKo-Christian/go-laya/internal/head"
)

// forwardTrace is Forward, returning the head's stages (head.Trace) instead
// of its two outputs: h after the type embedding and after each head layer,
// the gathered marker rows, the filled logits, their softmax and the act
// features, from which Trace.Logits and Trace.Act are Forward's outputs.
//
// It exists for the gated tests of invariants #35-38, which hold those
// stages on the real checkpoints to the ones PyTorch recorded in
// testdata/head_intermediates.jsonl (Task 8.8), and Forward has no place to
// return them. It repeats Forward's steps and context checks, with
// head.ForwardTrace, which shares head.Forward's code, in place of
// head.Forward; the tests check that both return the same outputs. Forward
// is left as it is because its PR (#49) is still under review; once it has
// landed, the two can share one helper.
func (b *Backend) forwardTrace(ctx context.Context, in backend.Batch) (*head.Trace, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	b.mu.RLock()
	defer b.mu.RUnlock()
	if b.m == nil {
		return nil, ErrClosed
	}
	if err := checkBatch(in); err != nil {
		return nil, err
	}
	h, err := b.m.enc.Forward(in.InputIDs, in.AttentionMask)
	if err != nil {
		return nil, fmt.Errorf("%w: encoder: %w", ErrBadBatch, err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	tr, err := b.m.head.ForwardTrace(h, in)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrBadBatch, err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return tr, nil
}
