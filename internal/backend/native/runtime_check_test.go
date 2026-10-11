package native

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/MeKo-Christian/go-laya/backend"
	"github.com/MeKo-Christian/go-laya/internal/head"
)

// headVariants are checkpoints that differ only in their head: the head
// layer count and the act width the file fixes, and Open builds.
func headVariants() []struct {
	name  string
	s     *synth
	shape HeadShape
} {
	oneLayer := newSynth(1)
	for name := range oneLayer.tensors {
		if strings.HasPrefix(name, "head.layers.1.") {
			delete(oneLayer.tensors, name)
		}
	}
	noHead := newSynth(1)
	for name := range noHead.tensors {
		if strings.HasPrefix(name, "head.layers.") {
			delete(noHead.tensors, name)
		}
	}
	fiveAct := newSynth(1)
	fiveAct.zeros("act_head.2.weight", 5, head.ActHidden)
	fiveAct.zeros("act_head.2.bias", 5)

	return []struct {
		name  string
		s     *synth
		shape HeadShape
	}{
		{"tiny", newSynth(1), HeadShape{Layers: tinyHeadLayers, ActWidth: tinyAct}},
		{"one head layer", oneLayer, HeadShape{Layers: 1, ActWidth: tinyAct}},
		{"no head layers", noHead, HeadShape{Layers: 0, ActWidth: tinyAct}},
		{"five act logits", fiveAct, HeadShape{Layers: tinyHeadLayers, ActWidth: 5}},
	}
}

// Task 8.10 (PR #50's review): Options.Head holds the head that
// model.safetensors fixes to the one the caller expects, on the very file
// Open decodes, so a file replaced between a separate check and the load
// cannot slip past it. A file that agrees opens and answers with as many
// act logits as expected; one that disagrees in either half is a
// HeadMismatchError carrying both shapes.
func TestOpenExpectedHead(t *testing.T) {
	for _, tc := range headVariants() {
		t.Run(tc.name, func(t *testing.T) {
			dir := tc.s.write(t)
			want := tc.shape
			b, err := Open(t.Context(), dir, Options{Head: &want})
			if err != nil {
				t.Fatalf("Open expecting %+v: %v", want, err)
			}
			_, act, err := b.Forward(t.Context(), tinyBatch())
			_ = b.Close()
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
			if len(act[0]) != want.ActWidth {
				t.Errorf("Forward gives %d act logits, want %d", len(act[0]), want.ActWidth)
			}

			for _, other := range []HeadShape{
				{Layers: want.Layers + 1, ActWidth: want.ActWidth},
				{Layers: want.Layers, ActWidth: want.ActWidth + 1},
			} {
				b, err := Open(t.Context(), dir, Options{Head: &other})
				var hm *HeadMismatchError
				if !errors.As(err, &hm) || !errors.Is(err, backend.ErrIncompatibleCheckpoint) || b != nil {
					t.Fatalf("Open expecting %+v = %v, %v; want a HeadMismatchError wrapping ErrIncompatibleCheckpoint", other, b, err)
				}
				if hm.Got != want || hm.Want != other || !strings.HasSuffix(hm.Path, weightsFile) {
					t.Errorf("HeadMismatchError = %+v, want Got %+v, Want %+v and the weights' path", *hm, want, other)
				}
			}
		})
	}
}

// errAfterFirst is a context whose Err is nil the first time it is asked
// and context.Canceled after that: Open checks ctx once on entry, and every
// tensor decode checks it again.
type errAfterFirst struct {
	context.Context //nolint:containedctx // a wrapper: it overrides Err alone

	calls atomic.Int32
}

func (c *errAfterFirst) Err() error {
	if c.calls.Add(1) == 1 {
		return nil
	}
	return context.Canceled
}

// Task 8.10: the expected head is held to the file before a weight is
// decoded, the 0.8-1.7 GB a shipped checkpoint holds. Any decode would ask
// ctx and see the cancellation; the mismatch must be what Open reports.
func TestOpenExpectedHeadBeforeDecode(t *testing.T) {
	dir := newSynth(1).write(t)
	other := HeadShape{Layers: tinyHeadLayers + 1, ActWidth: tinyAct}
	ctx := &errAfterFirst{Context: t.Context()}
	_, err := Open(ctx, dir, Options{Head: &other})
	var hm *HeadMismatchError
	if !errors.As(err, &hm) {
		t.Fatalf("Open = %v, want the HeadMismatchError before any tensor is decoded", err)
	}

	// The control: without Options.Head the same context stops the decode.
	ctx = &errAfterFirst{Context: t.Context()}
	if _, err := Open(ctx, dir, Options{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("Open without Head = %v, want context.Canceled from the first decode", err)
	}
}

// With Options.Head, a file Open cannot read the head of fails: no weights
// file, or no act_head.2.weight of rank 2.
func TestOpenExpectedHeadUnreadable(t *testing.T) {
	noAct := newSynth(1)
	delete(noAct.tensors, "act_head.2.weight")
	rank1 := newSynth(1)
	rank1.zeros("act_head.2.weight", tinyAct)
	noWeights := newSynth(1).write(t)
	if err := os.Remove(filepath.Join(noWeights, weightsFile)); err != nil {
		t.Fatal(err)
	}

	want := HeadShape{Layers: tinyHeadLayers, ActWidth: tinyAct}
	for _, tc := range []struct {
		name string
		dir  string
		says string
	}{
		{"no weights file", noWeights, weightsFile},
		{"no act_head.2.weight", noAct.write(t), "act_head.2.weight"},
		{"act_head.2.weight of rank 1", rank1.write(t), "act_head.2.weight"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b, err := Open(t.Context(), tc.dir, Options{Head: &want})
			if !errors.Is(err, backend.ErrIncompatibleCheckpoint) || !strings.Contains(err.Error(), tc.says) || b != nil {
				t.Fatalf("Open = %v, %v; want ErrIncompatibleCheckpoint naming %s", b, err, tc.says)
			}
		})
	}
}
