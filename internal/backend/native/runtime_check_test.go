package native

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/MeKo-Christian/go-laya/backend"
	"github.com/MeKo-Christian/go-laya/internal/head"
)

// Task 8.10 (PR #49's review finding 2): ReadHeadShape reads, from the
// header alone, the head layer count and the act width Open builds, so the
// loader can hold them to rl_agent_config.json before decoding a weight.
func TestReadHeadShape(t *testing.T) {
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

	for _, tc := range []struct {
		name string
		s    *synth
		want HeadShape
	}{
		{"tiny", newSynth(1), HeadShape{Layers: tinyHeadLayers, ActWidth: tinyAct}},
		{"one head layer", oneLayer, HeadShape{Layers: 1, ActWidth: tinyAct}},
		{"no head layers", noHead, HeadShape{Layers: 0, ActWidth: tinyAct}},
		{"five act logits", fiveAct, HeadShape{Layers: tinyHeadLayers, ActWidth: 5}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := tc.s.write(t)
			got, err := ReadHeadShape(dir)
			if err != nil {
				t.Fatalf("ReadHeadShape: %v", err)
			}
			if got != tc.want {
				t.Fatalf("ReadHeadShape = %+v, want %+v", got, tc.want)
			}

			// What Open builds from the same file: as many act logits per
			// row as ReadHeadShape says.
			b, err := Open(context.Background(), dir, Options{})
			if err != nil {
				t.Fatalf("Open: %v", err)
			}
			defer b.Close()
			_, act, err := b.Forward(context.Background(), tinyBatch())
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
			if len(act[0]) != got.ActWidth {
				t.Errorf("Forward gives %d act logits, ReadHeadShape %d", len(act[0]), got.ActWidth)
			}
		})
	}
}

// ReadHeadShape fails, as Open would, on a checkpoint it cannot read the
// head of: no weights file, or no act_head.2.weight of rank 2.
func TestReadHeadShapeErrors(t *testing.T) {
	noAct := newSynth(1)
	delete(noAct.tensors, "act_head.2.weight")
	rank1 := newSynth(1)
	rank1.zeros("act_head.2.weight", tinyAct)

	for _, tc := range []struct {
		name string
		dir  string
	}{
		{"no weights file", t.TempDir()},
		{"no act_head.2.weight", noAct.write(t)},
		{"act_head.2.weight of rank 1", rank1.write(t)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ReadHeadShape(tc.dir)
			if err == nil {
				t.Fatalf("ReadHeadShape = %+v, want an error", got)
			}
			if !errors.Is(err, backend.ErrIncompatibleCheckpoint) {
				t.Errorf("ReadHeadShape: %v, want ErrIncompatibleCheckpoint", err)
			}
		})
	}
}
