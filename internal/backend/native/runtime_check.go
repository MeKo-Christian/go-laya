package native

import (
	"fmt"

	"github.com/MeKo-Christian/go-laya/backend"
	"github.com/MeKo-Christian/go-laya/internal/head"
	"github.com/MeKo-Christian/go-laya/internal/safetensors"
)

// HeadShape is what a checkpoint's model.safetensors fixes of the decision
// head, and Open builds: the head's layer count and its act width.
// Upstream takes both from rl_agent_config.json instead (head_layers and
// len(act_costs)+1, common.py:137) and its strict load_state_dict refuses a
// file that disagrees, so a caller passes them as Options.Head.
type HeadShape struct {
	// Layers is how many head.layers.N the file numbers.
	Layers int
	// ActWidth is act_head.2.weight's row count, the act logits per row.
	ActWidth int
}

// HeadMismatchError is Open's error for a model.safetensors whose head is
// not the Options.Head the caller expects. It wraps
// backend.ErrIncompatibleCheckpoint.
type HeadMismatchError struct {
	// Path is the weights file's.
	Path string
	// Got is the head the file fixes, Want the caller's Options.Head.
	Got, Want HeadShape
}

func (e *HeadMismatchError) Error() string {
	return fmt.Sprintf("native backend: %s has %d head layers and %d act logits, want %d and %d: %v",
		e.Path, e.Got.Layers, e.Got.ActWidth, e.Want.Layers, e.Want.ActWidth, backend.ErrIncompatibleCheckpoint)
}

// Unwrap makes the error backend.ErrIncompatibleCheckpoint to errors.Is.
func (e *HeadMismatchError) Unwrap() error { return backend.ErrIncompatibleCheckpoint }

// checkHead holds the head that f, the weights file at path, fixes to want.
// It reads the header alone: the head layer count from the names, as load
// counts them, and the act width from act_head.2.weight's shape, as
// checkNames takes it. An act_head.2.weight that is missing or not of rank
// 2 wraps backend.ErrIncompatibleCheckpoint.
func checkHead(f *safetensors.File, path string, want HeadShape) error {
	const actName = "act_head.2.weight"
	info, ok := f.Info(actName)
	if !ok || len(info.Shape) != 2 {
		return fmt.Errorf("native backend: %s: tensor %q is missing or not [n_act, %d]: %w",
			path, actName, head.ActHidden, backend.ErrIncompatibleCheckpoint)
	}
	got := HeadShape{
		Layers:   len(layerIndices(f.Names(), headLayers)),
		ActWidth: int(info.Shape[0]),
	}
	if got != want {
		return &HeadMismatchError{Path: path, Got: got, Want: want}
	}
	return nil
}
