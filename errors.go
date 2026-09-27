package laya

import (
	"errors"

	"github.com/MeKo-Christian/go-laya/backend"
)

// The sentinel errors the public API returns. `.golangci.yml` disables err113
// on the understanding that dynamic context is wrapped around one of these
// rather than returned bare, so every error a caller might branch on lives
// here.
var (
	// ErrUnknownModel reports a model name that is neither one of the three
	// checkpoints nor one of their aliases. Python raises ValueError from
	// normalise_name (router.py:104-106).
	ErrUnknownModel = errors.New("laya: unknown model")

	// ErrNoLoader reports a Router asked to load a checkpoint without a
	// loader to build it with, which only WithLoader(nil) arranges: NewRouter
	// otherwise installs the default loader. Failing here beats caching a nil
	// agent that panics at the first use.
	ErrNoLoader = errors.New("laya: no agent loader configured")

	// ErrCheckpointNotFound reports a checkpoint location that does not exist:
	// a local path, a subfolder missing from a checkpoint directory or outside
	// it, or a Hub repo or subfolder with no files (wrapping hub's own
	// not-found error). Python raises FileNotFoundError (agent.py:117-121,
	// 131-135).
	ErrCheckpointNotFound = errors.New("laya: checkpoint not found")

	// ErrNoGraph reports a checkpoint with no ONNX export where the default
	// loader looks for one. The Hub checkpoints carry safetensors, which only
	// torch runs, so the graph is exported locally with
	// scripts/export_onnx.py (D24). The message names the path and the
	// command.
	ErrNoGraph = errors.New("laya: no ONNX export for the checkpoint")

	// ErrIncompatibleCheckpoint reports a checkpoint that is not a laya
	// decision model, wrapped with what was wrong. Python raises ValueError
	// from _verify_compatibility (agent.py:49-93). It is
	// backend.ErrIncompatibleCheckpoint, so errors.Is matches either name.
	ErrIncompatibleCheckpoint = backend.ErrIncompatibleCheckpoint
)
