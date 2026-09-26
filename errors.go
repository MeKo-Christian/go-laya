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
	// loader to build it with. Python builds an Agent inline
	// (router.py:175-178); until M6 provides that, a loader has to be
	// injected with WithLoader. Failing here beats caching a nil agent that
	// panics at the first use.
	ErrNoLoader = errors.New("laya: no agent loader configured")

	// ErrIncompatibleCheckpoint reports a checkpoint that is not a laya
	// decision model, wrapped with what was wrong. Python raises ValueError
	// from _verify_compatibility (agent.py:49-93). It is
	// backend.ErrIncompatibleCheckpoint, so errors.Is matches either name.
	ErrIncompatibleCheckpoint = backend.ErrIncompatibleCheckpoint
)
