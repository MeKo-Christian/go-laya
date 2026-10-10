package laya

import (
	"errors"
	"fmt"

	"github.com/MeKo-Christian/go-laya/backend"
	"github.com/MeKo-Christian/go-laya/internal/backend/onnx"
	"github.com/MeKo-Christian/go-laya/question"
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

	// ErrUnsupportedPlatform reports a platform the ONNX backend does not
	// build for: outside D21's list, Open fails with it. Python runs wherever
	// PyTorch does. It is the backend's own sentinel, which lives in an
	// internal package, so errors.Is matches either name.
	ErrUnsupportedPlatform = onnx.ErrUnsupportedPlatform

	// ErrUnknownDevice reports a WithDevice or WithRouterDevice value that
	// names no device, such as Python's "mps". It is never a fallback to the
	// CPU, so a typo fails at load time. It is the backend's own sentinel, as
	// ErrUnsupportedPlatform is.
	ErrUnknownDevice = onnx.ErrUnknownDevice

	// ErrEmptyQuestions reports a SystemOne call with no questions. Python
	// fails with a TypeError when collate_items returns None (agent.py:266).
	ErrEmptyQuestions = errors.New("laya: no questions")

	// ErrDuplicateQuestionID reports two questions under one id, which a
	// Python dict cannot hold. It is question.ErrDuplicateQuestionID.
	ErrDuplicateQuestionID = question.ErrDuplicateQuestionID

	// ErrOptionsExceedHeadBudget reports a question whose options did not all
	// fit into max_len, so some lost their marker. It is what an
	// *OptionBudgetError unwraps to.
	ErrOptionsExceedHeadBudget = errors.New("laya: question options exceed head_max_len")

	// ErrNoOptions reports a question that renders no options at all: a
	// choice question with an empty option list. Python gets as far as the
	// softmax and fails on an empty array (agent.py:306).
	ErrNoOptions = errors.New("laya: question has no options")

	// ErrInvalidLimits reports a max_len or head_max_len that is not
	// positive, passed to SetLimits or WithLimits. Python accepts either.
	// A max_len ≤ 0 then leaves build_sequence no option marker
	// (common.py:86), so a question with options raises a ValueError that
	// blames head_max_len (agent.py:262-263); a head_max_len ≤ 0 silently
	// cuts every option to 4 tokens, its mask included, and the question
	// head to 8 (common.py:70-75). The config's own values are held to the
	// same rule.
	ErrInvalidLimits = errors.New("laya: max_len and head_max_len must be positive")

	// ErrInvalidRevision reports a WithRevision or WithRouterRevision value
	// that is not safe to put into a Hub URL and a cache path: an absolute
	// path, a "." or ".." segment, an empty segment or a backslash. Open and
	// NewRouter refuse it as the option is applied.
	ErrInvalidRevision = errors.New("laya: invalid revision")

	// ErrUnknownRuntime reports a WithRuntime or WithRouterRuntime value that
	// is neither RuntimeONNX nor RuntimeNative.
	ErrUnknownRuntime = errors.New("laya: unknown runtime")

	// ErrConflictingOptions reports options that cannot apply together, one
	// of which would be silently ignored. For Open: WithBackend beside
	// WithGraph or WithDevice, which choose the graph and the device of the
	// ONNX backend WithBackend replaces, or beside any WithRuntime, which
	// selects the runtime it replaces; and WithRuntime(RuntimeNative) beside
	// WithGraph or a WithDevice other than "", "auto" and "cpu", since the
	// native backend runs no export and only on the CPU. For NewRouter:
	// WithRouterRuntime(RuntimeNative) beside WithONNXDir or such a
	// WithRouterDevice. WithLogger beside WithBackend is accepted and has
	// nothing to log, since only the ONNX backend warns. The message names
	// the options.
	ErrConflictingOptions = errors.New("laya: conflicting options")
)

// OptionBudgetError names the question whose options lost their markers,
// replacing Python's ValueError("question %r options exceed head_max_len=%d")
// at agent.py:262-263. HeadMaxLen is the value upstream names in the message;
// the markers are lost to max_len, which bounds the whole sequence.
type OptionBudgetError struct {
	QuestionID  string
	HeadMaxLen  int
	WantMarkers int
	GotMarkers  int
}

// Error words it as agent.py:263 does, with the marker counts after it.
func (e *OptionBudgetError) Error() string {
	return fmt.Sprintf("laya: question %q options exceed head_max_len=%d (%d of %d options kept a marker)",
		e.QuestionID, e.HeadMaxLen, e.GotMarkers, e.WantMarkers)
}

// Unwrap returns ErrOptionsExceedHeadBudget.
func (*OptionBudgetError) Unwrap() error { return ErrOptionsExceedHeadBudget }
