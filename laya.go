// Package laya is a Go port of the Python laya decision engine: a
// non-autoregressive System 1 model that answers typed questions (choice,
// score, noul) about arbitrary state in a single forward pass, returning
// calibrated probabilities rather than generated text.
//
// The port is inference only. The reinforcement-learning training math in the
// Python original has no counterpart here; see NOTICE for the full statement
// of modification.
package laya

import (
	"github.com/MeKo-Christian/go-laya/jsonx"
	"github.com/MeKo-Christian/go-laya/lang"
)

// Version is the single source of truth for this module's version. The release
// workflow asserts that a v-prefixed tag matches it exactly, so the constant and
// the tag can never drift. The Python original kept its version in three files;
// this is deliberately the only copy.
const Version = "0.1.0"

// Obj is an ordered JSON object, re-exported from jsonx so that callers of the
// Map methods on this package's types need not import it separately. Order is
// observable: it is the order Python's json.dumps writes, and the bytes are
// what the model is prompted with.
type Obj = jsonx.Obj

// Detection is the script and language analysis of a state, re-exported from
// lang. It is the `detection` block of a routing payload.
//
// An alias rather than a second struct: lang owns the type, and redeclaring it
// here would put every routing caller one conversion away from the package
// that produces it.
type Detection = lang.Detection

// Agent is a loaded checkpoint, as far as the Router is concerned.
//
// It is an interface, and today it carries only Close, because that is the
// whole of what the Router needs: it caches agents, evicts the least recently
// used, and must release the one it drops. The concrete agent -- the encoder,
// the head, the tokenizer -- arrives in M6/M7 and will widen this interface
// with SystemOne (PLAN.md D13).
//
// Modelling it as an interface now is also what makes the upstream LRU tests
// portable: they never build a real agent either, they pass a stub
// (test_router.py:167-172), and WithLoader is where it goes in.
type Agent interface {
	// Close releases the agent's backend. A leaked ONNX Runtime session is
	// hundreds of megabytes, so eviction calls it.
	Close() error
}
