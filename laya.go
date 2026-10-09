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
	"context"

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

// Predictor is a loaded checkpoint as the Router sees it: what it caches, runs
// and releases. The default loader builds one per checkpoint; a test double or
// another runtime can stand in for it through WithLoader.
//
// It is an interface rather than a struct because the Router needs two things
// of an agent and nothing else: to answer questions (Router.Predict runs
// SystemOne) and to release its backend when evicted. That keeps the upstream
// LRU tests portable -- they never build a real agent either, they pass a
// stub (test_router.py:167-172), and WithLoader is where it goes in. Widening
// it breaks third-party implementations, which D13 accepts before 1.0.
type Predictor interface {
	// SystemOne answers every question about state in one forward pass:
	// agent.system_one (agent.py:240-343). The answers come back in question
	// order; Routing is nil, since no router was involved.
	//
	// It must not unload, close or evict its own agent through the Router
	// running it: Router.Predict holds a lease on the agent for the length
	// of this call, and dropping it waits for that lease (D27).
	SystemOne(ctx context.Context, state any, qs Questions) (*Result, error)

	// Close releases the agent's backend. A leaked ONNX Runtime session is
	// hundreds of megabytes, so eviction calls it.
	Close() error
}
