// Package laya is a Go port of the Python laya decision engine: a
// non-autoregressive System 1 model that answers typed questions (choice,
// score, noul) about arbitrary state in a single forward pass, returning
// calibrated probabilities rather than generated text.
//
// The port is inference only. The reinforcement-learning training math in the
// Python original has no counterpart here; see NOTICE for the full statement
// of modification.
package laya

// Version is the single source of truth for this module's version. The release
// workflow asserts that a v-prefixed tag matches it exactly, so the constant and
// the tag can never drift. The Python original kept its version in three files;
// this is deliberately the only copy.
const Version = "0.1.0"
