// Package question defines the typed questions a laya model answers: choice,
// score and noul.
//
// It is a leaf. It depends on jsonx and on internal/prompt and on nothing
// heavier, so a caller that only builds questions -- presets, for instance --
// never pulls in the ONNX Runtime binding (PLAN.md D9). The root package
// re-exports every type here under its documented name, so the public API is
// the one docs/API.md describes either way.
//
// The Go design collapses two of Python's dynamically typed shapes. A choice
// question's criteria are a dict *or* a list of bare labels upstream; here they
// are always a []ChoiceOption, with Labels for the list form. A noul question's
// criteria are a dict that may or may not carry "true" and "false" keys; here
// they are two named fields, which removes any chance of getting the index
// order wrong. Index 0 is false and index 1 is true, always.
package question

import (
	"errors"
	"fmt"
	"strconv"

	"github.com/MeKo-Christian/go-laya/internal/prompt"
	"github.com/MeKo-Christian/go-laya/jsonx"
)

// QType is a decision primitive. The integer values are baked into the
// checkpoint's type embedding (common.py:11) and must not be renumbered: a
// question that arrives under the wrong type index still produces confident
// probabilities, just for a different question.
type QType uint8

// The three decision primitives, in the numbering the checkpoints were trained
// with.
const (
	Choice QType = 0
	Score  QType = 1
	Noul   QType = 2
)

// String returns the upstream spelling of the type, as QTYPE_NAMES holds it.
func (t QType) String() string {
	switch t {
	case Choice:
		return prompt.TypeChoice
	case Score:
		return prompt.TypeScore
	case Noul:
		return prompt.TypeNoul
	default:
		return "QType(" + strconv.Itoa(int(t)) + ")"
	}
}

// Criterion is a criterion description. A string renders verbatim; nil and ""
// both mean "no description"; anything else renders as Python-compatible
// compact JSON. Use jsonx.Obj rather than a Go map for an object, so that the
// key order the model is prompted with is the one that was written.
//
// Note what is *not* blank: 0, false and 0.0 are legitimate criterion values
// and render as "zero: 0" and "no: false" (invariant #14).
type Criterion any

// Errors a question can report. ErrDuplicateQuestionID and ErrDuplicateOption
// both describe states a Python dict cannot reach and a Go slice can, which is
// why validation exists at all: upstream gets these for free from its
// container and the Go port has to check for them.
var (
	// ErrDuplicateQuestionID reports two questions submitted under one id.
	// Python's questions are a dict, so this is impossible there; last-wins
	// here would silently return fewer answers than the caller asked for.
	ErrDuplicateQuestionID = errors.New("laya: duplicate question id")

	// ErrDuplicateOption reports two choice options sharing a key. Python's
	// criteria are a dict, so the two collapse into one option there; keeping
	// both here would emit an extra option slot and shift every later answer
	// index by one.
	ErrDuplicateOption = errors.New("laya: duplicate choice option key")
)

// Question is one typed question. The interface is closed on purpose: it has an
// unexported method, so the only implementations are the three in this package.
// The set of question types is fixed by the checkpoint's type embedding, and a
// fourth one would have no logits to read.
type Question interface {
	// Type is the decision primitive, and the index into the type embedding.
	Type() QType
	// Instructions is the prompt text the question is asked with.
	Instructions() string
	// RenderOptions reproduces common.render_options: the option texts in
	// label-index order, which is the order the output logits are indexed by.
	RenderOptions() []string

	validate() error
}

// ChoiceOption is one option of a choice question: a key, which is both the
// label in the prompt and the key the probability comes back under, and an
// optional description.
type ChoiceOption struct {
	Key  string
	Desc Criterion // nil or "" renders the bare key
}

// Labels is the list form of Python's choice criteria: bare option names with
// no descriptions, which is what Agent._to_internal builds from a list at
// agent.py:233-234.
func Labels(keys ...string) []ChoiceOption {
	opts := make([]ChoiceOption, len(keys))
	for i, k := range keys {
		opts[i] = ChoiceOption{Key: k}
	}
	return opts
}

// ChoiceQuestion picks one option out of several. Opts is ordered, and the
// order is observable twice over: it is the order the options appear in the
// prompt and the order the probabilities come back in.
type ChoiceQuestion struct {
	Ins  string
	Opts []ChoiceOption
}

// Type returns Choice.
func (q ChoiceQuestion) Type() QType { return Choice }

// Instructions returns the prompt text.
func (q ChoiceQuestion) Instructions() string { return q.Ins }

// RenderOptions renders "<key>" for an option with no description and
// "<key>: <rendered>" otherwise (invariant #14).
func (q ChoiceQuestion) RenderOptions() []string { return prompt.RenderOptions(q.internal()) }

func (q ChoiceQuestion) internal() prompt.Internal {
	crit := make(jsonx.Obj, len(q.Opts))
	for i, o := range q.Opts {
		crit[i] = jsonx.Field{Key: o.Key, Value: o.Desc}
	}
	return prompt.Internal{T: prompt.TypeChoice, Ins: q.Ins, Crit: crit}
}

// validate rejects only what Python's own container would have rejected for
// it. An empty option list is legal upstream and stays legal here.
func (q ChoiceQuestion) validate() error {
	seen := make(map[string]struct{}, len(q.Opts))
	for _, o := range q.Opts {
		if _, dup := seen[o.Key]; dup {
			return fmt.Errorf("%w: %q", ErrDuplicateOption, o.Key)
		}
		seen[o.Key] = struct{}{}
	}
	return nil
}

// ScoreQuestion rates state on an ordered scale. Levels[i] describes level i,
// counted from 0, and the index is the answer.
type ScoreQuestion struct {
	Ins    string
	Levels []Criterion
}

// Type returns Score.
func (q ScoreQuestion) Type() QType { return Score }

// Instructions returns the prompt text.
func (q ScoreQuestion) Instructions() string { return q.Ins }

// RenderOptions renders "level %d: %s" per level, with the index from 0
// (invariant #15).
func (q ScoreQuestion) RenderOptions() []string { return prompt.RenderOptions(q.internal()) }

func (q ScoreQuestion) internal() prompt.Internal {
	crit := make([]any, len(q.Levels))
	for i, l := range q.Levels {
		crit[i] = l
	}
	return prompt.Internal{T: prompt.TypeScore, Ins: q.Ins, Crit: crit}
}

func (q ScoreQuestion) validate() error { return nil }

// NoulQuestion asks whether a statement holds. It always has exactly two
// options and they are always in the order false, true -- the fields are named
// rather than keyed so that no caller can reverse them (invariant #16).
type NoulQuestion struct {
	Ins   string
	False Criterion // nil or "" renders "no, the statement does not hold"
	True  Criterion // nil or "" renders "yes, the statement holds"
}

// Type returns Noul.
func (q NoulQuestion) Type() QType { return Noul }

// Instructions returns the prompt text.
func (q NoulQuestion) Instructions() string { return q.Ins }

// RenderOptions renders exactly two options, false first, falling back to the
// upstream default texts where a criterion is absent (invariant #16).
func (q NoulQuestion) RenderOptions() []string { return prompt.RenderOptions(q.internal()) }

func (q NoulQuestion) internal() prompt.Internal {
	// The key order here is not the render order -- renderNoul reads the two by
	// name. It matches upstream's own dict for readability, nothing more.
	crit := jsonx.Obj{
		{Key: "false", Value: q.False},
		{Key: "true", Value: q.True},
	}
	return prompt.Internal{T: prompt.TypeNoul, Ins: q.Ins, Crit: crit}
}

func (q NoulQuestion) validate() error { return nil }

// NamedQuestion pairs a question with the id its answer comes back under.
type NamedQuestion struct {
	ID string
	Q  Question
}

// Questions is an ordered set of questions. It is a slice and not a map
// because the order is observable: it is the order of the rows in the batch and
// the order of the answers that come back.
type Questions []NamedQuestion

// IDs returns the question ids in definition order.
func (qs Questions) IDs() []string {
	ids := make([]string, len(qs))
	for i, nq := range qs {
		ids[i] = nq.ID
	}
	return ids
}

// Get returns the first question stored under id. First rather than last
// because Validate rejects duplicates, so there can only be one.
func (qs Questions) Get(id string) (Question, bool) {
	for _, nq := range qs {
		if nq.ID == id {
			return nq.Q, true
		}
	}
	return nil, false
}

// Validate reports the first problem in the set: a duplicate id, a missing
// question, or a per-question failure. It exists because Python gets all three
// checks from its dict container and a Go slice gets none of them.
func (qs Questions) Validate() error {
	seen := make(map[string]struct{}, len(qs))
	for _, nq := range qs {
		if _, dup := seen[nq.ID]; dup {
			return fmt.Errorf("%w: %q", ErrDuplicateQuestionID, nq.ID)
		}
		seen[nq.ID] = struct{}{}

		if nq.Q == nil {
			return fmt.Errorf("laya: question %q is nil; it would panic at render "+
				"time, long after the caller could act on it", nq.ID)
		}
		if err := nq.Q.validate(); err != nil {
			return fmt.Errorf("laya: question %q: %w", nq.ID, err)
		}
	}
	return nil
}
