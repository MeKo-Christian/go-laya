package laya

import "github.com/MeKo-Christian/go-laya/question"

// The question types live in their own leaf package and are re-exported here
// under the names docs/API.md gives them, so `laya.ChoiceQuestion` means what
// it always meant.
//
// The split is forced by D9 (PLAN.md:237). `presets` builds questions, and
// `go list -deps ./presets` has to stay free of the ONNX Runtime binding --
// which the root package is allowed to pull in and eventually will. Questions
// defined at the root would put every preset one import edge away from it.
// Aliases rather than wrappers, so laya.Question and question.Question are the
// same type and a value crosses the boundary unchanged.

// QType is a decision primitive. The integer values are baked into the
// checkpoint's type embedding and must not be renumbered.
type QType = question.QType

// The three decision primitives, in the numbering the checkpoints were trained
// with.
const (
	Choice = question.Choice
	Score  = question.Score
	Noul   = question.Noul
)

// Criterion is a criterion description: a string renders verbatim, nil and ""
// mean "no description", anything else renders as Python-compatible compact
// JSON.
type Criterion = question.Criterion

// Question is one typed question: a ChoiceQuestion, a ScoreQuestion or a
// NoulQuestion. The interface is closed, because the set of question types is
// fixed by the checkpoint.
type Question = question.Question

// ChoiceOption is one option of a choice question.
type ChoiceOption = question.ChoiceOption

// ChoiceQuestion picks one option out of several, in the order they are given.
type ChoiceQuestion = question.ChoiceQuestion

// ScoreQuestion rates state on an ordered scale, level 0 upwards.
type ScoreQuestion = question.ScoreQuestion

// NoulQuestion asks whether a statement holds; false is always index 0.
type NoulQuestion = question.NoulQuestion

// NamedQuestion pairs a question with the id its answer comes back under.
type NamedQuestion = question.NamedQuestion

// Questions is an ordered set of questions; the order is the answer order.
type Questions = question.Questions

// Labels is the list form of a choice question's criteria: bare option names
// with no descriptions.
func Labels(keys ...string) []ChoiceOption { return question.Labels(keys...) }
