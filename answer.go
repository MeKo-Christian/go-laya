package laya

import (
	"fmt"
	"strconv"

	"github.com/MeKo-Christian/go-laya/internal/calib"
	"github.com/MeKo-Christian/go-laya/jsonx"
	"github.com/MeKo-Christian/go-laya/question"
)

// Answer types, the "type" value of each answer shape.
const (
	answerChoice = "choice"
	answerScore  = "score"
	answerNoul   = "noul"

	// keyAction is the answer key of the action block.
	keyAction = "action"
)

// Action is the action head's output for one question: the probability, after
// a softmax over the act logits, that the agent should act on the answer.
type Action struct {
	// ActProbability is softmax(act)[0], rounded to four decimals.
	ActProbability float64 `json:"act_probability"`
}

// Map returns the action as the one-key object Python builds.
func (a Action) Map() Obj {
	return Obj{{Key: "act_probability", Value: a.ActProbability}}
}

// ProbEntry is one key of a Probs map.
type ProbEntry struct {
	// Key is the choice key, or the level index as a string for a score.
	Key string
	// P is the probability, rounded to four decimals.
	P float64
}

// Probs is an ordered probability map; key order mirrors Python's dict, which
// is the order of the options the question defined.
type Probs []ProbEntry

// Get returns the probability stored under key, or 0 when there is none.
func (p Probs) Get(key string) float64 {
	for _, e := range p {
		if e.Key == key {
			return e.P
		}
	}
	return 0
}

// Max returns the key with the highest probability. On a tie it returns the
// first, as numpy's argmax does (invariant #30a); an empty Probs gives "", 0.
func (p Probs) Max() (key string, prob float64) {
	for i, e := range p {
		if i == 0 || e.P > prob {
			key, prob = e.Key, e.P
		}
	}
	return key, prob
}

// MarshalJSON writes the map as a JSON object in key order.
func (p Probs) MarshalJSON() ([]byte, error) {
	b, err := jsonx.Marshal(p.obj())
	if err != nil {
		return nil, fmt.Errorf("laya: marshal probabilities: %w", err)
	}
	return b, nil
}

// obj returns the map as an ordered object. Never nil, so an empty Probs is
// written as {} rather than null.
func (p Probs) obj() Obj {
	o := make(Obj, len(p))
	for i, e := range p {
		o[i] = jsonx.Field{Key: e.Key, Value: e.P}
	}
	return o
}

// Answer is one question's answer. It carries all three shapes; which keys are
// emitted is decided by Type alone, not by which fields happen to be zero.
// There is deliberately no omitempty anywhere: a choice key of "" is legal in
// Python and must still be emitted as "choice", and an empty legend or
// probabilities map is still written as {} (invariant #30).
type Answer struct {
	// Type is "choice", "score" or "noul".
	Type string
	// Choice is the key of the most probable option (choice only).
	Choice string
	// Score is the expected level Σ i·p[i] (score only).
	Score *float64
	// Noul is P(true) (noul only).
	Noul *float64
	// Legend maps each level index to the raw criterion value the caller gave
	// (score only).
	Legend Obj
	// Probabilities is the per-option distribution (choice and score).
	Probabilities Probs
	// Confidence is the normalized-entropy confidence for choice and score,
	// max(p1, 1-p1) for noul.
	Confidence float64
	// Action is the action head's output.
	Action Action
}

// Map returns the answer as the ordered object agent.py builds, with the key
// set and order of its Type:
//
//	choice: type, choice, probabilities, confidence, action
//	score:  type, score, legend, probabilities, confidence, action
//	noul:   type, noul, confidence, action
//
// Hand this to jsonx.Marshal when the bytes have to match json.dumps.
func (a Answer) Map() Obj {
	o := Obj{{Key: "type", Value: a.Type}}
	switch a.Type {
	case answerChoice:
		o = append(
			o,
			jsonx.Field{Key: "choice", Value: a.Choice},
			jsonx.Field{Key: "probabilities", Value: a.Probabilities.obj()},
		)
	case answerScore:
		legend := a.Legend
		if legend == nil {
			legend = Obj{}
		}
		o = append(
			o,
			jsonx.Field{Key: "score", Value: floatOrNil(a.Score)},
			jsonx.Field{Key: "legend", Value: legend},
			jsonx.Field{Key: "probabilities", Value: a.Probabilities.obj()},
		)
	case answerNoul:
		o = append(o, jsonx.Field{Key: "noul", Value: floatOrNil(a.Noul)})
	}
	return append(
		o,
		jsonx.Field{Key: "confidence", Value: a.Confidence},
		jsonx.Field{Key: keyAction, Value: a.Action.Map()},
	)
}

// MarshalJSON emits the answer's per-type key set in Python's order.
//
// Beware D12: encoding/json compacts whatever a MarshalJSON returns, so
// json.Marshal(answer) does not produce Python's bytes. Use
// jsonx.Marshal(a.Map()) wherever byte equality is the requirement.
func (a Answer) MarshalJSON() ([]byte, error) {
	b, err := jsonx.Marshal(a.Map())
	if err != nil {
		return nil, fmt.Errorf("laya: marshal answer: %w", err)
	}
	return b, nil
}

// floatOrNil unwraps p so jsonx sees a float64 (and writes Python's repr)
// rather than a pointer, and nil as JSON null.
func floatOrNil(p *float64) any {
	if p == nil {
		return nil
	}
	return *p
}

// Usage is the token accounting of one SystemOne call.
type Usage struct {
	// InputTokens is the number of attended tokens in the encoder input.
	InputTokens int `json:"input_tokens"`
	// OutputTokens is always 0: the model generates nothing.
	OutputTokens int `json:"output_tokens"`
}

// Map returns the usage as the ordered object Python builds.
func (u Usage) Map() Obj {
	return Obj{
		{Key: "input_tokens", Value: u.InputTokens},
		{Key: "output_tokens", Value: u.OutputTokens},
	}
}

// NamedAnswer is one entry of an AnswerSet.
type NamedAnswer struct {
	// ID is the question id the answer belongs to.
	ID string
	// A is the answer.
	A Answer
}

// AnswerSet is an ordered id -> Answer map mirroring the question order.
type AnswerSet []NamedAnswer

// Get returns the answer for id.
func (as AnswerSet) Get(id string) (Answer, bool) {
	for _, na := range as {
		if na.ID == id {
			return na.A, true
		}
	}
	return Answer{}, false
}

// Map returns the set as an ordered object of each answer's Map.
func (as AnswerSet) Map() Obj {
	o := make(Obj, len(as))
	for i, na := range as {
		o[i] = jsonx.Field{Key: na.ID, Value: na.A.Map()}
	}
	return o
}

// MarshalJSON writes the answers as a JSON object in question order. D12
// applies as for Answer.MarshalJSON.
func (as AnswerSet) MarshalJSON() ([]byte, error) {
	b, err := jsonx.Marshal(as.Map())
	if err != nil {
		return nil, fmt.Errorf("laya: marshal answers: %w", err)
	}
	return b, nil
}

// Result is what SystemOne returns: the model name, the answers in question
// order, the token usage and, when the call went through a Router, the
// routing decision.
type Result struct {
	// Model is always "laya-rl-agent".
	Model string `json:"model"`
	// Answers holds one answer per question, in question order.
	Answers AnswerSet `json:"answers"`
	// Usage is the token accounting.
	Usage Usage `json:"usage"`
	// Routing is the router's decision, nil when no router was involved.
	Routing *RouteDecision `json:"routing,omitempty"`
}

// Map returns the result as the ordered object Python builds: model, answers,
// usage, and routing only when there is a routing decision.
func (r Result) Map() Obj {
	o := Obj{
		{Key: "model", Value: r.Model},
		{Key: "answers", Value: r.Answers.Map()},
		{Key: "usage", Value: r.Usage.Map()},
	}
	if r.Routing != nil {
		o = append(o, jsonx.Field{Key: "routing", Value: r.Routing.Map()})
	}
	return o
}

// MarshalJSON writes the result in Python's key order. D12 applies as for
// Answer.MarshalJSON.
func (r Result) MarshalJSON() ([]byte, error) {
	b, err := jsonx.Marshal(r.Map())
	if err != nil {
		return nil, fmt.Errorf("laya: marshal result: %w", err)
	}
	return b, nil
}

// formatAnswer is agent.py:302-337 for one question: logits are the head's
// output row for it (only the first k count), act its action logits, and k
// its number of option markers.
//
// The precision follows numpy (invariant #24a): the softmax and the entropy
// confidence are float32, the score is float64 over the float32 p, and the
// noul value and confidence are float64 from the float32 p[1]. The choice is
// the first maximum of p, as p.argmax() is (#30a).
//
// k is the marker count the prompt was built with, which for a choice
// question equals the number of options; an argmax past the options would be
// an IndexError upstream and panics here.
func formatAnswer(q question.Question, logits, act []float32, k int, temps calib.Temperatures) Answer {
	qt := q.Type()
	p := calib.Softmax(logits, k, temps.Scale(qt, k))
	// The criteria as Agent._to_internal flattens them: an ordered object for
	// choice, a list for score. Reading them there rather than off the typed
	// question is what upstream does, and covers pointer receivers too.
	crit := question.ToInternal(q).Crit

	// agent.py:295 applies torch.softmax to the act logits in float32, and
	// :310 keeps column 0. ActSoftmax is ATen's kernel, not numpy's (7.3.10).
	a := Answer{
		Type: qt.String(),
		Action: Action{
			ActProbability: jsonx.Round4(float64(calib.ActSoftmax(act)[0])),
		},
	}

	switch qt {
	case question.Choice:
		keys, _ := crit.(jsonx.Obj)
		best := 0
		for i := 1; i < len(p); i++ {
			if p[i] > p[best] {
				best = i
			}
		}
		a.Choice = keys[best].Key
		// zip(keys, p) stops at the shorter of the two.
		n := min(len(keys), len(p))
		a.Probabilities = make(Probs, n)
		for i := range n {
			a.Probabilities[i] = ProbEntry{Key: keys[i].Key, P: jsonx.Round4(float64(p[i]))}
		}
		a.Confidence = jsonx.Round4(calib.Confidence(p, k))
	case question.Score:
		levels, _ := crit.([]any)
		score := jsonx.Round4(calib.Expectation(p))
		a.Score = &score
		// The legend enumerates every criterion, not just the first k.
		a.Legend = make(Obj, len(levels))
		for i, c := range levels {
			a.Legend[i] = jsonx.Field{Key: strconv.Itoa(i), Value: c}
		}
		a.Probabilities = make(Probs, len(p))
		for i, v := range p {
			a.Probabilities[i] = ProbEntry{Key: strconv.Itoa(i), P: jsonx.Round4(float64(v))}
		}
		a.Confidence = jsonx.Round4(calib.Confidence(p, k))
	case question.Noul:
		// Index 1 is P(true) (invariant #27).
		noul := jsonx.Round4(float64(p[1]))
		a.Noul = &noul
		a.Confidence = jsonx.Round4(calib.NoulConfidence(p[1]))
	}
	return a
}
