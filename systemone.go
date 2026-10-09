package laya

import (
	"context"
	"fmt"

	"github.com/MeKo-Christian/go-laya/internal/prompt"
	"github.com/MeKo-Christian/go-laya/question"
)

// modelName is the constant "model" of every result (agent.py:340).
const modelName = "laya-rl-agent"

// SystemOne answers every question about state in one forward pass: Python's
// Agent.system_one (agent.py:240-343).
//
// state is a string, which is used as it is, or anything jsonx.Marshal
// accepts, which is serialized as json.dumps would (D25). The answers come
// back in question order.
//
// It fails before running the model on an empty or invalid question set, a
// state that cannot be serialized, a question with no options, or options
// that do not all fit into max_len (*OptionBudgetError).
func (a *Agent) SystemOne(ctx context.Context, state any, qs Questions) (*Result, error) {
	if err := qs.Validate(); err != nil {
		// Validate's errors already carry the "laya:" prefix.
		return nil, fmt.Errorf("%w", err)
	}
	if len(qs) == 0 {
		return nil, ErrEmptyQuestions
	}

	// One read for the whole call, so a concurrent SetLimits cannot give two
	// rows of one batch different budgets.
	maxLen, headMaxLen := a.limits()
	items := make([]prompt.Item, len(qs))
	for i, nq := range qs {
		q := question.ToInternal(nq.Q)
		if err := prompt.CheckCriteria(q); err != nil {
			return nil, fmt.Errorf("laya: question %q: %w", nq.ID, err)
		}
		want := len(prompt.RenderOptions(q))
		if want == 0 {
			return nil, fmt.Errorf("%w: %q", ErrNoOptions, nq.ID)
		}
		ids, markers, err := prompt.BuildSequence(a.tok, state, q, maxLen, headMaxLen, nil, false)
		if err != nil {
			return nil, fmt.Errorf("laya: state: %w", err)
		}
		// agent.py:262-263: a marker lost to max_len means an option the
		// model never scores.
		if len(markers) != want {
			return nil, &OptionBudgetError{
				QuestionID: nq.ID, HeadMaxLen: headMaxLen,
				WantMarkers: want, GotMarkers: len(markers),
			}
		}
		items[i] = prompt.Item{IDs: ids, Markers: markers, QType: int64(nq.Q.Type())}
	}

	batch := prompt.Collate(items, a.tok.PADID())
	logits, act, err := a.backend.Forward(ctx, batch)
	if err != nil {
		return nil, fmt.Errorf("laya: forward pass: %w", err)
	}
	if len(logits) != len(qs) || len(act) != len(qs) {
		return nil, fmt.Errorf("%w: forward pass returned %d logit rows and %d act rows for %d questions",
			ErrIncompatibleCheckpoint, len(logits), len(act), len(qs))
	}

	res := &Result{Model: modelName, Answers: make(AnswerSet, len(qs))}
	for r, nq := range qs {
		k := len(items[r].Markers)
		if len(logits[r]) < k {
			return nil, fmt.Errorf("%w: %d logits for %d options", ErrIncompatibleCheckpoint, len(logits[r]), k)
		}
		if len(act[r]) == 0 {
			return nil, fmt.Errorf("%w: an empty act row", ErrIncompatibleCheckpoint)
		}
		res.Answers[r] = NamedAnswer{ID: nq.ID, A: formatAnswer(nq.Q, logits[r], act[r], k, a.temps)}
	}
	// agent.py:298 counts the whole batch's real tokens.
	for _, row := range batch.AttentionMask {
		for _, v := range row {
			res.Usage.InputTokens += int(v)
		}
	}
	return res, nil
}

// Predict is SystemOne under upstream's other name (agent.py:345).
func (a *Agent) Predict(ctx context.Context, state any, qs Questions) (*Result, error) {
	return a.SystemOne(ctx, state, qs)
}
