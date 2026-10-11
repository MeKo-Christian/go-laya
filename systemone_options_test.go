package laya

import (
	"context"
	"errors"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/MeKo-Christian/go-laya/backend"
	"github.com/MeKo-Christian/go-laya/internal/golden"
	"github.com/MeKo-Christian/go-laya/jsonx"
)

// errForwardRan is what noForwardBackend returns, so a call that reaches it
// fails even where the test's own error is not checked.
var errForwardRan = errors.New("test: the forward pass ran")

// noForwardBackend fails the test if a forward pass reaches it.
type noForwardBackend struct{ t *testing.T }

func (b noForwardBackend) Forward(context.Context, backend.Batch) (logits, act [][]float32, err error) {
	b.t.Helper()
	b.t.Error("the forward pass ran on a batch SystemOne should have refused")
	return nil, nil, errForwardRan
}

func (noForwardBackend) Close() error { return nil }

// tooFewOptionBatches are batches with no question of two or more options:
// upstream's p.topk(2, -1) (common.py:122) raises on each.
func tooFewOptionBatches() []struct {
	name string
	qs   Questions
	ids  []string
} {
	return []struct {
		name string
		qs   Questions
		ids  []string
	}{
		{
			name: "one 1-option choice",
			qs:   Questions{{ID: "only", Q: ChoiceQuestion{Ins: "pick", Opts: Labels("yes")}}},
			ids:  []string{"only"},
		},
		{
			name: "one 1-level score",
			qs:   Questions{{ID: "level", Q: ScoreQuestion{Ins: "rate", Levels: []Criterion{"low"}}}},
			ids:  []string{"level"},
		},
		{
			name: "two 1-option questions",
			qs: Questions{
				{ID: "a", Q: ChoiceQuestion{Ins: "pick", Opts: Labels("yes")}},
				{ID: "b", Q: ScoreQuestion{Ins: "rate", Levels: []Criterion{"low"}}},
			},
			ids: []string{"a", "b"},
		},
	}
}

// checkTooFewOptions holds err to ErrTooFewOptions, naming every question id.
func checkTooFewOptions(t *testing.T, err error, ids []string) {
	t.Helper()
	if !errors.Is(err, ErrTooFewOptions) {
		t.Fatalf("err = %v, want ErrTooFewOptions", err)
	}
	for _, id := range ids {
		if !strings.Contains(err.Error(), `"`+id+`"`) {
			t.Errorf("err = %q does not name question %q", err, id)
		}
	}
}

// TestSystemOneTooFewOptions is B.12: a batch in which no question has two
// options is refused before the backend, on any backend, where upstream's
// topk(2) raises inside the forward pass.
func TestSystemOneTooFewOptions(t *testing.T) {
	for _, tc := range tooFewOptionBatches() {
		t.Run(tc.name, func(t *testing.T) {
			a := testAgent(t, "", noForwardBackend{t: t})
			_, err := a.SystemOne(context.Background(), "state", tc.qs)
			checkTooFewOptions(t, err, tc.ids)
		})
	}
}

// TestPredictTooFewOptions: Agent.Predict and both of the Router's paths reach
// the backend through SystemOne, so the one check covers them. The Router
// loads the agent first, as router.py:303-309 does before system_one raises.
func TestPredictTooFewOptions(t *testing.T) {
	tc := tooFewOptionBatches()[2]
	a := testAgent(t, "", noForwardBackend{t: t})

	_, err := a.Predict(context.Background(), "state", tc.qs)
	checkTooFewOptions(t, err, tc.ids)

	loads := 0
	r, err := NewRouter(WithLoader(func(context.Context, string, ModelSpec) (Predictor, error) {
		loads++
		return a, nil
	}))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = r.Close() })
	_, err = r.Predict(context.Background(), "state", tc.qs, ForModel(ModelEnglish))
	checkTooFewOptions(t, err, tc.ids)
	_, err = r.SystemOne(context.Background(), "state", tc.qs, ForModel(ModelEnglish))
	checkTooFewOptions(t, err, tc.ids)
	if loads != 1 {
		t.Errorf("the Router loaded the agent %d times, want 1", loads)
	}
}

// TestSystemOneSingleOptionInMixedBatch: a 1-option question beside one with
// two options runs, as it does upstream, where kmax is 2. Its row's second
// marker column is padding, so its softmax is over its one logit: p = [1],
// and confidence_from_probs gives 1 for k < 2 (common.py:202-203).
func TestSystemOneSingleOptionInMixedBatch(t *testing.T) {
	be := &recordingBackend{
		// Each 1-option row's second logit is padding and must not count.
		logits: [][]float32{{0.3, 5}, {0.1, 0.9}, {-2, 7}},
		act:    [][]float32{{0, 0}},
	}
	a := testAgent(t, "", be)
	qs := Questions{
		{ID: "only", Q: ChoiceQuestion{Ins: "pick", Opts: Labels("yes")}},
		{ID: "pair", Q: NoulQuestion{Ins: "yes?"}},
		{ID: "level", Q: ScoreQuestion{Ins: "rate", Levels: []Criterion{"low"}}},
	}
	res, err := a.SystemOne(context.Background(), "state", qs)
	if err != nil {
		t.Fatal(err)
	}
	if len(be.calls) != 1 {
		t.Fatalf("forward passes = %d, want 1", len(be.calls))
	}
	if want := [][]bool{{true, false}, {true, true}, {true, false}}; !slices.EqualFunc(be.calls[0].MarkerMask, want, slices.Equal) {
		t.Errorf("marker_mask = %v, want %v", be.calls[0].MarkerMask, want)
	}

	only, _ := res.Answers.Get("only")
	if only.Choice != "yes" || !slices.Equal(only.Probabilities, Probs{{Key: "yes", P: 1}}) || only.Confidence != 1 {
		t.Errorf("only = %+v, want choice yes at p 1 and confidence 1", only)
	}
	level, _ := res.Answers.Get("level")
	if level.Score == nil || *level.Score != 0 ||
		!slices.Equal(level.Probabilities, Probs{{Key: "0", P: 1}}) || level.Confidence != 1 {
		t.Errorf("level = %+v, want score 0 at p 1 and confidence 1", level)
	}
	if want := (Obj{{Key: "0", Value: "low"}}); !slices.Equal(level.Legend, want) {
		t.Errorf("legend = %v, want %v", level.Legend, want)
	}
	checkNoulInOpenUnit(t, res, "pair")
	if _, err := jsonx.Marshal(res.Map()); err != nil {
		t.Errorf("marshal the result: %v", err)
	}
}

// TestTooFewOptionsRealBackends is B.12 on both real backends with the english
// checkpoint: each refuses the batches before a forward pass, where the native
// head would fail in checkMarkers and ONNX Runtime at the graph's TopK node,
// each with its own error. A mixed batch then runs on the real weights. Each
// backend is closed before the next one loads.
func TestTooFewOptionsRealBackends(t *testing.T) {
	root := golden.SkipWithoutModels(t)
	dir := golden.CheckpointDir(root, golden.English)

	t.Run("onnx", func(t *testing.T) {
		exports := os.Getenv("LAYA_ONNX_DIR")
		if exports == "" {
			t.Skip("no exports (set LAYA_ONNX_DIR)")
		}
		graph, ok := findExport(exports, golden.English)
		if !ok {
			t.Skipf("no %s export in %s", golden.English, exports)
		}
		tooFewOptionsOn(t, dir, WithGraph(graph), WithDevice("cpu"))
	})
	t.Run("native", func(t *testing.T) {
		t.Setenv("LAYA_ONNX_DIR", t.TempDir()) // holds no export
		tooFewOptionsOn(t, dir, WithRuntime(RuntimeNative))
	})
}

func tooFewOptionsOn(t *testing.T, dir string, opts ...Option) {
	t.Helper()
	a, err := Open(context.Background(), dir, opts...)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	inner := a.backend
	defer func() {
		if err := inner.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
	}()

	// The guard goes between the agent and the backend Open built, so a
	// refused batch provably never reaches it.
	a.backend = noForwardBackend{t: t}
	for _, tc := range tooFewOptionBatches() {
		_, err := a.SystemOne(context.Background(), "I was charged twice.", tc.qs)
		checkTooFewOptions(t, err, tc.ids)
	}

	a.backend = inner
	res, err := a.SystemOne(context.Background(), "I was charged twice.", Questions{
		{ID: "only", Q: ChoiceQuestion{Ins: "Which team?", Opts: Labels("billing")}},
		{ID: "refund", Q: NoulQuestion{Ins: "Does the customer ask for money back?"}},
	})
	if err != nil {
		t.Fatalf("mixed batch: %v", err)
	}
	only, _ := res.Answers.Get("only")
	if only.Choice != "billing" || !slices.Equal(only.Probabilities, Probs{{Key: "billing", P: 1}}) ||
		only.Confidence != 1 {
		t.Errorf("only = %+v, want billing at p 1 and confidence 1", only)
	}
	// The two-option row is unaffected by the other row's padded column.
	checkNoulInOpenUnit(t, res, "refund")
}

// checkNoulInOpenUnit fails the test unless answer id carries a noul
// probability strictly between 0 and 1. The range check is in positive form
// so a NaN fails it too.
func checkNoulInOpenUnit(t *testing.T, res *Result, id string) {
	t.Helper()
	a, ok := res.Answers.Get(id)
	if !ok || a.Noul == nil {
		t.Errorf("%s = %+v (found %v), want a noul", id, a, ok)
		return
	}
	if p := *a.Noul; 0 < p && p < 1 {
		return
	}
	t.Errorf("%s noul = %v, want strictly between 0 and 1", id, *a.Noul)
}
