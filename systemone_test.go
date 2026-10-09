package laya

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/MeKo-Christian/go-laya/backend"
	"github.com/MeKo-Christian/go-laya/internal/backend/fake"
	"github.com/MeKo-Christian/go-laya/internal/checkpoint"
	"github.com/MeKo-Christian/go-laya/internal/golden"
	"github.com/MeKo-Christian/go-laya/jsonx"
	"github.com/MeKo-Christian/go-laya/question"
	"github.com/MeKo-Christian/go-laya/tokenizer"
)

// recordingBackend answers every forward pass with fixed per-row logits and
// remembers each batch it was given.
type recordingBackend struct {
	calls []backend.Batch
	// logits and act are per row; a row beyond them repeats the last one.
	logits, act [][]float32
}

func (b *recordingBackend) Forward(_ context.Context, in backend.Batch) (logits, act [][]float32, err error) {
	b.calls = append(b.calls, in)
	for r := range in.InputIDs {
		logits = append(logits, b.logits[min(r, len(b.logits)-1)])
		act = append(act, b.act[min(r, len(b.act)-1)])
	}
	return logits, act, nil
}

func (*recordingBackend) Close() error { return nil }

// testAgent builds an agent over the mini_en tokenizer and the given config.
func testAgent(t *testing.T, config string, be backend.Backend) *Agent {
	t.Helper()
	dir := t.TempDir()
	writeCheckpoint(t, dir)
	if config != "" {
		if err := os.WriteFile(filepath.Join(dir, checkpoint.ConfigFile), []byte(config), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	cfg, err := checkpoint.LoadConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.Open(filepath.Join(dir, "tokenizer"))
	if err != nil {
		t.Fatal(err)
	}
	a, err := newAgent(be, cfg, tok)
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func threeQuestions() Questions {
	return Questions{
		{ID: "intent", Q: ChoiceQuestion{Ins: "What does the customer want?", Opts: Labels("refund", "billing", "other")}},
		{ID: "urgency", Q: ScoreQuestion{Ins: "How urgent is it?", Levels: []Criterion{"low", "mid", "high"}}},
		{ID: "angry", Q: NoulQuestion{Ins: "Is the customer angry?"}},
	}
}

// TestSystemOneBatches is 7.3.6: every question goes through one forward pass,
// one row each, with its own qtype, and the answers come back in question
// order.
func TestSystemOneBatches(t *testing.T) {
	be := &recordingBackend{
		logits: [][]float32{{0.1, 2.0, -1.0}, {-1.0, 0.5, 3.0}, {0.2, -0.4, -10000}},
		act:    [][]float32{{1.5, -0.5, 0.25}},
	}
	a := testAgent(t, "", be)

	res, err := a.SystemOne(context.Background(), "I was charged twice and want my money back.", threeQuestions())
	if err != nil {
		t.Fatal(err)
	}
	if len(be.calls) != 1 {
		t.Fatalf("forward passes = %d, want 1", len(be.calls))
	}
	b := be.calls[0]
	if len(b.InputIDs) != 3 {
		t.Fatalf("rows = %d, want 3", len(b.InputIDs))
	}
	if want := []int64{0, 1, 2}; !slices.Equal(b.QType, want) {
		t.Errorf("qtype = %v, want %v", b.QType, want)
	}
	var tokens int
	for _, row := range b.AttentionMask {
		for _, v := range row {
			tokens += int(v)
		}
	}
	if res.Usage != (Usage{InputTokens: tokens, OutputTokens: 0}) {
		t.Errorf("usage = %+v, want %d input tokens", res.Usage, tokens)
	}
	if res.Model != "laya-rl-agent" {
		t.Errorf("model = %q", res.Model)
	}

	ids := make([]string, 0, len(res.Answers))
	for _, na := range res.Answers {
		ids = append(ids, na.ID)
	}
	if got := strings.Join(ids, ","); got != "intent,urgency,angry" {
		t.Errorf("answer order = %s", got)
	}
	intent, _ := res.Answers.Get("intent")
	if intent.Type != "choice" || intent.Choice != "billing" {
		t.Errorf("intent = %+v, want billing", intent)
	}
	urgency, _ := res.Answers.Get("urgency")
	if urgency.Type != "score" || urgency.Score == nil {
		t.Errorf("urgency = %+v", urgency)
	}
	// The noul row's third logit is padding: k is the marker count, 2.
	angry, _ := res.Answers.Get("angry")
	if angry.Type != "noul" || angry.Noul == nil || *angry.Noul != jsonx.Round4(1/(1+math.Exp(0.6))) {
		t.Errorf("angry = %+v", angry)
	}
	// Column 0 of the whole act row's softmax, the same for every answer.
	if intent.Action != angry.Action {
		t.Errorf("act differs across answers: %+v vs %+v", intent.Action, angry.Action)
	}
	if _, err := json.Marshal(res); err != nil {
		t.Errorf("marshal the result: %v", err)
	}
}

// TestSystemOneUsesConfig checks that the checkpoint's temperatures reach the
// softmax: a temperature of 2 halves every logit.
func TestSystemOneUsesConfig(t *testing.T) {
	be := &recordingBackend{logits: [][]float32{{0, 2}}, act: [][]float32{{0, 0}}}
	cfg := `{"encoder": "x", "head_layers": 2, "temperature": [1, 1, 2]}`
	a := testAgent(t, cfg, be)

	res, err := a.SystemOne(context.Background(), "state", Questions{{ID: "q", Q: NoulQuestion{Ins: "yes?"}}})
	if err != nil {
		t.Fatal(err)
	}
	got, _ := res.Answers.Get("q")
	if want := jsonx.Round4(1 / (1 + math.Exp(-1))); *got.Noul != want {
		t.Errorf("noul = %v, want %v (temperature 2)", *got.Noul, want)
	}
}

func TestSystemOneErrors(t *testing.T) {
	be := &recordingBackend{logits: [][]float32{{0, 1}}, act: [][]float32{{0, 0}}}

	tests := []struct {
		name   string
		config string
		state  any
		qs     Questions
		want   error
	}{
		{
			name: "no questions",
			qs:   nil,
			want: ErrEmptyQuestions,
		},
		{
			name: "duplicate id",
			qs: Questions{
				{ID: "q", Q: NoulQuestion{Ins: "a"}},
				{ID: "q", Q: NoulQuestion{Ins: "b"}},
			},
			want: ErrDuplicateQuestionID,
		},
		{
			name: "choice without options",
			qs:   Questions{{ID: "q", Q: ChoiceQuestion{Ins: "pick"}}},
			want: ErrNoOptions,
		},
		{
			// max_len 16 leaves room for the head but not for twenty options.
			name:   "options past max_len",
			config: `{"encoder": "x", "head_layers": 2, "max_len": 16}`,
			qs: Questions{{ID: "q", Q: ChoiceQuestion{Ins: "pick", Opts: Labels(
				"a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
				"k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
			)}}},
			want: ErrOptionsExceedHeadBudget,
		},
		{
			name:  "unordered state",
			state: map[string]any{"a": 1},
			qs:    Questions{{ID: "q", Q: NoulQuestion{Ins: "a"}}},
			want:  jsonx.ErrUnorderedMap,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			be.calls = nil
			a := testAgent(t, tc.config, be)
			state := tc.state
			if state == nil {
				state = "state"
			}
			_, err := a.SystemOne(context.Background(), state, tc.qs)
			if !errors.Is(err, tc.want) {
				t.Fatalf("err = %v, want %v", err, tc.want)
			}
			if len(be.calls) != 0 {
				t.Errorf("a failed request still ran %d forward passes", len(be.calls))
			}
		})
	}
}

func TestOptionBudgetError(t *testing.T) {
	be := &recordingBackend{logits: [][]float32{{0, 1}}, act: [][]float32{{0, 0}}}
	a := testAgent(t, `{"encoder": "x", "head_layers": 2, "max_len": 16}`, be)
	_, err := a.SystemOne(context.Background(), "s", Questions{{ID: "many", Q: ChoiceQuestion{
		Ins: "pick", Opts: Labels("a", "b", "c", "d", "e", "f", "g", "h", "i", "j"),
	}}})
	var be2 *OptionBudgetError
	if !errors.As(err, &be2) {
		t.Fatalf("err = %v, want an *OptionBudgetError", err)
	}
	if be2.QuestionID != "many" || be2.WantMarkers != 10 || be2.GotMarkers >= 10 || be2.HeadMaxLen != 192 {
		t.Errorf("OptionBudgetError = %+v", *be2)
	}
	// agent.py:263's message.
	if want := `laya: question "many" options exceed head_max_len=192`; !strings.HasPrefix(err.Error(), want) {
		t.Errorf("message = %q, want prefix %q", err.Error(), want)
	}
}

func TestPredictIsSystemOne(t *testing.T) {
	be := &recordingBackend{logits: [][]float32{{0, 1}}, act: [][]float32{{0, 0}}}
	a := testAgent(t, "", be)
	qs := Questions{{ID: "q", Q: NoulQuestion{Ins: "a"}}}
	r1, err := a.SystemOne(context.Background(), "s", qs)
	if err != nil {
		t.Fatal(err)
	}
	r2, err := a.Predict(context.Background(), "s", qs)
	if err != nil {
		t.Fatal(err)
	}
	b1, _ := jsonx.Marshal(r1.Map())
	b2, _ := jsonx.Marshal(r2.Map())
	if string(b1) != string(b2) {
		t.Errorf("Predict = %s, SystemOne = %s", b2, b1)
	}
}

// TestSystemOneReplay runs every logits.jsonl case through SystemOne with the
// real tokenizers and budgets, against the fake backend. The fake answers only
// a batch equal to the recorded one cell for cell, so a pass means the whole
// front half -- conversion, sequence, collate -- agrees with Python; the
// result, formatted from the recorded logits, must then be byte-equal to what
// Python's system_one returned for them.
func TestSystemOneReplay(t *testing.T) {
	root := golden.SkipWithoutModels(t)

	type logitsCase struct {
		Checkpoint  string          `json:"checkpoint"`
		State       json.RawMessage `json:"state"`
		Questions   json.RawMessage `json:"questions"`
		QIDs        []string        `json:"qids"`
		InputTokens int             `json:"input_tokens"`
		ResultJSON  string          `json:"result_json"`
	}
	agents := map[string]*Agent{}
	for _, rec := range golden.Load(t, "logits") {
		var c logitsCase
		rec.Unmarshal(t, &c)
		a, ok := agents[c.Checkpoint]
		if !ok {
			dir := golden.CheckpointDir(root, c.Checkpoint)
			cfg, err := checkpoint.LoadConfig(dir)
			if err != nil {
				t.Fatal(err)
			}
			tok, err := tokenizer.Open(filepath.Join(dir, "tokenizer"))
			if err != nil {
				t.Fatal(err)
			}
			if a, err = newAgent(fake.New(t, c.Checkpoint), cfg, tok); err != nil {
				t.Fatal(err)
			}
			agents[c.Checkpoint] = a
		}

		t.Run(rec.Name, func(t *testing.T) {
			state, err := jsonx.Decode(c.State)
			if err != nil {
				t.Fatal(err)
			}
			qs := recordedQuestions(t, c.Questions)
			res, err := a.SystemOne(context.Background(), state, qs)
			if err != nil {
				t.Fatal(err)
			}
			if res.Usage.InputTokens != c.InputTokens {
				t.Errorf("input_tokens = %d, want %d", res.Usage.InputTokens, c.InputTokens)
			}
			if len(res.Answers) != len(c.QIDs) {
				t.Fatalf("answers = %d, want %d", len(res.Answers), len(c.QIDs))
			}
			for i, na := range res.Answers {
				if na.ID != c.QIDs[i] {
					t.Errorf("answer %d is %q, want %q", i, na.ID, c.QIDs[i])
				}
			}
			// Task 7.4.3: the whole result, as Python's json.dumps writes it.
			got, err := jsonx.Marshal(res.Map())
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			if string(got) != c.ResultJSON {
				t.Errorf("result bytes\n got %s\nwant %s", got, c.ResultJSON)
			}
		})
	}
}

// recordedQuestions turns a fixture's question dict into typed questions, as a
// caller of the Go API would write them.
func recordedQuestions(t *testing.T, raw json.RawMessage) Questions {
	t.Helper()
	v, err := jsonx.Decode(raw)
	if err != nil {
		t.Fatal(err)
	}
	obj, ok := v.(jsonx.Obj)
	if !ok {
		t.Fatalf("questions are %T", v)
	}
	qs := make(Questions, 0, len(obj))
	for _, f := range obj {
		qdef, _ := f.Value.(jsonx.Obj)
		tv, _ := qdef.Get("type")
		iv, _ := qdef.Get("instructions")
		ins, _ := iv.(string)
		crit, _ := qdef.Get("criteria")
		var q Question
		switch tv {
		case "choice":
			cq := ChoiceQuestion{Ins: ins}
			switch c := crit.(type) {
			case []any:
				for _, l := range c {
					s, _ := l.(string)
					cq.Opts = append(cq.Opts, ChoiceOption{Key: s})
				}
			case jsonx.Obj:
				for _, o := range c {
					cq.Opts = append(cq.Opts, ChoiceOption{Key: o.Key, Desc: o.Value})
				}
			}
			q = cq
		case "score":
			levels, _ := crit.([]any)
			sq := ScoreQuestion{Ins: ins}
			for _, l := range levels {
				sq.Levels = append(sq.Levels, l)
			}
			q = sq
		case "noul":
			nq := NoulQuestion{Ins: ins}
			if c, isObj := crit.(jsonx.Obj); isObj {
				nq.False, _ = c.Get("false")
				nq.True, _ = c.Get("true")
			}
			q = nq
		default:
			t.Fatalf("question type %v", tv)
		}
		qs = append(qs, question.NamedQuestion{ID: f.Key, Q: q})
	}
	return qs
}

// TestSystemOneTypedNilQuestion: a nil *ChoiceQuestion is an error, not a
// panic, and never reaches the model.
func TestSystemOneTypedNilQuestion(t *testing.T) {
	be := &recordingBackend{logits: [][]float32{{0, 1}}, act: [][]float32{{0, 0}}}
	a := testAgent(t, "", be)
	_, err := a.SystemOne(context.Background(), "s", Questions{{ID: "q", Q: (*ChoiceQuestion)(nil)}})
	if err == nil {
		t.Fatal("SystemOne accepted a typed-nil question")
	}
	if len(be.calls) != 0 {
		t.Errorf("forward passes = %d, want 0", len(be.calls))
	}
}
