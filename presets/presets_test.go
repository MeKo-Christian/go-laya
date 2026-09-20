package presets

import (
	"encoding/json"
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/golden"
	"github.com/MeKo-Christian/go-laya/jsonx"
	"github.com/MeKo-Christian/go-laya/question"
)

// qdef renders a question back into the public definition dict upstream's
// preset constructors return -- {"type", "instructions", "criteria"} in that
// order. It is the inverse of Agent._to_internal (agent.py:229-238) and exists
// only so the presets can be compared against what the Python emitted.
func qdef(t *testing.T, q question.Question) jsonx.Obj {
	t.Helper()

	def := jsonx.Obj{
		{Key: "type", Value: q.Type().String()},
		{Key: "instructions", Value: q.Instructions()},
	}

	switch x := q.(type) {
	case question.ChoiceQuestion:
		crit := make(jsonx.Obj, 0, len(x.Opts))
		for _, o := range x.Opts {
			crit = append(crit, jsonx.Field{Key: o.Key, Value: o.Desc})
		}
		def = append(def, jsonx.Field{Key: "criteria", Value: crit})
	case question.ScoreQuestion:
		levels := make([]any, len(x.Levels))
		for i, l := range x.Levels {
			levels[i] = l
		}
		def = append(def, jsonx.Field{Key: "criteria", Value: levels})
	case question.NoulQuestion:
		// Upstream omits "criteria" entirely unless the question sets one;
		// only email's is_phishing does.
		if x.False != nil || x.True != nil {
			def = append(def, jsonx.Field{Key: "criteria", Value: jsonx.Obj{
				{Key: "true", Value: x.True},
				{Key: "false", Value: x.False},
			}})
		}
	default:
		t.Fatalf("unknown question type %T", q)
	}
	return def
}

func qsetJSON(t *testing.T, qs question.Questions) string {
	t.Helper()

	out := make(jsonx.Obj, 0, len(qs))
	for _, nq := range qs {
		out = append(out, jsonx.Field{Key: nq.ID, Value: qdef(t, nq.Q)})
	}
	b, err := jsonx.Marshal(out)
	if err != nil {
		t.Fatalf("marshal question set: %v", err)
	}
	return string(b)
}

// TestEmailQuestionsGolden is the only preset with recorded Python output, and
// it is real parity evidence rather than a shape test: testdata/mailtext.jsonl
// carries email_questions' two cases, dumped from upstream. Task 2.3.2 says
// email.email_questions is a byte-identical dead duplicate of this one, so the
// vectors pin the preset that survived.
func TestEmailQuestionsGolden(t *testing.T) {
	for _, c := range golden.ByFn(t, "mailtext", "email_questions") {
		t.Run(c.Name, func(t *testing.T) {
			var rec struct {
				Input  json.RawMessage `json:"input"`
				Output json.RawMessage `json:"output"`
			}
			c.Unmarshal(t, &rec)

			// The recorded output is an ordered object; decode and re-emit it
			// through the same encoder so the comparison is byte-for-byte and
			// not a re-formatting artefact.
			want, err := jsonx.Decode(rec.Output)
			if err != nil {
				t.Fatalf("decode recorded output: %v", err)
			}
			wantJSON, err := jsonx.Marshal(want)
			if err != nil {
				t.Fatalf("re-marshal recorded output: %v", err)
			}

			var got string
			if string(rec.Input) == "null" {
				got = qsetJSON(t, EmailQuestions())
			} else {
				in, err := jsonx.Decode(rec.Input)
				if err != nil {
					t.Fatalf("decode recorded input: %v", err)
				}
				obj, ok := in.(jsonx.Obj)
				if !ok {
					t.Fatalf("recorded input is %T, want an object of categories", in)
				}
				cats := make([]question.ChoiceOption, 0, len(obj))
				for _, f := range obj {
					cats = append(cats, question.ChoiceOption{Key: f.Key, Desc: f.Value})
				}
				got = qsetJSON(t, EmailQuestionsWith(cats))
			}

			if got != string(wantJSON) {
				t.Errorf("preset does not match the Python it was ported from\n got: %s\nwant: %s",
					got, wantJSON)
			}
		})
	}
}

// TestPresetsValidate is task 2.4.2: the shape test upstream never had. A
// preset that cannot pass the validation every caller-supplied question set
// goes through would fail at the first call rather than here.
func TestPresetsValidate(t *testing.T) {
	for name, qs := range All() {
		t.Run(name, func(t *testing.T) {
			if err := qs.Validate(); err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			if len(qs) == 0 {
				t.Fatal("empty preset")
			}
			for _, nq := range qs {
				if nq.ID == "" {
					t.Error("a question has an empty id")
				}
				if nq.Q.Instructions() == "" {
					t.Errorf("%s: empty instructions; the prompt would ask nothing", nq.ID)
				}
				if opts := nq.Q.RenderOptions(); len(opts) < 2 {
					t.Errorf("%s: %d rendered options, want at least 2 -- a decision "+
						"needs something to decide between", nq.ID, len(opts))
				}
			}
		})
	}
}

// TestPresetOptionOrder is task 2.4.3. The option index is the answer index:
// probabilities come back positionally, so reordering a preset's options
// silently relabels every answer it has ever produced.
func TestPresetOptionOrder(t *testing.T) {
	for _, tc := range []struct {
		preset, id string
		want       []string
	}{
		{"triage", "intent", []string{
			"refund: money returned or a duplicate charge reversed",
			"technical_help: a bug, outage or integration problem",
			"billing_question: a question about an invoice, plan or payment method",
			"information: general information, pricing or how-to",
			"cancellation: wants to cancel or downgrade",
			"other: none of the other options fits",
		}},
		{"guard", "topic", []string{
			"product_support", "coding", "general_knowledge",
			"personal_advice", "security_testing", "other",
		}},
		{"moderation", "severity", []string{
			"level 0: no rule-breaking: ordinary on-topic post",
			"level 1: mild: rude tone or off-topic, no target",
			"level 2: clear violation: insults, harassment or spam aimed at someone",
			"level 3: severe: threats, hate speech or calls for violence",
		}},
		{"email", "is_phishing", []string{
			"false: a legitimate email",
			"true: phishing, scam, or fraud",
		}},
	} {
		t.Run(tc.preset+"/"+tc.id, func(t *testing.T) {
			q, ok := All()[tc.preset].Get(tc.id)
			if !ok {
				t.Fatalf("%s has no question %q", tc.preset, tc.id)
			}
			got := q.RenderOptions()
			if len(got) != len(tc.want) {
				t.Fatalf("got %d options, want %d: %q", len(got), len(tc.want), got)
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Errorf("option %d = %q, want %q (the index is the answer index)",
						i, got[i], tc.want[i])
				}
			}
		})
	}
}

// TestEmailCategoriesFallback pins upstream's `categories or {...}`: an empty
// taxonomy is falsy in Python and selects the defaults rather than producing a
// choice question with no options.
func TestEmailCategoriesFallback(t *testing.T) {
	def := qsetJSON(t, EmailQuestions())
	for _, empty := range [][]question.ChoiceOption{nil, {}} {
		if got := qsetJSON(t, EmailQuestionsWith(empty)); got != def {
			t.Errorf("an empty category list must fall back to the defaults, like `categories or {...}`")
		}
	}
}

// TestAllIsNotShared guards the slice-aliasing trap a Python function call does
// not have: All returns fresh values, so a caller mutating one preset cannot
// change what the next caller sees.
func TestAllIsNotShared(t *testing.T) {
	a := TriageQuestions()
	before := a[0].ID
	a[0].ID = "mutated"
	if TriageQuestions()[0].ID != before {
		t.Error("presets share backing state; one caller can corrupt another's questions")
	}
}
