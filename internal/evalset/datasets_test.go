package evalset

import (
	"encoding/json"
	"fmt"
	"slices"
	"testing"

	"github.com/MeKo-Christian/go-laya/jsonx"
	"github.com/MeKo-Christian/go-laya/question"
)

// TestSpecsMapRows feeds each dataset one row exactly as the datasets-server
// rows API returns it, and checks the whole way to the question: the text and
// label pulled out of the row, the state the model reads, the question type
// and size, and that the label index lands on the option or level that means
// what the dataset's ClassLabel name says.
func TestSpecsMapRows(t *testing.T) {
	tests := []struct {
		name    string
		row     string
		text    string
		label   int
		field   string
		qtype   question.QType
		k       int
		meaning string // the option key or level text at the label index
	}{
		{
			name: "ag_news",
			row:  `{"text":"Wall St. Bears Claw Back","label":2}`,
			text: "Wall St. Bears Claw Back", label: 2,
			field: "article", qtype: question.Choice, k: 4, meaning: "business",
		},
		{
			name: "emotion",
			row:  `{"text":"i feel so alone","label":0}`,
			text: "i feel so alone", label: 0,
			field: "text", qtype: question.Choice, k: 6, meaning: "sadness",
		},
		{
			name: "emotion",
			row:  `{"text":"i was shocked","label":5}`,
			text: "i was shocked", label: 5,
			field: "text", qtype: question.Choice, k: 6, meaning: "surprise",
		},
		{
			name: "sst2",
			row:  `{"idx":3,"sentence":"the acting is astounding . ","label":1}`,
			text: "the acting is astounding . ", label: 1,
			field: "review", qtype: question.Noul, k: 2, meaning: "true",
		},
		{
			name: "sst2",
			row:  `{"idx":4,"sentence":"a dull mess . ","label":0}`,
			text: "a dull mess . ", label: 0,
			field: "review", qtype: question.Noul, k: 2, meaning: "false",
		},
		{
			name: "yelp",
			row:  `{"label":0,"text":"Never again."}`,
			text: "Never again.", label: 0,
			field: "review", qtype: question.Score, k: 5, meaning: "1 star: very negative",
		},
		{
			name: "yelp",
			row:  `{"label":4,"text":"Best tacos in town."}`,
			text: "Best tacos in town.", label: 4,
			field: "review", qtype: question.Score, k: 5, meaning: "5 stars: very positive",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name+"/"+tt.meaning, func(t *testing.T) {
			spec, ok := SpecByName(tt.name)
			if !ok {
				t.Fatalf("no spec %q", tt.name)
			}
			row, err := spec.Extract(17, json.RawMessage(tt.row))
			if err != nil {
				t.Fatalf("Extract: %v", err)
			}
			if row != (Row{Offset: 17, Text: tt.text, Label: tt.label}) {
				t.Fatalf("Extract = %+v, want offset 17, %q, label %d", row, tt.text, tt.label)
			}

			c, err := spec.Case(row)
			if err != nil {
				t.Fatalf("Case: %v", err)
			}
			if want := (jsonx.Obj{{Key: tt.field, Value: tt.text}}); !slices.Equal(c.State, want) {
				t.Errorf("state = %v, want %v", c.State, want)
			}
			if len(c.Questions) != 1 || c.Questions[0].ID != QuestionID {
				t.Fatalf("questions = %+v, want one named %q", c.Questions, QuestionID)
			}
			if err := c.Questions.Validate(); err != nil {
				t.Errorf("question does not validate: %v", err)
			}
			q := c.Questions[0].Q
			if q.Type() != tt.qtype || c.Type != tt.qtype {
				t.Errorf("type = %v / %v, want %v", q.Type(), c.Type, tt.qtype)
			}
			if c.Classes != tt.k {
				t.Errorf("classes = %d, want %d", c.Classes, tt.k)
			}
			if c.Label != tt.label {
				t.Errorf("label = %d, want %d", c.Label, tt.label)
			}
			if got := meaningOf(t, q, c.Label); got != tt.meaning {
				t.Errorf("label %d means %q, want %q", c.Label, got, tt.meaning)
			}
		})
	}
}

// meaningOf is what index i stands for in q: the option key of a choice, the
// level text of a score, false/true for a noul (index 1 is P(true)).
func meaningOf(t *testing.T, q question.Question, i int) string {
	t.Helper()
	if n := (Spec{Question: q}).classes(); i < 0 || i >= n {
		return fmt.Sprintf("<index %d of %d>", i, n)
	}
	switch q := q.(type) {
	case question.ChoiceQuestion:
		return q.Opts[i].Key
	case question.ScoreQuestion:
		s, _ := q.Levels[i].(string)
		return s
	case question.NoulQuestion:
		return [...]string{"false", "true"}[i]
	}
	t.Fatalf("unexpected question %T", q)
	return ""
}

// TestSpecClassNames pins each spec's class names to the ClassLabel names the
// rows API reported on 2026-10-11, in order. Fetch refuses a dataset whose
// names differ, which is what keeps label i meaning option i.
func TestSpecClassNames(t *testing.T) {
	want := map[string][]string{
		"ag_news": {"World", "Sports", "Business", "Sci/Tech"},
		"emotion": {"sadness", "joy", "love", "anger", "fear", "surprise"},
		"sst2":    {"negative", "positive"},
		"yelp":    {"1 star", "2 star", "3 stars", "4 stars", "5 stars"},
	}
	if len(Specs) != len(want) {
		t.Fatalf("%d specs, want %d", len(Specs), len(want))
	}
	for _, s := range Specs {
		if !slices.Equal(s.ClassNames, want[s.Name]) {
			t.Errorf("%s: class names = %q, want %q", s.Name, s.ClassNames, want[s.Name])
		}
		if s.classes() != len(s.ClassNames) {
			t.Errorf("%s: %d classes for %d names", s.Name, s.classes(), len(s.ClassNames))
		}
	}
}

func TestExtractRejects(t *testing.T) {
	spec, _ := SpecByName("ag_news")
	for name, row := range map[string]string{
		"label out of range": `{"text":"x","label":4}`,
		"negative label":     `{"text":"x","label":-1}`,
		"fractional label":   `{"text":"x","label":1.5}`,
		"no text":            `{"label":1}`,
		"empty text":         `{"text":"","label":1}`,
		"text not a string":  `{"text":3,"label":1}`,
		"not an object":      `[1,2]`,
	} {
		t.Run(name, func(t *testing.T) {
			if r, err := spec.Extract(0, json.RawMessage(row)); err == nil {
				t.Fatalf("Extract = %+v, want an error", r)
			}
		})
	}
	if _, err := spec.Case(Row{Text: "x", Label: 4}); err == nil {
		t.Error("Case accepted label 4 of 4 classes")
	}
}
