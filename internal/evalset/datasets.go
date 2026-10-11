package evalset

import (
	"bytes"
	"encoding/json"
	"fmt"
	"slices"
	"strconv"

	"github.com/MeKo-Christian/go-laya/jsonx"
	"github.com/MeKo-Christian/go-laya/question"
)

// QuestionID is the id every case asks its one question under.
const QuestionID = "q"

// The split and the fields most of the datasets share.
const (
	splitTest  = "test"
	textField  = "text"
	labelField = "label"
)

// Spec is one dataset: where it lives on the Hub, which row fields hold the
// text and the label, and the question each row becomes. Label i is option i
// of a choice, level i of a score, and false (0) or true (1) of a noul; that
// only holds while ClassNames is the dataset's ClassLabel order, which Fetch
// checks.
type Spec struct {
	Name       string
	Dataset    string
	Config     string
	Split      string
	TextField  string
	LabelField string
	ClassNames []string
	StateKey   string
	Question   question.Question
}

// Specs are B.5.3's four datasets, one per question type plus a second choice
// with more options. Each split is the one with public labels: SST-2's test
// labels are hidden (-1), so it uses validation.
var Specs = []Spec{
	{
		Name: "ag_news", Dataset: "fancyzhx/ag_news", Config: "default", Split: splitTest,
		TextField: textField, LabelField: labelField,
		ClassNames: []string{"World", "Sports", "Business", "Sci/Tech"},
		StateKey:   "article",
		Question: question.ChoiceQuestion{
			Ins: "What is the topic of this news article?",
			Opts: []question.ChoiceOption{
				{Key: "world", Desc: "world news: international affairs, politics, conflicts"},
				{Key: "sports", Desc: "sports"},
				{Key: "business", Desc: "business: companies, markets, the economy"},
				{Key: "sci_tech", Desc: "science and technology"},
			},
		},
	},
	{
		Name: "emotion", Dataset: "dair-ai/emotion", Config: "split", Split: splitTest,
		TextField: textField, LabelField: labelField,
		ClassNames: []string{"sadness", "joy", "love", "anger", "fear", "surprise"},
		StateKey:   "text",
		Question: question.ChoiceQuestion{
			Ins:  "Which emotion does the author express?",
			Opts: question.Labels("sadness", "joy", "love", "anger", "fear", "surprise"),
		},
	},
	{
		Name: "sst2", Dataset: "stanfordnlp/sst2", Config: "default", Split: "validation",
		TextField: "sentence", LabelField: labelField,
		ClassNames: []string{"negative", "positive"},
		StateKey:   "review",
		Question:   question.NoulQuestion{Ins: "Is the sentiment of this movie review positive?"},
	},
	{
		Name: "yelp", Dataset: "Yelp/yelp_review_full", Config: "yelp_review_full", Split: splitTest,
		TextField: textField, LabelField: labelField,
		ClassNames: []string{"1 star", "2 star", "3 stars", "4 stars", "5 stars"},
		StateKey:   "review",
		Question: question.ScoreQuestion{
			Ins: "How many stars does this review give?",
			Levels: []question.Criterion{
				"1 star: very negative", "2 stars: negative", "3 stars: mixed",
				"4 stars: positive", "5 stars: very positive",
			},
		},
	},
}

// SpecByName returns the dataset called name.
func SpecByName(name string) (Spec, bool) {
	i := slices.IndexFunc(Specs, func(s Spec) bool { return s.Name == name })
	if i < 0 {
		return Spec{}, false
	}
	return Specs[i], true
}

// Extract pulls the text and the label out of one row as the rows API
// returns it ("row": {...}). The text must be a non-empty string and the label
// an integer index of the question's answers.
func (s Spec) Extract(offset int, raw json.RawMessage) (Row, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return Row{}, fmt.Errorf("evalset: %s row %d: %w", s.Name, offset, err)
	}
	var text string
	if err := json.Unmarshal(fields[s.TextField], &text); err != nil || text == "" {
		return Row{}, fmt.Errorf("evalset: %s row %d: field %q is not a non-empty string", s.Name, offset, s.TextField)
	}
	dec := json.NewDecoder(bytes.NewReader(fields[s.LabelField]))
	dec.UseNumber()
	var num json.Number
	if err := dec.Decode(&num); err != nil {
		return Row{}, fmt.Errorf("evalset: %s row %d: label: %w", s.Name, offset, err)
	}
	label, err := strconv.Atoi(num.String())
	if err != nil {
		return Row{}, fmt.Errorf("evalset: %s row %d: label %s is not an integer", s.Name, offset, num)
	}
	r := Row{Offset: offset, Text: text, Label: label}
	if err := s.checkLabel(r); err != nil {
		return Row{}, err
	}
	return r, nil
}

// Case is one row as the model is asked it: the state, its one question, and
// the index of the right answer among the question's Classes answers.
type Case struct {
	State     jsonx.Obj
	Questions question.Questions
	Type      question.QType
	Classes   int
	Label     int
}

// Case turns r into the question the model answers.
func (s Spec) Case(r Row) (Case, error) {
	if err := s.checkLabel(r); err != nil {
		return Case{}, err
	}
	return Case{
		State:     jsonx.Obj{{Key: s.StateKey, Value: r.Text}},
		Questions: question.Questions{{ID: QuestionID, Q: s.Question}},
		Type:      s.Question.Type(),
		Classes:   s.classes(),
		Label:     r.Label,
	}, nil
}

// classes is the number of answers the question has, which every label must
// index.
func (s Spec) classes() int {
	switch q := s.Question.(type) {
	case question.ChoiceQuestion:
		return len(q.Opts)
	case question.ScoreQuestion:
		return len(q.Levels)
	default:
		return 2 // a noul is always false, true
	}
}

func (s Spec) checkLabel(r Row) error {
	if r.Label < 0 || r.Label >= s.classes() {
		return fmt.Errorf("evalset: %s row %d: label %d outside [0, %d)", s.Name, r.Offset, r.Label, s.classes())
	}
	return nil
}
