package calib

import (
	"bytes"
	"encoding/json"
	"math"
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/golden"
	"github.com/MeKo-Christian/go-laya/jsonx"
	"github.com/MeKo-Christian/go-laya/question"
)

// TestTempBucket pins invariant #22 at every threshold: the size is "2" up to
// k=2, "3-5" up to 5, "6-10" up to 10, "11+" beyond (common.py:209-211).
func TestTempBucket(t *testing.T) {
	sizes := []struct {
		k    int
		want string
	}{
		{1, "2"}, {2, "2"}, {3, "3-5"}, {5, "3-5"}, {6, "6-10"}, {10, "6-10"}, {11, "11+"}, {13, "11+"},
	}
	for _, qt := range []question.QType{question.Choice, question.Score, question.Noul} {
		for _, s := range sizes {
			want := qt.String() + ":" + s.want
			if got := TempBucket(qt, s.k); got != want {
				t.Errorf("TempBucket(%v, %d) = %q, want %q", qt, s.k, got, want)
			}
		}
	}
}

// TestScale pins invariant #23: a fitted bucket wins, otherwise the per-type
// temperature, and the defaults are [1, 1, 1] with no buckets at all -- which
// is what laya-multilingual ships (README:355).
func TestScale(t *testing.T) {
	temps := Temperatures{
		ByQType:   [3]float64{1.5, 2.5, 3.5},
		ByOptions: map[string]float64{"choice:3-5": 0.7, "noul:2": 0.9},
	}
	for _, tc := range []struct {
		qt   question.QType
		k    int
		want float64
	}{
		{question.Choice, 4, 0.7}, // bucket hit
		{question.Choice, 2, 1.5}, // choice:2 not fitted -> temperature[choice]
		{question.Score, 3, 2.5},  // score:3-5 not fitted -> temperature[score]
		{question.Noul, 2, 0.9},   // bucket hit
		{question.Noul, 3, 3.5},   // noul:3-5 not fitted
	} {
		if got := temps.Scale(tc.qt, tc.k); got != tc.want {
			t.Errorf("Scale(%v, %d) = %v, want %v", tc.qt, tc.k, got, tc.want)
		}
	}

	def := DefaultTemperatures()
	for _, qt := range []question.QType{question.Choice, question.Score, question.Noul} {
		if got := def.Scale(qt, 4); got != 1 {
			t.Errorf("default Scale(%v, 4) = %v, want 1", qt, got)
		}
	}
	// An empty map is multilingual's shape; it must fall back, not panic.
	empty := Temperatures{ByQType: [3]float64{1.5, 2.5, 3.5}, ByOptions: map[string]float64{}}
	if got := empty.Scale(question.Score, 11); got != 2.5 {
		t.Errorf("empty ByOptions Scale = %v, want 2.5", got)
	}
}

// TestTemperatureFloor pins the max(1e-3, t) guard (invariant #24): a zero or
// negative temperature divides by 1e-3, not by itself.
func TestTemperatureFloor(t *testing.T) {
	logits := []float32{0.001, 0.002, 0}
	want := Softmax(logits, 3, 1e-3)
	for _, temp := range []float64{0, -1, 1e-9} {
		got := Softmax(logits, 3, temp)
		for i := range want {
			if got[i] != want[i] || math.IsNaN(float64(got[i])) {
				t.Fatalf("Softmax(t=%v) = %v, want the t=1e-3 result %v", temp, got, want)
			}
		}
	}
	// And the floor is a floor, not a constant: a larger t is used as given.
	if a, b := Softmax(logits, 3, 1e-3), Softmax(logits, 3, 1e-2); a[0] == b[0] {
		t.Errorf("t=1e-2 gave the same result as t=1e-3: %v", a)
	}
}

// TestSoftmaxFirstK pins that only logits[:k] take part (invariant #24): the
// head emits a fixed-width row and the tail past k is padding.
func TestSoftmaxFirstK(t *testing.T) {
	a := Softmax([]float32{1, 2, 3, 0, 0}, 3, 1)
	b := Softmax([]float32{1, 2, 3, 100, -100}, 3, 1)
	if len(a) != 3 || len(b) != 3 {
		t.Fatalf("len = %d, %d, want 3", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("logits past k changed the result: %v vs %v", a, b)
		}
	}
	// Max-subtracted: a logit that would overflow exp() on its own still works.
	p := Softmax([]float32{200, 0, 0}, 3, 1)
	if p[0] != 1 || p[1] != 0 {
		t.Errorf("Softmax of a 200 logit = %v, want [1 0 0]", p)
	}
}

// TestConfidence pins invariant #25's edges and #26's separate path: for a
// uniform k=2 distribution the entropy form gives 0 while noul's
// max(p1, 1-p1) gives 0.5, so the two must never be unified.
func TestConfidence(t *testing.T) {
	uniform := []float32{0.5, 0.5}
	if got := Confidence(uniform, 2); got != 0 {
		t.Errorf("Confidence(uniform, 2) = %v, want 0", got)
	}
	if got := NoulConfidence(uniform[1]); got != 0.5 {
		t.Errorf("NoulConfidence(0.5) = %v, want 0.5", got)
	}
	if got := Confidence([]float32{1}, 1); got != 1 {
		t.Errorf("Confidence(k=1) = %v, want 1", got)
	}
	// A certain outcome hits the log clip (p=0 -> ln 1e-12) and still gives 1.
	if got := Confidence([]float32{1, 0, 0}, 3); got != 1 {
		t.Errorf("Confidence(certain) = %v, want 1", got)
	}
	// The [0, 1] clip is not decoration. In float32 a uniform distribution's
	// entropy can land a bit above ln k (k=6 gives 1 - H/ln k = -1.2e-07), and
	// Round4 would turn that into -0.0, which the JSON shows.
	for k := 2; k <= 16; k++ {
		if got := Confidence(Softmax(make([]float32, k), k, 1), k); got < 0 || math.Signbit(got) {
			t.Errorf("Confidence(uniform, %d) = %v, want >= +0", k, got)
		}
	}
	// Only the first k count.
	if a, b := Confidence([]float32{0.5, 0.5, 0.9}, 2), Confidence([]float32{0.5, 0.5}, 2); a != b {
		t.Errorf("Confidence read past k: %v vs %v", a, b)
	}
}

// answerCase is the part of an answers.jsonl record this package reproduces.
type answerCase struct {
	QType                string             `json:"qtype"`
	K                    int                `json:"k"`
	Logits               []float32          `json:"logits"`
	Temperature          [3]float64         `json:"temperature"`
	TemperatureByOptions map[string]float64 `json:"temperature_by_options"`
	Answer               struct {
		Probabilities json.RawMessage `json:"probabilities"`
		Confidence    float64         `json:"confidence"`
		Score         *float64        `json:"score"`
		Noul          *float64        `json:"noul"`
	} `json:"answer"`
}

// TestAnswersNumerics replays every testdata/answers.jsonl case through the
// temperature lookup, the softmax and the confidence, rounding with
// jsonx.Round4 exactly where agent.py:309-335 calls round(), and requires the
// emitted floats to be equal -- not close -- to Python's (invariants #22-#29).
//
// The act head and the answer shapes are Task 7.2's; only the numbers the
// calibration produces are asserted here.
func TestAnswersNumerics(t *testing.T) {
	cases := golden.Load(t, "answers")
	if len(cases) != 27 {
		t.Fatalf("answers.jsonl has %d cases, want 27", len(cases))
	}
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			var rec answerCase
			c.Unmarshal(t, &rec)

			qt, ok := qtypes[rec.QType]
			if !ok {
				t.Fatalf("unknown qtype %q", rec.QType)
			}
			temps := Temperatures{ByQType: rec.Temperature, ByOptions: rec.TemperatureByOptions}
			p := Softmax(rec.Logits, rec.K, temps.Scale(qt, rec.K))

			switch qt {
			case question.Choice, question.Score:
				want := orderedFloats(t, rec.Answer.Probabilities)
				if len(want) != len(p) {
					t.Fatalf("len(p) = %d, want %d", len(p), len(want))
				}
				for i, w := range want {
					if got := jsonx.Round4(float64(p[i])); got != w {
						t.Errorf("p[%d] = %v (round4 %v), want %v", i, p[i], got, w)
					}
				}
				if got := jsonx.Round4(Confidence(p, rec.K)); got != rec.Answer.Confidence {
					t.Errorf("confidence = %v, want %v", got, rec.Answer.Confidence)
				}
				if qt == question.Score {
					if rec.Answer.Score == nil {
						t.Fatal("score case without a score")
					}
					if got := jsonx.Round4(Expectation(p)); got != *rec.Answer.Score {
						t.Errorf("score = %v, want %v", got, *rec.Answer.Score)
					}
				}
			case question.Noul:
				if rec.Answer.Noul == nil {
					t.Fatal("noul case without a noul value")
				}
				// Index 1 is P(true) (invariant #27).
				if got := jsonx.Round4(float64(p[1])); got != *rec.Answer.Noul {
					t.Errorf("noul = %v, want %v", got, *rec.Answer.Noul)
				}
				if got := jsonx.Round4(NoulConfidence(p[1])); got != rec.Answer.Confidence {
					t.Errorf("confidence = %v, want %v", got, rec.Answer.Confidence)
				}
			}
		})
	}
}

var qtypes = map[string]question.QType{
	"choice": question.Choice,
	"score":  question.Score,
	"noul":   question.Noul,
}

// orderedFloats returns a JSON object's values in document order; the
// probabilities are keyed by criterion and the order is the criteria's.
func orderedFloats(t *testing.T, raw json.RawMessage) []float64 {
	t.Helper()
	dec := json.NewDecoder(bytes.NewReader(raw))
	if _, err := dec.Token(); err != nil { // {
		t.Fatalf("probabilities: %v", err)
	}
	var out []float64
	for dec.More() {
		if _, err := dec.Token(); err != nil { // key
			t.Fatalf("probabilities: %v", err)
		}
		var v float64
		if err := dec.Decode(&v); err != nil {
			t.Fatalf("probabilities: %v", err)
		}
		out = append(out, v)
	}
	return out
}
