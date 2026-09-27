package laya

import (
	"bytes"
	"context"
	"encoding/json"
	"slices"
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/golden"
	"github.com/MeKo-Christian/go-laya/jsonx"
)

// TestAnswerParity is Task 7.4: every answers.jsonl case, sent through
// SystemOne rather than straight into formatAnswer, must come back
// byte-equal to Python's json.dumps.
//
// The fixture holds no state and no batch, so internal/backend/fake, which
// answers only the recorded logits.jsonl batches, cannot serve it. A one-row
// stub returns the case's recorded logits instead; the config carries the
// case's temperatures, so the temperatures and k SystemOne picks are the
// ones the answer was computed with.
//
// noul always renders two options, so a noul case recorded with k != 2 is a
// state SystemOne cannot reach. Those stay TestAnswerGolden's alone.
func TestAnswerParity(t *testing.T) {
	const unreachable = 3 // noul/k03, noul/k06, noul/k11

	skipped := 0
	for _, c := range golden.Load(t, "answers") {
		var rec answerFixture
		c.Unmarshal(t, &rec)
		if rec.QType == "noul" && rec.K != 2 {
			skipped++
			continue
		}

		t.Run(c.Name, func(t *testing.T) {
			be := &recordingBackend{logits: [][]float32{rec.Logits}, act: [][]float32{rec.ActLogits}}
			a := testAgent(t, parityConfig(t, rec), be)

			res, err := a.SystemOne(context.Background(), "state",
				Questions{{ID: "q", Q: fixtureQuestion(t, rec)}})
			if err != nil {
				t.Fatal(err)
			}
			if len(be.calls) != 1 || len(be.calls[0].InputIDs) != 1 {
				t.Fatalf("forward passes = %d, want one with one row", len(be.calls))
			}
			if k := countTrue(be.calls[0].MarkerMask[0]); k != rec.K {
				t.Fatalf("markers = %d, want the recorded k = %d", k, rec.K)
			}

			ans := res.Answers[0].A
			got, err := jsonx.Marshal(ans.Map())
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			if string(got) != rec.AnswerJSON {
				t.Errorf("answer bytes\n got %s\nwant %s", got, rec.AnswerJSON)
			}
			if keys := objKeys(ans.Map()); !slices.Equal(keys, answerKeys[rec.QType]) {
				t.Errorf("keys = %q, want %q", keys, answerKeys[rec.QType])
			}

			viaStd, err := json.Marshal(ans)
			if err != nil {
				t.Fatalf("json.Marshal: %v", err)
			}
			var compact bytes.Buffer
			if err := json.Compact(&compact, []byte(rec.AnswerJSON)); err != nil {
				t.Fatalf("compact: %v", err)
			}
			if !bytes.Equal(viaStd, compact.Bytes()) {
				t.Errorf("json.Marshal\n got %s\nwant %s", viaStd, compact.Bytes())
			}
		})
	}
	// A skip that grows silently would hide cases.
	if skipped != unreachable {
		t.Errorf("skipped %d noul cases with k != 2, want %d", skipped, unreachable)
	}
}

// parityConfig is testConfig plus the case's temperatures.
func parityConfig(t *testing.T, rec answerFixture) string {
	t.Helper()
	var cfg map[string]any
	if err := json.Unmarshal([]byte(testConfig), &cfg); err != nil {
		t.Fatal(err)
	}
	cfg["temperature"] = rec.Temperature
	cfg["temperature_by_options"] = rec.TemperatureByOptions
	raw, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

func countTrue(row []bool) int {
	n := 0
	for _, b := range row {
		if b {
			n++
		}
	}
	return n
}
