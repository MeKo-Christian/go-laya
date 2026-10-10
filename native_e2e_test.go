package laya

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/golden"
	"github.com/MeKo-Christian/go-laya/jsonx"
)

// TestOpenNativeGolden is Task 8.10 end to end: laya.Open with
// WithRuntime(RuntimeNative) on each local checkpoint, with no export and no
// network, answers one recorded logits.jsonl case, and every answer's most
// probable option is the one Python's recorded logits pick. Task 8.8 holds
// the probabilities themselves to 1e-4 over every case; this is the public
// path's smoke test. Each checkpoint is closed before the next is loaded.
func TestOpenNativeGolden(t *testing.T) {
	root := golden.SkipWithoutModels(t)
	t.Setenv("LAYA_ONNX_DIR", t.TempDir()) // holds no export

	type logitsCase struct {
		Checkpoint string          `json:"checkpoint"`
		State      json.RawMessage `json:"state"`
		Questions  json.RawMessage `json:"questions"`
		QIDs       []string        `json:"qids"`
		Logits     json.RawMessage `json:"logits"`
	}
	cases := golden.Load(t, "logits")

	for _, ck := range []string{golden.English, golden.Multilingual, golden.TypedDecisions} {
		t.Run(ck, func(t *testing.T) {
			var c logitsCase
			var name string
			for _, rec := range cases {
				rec.Unmarshal(t, &c)
				if c.Checkpoint == ck {
					name = rec.Name
					break
				}
			}
			if name == "" {
				t.Fatalf("logits.jsonl has no case for %s", ck)
			}

			a, err := Open(t.Context(), golden.CheckpointDir(root, ck), WithRuntime(RuntimeNative))
			if err != nil {
				t.Fatalf("Open: %v", err)
			}
			defer func() {
				if err := a.Close(); err != nil {
					t.Errorf("Close: %v", err)
				}
			}()

			state, err := jsonx.Decode(c.State)
			if err != nil {
				t.Fatal(err)
			}
			res, err := a.SystemOne(context.Background(), state, recordedQuestions(t, c.Questions))
			if err != nil {
				t.Fatalf("%s: SystemOne: %v", name, err)
			}
			want := golden.Matrix[float32](t, c.Logits, name+" logits")
			if len(res.Answers) != len(c.QIDs) || len(want) != len(c.QIDs) {
				t.Fatalf("%s: %d answers and %d logit rows, recorded %d questions",
					name, len(res.Answers), len(want), len(c.QIDs))
			}
			for r, na := range res.Answers {
				if na.ID != c.QIDs[r] {
					t.Fatalf("%s: answer %d is %q, want %q", name, r, na.ID, c.QIDs[r])
				}
				// A noul answer is P(true), the second of its two columns
				// (false = 0, true = 1); the others carry every option's.
				var p []float32
				if na.A.Type == "noul" {
					p = []float32{float32(1 - *na.A.Noul), float32(*na.A.Noul)}
				} else {
					for _, e := range na.A.Probabilities {
						p = append(p, float32(e.P))
					}
				}
				// The masked columns hold -1e4, so the recorded row's argmax
				// is over the real options.
				got, wantArg := argmax(p), argmax(want[r])
				if got != wantArg {
					t.Errorf("%s %s: argmax %d, Python %d\n got %v\nlogits %v", name, na.ID, got, wantArg, p, want[r])
				}
				t.Logf("%s %s (%s): argmax %d of p %v, Python's logits %v", name, na.ID, na.A.Type, got, p, want[r])
			}
		})
	}
}
