package laya

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/MeKo-Christian/go-laya/backend"
	"github.com/MeKo-Christian/go-laya/internal/backend/onnx"
	"github.com/MeKo-Christian/go-laya/internal/calib"
	"github.com/MeKo-Christian/go-laya/internal/checkpoint"
	"github.com/MeKo-Christian/go-laya/internal/golden"
	"github.com/MeKo-Christian/go-laya/jsonx"
	"github.com/MeKo-Christian/go-laya/question"
	"github.com/MeKo-Christian/go-laya/tokenizer"
)

// e2eTol is the largest per-option probability difference TestE2EParity
// accepts between SystemOne over ONNX Runtime and the probabilities Python's
// recorded logits give, per checkpoint (Task 7.5.2).
//
// Measured over every logits.jsonl case (2026-09-27, ORT 1.23.0 on the dynamo
// exports): english 4.3e-06, multilingual 2.8e-06, typed-decisions 1.4e-06,
// with no argmax flipping. Each gets 2x headroom; TestE2EParity also holds
// every entry to the item's 1e-4 ceiling.
var e2eTol = map[string]float64{
	golden.English:        8.6e-6,
	golden.Multilingual:   5.6e-6,
	golden.TypedDecisions: 2.8e-6,
}

// e2eCeiling is Task 7.5.2's bound on any per-checkpoint tolerance.
const e2eCeiling = 1e-4

// roundStep is how far apart two values within e2eTol can land once both are
// rounded to four places, plus room for the float64 representation.
const roundStep = 1e-4 + 1e-9

// teeBackend forwards to a real backend and keeps what the last pass saw and
// returned, so the test can compare the logits behind an answer rather than
// its rounded JSON.
type teeBackend struct {
	backend.Backend

	batch       backend.Batch
	logits, act [][]float32
}

func (b *teeBackend) Forward(ctx context.Context, in backend.Batch) (logits, act [][]float32, err error) {
	logits, act, err = b.Backend.Forward(ctx, in)
	b.batch, b.logits, b.act = in, logits, act
	return logits, act, err
}

// TestE2EParity is Task 7.5.1-7.5.3: every logits.jsonl case through SystemOne
// with the real tokenizer and ONNX Runtime, for all three checkpoints. The
// per-option probabilities must agree with the ones Python's recorded logits
// give at the same temperature, and no row's argmax may flip.
//
// TestSystemOneReplay already proves the batch equals the recorded one, so
// what is left to differ here is the forward pass.
func TestE2EParity(t *testing.T) {
	root := golden.SkipWithoutModels(t)
	exports := os.Getenv("LAYA_ONNX_DIR")
	if exports == "" {
		t.Skip("no exports (set LAYA_ONNX_DIR)")
	}
	for ck, tol := range e2eTol {
		if tol > e2eCeiling {
			t.Errorf("e2eTol[%s] = %g exceeds the 1e-4 ceiling", ck, tol)
		}
	}

	type logitsCase struct {
		Checkpoint string          `json:"checkpoint"`
		State      json.RawMessage `json:"state"`
		Questions  json.RawMessage `json:"questions"`
		QIDs       []string        `json:"qids"`
		QTypes     []int           `json:"qtypes"`
		Logits     json.RawMessage `json:"logits"`
		ResultJSON string          `json:"result_json"`
	}
	cases := golden.Load(t, "logits")

	for _, ck := range []string{golden.English, golden.Multilingual, golden.TypedDecisions} {
		t.Run(ck, func(t *testing.T) {
			graph := filepath.Join(exports, "laya-"+ck+"-dynamo.onnx")
			if _, err := os.Stat(graph); err != nil {
				t.Skipf("no export: %v (run scripts/export_onnx.py --all --dynamo)", err)
			}
			tee, a := e2eAgent(t, golden.CheckpointDir(root, ck), graph)

			worst, n := 0.0, 0
			for _, rec := range cases {
				var c logitsCase
				rec.Unmarshal(t, &c)
				if c.Checkpoint != ck {
					continue
				}
				n++

				state, err := jsonx.Decode(c.State)
				if err != nil {
					t.Fatal(err)
				}
				res, err := a.SystemOne(context.Background(), state, recordedQuestions(t, c.Questions))
				if err != nil {
					t.Fatalf("%s: SystemOne: %v", rec.Name, err)
				}
				want := golden.Matrix[float32](t, c.Logits, rec.Name+" logits")
				if len(tee.logits) != len(want) || len(res.Answers) != len(c.QIDs) {
					t.Fatalf("%s: %d logit rows and %d answers, recorded %d and %d",
						rec.Name, len(tee.logits), len(res.Answers), len(want), len(c.QIDs))
				}

				var py struct {
					Answers map[string]struct {
						Choice        string             `json:"choice"`
						Noul          *float64           `json:"noul"`
						Probabilities map[string]float64 `json:"probabilities"`
					} `json:"answers"`
				}
				if err := json.Unmarshal([]byte(c.ResultJSON), &py); err != nil {
					t.Fatalf("%s: result_json: %v", rec.Name, err)
				}

				for r, na := range res.Answers {
					if na.ID != c.QIDs[r] {
						t.Fatalf("%s: answer %d is %q, want %q", rec.Name, r, na.ID, c.QIDs[r])
					}
					k := countTrue(tee.batch.MarkerMask[r])
					temp := a.temps.Scale(question.QType(c.QTypes[r]), k)
					pGo := calib.Softmax(tee.logits[r], k, temp)
					pPy := calib.Softmax(want[r], k, temp)

					for i := range pPy {
						d := math.Abs(float64(pGo[i]) - float64(pPy[i]))
						worst = max(worst, d)
						if d > e2eTol[ck] {
							t.Errorf("%s %s: p[%d] = %.8g, Python %.8g (diff %.3g > %g)",
								rec.Name, na.ID, i, pGo[i], pPy[i], d, e2eTol[ck])
						}
					}
					if g, p := argmax(pGo), argmax(pPy); g != p {
						t.Errorf("%s %s: argmax %d, Python %d\n got %v\nwant %v", rec.Name, na.ID, g, p, pGo, pPy)
					}

					// The formatted answer, against what Python's system_one
					// returned: the same decision, and probabilities that, both
					// rounded to four places, sit at most one step apart. That
					// is also what catches a wrong temperature, which moves no
					// argmax.
					pa := py.Answers[na.ID]
					for _, e := range na.A.Probabilities {
						if pp, ok := pa.Probabilities[e.Key]; !ok || math.Abs(e.P-pp) > roundStep {
							t.Errorf("%s %s: probability %s = %v, Python %v", rec.Name, na.ID, e.Key, e.P, pp)
						}
					}
					if len(na.A.Probabilities) != len(pa.Probabilities) {
						t.Errorf("%s %s: %d probabilities, Python %d",
							rec.Name, na.ID, len(na.A.Probabilities), len(pa.Probabilities))
					}
					switch na.A.Type {
					case "choice":
						if na.A.Choice != pa.Choice {
							t.Errorf("%s %s: choice %q, Python %q", rec.Name, na.ID, na.A.Choice, pa.Choice)
						}
					case "noul":
						if pa.Noul == nil || (*na.A.Noul > 0.5) != (*pa.Noul > 0.5) ||
							math.Abs(*na.A.Noul-*pa.Noul) > roundStep {
							t.Errorf("%s %s: noul %v, Python %v", rec.Name, na.ID, *na.A.Noul, pa.Noul)
						}
					}
				}
			}
			if n == 0 {
				t.Fatalf("logits.jsonl has no cases for %s", ck)
			}
			t.Logf("%d cases; worst per-option probability diff vs PyTorch: %.3g (tolerance %g)", n, worst, e2eTol[ck])
		})
	}
}

// e2eAgent builds the checkpoint's agent over the export at graph, as the
// default loader does, with the backend wrapped so the test sees its output.
func e2eAgent(t *testing.T, dir, graph string) (*teeBackend, *onnxAgent) {
	t.Helper()
	cfg, err := checkpoint.LoadConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	actWidth, err := cfg.ActWidth()
	if err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.Open(filepath.Join(dir, "tokenizer"))
	if err != nil {
		t.Fatal(err)
	}
	be, err := onnx.Open(graph, onnx.Options{Device: "cpu", ActWidth: actWidth})
	if err != nil {
		t.Fatalf("onnx.Open(%q): %v", graph, err)
	}
	t.Cleanup(func() {
		if err := be.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
	})
	tee := &teeBackend{Backend: be}
	a, err := newONNXAgent(tee, cfg, tok)
	if err != nil {
		t.Fatal(err)
	}
	return tee, a
}

// argmax is the first maximum, as numpy's and torch's argmax pick it.
func argmax(p []float32) int {
	best := 0
	for i := 1; i < len(p); i++ {
		if p[i] > p[best] {
			best = i
		}
	}
	return best
}
