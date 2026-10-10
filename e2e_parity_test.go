package laya

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"runtime/debug"
	"testing"

	"github.com/MeKo-Christian/go-laya/backend"
	"github.com/MeKo-Christian/go-laya/internal/backend/native"
	"github.com/MeKo-Christian/go-laya/internal/backend/onnx"
	"github.com/MeKo-Christian/go-laya/internal/calib"
	"github.com/MeKo-Christian/go-laya/internal/checkpoint"
	"github.com/MeKo-Christian/go-laya/internal/golden"
	"github.com/MeKo-Christian/go-laya/internal/runtime/tensor"
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

// e2eActTol is the largest relative act-logit difference, |go-py|/max(|py|,1),
// TestE2EParity accepts between the action head over ONNX Runtime and the act
// logits Python recorded, per checkpoint. The act logits sit in the thousands
// (about ±1100 to ±5000), where one float32 ulp is already up to 4.9e-4, so an
// absolute bound would have to scale with the checkpoint; a relative one does
// not.
//
// Measured over every logits.jsonl case (2026-09-27, ORT 1.23.0 on the dynamo
// exports): english 3.9e-06 (about 0.02 absolute), multilingual 1.4e-06,
// typed-decisions 1.2e-06, with no act argmax flipping. Each gets 2x headroom.
// The formatted confidence, score and act_probability matched Python's exactly
// in every case, so roundStep covers them with room to spare.
var e2eActTol = map[string]float64{
	golden.English:        7.8e-6,
	golden.Multilingual:   2.8e-6,
	golden.TypedDecisions: 2.5e-6,
}

// e2eNativeTol is e2eTol for the pure-Go native backend (Task 8.8): the
// largest per-option probability difference between SystemOne over the
// native backend and the probabilities Python's recorded logits give, per
// checkpoint.
//
// Measured over every logits.jsonl case (2026-10-10, amd64 with AVX2):
// english 2.2e-06, multilingual 3.5e-06, typed-decisions 9.2e-07, with no
// argmax flipping in any of the 30 rows of each. The logits are within
// TestForwardGolden's 1.1e-5 of torch's, and the softmax shrinks that. Each
// gets 2x headroom, as e2eTol does; TestE2EParity also holds every entry to
// Task 8.8's 1e-4 (e2eCeiling).
var e2eNativeTol = map[string]float64{
	golden.English:        4.3e-6,
	golden.Multilingual:   7.0e-6,
	golden.TypedDecisions: 1.9e-6,
}

// e2eNativeActTol is e2eActTol for the native backend (Task 8.8), relative
// for the reason given there.
//
// Measured over every logits.jsonl case (2026-10-10, amd64 with AVX2):
// english 1.4e-06, multilingual 4.7e-07, typed-decisions 4.1e-07, with no
// act argmax flipping. Each gets 2x headroom.
var e2eNativeActTol = map[string]float64{
	golden.English:        2.9e-6,
	golden.Multilingual:   9.5e-7,
	golden.TypedDecisions: 8.3e-7,
}

// e2eCeiling is Task 7.5.2's bound on any per-checkpoint tolerance, which
// Task 8.8 holds the native backend to as well.
const e2eCeiling = 1e-4

// e2eBackend is one backend TestE2EParity gates: its name, which e2eAgent
// opens it by, and its per-checkpoint bounds on the probabilities and the
// relative act logits.
type e2eBackend struct {
	name        string
	tol, actTol map[string]float64
}

// e2eBackends are the backends TestE2EParity runs: ONNX Runtime over the
// exports in $LAYA_ONNX_DIR (Task 7.5), and the native backend over the
// checkpoints' own weights (Task 8.8).
var e2eBackends = []e2eBackend{
	{name: "onnx", tol: e2eTol, actTol: e2eActTol},
	{name: "native", tol: e2eNativeTol, actTol: e2eNativeActTol},
}

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

// TestE2EParity is Task 7.5.1-7.5.3 and Task 8.8: every logits.jsonl case
// through SystemOne with the real tokenizer, for all three checkpoints, on
// each backend of e2eBackends. The per-option probabilities must agree with
// the ones Python's recorded logits give at the same temperature, and no
// row's argmax may flip.
//
// TestSystemOneReplay already proves the batch equals the recorded one, so
// what is left to differ here is the forward pass.
func TestE2EParity(t *testing.T) {
	root := golden.SkipWithoutModels(t)
	for _, be := range e2eBackends {
		for ck, tol := range be.tol {
			if tol > e2eCeiling {
				t.Errorf("%s tolerance[%s] = %g exceeds the 1e-4 ceiling", be.name, ck, tol)
			}
		}
	}

	type logitsCase struct {
		Checkpoint string          `json:"checkpoint"`
		State      json.RawMessage `json:"state"`
		Questions  json.RawMessage `json:"questions"`
		QIDs       []string        `json:"qids"`
		QTypes     []int           `json:"qtypes"`
		Logits     json.RawMessage `json:"logits"`
		ActLogits  json.RawMessage `json:"act_logits"`
		ResultJSON string          `json:"result_json"`
	}
	cases := golden.Load(t, "logits")

	for _, be := range e2eBackends {
		t.Run(be.name, func(t *testing.T) {
			for _, ck := range []string{golden.English, golden.Multilingual, golden.TypedDecisions} {
				t.Run(ck, func(t *testing.T) {
					tee, a := e2eAgent(t, be.name, root, ck)

					worst, worstAct, n, rows, flips := 0.0, 0.0, 0, 0, 0
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

						// The action head is a separate output of the same pass,
						// and the recorded act_probability is saturated at 1.0
						// in every case, so only its logits can expose a wrong
						// act output.
						wantAct := golden.Matrix[float32](t, c.ActLogits, rec.Name+" act_logits")
						if len(tee.act) != len(wantAct) {
							t.Fatalf("%s: %d act rows, recorded %d", rec.Name, len(tee.act), len(wantAct))
						}
						for r := range wantAct {
							worstAct = max(worstAct, compareActRow(t, rec.Name, r, tee.act[r], wantAct[r], be.actTol[ck]))
						}

						var py struct {
							Answers map[string]pyAnswer `json:"answers"`
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
								// A NaN compares false with every bound.
								if math.IsNaN(d) {
									d = math.Inf(1)
								}
								worst = max(worst, d)
								if d > be.tol[ck] {
									t.Errorf("%s %s: p[%d] = %.8g, Python %.8g (diff %.3g > %g)",
										rec.Name, na.ID, i, pGo[i], pPy[i], d, be.tol[ck])
								}
							}
							rows++
							if g, p := argmax(pGo), argmax(pPy); g != p {
								flips++
								t.Errorf("%s %s: argmax %d, Python %d\n got %v\nwant %v", rec.Name, na.ID, g, p, pGo, pPy)
							}

							pa, ok := py.Answers[na.ID]
							if !ok {
								t.Errorf("%s: Python has no answer %q", rec.Name, na.ID)
								continue
							}
							compareFormatted(t, rec.Name+" "+na.ID, na.A, pa)
						}
					}
					if n == 0 {
						t.Fatalf("logits.jsonl has no cases for %s", ck)
					}
					t.Logf("%s: %d cases; worst per-option probability diff vs PyTorch: %.3g (tolerance %g); argmax flips: %d of %d rows",
						be.name, n, worst, be.tol[ck], flips, rows)
					t.Logf("%s: worst relative act-logit diff vs PyTorch: %.3g (tolerance %g)", be.name, worstAct, be.actTol[ck])
				})
			}
		})
	}
}

// e2eAgent builds the checkpoint's agent over the backend kind names, with
// the backend wrapped so the test sees its output, or skips when the
// backend's inputs are not there.
//
//   - "onnx" opens the export of ck in $LAYA_ONNX_DIR and builds the agent
//     as the default loader does.
//   - "native" opens the checkpoint's model.safetensors with native.Open and
//     injects it with WithBackend, as a caller would; the native backend has
//     no public option yet (Task 8.10). The agent owns it and closes it with
//     the subtest, so one checkpoint's weights are in memory at a time.
func e2eAgent(t *testing.T, kind, root, ck string) (*teeBackend, *Agent) {
	t.Helper()
	dir := golden.CheckpointDir(root, ck)
	switch kind {
	case "onnx":
		exports := os.Getenv("LAYA_ONNX_DIR")
		if exports == "" {
			t.Skip("no exports (set LAYA_ONNX_DIR)")
		}
		graph, ok := findExport(exports, ck)
		if !ok {
			t.Skipf("no %s export in %s (run scripts/export_onnx.py --all --dynamo)", ck, exports)
		}
		return onnxE2EAgent(t, dir, graph)
	case "native":
		return nativeE2EAgent(t, dir)
	}
	t.Fatalf("unknown backend %q", kind)
	return nil, nil
}

// onnxE2EAgent builds the checkpoint's agent over the export at graph, as the
// default loader does, with the backend wrapped so the test sees its output.
func onnxE2EAgent(t *testing.T, dir, graph string) (*teeBackend, *Agent) {
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
	a, err := newAgent(tee, cfg, tok)
	if err != nil {
		t.Fatal(err)
	}
	return tee, a
}

// nativeE2EAgent opens the checkpoint in dir on the native backend and the
// agent over it through Open and WithBackend, the backend wrapped so the test
// sees its output.
func nativeE2EAgent(t *testing.T, dir string) (*teeBackend, *Agent) {
	t.Helper()
	// The kernels' worker count is process-wide (tensor.SetWorkers); the
	// test sets it only to finish sooner, and restores it.
	prev := tensor.Workers()
	tensor.SetWorkers(runtime.NumCPU())
	t.Cleanup(func() { tensor.SetWorkers(prev) })

	be, err := native.Open(t.Context(), dir, native.Options{})
	if err != nil {
		t.Fatalf("native.Open(%q): %v", dir, err)
	}
	tee := &teeBackend{Backend: be}
	a, err := Open(t.Context(), dir, WithBackend(tee))
	if err != nil {
		// A failed Open leaves the backend with its caller.
		_ = be.Close()
		t.Fatalf("Open(%q, WithBackend): %v", dir, err)
	}
	t.Cleanup(func() {
		if err := a.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
		runtime.GC()
		debug.FreeOSMemory()
	})
	return tee, a
}

// pyAnswer is one answer of Python's system_one result_json. Optional keys are
// pointers, so a key missing from Python's answer fails rather than reading
// as zero.
type pyAnswer struct {
	Choice        string             `json:"choice"`
	Score         *float64           `json:"score"`
	Legend        map[string]any     `json:"legend"`
	Noul          *float64           `json:"noul"`
	Probabilities map[string]float64 `json:"probabilities"`
	Confidence    *float64           `json:"confidence"`
	Action        *struct {
		ActProbability *float64 `json:"act_probability"`
	} `json:"action"`
}

// compareFormatted holds the formatted answer to what Python's system_one
// returned: the same decision, and probabilities that, both rounded to four
// places, sit at most one step apart. That is also what catches a wrong
// temperature, which moves no argmax. Score, confidence and act_probability
// are rounded the same way and get the same one-step bound.
func compareFormatted(t *testing.T, name string, got Answer, pa pyAnswer) {
	t.Helper()
	for _, e := range got.Probabilities {
		if pp, ok := pa.Probabilities[e.Key]; !ok || math.Abs(e.P-pp) > roundStep {
			t.Errorf("%s: probability %s = %v, Python %v", name, e.Key, e.P, pp)
		}
	}
	if len(got.Probabilities) != len(pa.Probabilities) {
		t.Errorf("%s: %d probabilities, Python %d", name, len(got.Probabilities), len(pa.Probabilities))
	}
	switch got.Type {
	case "choice":
		if got.Choice != pa.Choice {
			t.Errorf("%s: choice %q, Python %q", name, got.Choice, pa.Choice)
		}
	case "noul":
		if pa.Noul == nil || (*got.Noul > 0.5) != (*pa.Noul > 0.5) ||
			math.Abs(*got.Noul-*pa.Noul) > roundStep {
			t.Errorf("%s: noul %v, Python %v", name, *got.Noul, orMissing(pa.Noul))
		}
	case "score":
		if got.Score == nil || pa.Score == nil || math.Abs(*got.Score-*pa.Score) > roundStep {
			t.Errorf("%s: score %v, Python %v", name, orMissing(got.Score), orMissing(pa.Score))
		}
		compareLegend(t, name, got.Legend, pa.Legend)
	}
	// Confidence is present for every type, and act_probability in every
	// answer's action block.
	if pa.Confidence == nil || math.Abs(got.Confidence-*pa.Confidence) > roundStep {
		t.Errorf("%s: confidence %v, Python %v", name, got.Confidence, orMissing(pa.Confidence))
	}
	var pyAct *float64
	if pa.Action != nil {
		pyAct = pa.Action.ActProbability
	}
	if pyAct == nil || math.Abs(got.Action.ActProbability-*pyAct) > roundStep {
		t.Errorf("%s: act_probability %v, Python %v", name, got.Action.ActProbability, orMissing(pyAct))
	}
}

// compareActRow holds one row of the action head's logits to the recorded row:
// the same width, every entry within the relative tolerance, and the same
// argmax, which is what decides the act probability once it saturates. It
// returns the row's worst relative difference.
func compareActRow(t *testing.T, name string, r int, got, want []float32, tol float64) float64 {
	t.Helper()
	if len(got) != len(want) {
		t.Errorf("%s: act row %d has %d logits, recorded %d", name, r, len(got), len(want))
		return 0
	}
	worst := 0.0
	for i := range want {
		d := math.Abs(float64(got[i])-float64(want[i])) / max(math.Abs(float64(want[i])), 1)
		worst = max(worst, d)
		if d > tol {
			t.Errorf("%s: act[%d][%d] = %.8g, Python %.8g (relative diff %.3g > %g)",
				name, r, i, got[i], want[i], d, tol)
		}
	}
	if g, p := argmax(got), argmax(want); g != p {
		t.Errorf("%s: act row %d argmax %d, Python %d\n got %v\nwant %v", name, r, g, p, got, want)
	}
	return worst
}

// compareLegend holds a score answer's legend to Python's. The legend is the
// caller's criteria, not model output, so it has to match exactly; both sides
// go through JSON so the comparison sees what Python would have serialized.
func compareLegend(t *testing.T, name string, got Obj, want map[string]any) {
	t.Helper()
	if want == nil {
		t.Errorf("%s: Python has no legend", name)
		return
	}
	b, err := jsonx.Marshal(got)
	if err != nil {
		t.Fatalf("%s: marshal legend: %v", name, err)
	}
	var g map[string]any
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatalf("%s: legend %s: %v", name, b, err)
	}
	if !reflect.DeepEqual(g, want) {
		t.Errorf("%s: legend %v, Python %v", name, g, want)
	}
}

// orMissing formats an optional field for a failure message: its value, or
// "missing" rather than a pointer address.
func orMissing(p *float64) any {
	if p == nil {
		return "missing"
	}
	return *p
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
