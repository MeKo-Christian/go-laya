package evalset

// The int8 calibration harness, PLAN.md B.5.3. It is gated: it needs the
// checkpoints (LAYA_MODELS), the fp32 exports (LAYA_ONNX_DIR), the int8 ones
// (LAYA_ONNX_INT8_DIR, written by scripts/export_onnx.py --int8) and the
// subsets cmd/laya-evalfetch draws (LAYA_EVAL_DIR, default build/eval). It
// fails where int8 misses a threshold of metrics.go, which is B.5.4's gate.
// Run it one checkpoint per process, under a memory cap:
//
//	LAYA_EVAL_CHECKPOINTS=english systemd-run --user --scope -q -p MemoryMax=8G \
//	    -p MemorySwapMax=0 go test -run TestInt8Calibration -v -timeout 4h ./internal/evalset/

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/MeKo-Christian/go-laya"
	"github.com/MeKo-Christian/go-laya/internal/golden"
	"github.com/MeKo-Christian/go-laya/question"
)

var checkpoints = []string{golden.English, golden.Multilingual, golden.TypedDecisions}

// opener opens checkpoint ck on the exports in dir, which hold int8 graphs
// when quantized is set, and returns the predictor with the function that closes it.
type opener func(ctx context.Context, ck, dir string, quantized bool) (laya.Predictor, func() error, error)

// evaluate asks every case of ck through the fp32 exports, then through the
// int8 ones, holding one checkpoint open at a time, and pairs the answers.
func evaluate(ctx context.Context, ck string, cases []Case, fp32Dir, int8Dir string, open opener) ([]Pair, error) {
	fp32, err := answerAll(ctx, ck, cases, fp32Dir, false, open)
	if err != nil {
		return nil, err
	}
	quant, err := answerAll(ctx, ck, cases, int8Dir, true, open)
	if err != nil {
		return nil, err
	}
	pairs := make([]Pair, len(cases))
	for i, c := range cases {
		pairs[i] = Pair{Type: c.Type, Label: c.Label, FP32: fp32[i], Int8: quant[i]}
	}
	return pairs, nil
}

func answerAll(ctx context.Context, ck string, cases []Case, dir string, quantized bool, open opener) (_ []Outcome, err error) {
	p, closeFn, err := open(ctx, ck, dir, quantized)
	if err != nil {
		return nil, err
	}
	defer func() { err = errors.Join(err, closeFn()) }()
	out := make([]Outcome, len(cases))
	for i, c := range cases {
		res, err := p.SystemOne(ctx, c.State, c.Questions)
		if err != nil {
			return nil, fmt.Errorf("%s case %d: %w", ck, i, err)
		}
		a, ok := res.Answers.Get(QuestionID)
		if !ok {
			return nil, fmt.Errorf("%s case %d: no answer %q", ck, i, QuestionID)
		}
		if out[i], err = outcomeOf(c, a); err != nil {
			return nil, fmt.Errorf("%s case %d: %w", ck, i, err)
		}
	}
	return out, nil
}

// outcomeOf reads an answer as a caller gets it: after the checkpoint's
// temperatures, rounded to four decimals. The probabilities are looked up by
// key, as the notebook's p_ans["probabilities"].get(k) does, so label i reads
// option i whatever order they come back in.
func outcomeOf(c Case, a laya.Answer) (Outcome, error) {
	if a.Type != c.Type.String() {
		return Outcome{}, fmt.Errorf("a %s answer to a %v question", a.Type, c.Type)
	}
	if c.Type == question.Noul {
		if a.Noul == nil {
			return Outcome{}, errors.New("a noul answer without P(true)")
		}
		return NoulOutcome(*a.Noul), nil
	}
	if len(a.Probabilities) != len(c.Keys) {
		return Outcome{}, fmt.Errorf("%s answer has %d probabilities for %d classes", a.Type, len(a.Probabilities), len(c.Keys))
	}
	probs := make([]float64, len(c.Keys))
	for i, k := range c.Keys {
		j := slices.IndexFunc(a.Probabilities, func(e laya.ProbEntry) bool { return e.Key == k })
		if j < 0 {
			return Outcome{}, fmt.Errorf("%s answer has no probability for %q", a.Type, k)
		}
		probs[i] = a.Probabilities[j].P
	}
	if c.Type == question.Score {
		if a.Score == nil {
			return Outcome{}, errors.New("a score answer without its score")
		}
		return ScoreOutcome(probs, *a.Score), nil
	}
	i := slices.Index(c.Keys, a.Choice)
	if i < 0 {
		return Outcome{}, fmt.Errorf("choice %q is not an option", a.Choice)
	}
	return ChoiceOutcome(probs, i), nil
}

// openRouter is the real opener: a Router over the local checkpoint, with
// WithONNXDir selecting the precision, as a caller opts into int8. It first
// checks that dir's graph is the precision asked for, so a misconfigured run
// cannot compare fp32 against itself.
func openRouter(models string) opener {
	return func(ctx context.Context, ck, dir string, quantized bool) (laya.Predictor, func() error, error) {
		if err := checkPrecision(filepath.Join(dir, "laya-"+ck+".onnx"), quantized); err != nil {
			return nil, nil, err
		}
		r, err := laya.NewRouter(
			laya.WithModels(map[string]laya.ModelSpec{ck: laya.ModelSpecFromString(golden.CheckpointDir(models, ck))}),
			laya.WithONNXDir(dir),
			laya.WithRouterDevice("cpu"),
		)
		if err != nil {
			return nil, nil, err
		}
		p, err := r.Load(ctx, ck)
		if err != nil {
			return nil, nil, errors.Join(err, r.Close())
		}
		return p, r.Close, nil
	}
}

// checkPrecision reads the graph file (64 MiB at most, its weights being
// external) and holds it to the precision asked for: quantize_dynamic's
// MatMulInteger is in an int8 graph and in no fp32 one.
func checkPrecision(graph string, quantized bool) error {
	b, err := os.ReadFile(graph) // #nosec G304 -- the developer's own export
	if err != nil {
		return err
	}
	if got := bytes.Contains(b, []byte("MatMulInteger")); got != quantized {
		return fmt.Errorf("%s: MatMulInteger present = %v, want %v", graph, got, quantized)
	}
	return nil
}

// stageFP32 returns a directory holding fp32Dir's export of ck as
// laya-<ck>.onnx, the name the loader reads. A canonical directory is used as
// it is. Otherwise the -dynamo export is hard-linked beside fp32Dir, never
// into it: the loader refuses a symlink, and a hard link must stay on one
// filesystem.
func stageFP32(t *testing.T, fp32Dir, ck string) string {
	t.Helper()
	if _, err := os.Stat(filepath.Join(fp32Dir, "laya-"+ck+".onnx")); err == nil {
		return fp32Dir
	}
	src := filepath.Join(fp32Dir, "laya-"+ck+"-dynamo.onnx")
	if _, err := os.Stat(src); err != nil {
		t.Skipf("no fp32 %s export in %s", ck, fp32Dir)
	}
	staged, err := os.MkdirTemp(filepath.Dir(filepath.Clean(fp32Dir)), "evalset-fp32-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(staged) })
	links := map[string]string{"laya-" + ck + ".onnx": src, filepath.Base(src) + ".data": src + ".data"}
	for dst, from := range links {
		if err := os.Link(from, filepath.Join(staged, dst)); err != nil {
			t.Skipf("cannot hard-link the %s export: %v", ck, err)
		}
	}
	return staged
}

// loadCases reads every subset the manifest names, sha first, and turns its
// rows into cases.
func loadCases(t *testing.T, manifest, dir string) map[string][]Case {
	t.Helper()
	m, err := LoadManifest(manifest)
	if err != nil {
		t.Fatal(err)
	}
	cases := map[string][]Case{}
	for _, s := range Specs {
		e, ok := m.Entry(s.Name)
		if !ok {
			t.Fatalf("the manifest has no %s", s.Name)
		}
		if e.Dataset != s.Dataset || e.Config != s.Config || e.Split != s.Split {
			t.Fatalf("the manifest's %s is %s %s/%s, the spec's %s %s/%s",
				s.Name, e.Dataset, e.Config, e.Split, s.Dataset, s.Config, s.Split)
		}
		rows, err := ReadSubset(dir, e)
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range rows {
			c, err := s.Case(r)
			if err != nil {
				t.Fatal(err)
			}
			cases[s.Name] = append(cases[s.Name], c)
		}
	}
	return cases
}

func TestInt8Calibration(t *testing.T) {
	models := golden.SkipWithoutModels(t)
	fp32Dir, int8Dir := os.Getenv("LAYA_ONNX_DIR"), os.Getenv("LAYA_ONNX_INT8_DIR")
	if fp32Dir == "" || int8Dir == "" {
		t.Skip("set LAYA_ONNX_DIR (fp32) and LAYA_ONNX_INT8_DIR (scripts/export_onnx.py --int8)")
	}
	root := golden.Root(t)
	evalDir := os.Getenv("LAYA_EVAL_DIR")
	if evalDir == "" {
		evalDir = filepath.Join(root, "build", "eval")
	}
	if _, err := os.Stat(evalDir); err != nil {
		t.Skipf("no subsets in %s (go run ./cmd/laya-evalfetch -out %s, or set LAYA_EVAL_DIR)", evalDir, evalDir)
	}
	cases := loadCases(t, filepath.Join(root, "docs", "benchmarks", "eval-manifest.json"), evalDir)

	selected := checkpoints
	if s := os.Getenv("LAYA_EVAL_CHECKPOINTS"); s != "" {
		selected = strings.Split(s, ",")
	}
	results := map[string]map[string]any{}
	for _, ck := range selected {
		t.Run(ck, func(t *testing.T) {
			fp32 := stageFP32(t, fp32Dir, ck)
			// One run per precision over every dataset, so each side opens
			// the checkpoint once.
			var all []Case
			for _, s := range Specs {
				all = append(all, cases[s.Name]...)
			}
			start := time.Now()
			pairs, err := evaluate(t.Context(), ck, all, fp32, int8Dir, openRouter(models))
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("%s: %d cases, both precisions, in %v", ck, len(pairs), time.Since(start).Round(time.Second))

			var sb strings.Builder
			sb.WriteString("\n| checkpoint | set | n | ECE fp32 | ECE int8 | Brier fp32 | Brier int8 | acc fp32 | acc int8 " +
				"| MAE fp32 | MAE int8 | flips | max Δp | verdict |\n")
			sb.WriteString("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
			out := map[string]any{}
			for _, s := range ByType(pairs) {
				ok, why := s.Verdict()
				verdict := "pass"
				if !ok {
					// The gate of B.5.4 and D33: int8 holds up only where
					// every cell passes.
					verdict = "FAIL: " + strings.Join(why, "; ")
					t.Errorf("%s %v: int8 fails the thresholds: %s", ck, s.Type, strings.Join(why, "; "))
				}
				out[s.Type.String()] = jsonSummary(s, verdict)
				sb.WriteString(tableRow(ck, "**"+s.Type.String()+"**", s, verdict))
			}
			rest := pairs
			for _, s := range Specs {
				d := Summarize(rest[:len(cases[s.Name])])
				rest = rest[len(cases[s.Name]):]
				out[s.Name] = jsonSummary(d, "")
				sb.WriteString(tableRow(ck, s.Name, d, ""))
			}
			t.Log(sb.String())
			if slices.IndexFunc(ByType(pairs), func(s Summary) bool { return s.MaxDiff > 0 }) < 0 {
				t.Errorf("int8 answered every case exactly as fp32: the same graph on both sides?")
			}
			results[ck] = out
		})
	}
	if out := os.Getenv("LAYA_EVAL_OUT"); out != "" {
		b, err := json.MarshalIndent(results, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(out, append(b, '\n'), 0o600); err != nil {
			t.Fatal(err)
		}
	}
}

// jsonSummary is s for encoding/json, which refuses the NaN of a missing MAE.
func jsonSummary(s Summary, verdict string) map[string]any {
	num := func(v float64) any {
		if math.IsNaN(v) {
			return nil
		}
		return v
	}
	sideOf := func(d Side) map[string]any {
		return map[string]any{"ece": num(d.ECE), "brier": num(d.Brier), "accuracy": d.Accuracy, "mae": num(d.MAE)}
	}
	return map[string]any{
		"type": s.Type.String(), "n": s.N, "fp32": sideOf(s.FP32), "int8": sideOf(s.Int8),
		"flips": s.Flips, "max_diff": s.MaxDiff, "verdict": verdict,
	}
}

func tableRow(ck, set string, s Summary, verdict string) string {
	f := func(v float64) string {
		if math.IsNaN(v) {
			return "—"
		}
		return fmt.Sprintf("%.4f", v)
	}
	return fmt.Sprintf("| %s | %s | %d | %s | %s | %s | %s | %.3f | %.3f | %s | %s | %d (%.1f%%) | %.4f | %s |\n",
		ck, set, s.N, f(s.FP32.ECE), f(s.Int8.ECE), f(s.FP32.Brier), f(s.Int8.Brier),
		s.FP32.Accuracy, s.Int8.Accuracy, f(s.FP32.MAE), f(s.Int8.MAE),
		s.Flips, 100*s.FlipRate(), s.MaxDiff, verdict)
}
