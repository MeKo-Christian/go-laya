package evalset

// The int8 calibration harness, PLAN.md B.5.3. It is gated: it needs the
// checkpoints (LAYA_MODELS), the fp32 exports (LAYA_ONNX_DIR), the int8 ones
// (LAYA_ONNX_INT8_DIR, written by scripts/export_onnx.py --int8) and the
// subsets cmd/laya-evalfetch draws (LAYA_EVAL_DIR, default build/eval). It
// fails where int8 misses a threshold of metrics.go, which is B.5.4's gate.
// The fp32 side is the export each int8 graph's laya.int8 stamp names as its
// source (stageBaseline), so the deltas compare one graph at two precisions.
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

	"google.golang.org/protobuf/encoding/protowire"

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

// int8Stamp is the key scripts/export_onnx.py --int8 stamps its recipe under,
// in the int8 graph's metadata_props; the stamp's "source" is the file name of
// the fp32 export it quantized.
const int8Stamp = "laya.int8"

// int8Source reads the int8 graph at path (64 MiB at most, its weights being
// external) and returns the fp32 file name its laya.int8 stamp records.
func int8Source(path string) (string, error) {
	fi, err := os.Stat(path)
	if err != nil {
		return "", err
	}
	if !fi.Mode().IsRegular() || fi.Size() > 64<<20 {
		return "", fmt.Errorf("%s: not a graph file of at most 64 MiB", path)
	}
	b, err := os.ReadFile(path) // #nosec G304 -- the developer's own export
	if err != nil {
		return "", err
	}
	stamp, err := metadataProp(b, int8Stamp)
	if err != nil {
		return "", fmt.Errorf("%s: %w", path, err)
	}
	var v struct {
		Source string `json:"source"`
	}
	if err := json.Unmarshal([]byte(stamp), &v); err != nil {
		return "", fmt.Errorf("%s: %s stamp: %w", path, int8Stamp, err)
	}
	if v.Source == "" || filepath.Base(v.Source) != v.Source || v.Source == "." || v.Source == ".." {
		return "", fmt.Errorf("%s: %s source %q is not a plain name", path, int8Stamp, v.Source)
	}
	return v.Source, nil
}

// metadataProp returns the value of the ModelProto metadata_props entry
// (field 14, a StringStringEntryProto) called key.
func metadataProp(model []byte, key string) (string, error) {
	for b := model; len(b) > 0; {
		num, typ, n := protowire.ConsumeTag(b)
		if n < 0 {
			return "", protowire.ParseError(n)
		}
		b = b[n:]
		if num != 14 || typ != protowire.BytesType {
			if n = protowire.ConsumeFieldValue(num, typ, b); n < 0 {
				return "", protowire.ParseError(n)
			}
			b = b[n:]
			continue
		}
		entry, n := protowire.ConsumeBytes(b)
		if n < 0 {
			return "", protowire.ParseError(n)
		}
		b = b[n:]
		var k, v string
		for e := entry; len(e) > 0; {
			enum, etyp, m := protowire.ConsumeTag(e)
			if m < 0 {
				return "", protowire.ParseError(m)
			}
			e = e[m:]
			if etyp != protowire.BytesType {
				if m = protowire.ConsumeFieldValue(enum, etyp, e); m < 0 {
					return "", protowire.ParseError(m)
				}
				e = e[m:]
				continue
			}
			s, m := protowire.ConsumeString(e)
			if m < 0 {
				return "", protowire.ParseError(m)
			}
			e = e[m:]
			switch enum { //nolint:exhaustive // StringStringEntryProto has only key (1) and value (2)
			case 1:
				k = s
			case 2:
				v = s
			}
		}
		if k == key {
			return v, nil
		}
	}
	return "", fmt.Errorf("no %s metadata: not a graph scripts/export_onnx.py --int8 wrote", key)
}

// stageBaseline returns a directory, made in parent, that holds the fp32
// export int8Graph was quantized from as laya-<ck>.onnx, the name the loader
// reads, and that file's path in fp32Dir. The source is the one the int8
// graph's laya.int8 stamp names, never whatever fp32Dir holds under the
// loader's name: a canonical export beside the -dynamo one quantize_int8 read
// is another exporter's graph, and every difference between the two would be
// counted against int8. The files are hard-linked, as the loader refuses a
// symlink, and the external data keeps the name the graph references.
func stageBaseline(fp32Dir, int8Graph, ck, parent string) (staged, src string, err error) {
	name, err := int8Source(int8Graph)
	if err != nil {
		return "", "", err
	}
	src = filepath.Join(fp32Dir, name)
	for _, f := range []string{src, src + ".data"} {
		if _, err := os.Stat(f); err != nil {
			return "", "", fmt.Errorf("%s was quantized from %s, which %s does not hold: %w", int8Graph, name, fp32Dir, err)
		}
	}
	staged, err = os.MkdirTemp(parent, "evalset-fp32-")
	if err != nil {
		return "", "", err
	}
	links := map[string]string{"laya-" + ck + ".onnx": src, name + ".data": src + ".data"}
	for dst, from := range links {
		if err := os.Link(from, filepath.Join(staged, dst)); err != nil {
			return "", "", errors.Join(fmt.Errorf("hard-linking the %s export: %w", ck, err), os.RemoveAll(staged))
		}
	}
	return staged, src, nil
}

// stageFP32 stages the fp32 baseline of ck beside fp32Dir, never into it,
// a hard link having to stay on one filesystem.
func stageFP32(t *testing.T, fp32Dir, int8Dir, ck string) string {
	t.Helper()
	int8Graph := filepath.Join(int8Dir, "laya-"+ck+".onnx")
	if _, err := os.Stat(int8Graph); err != nil {
		t.Skipf("no int8 %s export in %s", ck, int8Dir)
	}
	staged, src, err := stageBaseline(fp32Dir, int8Graph, ck, filepath.Dir(filepath.Clean(fp32Dir)))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(staged) })
	t.Logf("%s: fp32 baseline %s, the source %s's %s stamp records", ck, src, int8Graph, int8Stamp)
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
			fp32 := stageFP32(t, fp32Dir, int8Dir, ck)
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
