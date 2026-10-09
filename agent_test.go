package laya

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/MeKo-Christian/go-laya/backend"
	"github.com/MeKo-Christian/go-laya/internal/backend/onnx"
	"github.com/MeKo-Christian/go-laya/internal/golden"
	"github.com/MeKo-Christian/go-laya/jsonx"
)

// openStub is what openAgent's loader recorded: the settings Open handed it,
// the snapshots it asked for and the graphs it opened.
type openStub struct {
	settings  loaderSettings
	token     string
	hubDir    string
	snapshots []snapshotCall
	opens     []openCall
	closer    *closeCounter
}

type snapshotCall struct {
	repo, rev string
	allow     []string
}

// stubOpen is openAgent with its network and runtime stubbed: a snapshot
// answers with root, laid out by the caller, and the backend is a
// closeCounter.
func stubOpen(t *testing.T, root string) (func(context.Context, string, ...Option) (*Agent, error), *openStub) {
	t.Helper()
	rec := &openStub{closer: &closeCounter{}}
	newLoader := func(s loaderSettings) *defaultLoader {
		l := newDefaultLoader(s)
		rec.settings, rec.token, rec.hubDir = s, l.hub.Token, l.hub.Dir
		l.snapshot = func(_ context.Context, repo, rev string, allow []string) (string, error) {
			rec.snapshots = append(rec.snapshots, snapshotCall{repo, rev, allow})
			return root, nil
		}
		l.open = func(path string, o onnx.Options) (backend.Backend, error) {
			rec.opens = append(rec.opens, openCall{path, o})
			return rec.closer, nil
		}
		return l
	}
	return func(ctx context.Context, ref string, opts ...Option) (*Agent, error) {
		return openAgent(ctx, ref, opts, newLoader)
	}, rec
}

// Task 7.7.5: Open is laya.load (agent.py:351-360). The repo and subfolder
// reach the snapshot as the Router's loader sends them, and the graph is the
// export of the checkpoint the spec names in either registry.
func TestOpenHub(t *testing.T) {
	cases := []struct {
		name      string
		ref       string
		opts      []Option
		layout    string // where the checkpoint sits under the snapshot root
		wantRepo  string
		wantRev   string
		wantAllow []string
		wantGraph string
	}{
		{
			name: "empty ref is the bundle repo", ref: "",
			wantRepo: bundleRepo, wantRev: pinnedRevision,
			wantAllow: allowPatterns(""), wantGraph: ModelEnglish,
		},
		{
			name: "bundle subfolder", ref: bundleRepo, opts: []Option{WithSubfolder(ModelMultilingual)},
			layout:   ModelMultilingual,
			wantRepo: bundleRepo, wantRev: pinnedRevision,
			wantAllow: allowPatterns(ModelMultilingual), wantGraph: ModelMultilingual,
		},
		{
			name: "standalone repo", ref: standaloneModels[ModelTypedDecisions].Repo,
			wantRepo: standaloneModels[ModelTypedDecisions].Repo, wantRev: "main",
			wantAllow: allowPatterns(""), wantGraph: ModelTypedDecisions,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			root, graphs := t.TempDir(), t.TempDir()
			writeCheckpoint(t, filepath.Join(root, c.layout))
			want := writeGraph(t, graphs, c.wantGraph)
			t.Setenv("LAYA_ONNX_DIR", graphs)

			open, rec := stubOpen(t, root)
			a, err := open(context.Background(), c.ref, c.opts...)
			if err != nil {
				t.Fatalf("Open: %v", err)
			}
			if len(rec.snapshots) != 1 {
				t.Fatalf("snapshot called %d times, want 1", len(rec.snapshots))
			}
			s := rec.snapshots[0]
			if s.repo != c.wantRepo || s.rev != c.wantRev || !slices.Equal(s.allow, c.wantAllow) {
				t.Errorf("snapshot(%q, %q, %q), want (%q, %q, %q)",
					s.repo, s.rev, s.allow, c.wantRepo, c.wantRev, c.wantAllow)
			}
			if len(rec.opens) != 1 || rec.opens[0].path != want {
				t.Fatalf("opened %v, want %q", rec.opens, want)
			}
			if err := a.Close(); err != nil || rec.closer.n != 1 {
				t.Errorf("Close = %v, backend closed %d times; want nil, 1", err, rec.closer.n)
			}
		})
	}
}

// Task 7.7.5: device and token reach the loader as laya.load's keyword
// arguments do (agent.py:351-352); the token falls back to $HF_TOKEN.
func TestOpenDeviceAndToken(t *testing.T) {
	ck := t.TempDir()
	writeCheckpoint(t, ck)
	graph := writeGraph(t, t.TempDir(), "x")
	t.Setenv("HF_TOKEN", "from-env")

	open, rec := stubOpen(t, "")
	if _, err := open(context.Background(), ck, WithGraph(graph), WithDevice("cpu"), WithHFToken("explicit")); err != nil {
		t.Fatalf("Open: %v", err)
	}
	if got := rec.opens[0].opts; got.Device != "cpu" || got.ActWidth != 3 {
		t.Errorf("onnx.Options = %+v, want Device cpu and ActWidth 3", got)
	}
	if rec.token != "explicit" {
		t.Errorf("token = %q, want %q", rec.token, "explicit")
	}

	open, rec = stubOpen(t, "")
	if _, err := open(context.Background(), ck, WithGraph(graph)); err != nil {
		t.Fatalf("Open: %v", err)
	}
	if rec.token != "from-env" {
		t.Errorf("token without WithHFToken = %q, want $HF_TOKEN's %q", rec.token, "from-env")
	}
	if rec.opens[0].opts.Device != "" {
		t.Errorf("Device without WithDevice = %q, want \"\" (auto)", rec.opens[0].opts.Device)
	}
}

// Task 7.7.5: WithGraph wins over the registry lookup; a checkpoint no
// registry names has no export name, so without WithGraph it is ErrNoGraph
// and no backend is opened.
func TestOpenGraph(t *testing.T) {
	t.Run("WithGraph wins", func(t *testing.T) {
		root, graphs := t.TempDir(), t.TempDir()
		writeCheckpoint(t, root)
		writeGraph(t, graphs, ModelEnglish)
		t.Setenv("LAYA_ONNX_DIR", graphs)
		explicit := writeGraph(t, t.TempDir(), "mine")

		open, rec := stubOpen(t, root)
		if _, err := open(context.Background(), "", WithGraph(explicit)); err != nil {
			t.Fatalf("Open: %v", err)
		}
		if rec.opens[0].path != explicit {
			t.Errorf("opened %q, want %q", rec.opens[0].path, explicit)
		}
	})
	t.Run("local dir needs WithGraph", func(t *testing.T) {
		ck := t.TempDir()
		writeCheckpoint(t, ck)
		t.Setenv("LAYA_ONNX_DIR", t.TempDir())

		open, rec := stubOpen(t, "")
		if _, err := open(context.Background(), ck); !errors.Is(err, ErrNoGraph) {
			t.Errorf("err = %v, want ErrNoGraph", err)
		}
		if len(rec.opens) != 0 {
			t.Errorf("opened %v, want nothing", rec.opens)
		}
	})
	t.Run("unknown repo needs WithGraph", func(t *testing.T) {
		root := t.TempDir()
		writeCheckpoint(t, root)
		t.Setenv("LAYA_ONNX_DIR", t.TempDir())

		open, rec := stubOpen(t, root)
		if _, err := open(context.Background(), "someone/their-model"); !errors.Is(err, ErrNoGraph) {
			t.Errorf("err = %v, want ErrNoGraph", err)
		}
		if len(rec.opens) != 0 {
			t.Errorf("opened %v, want nothing", rec.opens)
		}
	})
	t.Run("missing WithGraph file", func(t *testing.T) {
		ck := t.TempDir()
		writeCheckpoint(t, ck)

		open, rec := stubOpen(t, "")
		_, err := open(context.Background(), ck, WithGraph(filepath.Join(t.TempDir(), "absent.onnx")))
		if !errors.Is(err, ErrNoGraph) {
			t.Errorf("err = %v, want ErrNoGraph", err)
		}
		if len(rec.opens) != 0 {
			t.Errorf("opened %v, want nothing", rec.opens)
		}
	})
}

// Task 7.7.5: a subfolder outside the checkpoint is refused before anything
// is downloaded, as the Router's loader refuses it.
func TestOpenSubfolderEscape(t *testing.T) {
	open, rec := stubOpen(t, t.TempDir())
	_, err := open(context.Background(), bundleRepo, WithSubfolder("../other"))
	if !errors.Is(err, ErrCheckpointNotFound) {
		t.Errorf("err = %v, want ErrCheckpointNotFound", err)
	}
	if len(rec.snapshots) != 0 || len(rec.opens) != 0 {
		t.Errorf("snapshots %v, opens %v; want neither", rec.snapshots, rec.opens)
	}
}

// Task 7.7.5: Open's result is what the Router caches.
func TestOpenSatisfiesPredictor(t *testing.T) {
	ck := t.TempDir()
	writeCheckpoint(t, ck)
	graph := writeGraph(t, t.TempDir(), "x")
	open, _ := stubOpen(t, "")
	a, err := open(context.Background(), ck, WithGraph(graph))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	r, err := NewRouter(WithLoader(nil))
	if err != nil {
		t.Fatal(err)
	}
	if err := r.Attach(ModelEnglish, a); err != nil {
		t.Fatalf("Attach: %v", err)
	}
}

// Task 7.7.5: Open on each real checkpoint, run on ONNX Runtime, answers the
// first logits.jsonl case of that checkpoint as Python's system_one did:
// compareFormatted holds the decision and every rounded value to one step.
// Gated like TestE2EParity, which covers every case and the logits.
func TestOpenReal(t *testing.T) {
	root := golden.SkipWithoutModels(t)
	exports := os.Getenv("LAYA_ONNX_DIR")
	if exports == "" {
		t.Skip("no exports (set LAYA_ONNX_DIR)")
	}
	type logitsCase struct {
		Checkpoint string          `json:"checkpoint"`
		State      json.RawMessage `json:"state"`
		Questions  json.RawMessage `json:"questions"`
		ResultJSON string          `json:"result_json"`
	}
	cases := golden.Load(t, "logits")

	for _, ck := range []string{golden.English, golden.Multilingual, golden.TypedDecisions} {
		t.Run(ck, func(t *testing.T) {
			graph, ok := findExport(exports, ck)
			if !ok {
				t.Skipf("no %s export in %s", ck, exports)
			}
			var c logitsCase
			name := ""
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

			a, err := Open(context.Background(), golden.CheckpointDir(root, ck), WithGraph(graph), WithDevice("cpu"))
			if err != nil {
				t.Fatalf("Open: %v", err)
			}
			t.Cleanup(func() {
				if err := a.Close(); err != nil {
					t.Errorf("Close: %v", err)
				}
			})
			state, err := jsonx.Decode(c.State)
			if err != nil {
				t.Fatal(err)
			}
			res, err := a.Predict(context.Background(), state, recordedQuestions(t, c.Questions))
			if err != nil {
				t.Fatalf("Predict: %v", err)
			}
			var py struct {
				Answers map[string]pyAnswer `json:"answers"`
			}
			if err := json.Unmarshal([]byte(c.ResultJSON), &py); err != nil {
				t.Fatalf("%s: result_json: %v", name, err)
			}
			if len(res.Answers) != len(py.Answers) {
				t.Fatalf("%s: %d answers, Python %d", name, len(res.Answers), len(py.Answers))
			}
			for _, na := range res.Answers {
				pa, ok := py.Answers[na.ID]
				if !ok {
					t.Fatalf("%s: Python has no answer %q", name, na.ID)
				}
				compareFormatted(t, name+" "+na.ID, na.A, pa)
			}
		})
	}
}

// fixedBackend answers every row with the same logits and keeps nothing, so
// concurrent passes share no state of the test's own.
type fixedBackend struct{}

func (fixedBackend) Forward(_ context.Context, in backend.Batch) (logits, act [][]float32, err error) {
	for range in.InputIDs {
		logits = append(logits, []float32{0.5, -0.5, 0.25, 1, 0, 0, 0, 0, 0, 0})
		act = append(act, []float32{1, 0})
	}
	return logits, act, nil
}

func (fixedBackend) Close() error { return nil }

// Task 7.7.5: SetLimits replaces Python's agent.cfg["max_len"] and
// agent.cfg["head_max_len"] mutation (agent.py:256-257 reads them per call),
// so a change reaches the next SystemOne. head_max_len bounds the
// instructions (common.py:74), max_len the whole sequence (common.py:86).
func TestAgentSetLimits(t *testing.T) {
	be := &recordingBackend{logits: [][]float32{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}}, act: [][]float32{{0, 0}}}
	a := testAgent(t, "", be)
	if a.MaxLen() != 512 || a.HeadMaxLen() != 192 {
		t.Fatalf("limits = %d, %d; want the config defaults 512, 192", a.MaxLen(), a.HeadMaxLen())
	}
	long := Questions{{ID: "q", Q: NoulQuestion{Ins: strings.Repeat("is this a long instruction ", 2)}}}
	rowLen := func() int {
		t.Helper()
		if _, err := a.SystemOne(context.Background(), "s", long); err != nil {
			t.Fatalf("SystemOne: %v", err)
		}
		return len(be.calls[len(be.calls)-1].InputIDs[0])
	}
	before := rowLen()

	if err := a.SetLimits(512, 24); err != nil {
		t.Fatalf("SetLimits: %v", err)
	}
	if a.MaxLen() != 512 || a.HeadMaxLen() != 24 {
		t.Errorf("limits = %d, %d; want 512, 24", a.MaxLen(), a.HeadMaxLen())
	}
	if after := rowLen(); after >= before {
		t.Errorf("row is %d tokens at head_max_len 24, %d at 192; want it shorter", after, before)
	}

	if err := a.SetLimits(16, 192); err != nil {
		t.Fatalf("SetLimits: %v", err)
	}
	_, err := a.SystemOne(context.Background(), "s", Questions{{ID: "many", Q: ChoiceQuestion{
		Ins: "pick", Opts: Labels("a", "b", "c", "d", "e", "f", "g", "h", "i", "j"),
	}}})
	if !errors.Is(err, ErrOptionsExceedHeadBudget) {
		t.Errorf("err at max_len 16 = %v, want ErrOptionsExceedHeadBudget", err)
	}

	for _, bad := range [][2]int{{0, 192}, {512, 0}, {-1, -1}} {
		if err := a.SetLimits(bad[0], bad[1]); !errors.Is(err, ErrInvalidLimits) {
			t.Errorf("SetLimits(%d, %d) = %v, want ErrInvalidLimits", bad[0], bad[1], err)
		}
		if a.MaxLen() != 16 || a.HeadMaxLen() != 192 {
			t.Errorf("after SetLimits(%d, %d) limits = %d, %d; want 16, 192 kept",
				bad[0], bad[1], a.MaxLen(), a.HeadMaxLen())
		}
	}
}

// rowBackend answers like fixedBackend and records each pass's first row,
// under a lock, since passes run concurrently.
type rowBackend struct {
	fixedBackend

	mu   sync.Mutex
	rows []string
}

func (b *rowBackend) Forward(ctx context.Context, in backend.Batch) (logits, act [][]float32, err error) {
	b.mu.Lock()
	b.rows = append(b.rows, fmt.Sprint(in.InputIDs[0]))
	b.mu.Unlock()
	return b.fixedBackend.Forward(ctx, in)
}

// Task 7.7.5: SetLimits may run while passes are in flight, and a pass sees
// one complete pair: the limits of one SetLimits call, never max_len from one
// and head_max_len from another. Under -race this is also the check that
// SystemOne reads the limits under the lock.
//
// The two pairs differ in both limits, and every combination of them gives a
// different first row (checked first), so a pass that mixed the pairs would
// record a row neither pair produces.
func TestAgentSetLimitsConcurrent(t *testing.T) {
	pairs := [2][2]int{{256, 24}, {192, 192}}
	qs := Questions{{ID: "q", Q: NoulQuestion{Ins: strings.Repeat("is this a long instruction ", 2)}}}
	state := strings.Repeat("the customer wrote a long message ", 40)

	// The row each combination produces, sequentially.
	ref := &rowBackend{}
	a := testAgent(t, "", ref)
	rowOf := map[[2]int]string{}
	for _, m := range []int{pairs[0][0], pairs[1][0]} {
		for _, h := range []int{pairs[0][1], pairs[1][1]} {
			if err := a.SetLimits(m, h); err != nil {
				t.Fatal(err)
			}
			if _, err := a.SystemOne(context.Background(), state, qs); err != nil {
				t.Fatalf("SystemOne at (%d, %d): %v", m, h, err)
			}
			rowOf[[2]int{m, h}] = ref.rows[len(ref.rows)-1]
		}
	}
	allowed := map[string]bool{rowOf[pairs[0]]: true, rowOf[pairs[1]]: true}
	distinct := map[string]bool{}
	for _, r := range rowOf {
		distinct[r] = true
	}
	if len(distinct) != 4 {
		t.Fatalf("the four limit combinations give %d distinct rows, want 4", len(distinct))
	}

	be := &rowBackend{}
	a = testAgent(t, "", be)
	if err := a.SetLimits(pairs[0][0], pairs[0][1]); err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for i := range 8 {
		wg.Go(func() {
			for j := range 200 {
				if i%2 == 0 {
					p := pairs[j%2]
					if err := a.SetLimits(p[0], p[1]); err != nil {
						t.Errorf("SetLimits: %v", err)
					}
					continue
				}
				if _, err := a.SystemOne(context.Background(), state, qs); err != nil {
					t.Errorf("SystemOne: %v", err)
				}
			}
		})
	}
	wg.Wait()

	for _, r := range be.rows {
		if allowed[r] {
			continue
		}
		for pair, row := range rowOf {
			if row == r {
				t.Fatalf("a pass ran on max_len %d with head_max_len %d, a pair no SetLimits call set",
					pair[0], pair[1])
			}
		}
		t.Fatalf("a pass ran on a row no limit combination gives (%d tokens)", len(strings.Fields(r)))
	}
}

// Task 7.7.5: WithLimits sets both limits at Open, and a bad pair is refused
// before anything is downloaded or opened.
func TestOpenWithLimits(t *testing.T) {
	ck := t.TempDir()
	writeCheckpoint(t, ck)
	graph := writeGraph(t, t.TempDir(), "x")

	open, _ := stubOpen(t, "")
	a, err := open(context.Background(), ck, WithGraph(graph), WithLimits(1024, 256))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if a.MaxLen() != 1024 || a.HeadMaxLen() != 256 {
		t.Errorf("limits = %d, %d; want 1024, 256", a.MaxLen(), a.HeadMaxLen())
	}

	open, rec := stubOpen(t, "")
	if _, err := open(context.Background(), ck, WithGraph(graph), WithLimits(1024, 0)); !errors.Is(err, ErrInvalidLimits) {
		t.Errorf("err = %v, want ErrInvalidLimits", err)
	}
	if len(rec.opens) != 0 {
		t.Errorf("opened %v, want nothing", rec.opens)
	}
}
