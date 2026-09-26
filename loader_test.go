package laya

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/MeKo-Christian/go-laya/backend"
	"github.com/MeKo-Christian/go-laya/internal/backend/onnx"
	"github.com/MeKo-Christian/go-laya/internal/golden"
)

// testConfig is the smallest rl_agent_config.json LoadConfig accepts, with
// two act costs so ActWidth is 3 rather than the keyless default of 1.
const testConfig = `{"encoder": "answerdotai/ModernBERT-large", "head_layers": 2, "act_costs": {"ask": 0.1, "escalate": 0.5}}`

// writeCheckpoint lays out what the loader reads from a checkpoint: the config
// and the tokenizer/ directory, here the mini_en fixture.
func writeCheckpoint(t *testing.T, dir string) {
	t.Helper()
	tok := filepath.Join(dir, "tokenizer")
	if err := os.MkdirAll(tok, 0o750); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "rl_agent_config.json"), []byte(testConfig), 0o600); err != nil {
		t.Fatal(err)
	}
	for _, f := range []string{"tokenizer.json", "tokenizer_config.json"} {
		raw, err := os.ReadFile(filepath.Join("tokenizer", "testdata", "mini_en", f))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(tok, f), raw, 0o600); err != nil {
			t.Fatal(err)
		}
	}
}

// writeGraph puts an empty stand-in for an export where the loader looks. The
// stubbed open never parses it.
func writeGraph(t *testing.T, dir, name string) string {
	t.Helper()
	p := filepath.Join(dir, "laya-"+name+".onnx")
	if err := os.WriteFile(p, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

// closeCounter is a backend.Backend that only counts Close calls.
type closeCounter struct{ n int }

func (*closeCounter) Forward(context.Context, backend.Batch) (logits, act [][]float32, err error) {
	return nil, nil, errors.New("closeCounter: no forward pass")
}

func (c *closeCounter) Close() error {
	c.n++
	return nil
}

// openCall records what the loader asked ONNX Runtime for.
type openCall struct {
	path string
	opts onnx.Options
}

// stubbedLoader is a default loader whose network and runtime seams are
// stubs: snapshot fails the test unless a case replaces it.
func stubbedLoader(t *testing.T, onnxDir string, opts ...RouterOption) (*defaultLoader, *[]openCall, *closeCounter) {
	t.Helper()
	cfg := routerConfig{onnxDir: onnxDir}
	for _, o := range opts {
		if err := o(&cfg); err != nil {
			t.Fatal(err)
		}
	}
	l := newDefaultLoader(cfg)
	var calls []openCall
	closer := &closeCounter{}
	l.open = func(path string, o onnx.Options) (backend.Backend, error) {
		calls = append(calls, openCall{path, o})
		return closer, nil
	}
	l.snapshot = func(context.Context, string, string, []string) (string, error) {
		t.Fatal("snapshot called for a local checkpoint")
		return "", nil
	}
	return l, &calls, closer
}

// Task 7.2.1: a local directory is used as it is (agent.py:115-116), the
// graph comes from the ONNX directory under the router's name for the
// checkpoint (D24), and the act width the config derives reaches the backend.
func TestDefaultLoaderLocalDir(t *testing.T) {
	ck, graphs := t.TempDir(), t.TempDir()
	writeCheckpoint(t, ck)
	want := writeGraph(t, graphs, ModelEnglish)

	l, calls, closer := stubbedLoader(t, graphs)
	a, err := l.load(context.Background(), ModelEnglish, ModelSpecFromString(ck))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(*calls) != 1 {
		t.Fatalf("open called %d times, want 1", len(*calls))
	}
	got := (*calls)[0]
	if got.path != want {
		t.Errorf("graph = %q, want %q", got.path, want)
	}
	if got.opts.ActWidth != 3 {
		t.Errorf("ActWidth = %d, want 3 (two act_costs + 1)", got.opts.ActWidth)
	}
	if err := a.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if closer.n != 1 {
		t.Errorf("backend closed %d times, want 1", closer.n)
	}
}

// Task 7.2.1: the subfolder is joined to the checkpoint directory
// (agent.py:130-131), locally as after a download.
func TestDefaultLoaderLocalSubfolder(t *testing.T) {
	root, graphs := t.TempDir(), t.TempDir()
	writeCheckpoint(t, filepath.Join(root, ModelMultilingual))
	writeGraph(t, graphs, ModelMultilingual)

	l, calls, _ := stubbedLoader(t, graphs)
	if _, err := l.load(context.Background(), ModelMultilingual,
		ModelSpec{Repo: root, Subfolder: ModelMultilingual}); err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(*calls) != 1 {
		t.Fatalf("open called %d times, want 1", len(*calls))
	}
}

// Task 7.2.1 under D24: a Hub checkpoint downloads only the config and the
// tokenizer, since the weights are in the local ONNX export. The bundle repo is
// pinned to D17's revision; any other repo follows main, because the pin is a
// commit of the bundle repo alone.
func TestDefaultLoaderHubSnapshot(t *testing.T) {
	for _, tc := range []struct {
		name  string
		key   string
		spec  ModelSpec
		rev   string
		allow []string
	}{
		{
			"bundle root", ModelEnglish,
			ModelSpec{Repo: bundleRepo},
			pinnedRevision,
			[]string{"rl_agent_config.json", "tokenizer/*"},
		},
		{
			"bundle subfolder", ModelMultilingual,
			ModelSpec{Repo: bundleRepo, Subfolder: ModelMultilingual},
			pinnedRevision,
			[]string{"multilingual/rl_agent_config.json", "multilingual/tokenizer/*"},
		},
		{
			"standalone repo", ModelMultilingual,
			ModelSpec{Repo: "convaiinnovations/laya-multilingual"},
			"main",
			[]string{"rl_agent_config.json", "tokenizer/*"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			snap, graphs := t.TempDir(), t.TempDir()
			writeCheckpoint(t, filepath.Join(snap, tc.spec.Subfolder))
			writeGraph(t, graphs, tc.key)

			l, calls, _ := stubbedLoader(t, graphs)
			var gotRepo, gotRev string
			var gotAllow []string
			l.snapshot = func(_ context.Context, repo, rev string, allow []string) (string, error) {
				gotRepo, gotRev, gotAllow = repo, rev, allow
				return snap, nil
			}
			if _, err := l.load(context.Background(), tc.key, tc.spec); err != nil {
				t.Fatalf("load: %v", err)
			}
			if gotRepo != tc.spec.Repo || gotRev != tc.rev || !slices.Equal(gotAllow, tc.allow) {
				t.Errorf("Snapshot(%q, %q, %q), want (%q, %q, %q)",
					gotRepo, gotRev, gotAllow, tc.spec.Repo, tc.rev, tc.allow)
			}
			if len(*calls) != 1 {
				t.Fatalf("open called %d times, want 1", len(*calls))
			}
		})
	}
}

// Task 7.2.1: each way a load can fail says which, and none reaches the
// runtime.
func TestDefaultLoaderErrors(t *testing.T) {
	t.Run("no graph", func(t *testing.T) {
		ck := t.TempDir()
		writeCheckpoint(t, ck)
		l, calls, _ := stubbedLoader(t, t.TempDir())
		_, err := l.load(context.Background(), ModelEnglish, ModelSpecFromString(ck))
		if !errors.Is(err, ErrNoGraph) {
			t.Fatalf("err = %v, want ErrNoGraph", err)
		}
		if !strings.Contains(err.Error(), "export_onnx.py") {
			t.Errorf("err = %q, want it to name the export script", err)
		}
		if len(*calls) != 0 {
			t.Error("open called without a graph")
		}
	})
	t.Run("missing local path", func(t *testing.T) {
		// agent.py:117-121: a path-shaped id that does not exist is an error,
		// not a Hub repo to download.
		for _, p := range []string{"./nope", "../nope", filepath.Join(t.TempDir(), "nope")} {
			l, _, _ := stubbedLoader(t, t.TempDir())
			if _, err := l.load(context.Background(), ModelEnglish, ModelSpecFromString(p)); !errors.Is(err, ErrModelNotFound) {
				t.Errorf("%s: err = %v, want ErrModelNotFound", p, err)
			}
		}
	})
	t.Run("missing subfolder", func(t *testing.T) {
		// agent.py:131-135.
		l, _, _ := stubbedLoader(t, t.TempDir())
		_, err := l.load(context.Background(), ModelMultilingual, ModelSpec{Repo: t.TempDir(), Subfolder: "nope"})
		if !errors.Is(err, ErrModelNotFound) {
			t.Errorf("err = %v, want ErrModelNotFound", err)
		}
	})
	t.Run("no config", func(t *testing.T) {
		l, _, _ := stubbedLoader(t, t.TempDir())
		_, err := l.load(context.Background(), ModelEnglish, ModelSpecFromString(t.TempDir()))
		if !errors.Is(err, ErrIncompatibleCheckpoint) {
			t.Errorf("err = %v, want ErrIncompatibleCheckpoint", err)
		}
	})
}

// Task 7.2.1: WithONNXDir wins, then $LAYA_ONNX_DIR, then onnx/ under the
// laya cache.
func TestDefaultLoaderONNXDir(t *testing.T) {
	cache, env, opt := t.TempDir(), t.TempDir(), t.TempDir()
	t.Setenv("LAYA_CACHE", cache)

	t.Setenv("LAYA_ONNX_DIR", "")
	if got, err := newDefaultLoader(routerConfig{}).graphDir(); err != nil || got != filepath.Join(cache, "onnx") {
		t.Errorf("default graphDir = %q, %v; want %q", got, err, filepath.Join(cache, "onnx"))
	}
	t.Setenv("LAYA_ONNX_DIR", env)
	if got, _ := newDefaultLoader(routerConfig{}).graphDir(); got != env {
		t.Errorf("graphDir with LAYA_ONNX_DIR = %q, want %q", got, env)
	}
	cfg := routerConfig{}
	if err := WithONNXDir(opt)(&cfg); err != nil {
		t.Fatal(err)
	}
	if got, _ := newDefaultLoader(cfg).graphDir(); got != opt {
		t.Errorf("graphDir with WithONNXDir = %q, want %q", got, opt)
	}
}

// Task 7.2.1: NewRouter installs the default loader; WithLoader(nil) is the
// explicit way to have none.
func TestNewRouterInstallsDefaultLoader(t *testing.T) {
	r, err := NewRouter()
	if err != nil {
		t.Fatalf("NewRouter: %v", err)
	}
	if r.loader == nil {
		t.Error("NewRouter() has no loader")
	}
}

// Task 7.2.1 end to end, gated like the backend's own tests: the default
// loader builds a real agent from the downloaded english checkpoint
// ($LAYA_MODELS) and its export ($LAYA_ONNX_DIR), with ONNX Runtime resolved
// the default way. The S1 exports carry a -dynamo suffix, so the graph gets a
// hard link under the name the loader expects; a symlink would not do, since
// the header check rejects them. Its external data keeps the name the graph
// refers to.
func TestDefaultLoaderReal(t *testing.T) {
	if testing.Short() {
		t.Skip("-short: needs the checkpoints, an export and ONNX Runtime")
	}
	models, ok := golden.ModelsRoot()
	if !ok {
		t.Skip("no checkpoints (set LAYA_MODELS)")
	}
	exports := os.Getenv("LAYA_ONNX_DIR")
	if exports == "" {
		t.Skip("no export (set LAYA_ONNX_DIR)")
	}

	graphs := exports
	if _, err := os.Stat(filepath.Join(exports, "laya-english.onnx")); err != nil {
		src := filepath.Join(exports, "laya-english-dynamo.onnx")
		if _, err := os.Stat(src); err != nil {
			t.Skipf("no english export in %s", exports)
		}
		// Same directory as the export, so the hardlinks stay on one filesystem.
		graphs, err = os.MkdirTemp(exports, "loader-test-")
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = os.RemoveAll(graphs) })
		for dst, from := range map[string]string{
			"laya-english.onnx":             src,
			"laya-english-dynamo.onnx.data": src + ".data",
		} {
			if err := os.Link(from, filepath.Join(graphs, dst)); err != nil {
				t.Skipf("cannot hardlink the export: %v", err)
			}
		}
	}

	r, err := NewRouter(
		WithONNXDir(graphs),
		WithModels(map[string]ModelSpec{ModelEnglish: ModelSpecFromString(filepath.Join(models, "laya"))}),
	)
	if err != nil {
		t.Fatalf("NewRouter: %v", err)
	}
	a, err := r.Load(context.Background(), ModelEnglish)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	oa, ok := a.(*onnxAgent)
	if !ok {
		t.Fatalf("Load returned %T, want *onnxAgent", a)
	}
	if w, _ := oa.cfg.ActWidth(); w < 1 {
		t.Errorf("ActWidth = %d", w)
	}
	if err := r.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

// Task 7.2.2: the token goes to the Hub client, falling back to $HF_TOKEN as
// `token or os.environ.get("HF_TOKEN")` does (router.py:159). The fallback is
// read when the Router is built, as Python reads it in __init__.
func TestRouterToken(t *testing.T) {
	for _, tc := range []struct {
		name, env, opt, want string
	}{
		{"none", "", "", ""},
		{"env only", "hf_env", "", "hf_env"},
		{"explicit", "", "hf_opt", "hf_opt"},
		{"explicit wins", "hf_env", "hf_opt", "hf_opt"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("HF_TOKEN", tc.env)
			l, _, _ := stubbedLoader(t, t.TempDir(), WithRouterToken(tc.opt))
			if l.hub.Token != tc.want {
				t.Errorf("token = %q, want %q", l.hub.Token, tc.want)
			}
		})
	}
}

// Task 7.2.2: the device reaches the backend unchanged. Validating it is the
// backend's job (onnx.ErrUnknownDevice), and like Python's Agent
// construction it happens at load time.
func TestRouterDevice(t *testing.T) {
	ck, graphs := t.TempDir(), t.TempDir()
	writeCheckpoint(t, ck)
	writeGraph(t, graphs, ModelEnglish)

	l, calls, _ := stubbedLoader(t, graphs, WithRouterDevice("cuda:1"))
	if _, err := l.load(context.Background(), ModelEnglish, ModelSpecFromString(ck)); err != nil {
		t.Fatalf("load: %v", err)
	}
	if got := (*calls)[0].opts.Device; got != "cuda:1" {
		t.Errorf("Device = %q, want %q", got, "cuda:1")
	}
}

// Task 7.2.4: offline from the environment. HF_HUB_OFFLINE is read exactly as
// huggingface_hub reads it -- `_is_true(HF_HUB_OFFLINE or TRANSFORMERS_OFFLINE)`
// (constants.py:194), so an empty value falls through to the second variable
// but a set, falsy one does not -- and LAYA_OFFLINE turns it on as well.
func TestDefaultLoaderOffline(t *testing.T) {
	for _, tc := range []struct {
		hf, transformers, laya string
		want                   bool
	}{
		{"", "", "", false},
		{"1", "", "", true},
		{"true", "", "", true},
		{"Yes", "", "", true},
		{"ON", "", "", true},
		{"0", "", "", false},
		{"false", "", "", false},
		{"2", "", "", false},
		{"", "1", "", true},
		{"0", "1", "", false}, // `or` picks the first non-empty value
		{"", "", "1", true},
		{"", "", "no", false},
		{"0", "", "TRUE", true},
	} {
		t.Setenv("HF_HUB_OFFLINE", tc.hf)
		t.Setenv("TRANSFORMERS_OFFLINE", tc.transformers)
		t.Setenv("LAYA_OFFLINE", tc.laya)
		l, _, _ := stubbedLoader(t, t.TempDir())
		if l.hub.Offline != tc.want {
			t.Errorf("HF_HUB_OFFLINE=%q TRANSFORMERS_OFFLINE=%q LAYA_OFFLINE=%q: Offline = %v, want %v",
				tc.hf, tc.transformers, tc.laya, l.hub.Offline, tc.want)
		}
	}
}
