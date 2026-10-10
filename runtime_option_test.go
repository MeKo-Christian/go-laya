package laya

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/MeKo-Christian/go-laya/backend"
	"github.com/MeKo-Christian/go-laya/internal/backend/onnx"
	"github.com/MeKo-Christian/go-laya/internal/head"
	"github.com/MeKo-Christian/go-laya/internal/hub"
)

// writeWeights puts what the native runtime reads beside a checkpoint that
// writeCheckpoint laid out: an encoder/config.json and a model.safetensors
// whose header numbers headLayers head layers and gives act_head.2 actWidth
// rows. The stubbed native open never reads the rest, so the file holds
// nothing else; the loader's head check reads only its header.
func writeWeights(t *testing.T, dir string, headLayers, actWidth int) {
	t.Helper()
	type entry struct {
		DType   string   `json:"dtype"`
		Shape   []int64  `json:"shape"`
		Offsets [2]int64 `json:"data_offsets"`
	}
	header := map[string]entry{}
	for i := range headLayers {
		header[fmt.Sprintf("head.layers.%d.norm1.weight", i)] = entry{"F32", []int64{0}, [2]int64{0, 0}}
	}
	n := int64(actWidth) * head.ActHidden * 2
	header["act_head.2.weight"] = entry{"F16", []int64{int64(actWidth), head.ActHidden}, [2]int64{0, n}}
	raw, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	buf := binary.LittleEndian.AppendUint64(nil, uint64(len(raw)))
	buf = append(append(buf, raw...), make([]byte, n)...)
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), buf, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(dir, "encoder"), 0o750); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "encoder", "config.json"), []byte(`{}`), 0o600); err != nil {
		t.Fatal(err)
	}
}

// writeNativeCheckpoint is a checkpoint both runtimes can load: testConfig's
// two head layers and three act logits, in the config and in the weights.
func writeNativeCheckpoint(t *testing.T, dir string) {
	t.Helper()
	writeCheckpoint(t, dir)
	writeWeights(t, dir, 2, 3)
}

// nativeBackend stands in for the native backend: it answers every row with
// two option logits and as many act logits as its act field says, and
// counts Close calls.
type nativeBackend struct {
	act    int
	closed int
}

func (b *nativeBackend) Forward(_ context.Context, in backend.Batch) (logits, act [][]float32, err error) {
	for range in.QType {
		logits = append(logits, make([]float32, 2))
		act = append(act, make([]float32, b.act))
	}
	return logits, act, nil
}

func (b *nativeBackend) Close() error {
	b.closed++
	return nil
}

// runtimeRec records which runtime a stubbed loader opened, and on what.
type runtimeRec struct {
	mu          sync.Mutex
	snapshots   []snapshotCall
	onnxOpens   []string
	nativeOpens []string
	native      *nativeBackend
}

// stubRuntimes makes newDefaultLoader's constructor stub both runtimes: a
// snapshot answers with root (or fails the test when root is ""), the ONNX
// open records the graph and the native open the directory. Both return
// rec.native.
func stubRuntimes(t *testing.T, root string, rec *runtimeRec) func(loaderSettings) *defaultLoader {
	t.Helper()
	if rec.native == nil {
		rec.native = &nativeBackend{act: 3}
	}
	return func(s loaderSettings) *defaultLoader {
		l := newDefaultLoader(s)
		l.snapshot = func(_ context.Context, repo, rev string, allow []string) (string, error) {
			if root == "" {
				t.Fatalf("snapshot of %s requested for a local checkpoint", repo)
			}
			rec.mu.Lock()
			defer rec.mu.Unlock()
			rec.snapshots = append(rec.snapshots, snapshotCall{repo, rev, allow})
			return root, nil
		}
		l.open = func(path string, _ onnx.Options) (backend.Backend, error) {
			rec.mu.Lock()
			defer rec.mu.Unlock()
			rec.onnxOpens = append(rec.onnxOpens, path)
			return rec.native, nil
		}
		l.openNative = func(_ context.Context, dir string) (backend.Backend, error) {
			rec.mu.Lock()
			defer rec.mu.Unlock()
			rec.nativeOpens = append(rec.nativeOpens, dir)
			return rec.native, nil
		}
		return l
	}
}

// Task 8.10 (D30): Runtime has two values, RuntimeONNX the zero one, and
// any other is refused as the option is applied, before anything is read.
func TestRuntimeValues(t *testing.T) {
	if Runtime(0) != RuntimeONNX {
		t.Errorf("the zero Runtime is %v, want RuntimeONNX", Runtime(0))
	}
	for rt, want := range map[Runtime]string{RuntimeONNX: "onnx", RuntimeNative: "native", Runtime(7): "Runtime(7)"} {
		if got := rt.String(); got != want {
			t.Errorf("Runtime(%d).String() = %q, want %q", int(rt), got, want)
		}
	}
	for _, rt := range []Runtime{-1, 2, 7} {
		rec := &runtimeRec{}
		_, err := openAgent(context.Background(), "", []Option{WithRuntime(rt)}, stubRuntimes(t, "", rec))
		if !errors.Is(err, ErrUnknownRuntime) {
			t.Errorf("Open(WithRuntime(%d)) = %v, want ErrUnknownRuntime", int(rt), err)
		}
		if _, err := NewRouter(WithRouterRuntime(rt)); !errors.Is(err, ErrUnknownRuntime) {
			t.Errorf("NewRouter(WithRouterRuntime(%d)) = %v, want ErrUnknownRuntime", int(rt), err)
		}
	}
}

// Task 8.10 (D30): beside RuntimeNative, the options that only the ONNX
// backend reads would be silently ignored, and beside WithBackend so would
// any WithRuntime. Open refuses each pair, in either order, before anything
// is downloaded or opened.
func TestOpenRuntimeConflicts(t *testing.T) {
	native := WithRuntime(RuntimeNative)
	b := &closeCounter{}
	for _, tc := range []struct {
		name string
		opts []Option
		want []string // substrings of the message
	}{
		{"native and graph", []Option{native, WithGraph("x.onnx")}, []string{"RuntimeNative", "WithGraph"}},
		{"graph and native", []Option{WithGraph("x.onnx"), native}, []string{"RuntimeNative", "WithGraph"}},
		{"native and cuda", []Option{native, WithDevice("cuda")}, []string{"RuntimeNative", `WithDevice("cuda")`}},
		{"native and cuda:1", []Option{WithDevice("cuda:1"), native}, []string{"RuntimeNative", `WithDevice("cuda:1")`}},
		{"native and coreml", []Option{native, WithDevice("coreml")}, []string{"RuntimeNative", `WithDevice("coreml")`}},
		{"native and graph and cuda", []Option{native, WithGraph("x.onnx"), WithDevice("cuda")}, []string{"WithGraph and WithDevice"}},
		{"native and backend", []Option{native, WithBackend(b)}, []string{"WithBackend", "WithRuntime"}},
		{"backend and native", []Option{WithBackend(b), native}, []string{"WithBackend", "WithRuntime"}},
		{"explicit onnx and backend", []Option{WithRuntime(RuntimeONNX), WithBackend(b)}, []string{"WithBackend", "WithRuntime"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rec := &runtimeRec{}
			_, err := openAgent(context.Background(), "", tc.opts, stubRuntimes(t, t.TempDir(), rec))
			if !errors.Is(err, ErrConflictingOptions) {
				t.Fatalf("Open = %v, want ErrConflictingOptions", err)
			}
			for _, s := range tc.want {
				if !strings.Contains(err.Error(), s) {
					t.Errorf("Open = %v, want it to name %s", err, s)
				}
			}
			if len(rec.snapshots)+len(rec.onnxOpens)+len(rec.nativeOpens) != 0 || b.n != 0 {
				t.Errorf("snapshots %v, ONNX opens %v, native opens %v, backend closed %d times; want none",
					rec.snapshots, rec.onnxOpens, rec.nativeOpens, b.n)
			}
		})
	}
}

// Task 8.10: the CPU device names, a nil WithBackend and an explicit
// RuntimeONNX beside the ONNX options do not conflict.
func TestOpenRuntimeCompatible(t *testing.T) {
	ck := t.TempDir()
	writeNativeCheckpoint(t, ck)
	graph := writeGraph(t, t.TempDir(), "x")
	for _, tc := range []struct {
		name       string
		opts       []Option
		wantNative bool
	}{
		{"native", []Option{WithRuntime(RuntimeNative)}, true},
		{"native on the empty device", []Option{WithRuntime(RuntimeNative), WithDevice("")}, true},
		{"native on auto", []Option{WithDevice("auto"), WithRuntime(RuntimeNative)}, true},
		{"native on cpu", []Option{WithRuntime(RuntimeNative), WithDevice("cpu")}, true},
		{"native and a nil backend", []Option{WithBackend(nil), WithRuntime(RuntimeNative)}, true},
		{"onnx with graph and device", []Option{WithRuntime(RuntimeONNX), WithGraph(graph), WithDevice("cuda")}, false},
		{"native then onnx", []Option{WithRuntime(RuntimeNative), WithRuntime(RuntimeONNX), WithGraph(graph)}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rec := &runtimeRec{}
			a, err := openAgent(context.Background(), ck, tc.opts, stubRuntimes(t, "", rec))
			if err != nil {
				t.Fatalf("Open: %v", err)
			}
			defer a.Close()
			if got := len(rec.nativeOpens) == 1 && len(rec.onnxOpens) == 0; got != tc.wantNative {
				t.Errorf("native opens %v, ONNX opens %v; want native %v", rec.nativeOpens, rec.onnxOpens, tc.wantNative)
			}
		})
	}
}

// Task 8.10 (D30): the Router's options conflict as Open's do: beside
// RuntimeNative, WithONNXDir and a device other than the CPU. A custom
// WithLoader ignores WithRouterRuntime, as it ignores every other setting
// of the default loader, but the pairs that conflict still do.
func TestRouterRuntimeConflicts(t *testing.T) {
	native := WithRouterRuntime(RuntimeNative)
	loader := WithLoader(func(context.Context, string, ModelSpec) (Predictor, error) { return nil, ErrNoLoader })
	for _, tc := range []struct {
		name string
		opts []RouterOption
		want []string // nil: no error
	}{
		{"native and onnx dir", []RouterOption{native, WithONNXDir("x")}, []string{"RuntimeNative", "WithONNXDir"}},
		{"onnx dir and native", []RouterOption{WithONNXDir("x"), native}, []string{"RuntimeNative", "WithONNXDir"}},
		{"native and cuda", []RouterOption{native, WithRouterDevice("cuda")}, []string{"RuntimeNative", `WithRouterDevice("cuda")`}},
		{"native and coreml", []RouterOption{WithRouterDevice("coreml"), native}, []string{`WithRouterDevice("coreml")`}},
		{"native, loader and onnx dir", []RouterOption{native, loader, WithONNXDir("x")}, []string{"WithONNXDir"}},
		{"native", []RouterOption{native}, nil},
		{"native on cpu", []RouterOption{native, WithRouterDevice("cpu")}, nil},
		{"native on auto", []RouterOption{WithRouterDevice("auto"), native}, nil},
		{"native and a loader", []RouterOption{native, loader}, nil},
		{"onnx with onnx dir and cuda", []RouterOption{WithRouterRuntime(RuntimeONNX), WithONNXDir("x"), WithRouterDevice("cuda")}, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewRouter(tc.opts...)
			if tc.want == nil {
				if err != nil {
					t.Fatalf("NewRouter: %v", err)
				}
				return
			}
			if !errors.Is(err, ErrConflictingOptions) {
				t.Fatalf("NewRouter = %v, want ErrConflictingOptions", err)
			}
			for _, s := range tc.want {
				if !strings.Contains(err.Error(), s) {
					t.Errorf("NewRouter = %v, want it to name %s", err, s)
				}
			}
		})
	}
}

// Task 8.10: RuntimeNative adds the weights and the encoder config to the
// download, anchored under the subfolder as the config and the tokenizer
// are; RuntimeONNX downloads what it always has (D24).
func TestAllowPatternsForRuntime(t *testing.T) {
	for _, tc := range []struct {
		sub  string
		rt   Runtime
		want []string
	}{
		{"", RuntimeONNX, []string{"rl_agent_config.json", "tokenizer/*"}},
		{"multilingual", RuntimeONNX, []string{"multilingual/rl_agent_config.json", "multilingual/tokenizer/*"}},
		{"", RuntimeNative, []string{"rl_agent_config.json", "tokenizer/*", "model.safetensors", "encoder/config.json"}},
		{"multilingual", RuntimeNative, []string{
			"multilingual/rl_agent_config.json", "multilingual/tokenizer/*",
			"multilingual/model.safetensors", "multilingual/encoder/config.json",
		}},
	} {
		if got := allowPatternsFor(tc.sub, tc.rt); !slices.Equal(got, tc.want) {
			t.Errorf("allowPatternsFor(%q, %v) = %q, want %q", tc.sub, tc.rt, got, tc.want)
		}
		if tc.rt == RuntimeONNX && !slices.Equal(allowPatterns(tc.sub), tc.want) {
			t.Errorf("allowPatterns(%q) = %q, want %q", tc.sub, allowPatterns(tc.sub), tc.want)
		}
	}
}

// fakeHub serves one repo at one commit through the Hub's API, as the real
// hub client sees it: the listing, and per file a HEAD carrying the commit
// and the content's sha256, and a GET. It records every file downloaded.
type fakeHub struct {
	repo   string
	commit string
	files  map[string][]byte

	mu  sync.Mutex
	got []string
}

func (h *fakeHub) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	listing := "/api/models/" + h.repo + "/revision/"
	if strings.HasPrefix(r.URL.Path, listing) {
		type sibling struct {
			Name string `json:"rfilename"`
		}
		l := struct {
			SHA      string    `json:"sha"`
			Siblings []sibling `json:"siblings"`
		}{SHA: h.commit}
		for _, name := range slices.Sorted(func(yield func(string) bool) {
			for n := range h.files {
				if !yield(n) {
					return
				}
			}
		}) {
			l.Siblings = append(l.Siblings, sibling{name})
		}
		if err := json.NewEncoder(w).Encode(l); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
		return
	}
	path, ok := strings.CutPrefix(r.URL.Path, "/"+h.repo+"/resolve/"+h.commit+"/")
	body, found := h.files[path]
	if !ok || !found {
		http.NotFound(w, r)
		return
	}
	sum := sha256.Sum256(body)
	w.Header().Set("X-Repo-Commit", h.commit)
	w.Header().Set("X-Linked-Etag", `"`+hex.EncodeToString(sum[:])+`"`)
	if r.Method == http.MethodGet {
		h.mu.Lock()
		h.got = append(h.got, path)
		h.mu.Unlock()
		_, _ = w.Write(body)
	}
}

// Task 8.10 (D24 as amended): through the real hub client, hash-verified,
// RuntimeNative downloads exactly the ONNX set plus the checkpoint's own
// model.safetensors and encoder/config.json: not another checkpoint's, not
// any other file of its encoder/ directory, and nothing else of the repo.
// RuntimeONNX downloads what it did before.
func TestRuntimeDownloadSet(t *testing.T) {
	src := t.TempDir()
	writeNativeCheckpoint(t, src)
	files := map[string][]byte{}
	for _, rel := range []string{
		"rl_agent_config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
		"model.safetensors", "encoder/config.json",
	} {
		raw, err := os.ReadFile(filepath.Join(src, filepath.FromSlash(rel)))
		if err != nil {
			t.Fatal(err)
		}
		files[rel] = raw
		files["multilingual/"+rel] = raw
	}
	for _, decoy := range []string{
		"README.md", "training_args.bin", "pytorch_model.bin", "encoder/model.safetensors",
		"encoder/tokenizer.json", "onnx/model.safetensors", "multilingual/encoder/extra.json",
		"typed-decisions/model.safetensors", "multilingual/model.safetensors.index.json",
	} {
		files[decoy] = []byte("decoy " + decoy)
	}

	base := []string{"rl_agent_config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"}
	weights := []string{"encoder/config.json", "model.safetensors"}
	under := func(sub string, rels ...[]string) []string {
		var out []string
		for _, rs := range rels {
			for _, r := range rs {
				if sub != "" {
					r = sub + "/" + r
				}
				out = append(out, r)
			}
		}
		slices.Sort(out)
		return out
	}
	for _, tc := range []struct {
		name string
		sub  string
		rt   Runtime
		want []string
	}{
		{"onnx root", "", RuntimeONNX, under("", base)},
		{"onnx subfolder", ModelMultilingual, RuntimeONNX, under(ModelMultilingual, base)},
		{"native root", "", RuntimeNative, under("", base, weights)},
		{"native subfolder", ModelMultilingual, RuntimeNative, under(ModelMultilingual, base, weights)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			hub := &fakeHub{repo: bundleRepo, commit: pinnedRevision, files: files}
			srv := httptest.NewServer(hub)
			defer srv.Close()
			graphs := t.TempDir()
			writeGraph(t, graphs, ModelEnglish)
			writeGraph(t, graphs, ModelMultilingual)
			t.Setenv("LAYA_ONNX_DIR", graphs)
			t.Setenv("LAYA_OFFLINE", "")
			t.Setenv("HF_HUB_OFFLINE", "")
			t.Setenv("TRANSFORMERS_OFFLINE", "")

			rec := &runtimeRec{}
			cache := t.TempDir()
			newLoader := func(s loaderSettings) *defaultLoader {
				l := stubRuntimes(t, "", rec)(s)
				l.hub.Endpoint, l.hub.Dir, l.hub.Offline = srv.URL, cache, false
				l.snapshot = l.hub.Snapshot
				return l
			}
			a, err := openAgent(context.Background(), "",
				[]Option{WithSubfolder(tc.sub), WithRuntime(tc.rt)}, newLoader)
			if err != nil {
				t.Fatalf("Open: %v", err)
			}
			defer a.Close()

			got := slices.Sorted(slices.Values(hub.got))
			if !slices.Equal(got, tc.want) {
				t.Errorf("downloaded %q, want %q", got, tc.want)
			}
			snap := filepath.Join(cache, filepath.FromSlash(bundleRepo), pinnedRevision, tc.sub)
			if tc.rt == RuntimeNative {
				if !slices.Equal(rec.nativeOpens, []string{snap}) || len(rec.onnxOpens) != 0 {
					t.Errorf("native opens %q, ONNX opens %q; want native on %q only", rec.nativeOpens, rec.onnxOpens, snap)
				}
			} else if len(rec.nativeOpens) != 0 || len(rec.onnxOpens) != 1 {
				t.Errorf("native opens %q, ONNX opens %q; want one ONNX open", rec.nativeOpens, rec.onnxOpens)
			}
		})
	}
}

// Task 8.10: with RuntimeNative, Open runs the native backend on the
// checkpoint directory, wrapped in the same output-shape check as a
// caller's backend, and needs no export. A local directory is read without
// the network; a Hub one is snapshotted with the native download set. The
// agent owns the backend and closes it.
func TestOpenNative(t *testing.T) {
	root := t.TempDir()
	writeNativeCheckpoint(t, root)
	writeNativeCheckpoint(t, filepath.Join(root, ModelMultilingual))
	t.Setenv("LAYA_ONNX_DIR", t.TempDir()) // holds no export

	for _, tc := range []struct {
		name      string
		ref       string
		sub       string
		snapRoot  string // "" fails the test on a snapshot
		wantAllow []string
		wantDir   string
	}{
		{"local directory", root, "", "", nil, root},
		{"local subfolder", root, ModelMultilingual, "", nil, filepath.Join(root, ModelMultilingual)},
		{"hub root", "", "", root, allowPatternsFor("", RuntimeNative), root},
		{"hub subfolder", bundleRepo, ModelMultilingual, root, allowPatternsFor(ModelMultilingual, RuntimeNative), filepath.Join(root, ModelMultilingual)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rec := &runtimeRec{native: &nativeBackend{act: 3}}
			a, err := openAgent(context.Background(), tc.ref,
				[]Option{WithSubfolder(tc.sub), WithRuntime(RuntimeNative)}, stubRuntimes(t, tc.snapRoot, rec))
			if err != nil {
				t.Fatalf("Open: %v", err)
			}
			if tc.wantAllow != nil && (len(rec.snapshots) != 1 || !slices.Equal(rec.snapshots[0].allow, tc.wantAllow)) {
				t.Errorf("snapshots %v, want one with %q", rec.snapshots, tc.wantAllow)
			}
			if !slices.Equal(rec.nativeOpens, []string{tc.wantDir}) || len(rec.onnxOpens) != 0 {
				t.Errorf("native opens %q, ONNX opens %q; want native on %q only", rec.nativeOpens, rec.onnxOpens, tc.wantDir)
			}
			res, err := a.SystemOne(context.Background(), "hello", Questions{{ID: "q", Q: NoulQuestion{Ins: "Is it?"}}})
			if err != nil || len(res.Answers) != 1 {
				t.Errorf("SystemOne = %v, %v; want one answer", res, err)
			}
			if err := a.Close(); err != nil {
				t.Fatal(err)
			}
			if rec.native.closed != 1 {
				t.Errorf("native backend closed %d times by Agent.Close, want 1", rec.native.closed)
			}
		})
	}
}

// Task 8.10: the native backend's outputs are held to the config's act
// width, as a caller's backend's are: a row of the wrong width fails rather
// than answering plausibly.
func TestOpenNativeShapeChecked(t *testing.T) {
	ck := t.TempDir()
	writeNativeCheckpoint(t, ck)
	rec := &runtimeRec{native: &nativeBackend{act: 2}}
	a, err := openAgent(context.Background(), ck, []Option{WithRuntime(RuntimeNative)}, stubRuntimes(t, "", rec))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer a.Close()
	_, err = a.SystemOne(context.Background(), "hello", Questions{{ID: "q", Q: NoulQuestion{Ins: "Is it?"}}})
	if !errors.Is(err, ErrIncompatibleCheckpoint) {
		t.Fatalf("SystemOne = %v, want ErrIncompatibleCheckpoint for 2 act logits where the config has 3", err)
	}
}

// Task 8.10 (PR #49's review finding 2): upstream builds the head from
// rl_agent_config.json, head_layers layers (none when it is not positive)
// and len(act_costs)+1 act logits, and its strict load refuses weights that
// disagree. The loader holds the file to both before the native backend
// decodes a weight.
func TestOpenNativeHeadMismatch(t *testing.T) {
	for _, tc := range []struct {
		name        string
		config      string // "" is testConfig: two head layers, act width 3
		layers, act int
		want        []string // substrings of the error; nil: no error
	}{
		{"agree", "", 2, 3, nil},
		{"one head layer fewer", "", 1, 3, []string{"1 head layers", "head_layers 2"}},
		{"one head layer more", "", 3, 3, []string{"3 head layers", "head_layers 2"}},
		{"act width short", "", 2, 2, []string{"2 act logits", "act_costs give 3"}},
		{"act width long", "", 2, 4, []string{"4 act logits", "act_costs give 3"}},
		{"negative head_layers is none", `{"encoder": "x", "head_layers": -1, "act_costs": {"a": 1, "b": 2}}`, 0, 3, nil},
		{"zero head_layers, one in the file", `{"encoder": "x", "head_layers": 0, "act_costs": {"a": 1, "b": 2}}`, 1, 3, []string{"1 head layers", "head_layers 0"}},
		{"head_layers not an integer", `{"encoder": "x", "head_layers": "2", "act_costs": {"a": 1, "b": 2}}`, 2, 3, []string{"head_layers"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ck := t.TempDir()
			writeCheckpoint(t, ck)
			if tc.config != "" {
				if err := os.WriteFile(filepath.Join(ck, "rl_agent_config.json"), []byte(tc.config), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			writeWeights(t, ck, tc.layers, tc.act)

			rec := &runtimeRec{}
			a, err := openAgent(context.Background(), ck, []Option{WithRuntime(RuntimeNative)}, stubRuntimes(t, "", rec))
			if tc.want == nil {
				if err != nil {
					t.Fatalf("Open: %v", err)
				}
				_ = a.Close()
				return
			}
			if !errors.Is(err, ErrIncompatibleCheckpoint) {
				t.Fatalf("Open = %v, want ErrIncompatibleCheckpoint", err)
			}
			for _, s := range tc.want {
				if !strings.Contains(err.Error(), s) {
					t.Errorf("Open = %v, want it to say %q", err, s)
				}
			}
			if len(rec.nativeOpens) != 0 {
				t.Errorf("native backend opened on %q; the check must come first", rec.nativeOpens)
			}
		})
	}
}

// Task 8.10: a native backend the agent cannot be built on is closed, not
// leaked, and a failed native open is reported as it is.
func TestOpenNativeFailures(t *testing.T) {
	t.Run("agent rejects the config", func(t *testing.T) {
		ck := t.TempDir()
		writeNativeCheckpoint(t, ck)
		if err := os.WriteFile(filepath.Join(ck, "rl_agent_config.json"),
			[]byte(`{"encoder": "x", "head_layers": 2, "act_costs": {"a": 1, "b": 2}, "max_len": 0}`), 0o600); err != nil {
			t.Fatal(err)
		}
		rec := &runtimeRec{}
		if _, err := openAgent(context.Background(), ck, []Option{WithRuntime(RuntimeNative)}, stubRuntimes(t, "", rec)); !errors.Is(err, ErrIncompatibleCheckpoint) {
			t.Fatalf("Open = %v, want ErrIncompatibleCheckpoint", err)
		}
		if len(rec.nativeOpens) != 1 || rec.native.closed != 1 {
			t.Errorf("native opened %d times and closed %d; want 1 and 1", len(rec.nativeOpens), rec.native.closed)
		}
	})
	t.Run("native open fails", func(t *testing.T) {
		ck := t.TempDir()
		writeNativeCheckpoint(t, ck)
		boom := errors.New("boom")
		newLoader := func(s loaderSettings) *defaultLoader {
			l := stubRuntimes(t, "", &runtimeRec{})(s)
			l.openNative = func(context.Context, string) (backend.Backend, error) { return nil, boom }
			return l
		}
		if _, err := openAgent(context.Background(), ck, []Option{WithRuntime(RuntimeNative)}, newLoader); !errors.Is(err, boom) {
			t.Fatalf("Open = %v, want the native open's error", err)
		}
	})
	t.Run("no weights", func(t *testing.T) {
		ck := t.TempDir()
		writeCheckpoint(t, ck)
		rec := &runtimeRec{}
		if _, err := openAgent(context.Background(), ck, []Option{WithRuntime(RuntimeNative)}, stubRuntimes(t, "", rec)); !errors.Is(err, ErrIncompatibleCheckpoint) {
			t.Fatalf("Open = %v without model.safetensors, want ErrIncompatibleCheckpoint", err)
		}
		if len(rec.nativeOpens) != 0 {
			t.Errorf("native backend opened on %q", rec.nativeOpens)
		}
	})
}

// Task 8.10: the Router's default loader runs RuntimeNative as Open does,
// and needs no export for it.
func TestRouterNative(t *testing.T) {
	t.Run("through NewRouter", func(t *testing.T) {
		// The real native backend on a header-only fixture: it fails, but
		// as the native backend, where the ONNX path would want an export.
		ck := t.TempDir()
		writeNativeCheckpoint(t, ck)
		t.Setenv("LAYA_ONNX_DIR", t.TempDir()) // holds no export
		r, err := NewRouter(WithRouterRuntime(RuntimeNative),
			WithModels(map[string]ModelSpec{ModelEnglish: ModelSpecFromString(ck)}))
		if err != nil {
			t.Fatal(err)
		}
		defer r.Close()
		_, err = r.Load(context.Background(), ModelEnglish)
		if !errors.Is(err, ErrIncompatibleCheckpoint) || errors.Is(err, ErrNoGraph) ||
			!strings.Contains(err.Error(), "native backend") {
			t.Fatalf("Load = %v, want the native backend's ErrIncompatibleCheckpoint", err)
		}
	})

	ck := t.TempDir()
	writeNativeCheckpoint(t, ck)
	cfg := routerConfig{}
	for _, o := range []RouterOption{WithRouterRuntime(RuntimeNative), WithRouterDevice("cpu")} {
		if err := o(&cfg); err != nil {
			t.Fatal(err)
		}
	}
	t.Setenv("LAYA_ONNX_DIR", t.TempDir()) // holds no export
	rec := &runtimeRec{}
	l := stubRuntimes(t, "", rec)(cfg.loaderSettings)
	if l.runtime != RuntimeNative {
		t.Fatalf("loader runtime %v, want native", l.runtime)
	}
	p, err := l.load(context.Background(), ModelEnglish, ModelSpecFromString(ck))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer p.(*Agent).Close()
	if !slices.Equal(rec.nativeOpens, []string{ck}) || len(rec.onnxOpens) != 0 {
		t.Errorf("native opens %q, ONNX opens %q; want native on %q only", rec.nativeOpens, rec.onnxOpens, ck)
	}
}

// Task 8.10: offline, RuntimeNative needs its weights in the cache. A cache
// an ONNX load filled has the config and the tokenizer only, so the native
// load fails with the hub's not-cached error rather than downloading, and
// the ONNX load from the same cache still works.
func TestNativeOfflineNeedsWeights(t *testing.T) {
	src := t.TempDir()
	writeNativeCheckpoint(t, src)
	files := map[string][]byte{}
	for _, rel := range []string{
		"rl_agent_config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
		"model.safetensors", "encoder/config.json",
	} {
		raw, err := os.ReadFile(filepath.Join(src, filepath.FromSlash(rel)))
		if err != nil {
			t.Fatal(err)
		}
		files[rel] = raw
	}
	srv := httptest.NewServer(&fakeHub{repo: bundleRepo, commit: pinnedRevision, files: files})
	defer srv.Close()
	graphs := t.TempDir()
	writeGraph(t, graphs, ModelEnglish)
	t.Setenv("LAYA_ONNX_DIR", graphs)
	t.Setenv("HF_HUB_OFFLINE", "")
	t.Setenv("TRANSFORMERS_OFFLINE", "")
	cache := t.TempDir()

	open := func(offline bool, rt Runtime) error {
		t.Helper()
		if offline {
			t.Setenv("LAYA_OFFLINE", "1")
		} else {
			t.Setenv("LAYA_OFFLINE", "")
		}
		newLoader := func(s loaderSettings) *defaultLoader {
			l := stubRuntimes(t, "", &runtimeRec{})(s)
			l.hub.Endpoint, l.hub.Dir = srv.URL, cache
			l.snapshot = l.hub.Snapshot
			return l
		}
		a, err := openAgent(context.Background(), "", []Option{WithRuntime(rt)}, newLoader)
		if err == nil {
			_ = a.Close()
		}
		return err
	}
	if err := open(false, RuntimeONNX); err != nil {
		t.Fatalf("online ONNX Open: %v", err)
	}
	if err := open(true, RuntimeNative); !errors.Is(err, hub.ErrNotCached) {
		t.Fatalf("offline native Open = %v, want hub.ErrNotCached", err)
	}
	if err := open(true, RuntimeONNX); err != nil {
		t.Fatalf("offline ONNX Open: %v", err)
	}
	if err := open(false, RuntimeNative); err != nil {
		t.Fatalf("online native Open: %v", err)
	}
	if err := open(true, RuntimeNative); err != nil {
		t.Fatalf("offline native Open after an online one: %v", err)
	}
}
