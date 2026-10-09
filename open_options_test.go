package laya

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/MeKo-Christian/go-laya/backend"
	"github.com/MeKo-Christian/go-laya/internal/hub"
)

// Task 7.2.5 (D29): WithRevision is the opt-in D17 promises. Unset, the
// bundle repo stays at the pin and every other repo at main; set, the one
// revision is used for every Hub repo, the bundle repo included.
func TestOpenRevision(t *testing.T) {
	const sha = "0123456789abcdef0123456789abcdef01234567"
	standalone := standaloneModels[ModelTypedDecisions].Repo
	cases := []struct {
		name    string
		ref     string
		opts    []Option
		graph   string
		wantRev string
	}{
		{"bundle at the pin", bundleRepo, nil, ModelEnglish, pinnedRevision},
		{"standalone at main", standalone, nil, ModelTypedDecisions, "main"},
		{"empty keeps the pin", bundleRepo, []Option{WithRevision("")}, ModelEnglish, pinnedRevision},
		{"bundle follows main", bundleRepo, []Option{WithRevision("main")}, ModelEnglish, "main"},
		{"bundle at a sha", "", []Option{WithRevision(sha)}, ModelEnglish, sha},
		{"standalone at a tag", standalone, []Option{WithRevision("v1")}, ModelTypedDecisions, "v1"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			root, graphs := t.TempDir(), t.TempDir()
			writeCheckpoint(t, root)
			writeGraph(t, graphs, c.graph)
			t.Setenv("LAYA_ONNX_DIR", graphs)

			open, rec := stubOpen(t, root)
			if _, err := open(context.Background(), c.ref, c.opts...); err != nil {
				t.Fatalf("Open: %v", err)
			}
			if len(rec.snapshots) != 1 || rec.snapshots[0].rev != c.wantRev {
				t.Fatalf("snapshots %v, want one at revision %q", rec.snapshots, c.wantRev)
			}
		})
	}
}

// Task 7.2.5: the Router's form reaches every checkpoint its default loader
// downloads, the bundle repo's subfolders included.
func TestRouterRevision(t *testing.T) {
	for _, spec := range []ModelSpec{
		{Repo: bundleRepo},
		{Repo: bundleRepo, Subfolder: ModelMultilingual},
		standaloneModels[ModelMultilingual],
	} {
		t.Run(spec.String(), func(t *testing.T) {
			snap, graphs := t.TempDir(), t.TempDir()
			writeCheckpoint(t, filepath.Join(snap, spec.Subfolder))
			writeGraph(t, graphs, ModelMultilingual)

			l, _, _ := stubbedLoader(t, graphs, WithRouterRevision("main"))
			var gotRev string
			l.snapshot = func(_ context.Context, _, rev string, _ []string) (string, error) {
				gotRev = rev
				return snap, nil
			}
			if _, err := l.load(context.Background(), ModelMultilingual, spec); err != nil {
				t.Fatalf("load: %v", err)
			}
			if gotRev != "main" {
				t.Errorf("revision = %q, want main", gotRev)
			}
		})
	}
}

// Task 7.2.5: a local directory is read as it is; a revision has nothing to
// select there. stubbedLoader fails the test if a snapshot is asked for.
func TestRevisionIgnoredForLocalDir(t *testing.T) {
	ck, graphs := t.TempDir(), t.TempDir()
	writeCheckpoint(t, ck)
	writeGraph(t, graphs, ModelEnglish)

	l, calls, _ := stubbedLoader(t, graphs, WithRouterRevision("main"))
	if _, err := l.load(context.Background(), ModelEnglish, ModelSpecFromString(ck)); err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(*calls) != 1 {
		t.Errorf("open called %d times, want 1", len(*calls))
	}
}

// Task 7.2.5: a revision is spliced into a URL and a cache path, so one that
// climbs out of either fails with hub.ErrInvalidPath before a single request.
// The real Hub client runs here, against a server that fails the test.
func TestOpenRevisionInvalid(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		t.Errorf("request to %s with an invalid revision", r.URL)
	}))
	defer srv.Close()
	t.Setenv("LAYA_CACHE", t.TempDir())
	t.Setenv("LAYA_ONNX_DIR", t.TempDir())

	newLoader := func(s loaderSettings) *defaultLoader {
		l := newDefaultLoader(s)
		l.hub.Endpoint, l.hub.Offline = srv.URL, false
		return l
	}
	_, err := openAgent(context.Background(), "", []Option{WithRevision("../x")}, newLoader)
	if !errors.Is(err, hub.ErrInvalidPath) {
		t.Fatalf("Open(WithRevision(%q)) = %v, want hub.ErrInvalidPath", "../x", err)
	}
}

// Task 7.7.7 (D29): WithCacheDir moves all three things the laya cache holds:
// the Hub snapshots, the default export directory <cache>/onnx, and the ORT
// library lookup. $LAYA_CACHE, which it replaces, is set to somewhere else.
func TestOpenCacheDir(t *testing.T) {
	cache, root := t.TempDir(), t.TempDir()
	t.Setenv("LAYA_CACHE", t.TempDir())
	t.Setenv("LAYA_ONNX_DIR", "")
	writeCheckpoint(t, root)
	if err := os.Mkdir(filepath.Join(cache, "onnx"), 0o750); err != nil {
		t.Fatal(err)
	}
	want := writeGraph(t, filepath.Join(cache, "onnx"), ModelEnglish)

	open, rec := stubOpen(t, root)
	if _, err := open(context.Background(), "", WithCacheDir(cache)); err != nil {
		t.Fatalf("Open: %v", err)
	}
	if rec.hubDir != cache {
		t.Errorf("hub.Client.Dir = %q, want %q", rec.hubDir, cache)
	}
	if len(rec.opens) != 1 || rec.opens[0].path != want {
		t.Fatalf("opened %v, want %q", rec.opens, want)
	}
	if got := rec.opens[0].opts.CacheDir; got != cache {
		t.Errorf("onnx.Options.CacheDir = %q, want %q", got, cache)
	}
}

// Task 7.7.7: without WithCacheDir every part of the cache stays where it
// was, hub.DefaultDir(): the clients get "" and resolve it themselves.
func TestOpenCacheDirDefault(t *testing.T) {
	env, root := t.TempDir(), t.TempDir()
	t.Setenv("LAYA_CACHE", env)
	t.Setenv("LAYA_ONNX_DIR", "")
	writeCheckpoint(t, root)
	if err := os.Mkdir(filepath.Join(env, "onnx"), 0o750); err != nil {
		t.Fatal(err)
	}
	want := writeGraph(t, filepath.Join(env, "onnx"), ModelEnglish)

	open, rec := stubOpen(t, root)
	if _, err := open(context.Background(), ""); err != nil {
		t.Fatalf("Open: %v", err)
	}
	if rec.hubDir != "" || len(rec.opens) != 1 || rec.opens[0].path != want || rec.opens[0].opts.CacheDir != "" {
		t.Errorf("hub dir %q, opens %+v; want \"\", one open of %q with CacheDir \"\"", rec.hubDir, rec.opens, want)
	}
}

// Task 7.7.7: WithONNXDir and $LAYA_ONNX_DIR still win over the cache
// directory's onnx/, in the existing order; WithRouterCacheDir is the same
// setting for the Router.
func TestRouterCacheDirONNXOrder(t *testing.T) {
	cache, env, opt := t.TempDir(), t.TempDir(), t.TempDir()
	t.Setenv("LAYA_CACHE", t.TempDir())

	t.Setenv("LAYA_ONNX_DIR", "")
	l, _, _ := stubbedLoader(t, "", WithRouterCacheDir(cache))
	if got, err := l.graphDir(); err != nil || got != filepath.Join(cache, "onnx") {
		t.Errorf("graphDir = %q, %v; want %q", got, err, filepath.Join(cache, "onnx"))
	}
	if l.hub.Dir != cache {
		t.Errorf("hub.Client.Dir = %q, want %q", l.hub.Dir, cache)
	}
	t.Setenv("LAYA_ONNX_DIR", env)
	if got, _ := newDefaultLoader(loaderSettings{cacheDir: cache}).graphDir(); got != env {
		t.Errorf("graphDir with LAYA_ONNX_DIR = %q, want %q", got, env)
	}
	if got, _ := newDefaultLoader(loaderSettings{cacheDir: cache, onnxDir: opt}).graphDir(); got != opt {
		t.Errorf("graphDir with WithONNXDir = %q, want %q", got, opt)
	}
}

// Task 7.7.7: the Router's cache directory reaches the ORT library lookup of
// every backend its default loader opens.
func TestRouterCacheDirReachesBackend(t *testing.T) {
	ck, graphs, cache := t.TempDir(), t.TempDir(), t.TempDir()
	writeCheckpoint(t, ck)
	writeGraph(t, graphs, ModelEnglish)

	l, calls, _ := stubbedLoader(t, graphs, WithRouterCacheDir(cache))
	if _, err := l.load(context.Background(), ModelEnglish, ModelSpecFromString(ck)); err != nil {
		t.Fatalf("load: %v", err)
	}
	if got := (*calls)[0].opts.CacheDir; got != cache {
		t.Errorf("onnx.Options.CacheDir = %q, want %q", got, cache)
	}
}

// Task 7.7.7 (D29): the logger reaches onnx.Options.Logger, which receives
// the device-fallback warnings. Emitting those needs a real ONNX Runtime
// (device_ort_test.go), so this checks that the caller's own handler is the
// one the backend gets. nil keeps slog.Default(), which the backend picks.
func TestOpenLogger(t *testing.T) {
	ck := t.TempDir()
	writeCheckpoint(t, ck)
	graph := writeGraph(t, t.TempDir(), "x")
	lg := slog.New(slog.NewTextHandler(&bytes.Buffer{}, nil))

	open, rec := stubOpen(t, "")
	if _, err := open(context.Background(), ck, WithGraph(graph), WithLogger(lg)); err != nil {
		t.Fatalf("Open: %v", err)
	}
	if got := rec.opens[0].opts.Logger; got != lg {
		t.Fatalf("onnx.Options.Logger = %p, want the caller's %p", got, lg)
	}

	open, rec = stubOpen(t, "")
	if _, err := open(context.Background(), ck, WithGraph(graph), WithLogger(nil)); err != nil {
		t.Fatalf("Open: %v", err)
	}
	if rec.opens[0].opts.Logger != nil {
		t.Errorf("Logger with WithLogger(nil) = %p, want nil (slog.Default())", rec.opens[0].opts.Logger)
	}
}

// Task 7.7.7: the Router's logger reaches every backend its default loader
// opens.
func TestRouterLogger(t *testing.T) {
	ck, graphs := t.TempDir(), t.TempDir()
	writeCheckpoint(t, ck)
	writeGraph(t, graphs, ModelEnglish)
	lg := slog.New(slog.NewTextHandler(&bytes.Buffer{}, nil))

	l, calls, _ := stubbedLoader(t, graphs, WithRouterLogger(lg))
	if _, err := l.load(context.Background(), ModelEnglish, ModelSpecFromString(ck)); err != nil {
		t.Fatalf("load: %v", err)
	}
	if got := (*calls)[0].opts.Logger; got != lg {
		t.Errorf("onnx.Options.Logger = %p, want the caller's %p", got, lg)
	}
}

// Task 7.7.7 (D29): WithBackend runs the checkpoint on the caller's backend.
// The config and the tokenizer still come from the Hub, but no export is
// looked for and no ONNX backend is opened; the agent then owns the backend,
// and its Close closes it.
func TestOpenWithBackend(t *testing.T) {
	root := t.TempDir()
	writeCheckpoint(t, root)
	t.Setenv("LAYA_ONNX_DIR", t.TempDir()) // holds no export
	b := &closeCounter{}

	open, rec := stubOpen(t, root)
	a, err := open(context.Background(), "", WithBackend(b))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if len(rec.snapshots) != 1 || rec.snapshots[0].repo != bundleRepo {
		t.Errorf("snapshots %v, want the bundle repo's config and tokenizer", rec.snapshots)
	}
	if len(rec.opens) != 0 {
		t.Errorf("opened %v, want no ONNX backend", rec.opens)
	}
	_, err = a.SystemOne(context.Background(), "hello", Questions{{ID: "q", Q: NoulQuestion{Ins: "Is it?"}}})
	if err == nil || !strings.Contains(err.Error(), "closeCounter") {
		t.Errorf("SystemOne = %v, want the caller's backend's forward error", err)
	}
	if b.n != 0 {
		t.Fatalf("backend closed %d times before Agent.Close", b.n)
	}
	if err := a.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if b.n != 1 {
		t.Errorf("backend closed %d times by Agent.Close, want 1", b.n)
	}
}

// Task 7.7.7: a WithBackend Open that fails leaves the backend with the
// caller, open, whether it failed before the checkpoint was read or after.
func TestOpenWithBackendFailureLeavesBackendOpen(t *testing.T) {
	badLimits := t.TempDir()
	writeCheckpoint(t, badLimits)
	if err := os.WriteFile(filepath.Join(badLimits, "rl_agent_config.json"),
		[]byte(`{"encoder": "answerdotai/ModernBERT-large", "head_layers": 2, "max_len": 0}`), 0o600); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		ref  string
		want error
	}{
		{"missing checkpoint", filepath.Join(t.TempDir(), "nope"), ErrCheckpointNotFound},
		{"agent rejects the config", badLimits, ErrIncompatibleCheckpoint},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b := &closeCounter{}
			open, _ := stubOpen(t, "")
			if _, err := open(context.Background(), tc.ref, WithBackend(b)); !errors.Is(err, tc.want) {
				t.Fatalf("Open = %v, want %v", err, tc.want)
			}
			if b.n != 0 {
				t.Errorf("failed Open closed the caller's backend %d times, want 0", b.n)
			}
		})
	}
}

// Task 7.7.7 (D29): WithGraph and WithDevice configure the ONNX backend that
// WithBackend replaces, so either would be silently ignored. Open refuses the
// pair, in either order, before anything is downloaded or closed.
func TestOpenWithBackendConflictingOptions(t *testing.T) {
	for _, tc := range []struct {
		name string
		opts func(backend.Backend) []Option
	}{
		{"graph", func(b backend.Backend) []Option { return []Option{WithBackend(b), WithGraph("x.onnx")} }},
		{"graph first", func(b backend.Backend) []Option { return []Option{WithGraph("x.onnx"), WithBackend(b)} }},
		{"device", func(b backend.Backend) []Option { return []Option{WithBackend(b), WithDevice("cpu")} }},
		{"auto device", func(b backend.Backend) []Option { return []Option{WithDevice("auto"), WithBackend(b)} }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b := &closeCounter{}
			open, rec := stubOpen(t, t.TempDir())
			_, err := open(context.Background(), "", tc.opts(b)...)
			if !errors.Is(err, ErrConflictingOptions) {
				t.Fatalf("Open = %v, want ErrConflictingOptions", err)
			}
			if len(rec.snapshots) != 0 || len(rec.opens) != 0 || b.n != 0 {
				t.Errorf("snapshots %v, opens %v, backend closed %d times; want none of them",
					rec.snapshots, rec.opens, b.n)
			}
		})
	}
}

// Task 7.7.7: WithBackend(nil) is Open without it, as a nil WithLogger and
// an empty WithDevice, WithCacheDir or WithRevision are their defaults: the
// ONNX backend is opened on the export.
func TestOpenWithBackendNil(t *testing.T) {
	ck := t.TempDir()
	writeCheckpoint(t, ck)
	graph := writeGraph(t, t.TempDir(), "x")

	open, rec := stubOpen(t, "")
	if _, err := open(context.Background(), ck, WithBackend(nil), WithGraph(graph), WithDevice("cpu")); err != nil {
		t.Fatalf("Open: %v", err)
	}
	if len(rec.opens) != 1 || rec.opens[0].path != graph {
		t.Errorf("opened %v, want the ONNX backend on %q", rec.opens, graph)
	}
}

// Task 7.7.7: beside WithBackend, the options that shape the download and
// the agent still apply: the revision and the cache directory reach the Hub
// client, and WithLimits the agent built on the caller's backend.
func TestOpenWithBackendKeepsDownloadOptions(t *testing.T) {
	root, cache := t.TempDir(), t.TempDir()
	writeCheckpoint(t, root)

	open, rec := stubOpen(t, root)
	a, err := open(context.Background(), "", WithBackend(&closeCounter{}),
		WithRevision("main"), WithCacheDir(cache), WithLimits(64, 32))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if len(rec.snapshots) != 1 || rec.snapshots[0].rev != "main" || rec.hubDir != cache {
		t.Errorf("snapshots %v in %q, want one at main in %q", rec.snapshots, rec.hubDir, cache)
	}
	if a.MaxLen() != 64 || a.HeadMaxLen() != 32 {
		t.Errorf("limits = %d, %d; want 64, 32", a.MaxLen(), a.HeadMaxLen())
	}
}
