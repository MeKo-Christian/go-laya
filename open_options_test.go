package laya

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

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
