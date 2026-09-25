package hub

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
)

// fakeRepo is a Hub holding one multi-file repo, org/model, as the live Hub
// lists and serves it (2026-09-26): GET /api/models/{repo}/revision/{rev}
// answers {sha, siblings[].rfilename}, and each resolve URL answers like
// fakeHub's first hop, serving the body directly.
type fakeRepo struct {
	srv *httptest.Server

	sha        string
	files      map[string]string
	listed     []string          // the listing's names; the keys of files when nil
	commits    map[string]string // a file's X-Repo-Commit, when not sha
	listStatus int               // the listing's status, when not 0

	mu   sync.Mutex
	reqs []seen
}

// checkpoints mirrors the bundle repo's layout: the English checkpoint at the
// root, two more in subfolders, and a sibling folder sharing a prefix.
var checkpoints = map[string]string{
	"model.safetensors":                     "root weights\n",
	"tokenizer/tokenizer.json":              "root tokenizer\n",
	"multilingual/model.safetensors":        "multilingual weights\n",
	"multilingual/tokenizer/tokenizer.json": "multilingual tokenizer\n",
	"typed-decisions/model.safetensors":     "typed weights\n",
	"multilingualx/a":                       "not multilingual\n",
}

func newFakeRepo(t *testing.T) *fakeRepo {
	t.Helper()

	f := &fakeRepo{sha: commit, files: checkpoints, commits: map[string]string{}}
	f.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.mu.Lock()
		f.reqs = append(f.reqs, seen{"hub", r.Method, r.URL.EscapedPath(), r.Header.Get("Authorization")})
		f.mu.Unlock()

		if _, ok := strings.CutPrefix(r.URL.EscapedPath(), "/api/models/org/model/revision/"); ok {
			f.serveListing(w)
			return
		}
		rest, ok := strings.CutPrefix(r.URL.Path, "/org/model/resolve/")
		if !ok {
			http.NotFound(w, r)
			return
		}
		_, path, _ := strings.Cut(rest, "/")
		body, ok := f.files[path]
		if !ok {
			http.NotFound(w, r)
			return
		}
		c := f.sha
		if o, ok := f.commits[path]; ok {
			c = o
		}
		sum := sha256.Sum256([]byte(body))
		w.Header().Set("X-Repo-Commit", c)
		w.Header().Set("X-Linked-Etag", `"`+hex.EncodeToString(sum[:])+`"`)
		if r.Method == http.MethodGet {
			_, _ = w.Write([]byte(body))
		}
	}))
	t.Cleanup(f.srv.Close)
	return f
}

func (f *fakeRepo) serveListing(w http.ResponseWriter) {
	if f.listStatus != 0 {
		w.WriteHeader(f.listStatus)
		return
	}
	names := f.listed
	if names == nil {
		for name := range f.files {
			names = append(names, name)
		}
		slices.Sort(names)
	}
	type sibling struct {
		Rfilename string `json:"rfilename"`
	}
	sibs := make([]sibling, len(names))
	for i, n := range names {
		sibs[i] = sibling{n}
	}
	err := json.NewEncoder(w).Encode(struct {
		ID       string    `json:"id"`
		SHA      string    `json:"sha"`
		Siblings []sibling `json:"siblings"`
	}{"org/model", f.sha, sibs})
	if err != nil {
		panic(err)
	}
}

func (f *fakeRepo) seen() []seen {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]seen(nil), f.reqs...)
}

// downloads lists the files GET was asked for, in order.
func (f *fakeRepo) downloads() []string {
	var out []string
	for _, s := range f.seen() {
		if s.method == http.MethodGet && strings.Contains(s.path, "/resolve/") {
			out = append(out, s.path)
		}
	}
	return out
}

func (f *fakeRepo) client(t *testing.T, token string) *Client {
	t.Helper()
	return &Client{Endpoint: f.srv.URL, Token: token, Dir: t.TempDir()}
}

// TestMatch pins allow-pattern matching to Python's fnmatch, which
// huggingface_hub's filter_repo_objects uses: "*" crosses "/", so upstream's
// f"{subfolder}/*" takes the whole subfolder, nested files included.
func TestMatch(t *testing.T) {
	for _, c := range []struct {
		pattern, name string
		want          bool
	}{
		{"multilingual/*", "multilingual/model.safetensors", true},
		{"multilingual/*", "multilingual/tokenizer/tokenizer.json", true},
		{"multilingual/*", "multilingualx/a", false},
		{"multilingual/*", "multilingual", false},
		{"multilingual/*", "x/multilingual/a", false},
		{"*", "a/b/c", true},
		{"*.json", "tokenizer/tokenizer.json", true},
		{"*.json", "model.safetensors", false},
		{"model.safetensors", "model.safetensors", true},
		{"model.safetensor?", "model.safetensors", true},
		{"model.safetensor?", "model.safetensor", false},
		{"[mt]*/model.safetensors", "multilingual/model.safetensors", true},
		{"[mt]*/model.safetensors", "typed-decisions/model.safetensors", true},
		{"[!m]*/model.safetensors", "multilingual/model.safetensors", false},
		{"[!m]*/model.safetensors", "typed-decisions/model.safetensors", true},
		{"[a-c]", "b", true},
		{"[a-c]", "d", false},
		{"a[", "a[", true}, // an unclosed bracket is a literal
		{"a.b", "axb", false},
		{`a\b`, `a\b`, true}, // fnmatch has no escape character
		{"(a)+", "(a)+", true},
	} {
		if got := match(c.pattern, c.name); got != c.want {
			t.Errorf("match(%q, %q) = %v, want %v", c.pattern, c.name, got, c.want)
		}
	}
}

// TestSnapshotAllowPatterns is Task 6.2.4: a subfolder request downloads that
// subfolder and nothing else, every file at the listing's commit rather than
// at the moving revision it was asked for.
func TestSnapshotAllowPatterns(t *testing.T) {
	t.Run("subfolder", func(t *testing.T) {
		f := newFakeRepo(t)
		cl := f.client(t, "")
		dir, err := cl.Snapshot(context.Background(), "org/model", "main", []string{"multilingual/*"})
		if err != nil {
			t.Fatal(err)
		}
		if want := filepath.Join(cl.Dir, "org", "model", commit); dir != want {
			t.Errorf("dir = %s, want %s", dir, want)
		}

		wantGets := []string{
			"/org/model/resolve/" + commit + "/multilingual/model.safetensors",
			"/org/model/resolve/" + commit + "/multilingual/tokenizer/tokenizer.json",
		}
		if got := f.downloads(); !slices.Equal(got, wantGets) {
			t.Errorf("downloaded %q, want %q", got, wantGets)
		}
		got := files(t, filepath.Join(cl.Dir, "org", "model", commit))
		want := []string{"multilingual/model.safetensors", "multilingual/tokenizer/tokenizer.json"}
		if !slices.Equal(got, want) {
			t.Errorf("snapshot holds %q, want %q", got, want)
		}
		for _, name := range want {
			b, err := os.ReadFile(filepath.Join(dir, filepath.FromSlash(name)))
			if err != nil || string(b) != checkpoints[name] {
				t.Errorf("%s = %q, %v; want %q", name, b, err, checkpoints[name])
			}
		}
	})

	t.Run("everything", func(t *testing.T) {
		f := newFakeRepo(t)
		cl := f.client(t, "")
		dir, err := cl.Snapshot(context.Background(), "org/model", "main", nil)
		if err != nil {
			t.Fatal(err)
		}
		got := files(t, dir)
		want := make([]string, 0, len(checkpoints))
		for name := range checkpoints {
			want = append(want, name)
		}
		slices.Sort(want)
		if !slices.Equal(got, want) {
			t.Errorf("snapshot holds %q, want %q", got, want)
		}
	})

	t.Run("trailing slash", func(t *testing.T) {
		// filter_repo_objects appends "*" to a pattern ending in "/".
		f := newFakeRepo(t)
		cl := f.client(t, "")
		dir, err := cl.Snapshot(context.Background(), "org/model", "main", []string{"typed-decisions/"})
		if err != nil {
			t.Fatal(err)
		}
		if got := files(t, dir); !slices.Equal(got, []string{"typed-decisions/model.safetensors"}) {
			t.Errorf("snapshot holds %q", got)
		}
	})

	t.Run("second call downloads nothing", func(t *testing.T) {
		f := newFakeRepo(t)
		cl := f.client(t, "")
		for range 2 {
			if _, err := cl.Snapshot(context.Background(), "org/model", "main", []string{"multilingual/*"}); err != nil {
				t.Fatal(err)
			}
		}
		if got := f.downloads(); len(got) != 2 {
			t.Errorf("two snapshots downloaded %d files, want 2", len(got))
		}
	})
}

// TestSnapshotListing: the listing request goes to the revision endpoint with
// the revision escaped whole, carries the token, and its failures are named.
func TestSnapshotListing(t *testing.T) {
	const tok = "hf_secret_token"
	f := newFakeRepo(t)
	cl := f.client(t, tok)
	if _, err := cl.Snapshot(context.Background(), "org/model", "refs/pr/1", []string{"multilingual/*"}); err != nil {
		t.Fatal(err)
	}
	first := f.seen()[0]
	want := seen{"hub", http.MethodGet, "/api/models/org/model/revision/refs%2Fpr%2F1", "Bearer " + tok}
	if first != want {
		t.Errorf("first request = %+v, want %+v", first, want)
	}

	for _, c := range []struct {
		name   string
		status int
		sha    string
		want   error
	}{
		{"missing repo", http.StatusNotFound, commit, ErrNotFound},
		{"server error", http.StatusInternalServerError, commit, nil},
		{"made-up commit", 0, "../../evil", ErrInvalidPath},
	} {
		f := newFakeRepo(t)
		f.listStatus, f.sha = c.status, c.sha
		cl := f.client(t, tok)
		_, err := cl.Snapshot(context.Background(), "org/model", "main", nil)
		switch {
		case err == nil:
			t.Errorf("%s: no error", c.name)
		case c.want != nil && !errors.Is(err, c.want):
			t.Errorf("%s: err = %v, want %v", c.name, err, c.want)
		}
		if err != nil && strings.Contains(err.Error(), tok) {
			t.Errorf("%s: error leaks the token: %v", c.name, err)
		}
		if got := f.downloads(); len(got) != 0 {
			t.Errorf("%s: downloaded %q", c.name, got)
		}
		if got := files(t, cl.Dir); len(got) != 0 {
			t.Errorf("%s: left files %q", c.name, got)
		}
	}
}

// TestSnapshotPinsCommit: a file whose resolve answer names another commit
// than the listing fails the snapshot, so a revision moving mid-download
// cannot mix two commits in one directory, and no ref points at it.
func TestSnapshotPinsCommit(t *testing.T) {
	f := newFakeRepo(t)
	f.commits["multilingual/tokenizer/tokenizer.json"] = strings.Repeat("a", 40)
	cl := f.client(t, "")
	_, err := cl.Snapshot(context.Background(), "org/model", "main", []string{"multilingual/*"})
	if !errors.Is(err, ErrCommitMismatch) {
		t.Fatalf("err = %v, want ErrCommitMismatch", err)
	}
	for _, s := range f.seen()[1:] {
		if !strings.HasPrefix(s.path, "/org/model/resolve/"+commit+"/") {
			t.Errorf("%s %s: not at the listing's commit", s.method, s.path)
		}
	}
	for _, name := range files(t, cl.Dir) {
		if strings.Contains(name, "tokenizer") || strings.Contains(name, "refs/") || strings.Contains(name, "listings/") {
			t.Errorf("a failed snapshot left %s", name)
		}
	}
}

// TestSnapshotRejectsUnsafeListing: the names come from the Hub, so one that
// could climb out of the cache fails the whole snapshot before any download.
func TestSnapshotRejectsUnsafeListing(t *testing.T) {
	for _, bad := range []string{"../x", "multilingual/../../x", "/etc/passwd", `multilingual\x`, "", "a//b", "./a"} {
		f := newFakeRepo(t)
		f.listed = []string{"multilingual/model.safetensors", bad}
		cl := f.client(t, "")
		_, err := cl.Snapshot(context.Background(), "org/model", "main", nil)
		if !errors.Is(err, ErrInvalidPath) {
			t.Errorf("%q: err = %v, want ErrInvalidPath", bad, err)
		}
		if got := f.downloads(); len(got) != 0 {
			t.Errorf("%q: downloaded %q", bad, got)
		}
	}
}

// TestSnapshotRejectsUnsafeRevision: the revision becomes refs/<rev>, so a
// backslash, a separator on Windows, is as invalid as "..".
func TestSnapshotRejectsUnsafeRevision(t *testing.T) {
	for _, rev := range []string{`..\..\evil`, `a\b`, "..", "a/../b", ""} {
		for _, offline := range []bool{false, true} {
			f := newFakeRepo(t)
			cl := f.client(t, "")
			cl.Offline = offline
			if _, err := cl.Snapshot(context.Background(), "org/model", rev, nil); !errors.Is(err, ErrInvalidPath) {
				t.Errorf("rev %q, offline %v: err = %v, want ErrInvalidPath", rev, offline, err)
			}
			if got := f.seen(); len(got) != 0 {
				t.Errorf("rev %q: reached the Hub: %+v", rev, got)
			}
		}
	}
}

// TestSnapshotNoMatch: patterns matching nothing are ErrNotFound, where
// upstream downloads nothing and fails later on the missing subfolder.
func TestSnapshotNoMatch(t *testing.T) {
	f := newFakeRepo(t)
	cl := f.client(t, "")
	_, err := cl.Snapshot(context.Background(), "org/model", "main", []string{"english/*"})
	if !errors.Is(err, ErrNotFound) || !strings.Contains(err.Error(), "english/*") {
		t.Errorf("err = %v, want ErrNotFound naming the pattern", err)
	}
	if got := f.downloads(); len(got) != 0 {
		t.Errorf("downloaded %q", got)
	}
}

type failTransport struct{ t *testing.T }

func (f failTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	f.t.Errorf("offline client sent %s %s", r.Method, r.URL.Redacted())
	return nil, errors.New("offline")
}

// TestOffline is Task 6.2.6: once a snapshot is cached, an offline client
// resolves it, by revision or by commit, without a single request; anything
// the cache cannot fully answer is ErrNotCached, never a partial directory.
func TestOffline(t *testing.T) {
	ctx := context.Background()
	f := newFakeRepo(t)
	online := f.client(t, "")
	want, err := online.Snapshot(ctx, "org/model", "main", []string{"multilingual/*"})
	if err != nil {
		t.Fatal(err)
	}
	offline := func() *Client {
		return &Client{
			Endpoint: f.srv.URL, Dir: online.Dir, Offline: true,
			HTTP: &http.Client{Transport: failTransport{t}},
		}
	}
	before := len(f.seen())

	for _, rev := range []string{"main", commit} {
		got, err := offline().Snapshot(ctx, "org/model", rev, []string{"multilingual/*"})
		if err != nil || got != want {
			t.Errorf("Snapshot at %s = %q, %v; want %q", rev, got, err, want)
		}
	}
	local, err := offline().Fetch(ctx, "org/model", "main", "multilingual/model.safetensors")
	if wantLocal := filepath.Join(want, "multilingual", "model.safetensors"); err != nil || local != wantLocal {
		t.Errorf("Fetch = %q, %v; want %q", local, err, wantLocal)
	}

	for _, c := range []struct {
		name  string
		rev   string
		allow []string
	}{
		{"uncached subfolder", "main", []string{"typed-decisions/*"}},
		{"everything", "main", nil},
		{"unknown revision", "v2", []string{"multilingual/*"}},
		{"unknown commit", strings.Repeat("b", 40), []string{"multilingual/*"}},
	} {
		if _, err := offline().Snapshot(ctx, "org/model", c.rev, c.allow); !errors.Is(err, ErrNotCached) {
			t.Errorf("%s: err = %v, want ErrNotCached", c.name, err)
		}
	}
	if _, err := offline().Fetch(ctx, "org/model", "main", "model.safetensors"); !errors.Is(err, ErrNotCached) {
		t.Errorf("Fetch of an uncached file: err = %v, want ErrNotCached", err)
	}

	// A file lost from the cache, or swapped for a symlink to unverified
	// bytes, makes the snapshot incomplete.
	victim := filepath.Join(want, "multilingual", "tokenizer", "tokenizer.json")
	outside := filepath.Join(t.TempDir(), "evil")
	if err := os.WriteFile(outside, []byte("evil"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(victim); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, victim); err != nil {
		t.Fatal(err)
	}
	if _, err := offline().Snapshot(ctx, "org/model", "main", []string{"multilingual/*"}); !errors.Is(err, ErrNotCached) {
		t.Errorf("symlinked file: err = %v, want ErrNotCached", err)
	}
	if err := os.Remove(victim); err != nil {
		t.Fatal(err)
	}
	if _, err := offline().Snapshot(ctx, "org/model", "main", []string{"multilingual/*"}); !errors.Is(err, ErrNotCached) {
		t.Errorf("missing file: err = %v, want ErrNotCached", err)
	}

	// Checking the leaf alone would follow a symlinked parent: a subfolder,
	// the ref or the listing linked outside the cache is not a cached snapshot.
	full := filepath.Join(t.TempDir(), "full")
	if err := os.CopyFS(full, os.DirFS(online.Dir)); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(victim, []byte(checkpoints["multilingual/tokenizer/tokenizer.json"]), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := offline().Snapshot(ctx, "org/model", "main", []string{"multilingual/*"}); err != nil {
		t.Fatalf("restored cache: %v", err)
	}
	base := filepath.Join(online.Dir, "org", "model")
	for _, rel := range []string{
		filepath.Join(commit, "multilingual"),
		filepath.Join("refs", "main"),
		filepath.Join("listings", commit+".json"),
	} {
		t.Run("symlinked "+filepath.ToSlash(rel), func(t *testing.T) {
			orig := filepath.Join(base, rel)
			moved := filepath.Join(t.TempDir(), "moved")
			if err := os.Rename(orig, moved); err != nil {
				t.Fatal(err)
			}
			if err := os.Symlink(moved, orig); err != nil {
				t.Skipf("no symlinks here: %v", err)
			}
			defer func() {
				_ = os.Remove(orig)
				if err := os.Rename(moved, orig); err != nil {
					t.Fatal(err)
				}
			}()
			if _, err := offline().Snapshot(ctx, "org/model", "main", []string{"multilingual/*"}); !errors.Is(err, ErrNotCached) {
				t.Errorf("Snapshot: err = %v, want ErrNotCached", err)
			}
		})
	}

	if after := len(f.seen()); after != before {
		t.Errorf("offline calls reached the Hub: %d requests", after-before)
	}
}
