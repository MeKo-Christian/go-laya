package hub

import (
	"context"
	"errors"
	"io/fs"
	"maps"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// The expected hashes are literals, taken from `git hash-object` and
// `sha256sum` over "hello\n", never computed by the code under test.
const (
	hello       = "hello\n"
	helloSHA1   = "ce013625030ba8dba906f756967f9e9ca394464a"
	helloSHA256 = "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"
	commit      = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
)

type seen struct{ server, method, path, auth string }

// fakeHub plays both hops of a Hub download as the live Hub does them
// (2026-09-26): the resolve URL answers HEAD and GET with X-Repo-Commit and
// X-Linked-Etag, and either serves the body, redirects relative (307, a
// regular file) or redirects to another host (302, an LFS file on the CDN).
type fakeHub struct {
	hub, cdn *httptest.Server

	body     string
	header   http.Header // first-hop headers
	status   int         // first-hop status, when not 0
	redirect string      // "", "relative" or "cdn"
	// serve replaces the plain body write, to stall or break a download.
	serve func(w http.ResponseWriter, r *http.Request)

	mu   sync.Mutex
	reqs []seen
	gets int
}

func newFakeHub(t *testing.T) *fakeHub {
	t.Helper()

	f := &fakeHub{
		body: hello,
		header: http.Header{
			"X-Repo-Commit": {commit},
			"X-Linked-Etag": {`"` + helloSHA256 + `"`},
		},
	}
	f.hub = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.record("hub", r)
		if strings.HasPrefix(r.URL.Path, "/api/resolve-cache/") {
			f.serveBody(w, r)
			return
		}
		maps.Copy(w.Header(), f.header)
		switch {
		case f.status != 0:
			w.WriteHeader(f.status)
		case f.redirect == "relative":
			w.Header().Set("Location", "/api/resolve-cache/models/x")
			w.WriteHeader(http.StatusTemporaryRedirect)
		case f.redirect == "cdn":
			w.Header().Set("Location", f.cdn.URL+"/blob")
			w.WriteHeader(http.StatusFound)
		case r.Method == http.MethodHead:
			w.WriteHeader(http.StatusOK)
		default:
			f.serveBody(w, r)
		}
	}))
	f.cdn = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.record("cdn", r)
		f.serveBody(w, r)
	}))
	t.Cleanup(f.hub.Close)
	t.Cleanup(f.cdn.Close)
	return f
}

func (f *fakeHub) record(server string, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.reqs = append(f.reqs, seen{server, r.Method, r.URL.EscapedPath(), r.Header.Get("Authorization")})
}

func (f *fakeHub) serveBody(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	f.gets++
	f.mu.Unlock()
	if f.serve != nil {
		f.serve(w, r)
		return
	}
	_, _ = w.Write([]byte(f.body))
}

func (f *fakeHub) seen() []seen {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]seen(nil), f.reqs...)
}

func (f *fakeHub) client(t *testing.T, token string) *Client {
	t.Helper()
	return &Client{Endpoint: f.hub.URL, Token: token, Dir: t.TempDir()}
}

// files lists every regular file under dir, relative to it.
func files(t *testing.T, dir string) []string {
	t.Helper()

	var out []string
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() {
			rel, _ := filepath.Rel(dir, p)
			out = append(out, filepath.ToSlash(rel))
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk %s: %v", dir, err)
	}
	return out
}

// TestFetchResolvesURL is Task 6.2.1: both requests go to
// {endpoint}/{repo}/resolve/{rev}/{path}, with the revision path-escaped as
// huggingface_hub quotes it, so "refs/pr/1" stays one segment.
func TestFetchResolvesURL(t *testing.T) {
	for _, c := range []struct{ rev, want string }{
		{"main", "/org/model/resolve/main/sub/config.json"},
		{"refs/pr/1", "/org/model/resolve/refs%2Fpr%2F1/sub/config.json"},
	} {
		t.Run(c.rev, func(t *testing.T) {
			f := newFakeHub(t)
			if _, err := f.client(t, "").Fetch(context.Background(), "org/model", c.rev, "sub/config.json"); err != nil {
				t.Fatalf("Fetch: %v", err)
			}
			got := f.seen()
			if len(got) != 2 || got[0].method != http.MethodHead || got[1].method != http.MethodGet {
				t.Fatalf("requests = %+v, want one HEAD and one GET", got)
			}
			for _, r := range got {
				if r.path != c.want {
					t.Errorf("%s %s, want %s", r.method, r.path, c.want)
				}
			}
		})
	}
}

// TestFetchAuth is Task 6.2.1's optional bearer token, and the rule that keeps
// it on the Hub: an LFS download redirects to a CDN host, and a presigned URL
// there must not receive the user's token.
func TestFetchAuth(t *testing.T) {
	const tok = "hf_test_token"
	cases := []struct {
		name, token, redirect string
		want                  map[string]string // server -> Authorization it must see
	}{
		{"no token", "", "", map[string]string{"hub": ""}},
		{"token", tok, "", map[string]string{"hub": "Bearer " + tok}},
		{"relative redirect keeps it", tok, "relative", map[string]string{"hub": "Bearer " + tok}},
		{"cdn redirect drops it", tok, "cdn", map[string]string{"hub": "Bearer " + tok, "cdn": ""}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			f := newFakeHub(t)
			f.redirect = c.redirect
			if _, err := f.client(t, c.token).Fetch(context.Background(), "org/model", "main", "a.bin"); err != nil {
				t.Fatalf("Fetch: %v", err)
			}
			servers := map[string]bool{}
			for _, r := range f.seen() {
				servers[r.server] = true
				if r.auth != c.want[r.server] {
					t.Errorf("%s %s %s: Authorization %q, want %q", r.server, r.method, r.path, r.auth, c.want[r.server])
				}
			}
			for s := range c.want {
				if !servers[s] {
					t.Errorf("no request reached %s", s)
				}
			}
		})
	}
}

// TestFetchVerifies is Task 6.2.2 (R7): X-Linked-Etag is the git blob sha1 of
// a regular file and the sha256 of an LFS file, and a download that matches
// neither is never cached. A weak or missing tag is not a content hash -- the
// Hub's 404 carries a weak one -- so it fails closed rather than skipping the
// check.
func TestFetchVerifies(t *testing.T) {
	cases := []struct {
		name   string
		body   string
		header http.Header
		want   error
	}{
		{"lfs sha256", hello, http.Header{"X-Linked-Etag": {`"` + helloSHA256 + `"`}}, nil},
		{"lfs sha256 mismatch", "hellp\n", http.Header{"X-Linked-Etag": {`"` + helloSHA256 + `"`}}, ErrHashMismatch},
		{"git sha1", hello, http.Header{"X-Linked-Etag": {`"` + helloSHA1 + `"`}}, nil},
		{"git sha1 mismatch", "hellp\n", http.Header{"X-Linked-Etag": {`"` + helloSHA1 + `"`}}, ErrHashMismatch},
		{"plain etag", hello, http.Header{"Etag": {`"` + helloSHA1 + `"`}}, nil},
		{"linked etag wins", hello, http.Header{"X-Linked-Etag": {`"` + helloSHA256 + `"`}, "Etag": {`"0000"`}}, nil},
		{"weak etag", hello, http.Header{"Etag": {`W/"` + helloSHA1 + `"`}}, ErrUnverifiable},
		{"no etag", hello, http.Header{}, ErrUnverifiable},
		{"md5-sized etag", hello, http.Header{"Etag": {`"d41d8cd98f00b204e9800998ecf8427e"`}}, ErrUnverifiable},
	}
	for _, c := range cases {
		for _, redirect := range []string{"", "cdn"} {
			t.Run(c.name+"/"+redirect, func(t *testing.T) {
				f := newFakeHub(t)
				f.body, f.redirect = c.body, redirect
				f.header = c.header.Clone()
				f.header.Set("X-Repo-Commit", commit)
				cl := f.client(t, "")

				local, err := cl.Fetch(context.Background(), "org/model", "main", "a.bin")
				if !errors.Is(err, c.want) {
					t.Fatalf("err = %v, want %v", err, c.want)
				}
				if c.want != nil {
					if got := files(t, cl.Dir); len(got) != 0 {
						t.Errorf("a rejected download left files: %v", got)
					}
					return
				}
				if b, err := os.ReadFile(local); err != nil || string(b) != hello {
					t.Errorf("cached %q, %v; want %q", b, err, hello)
				}
			})
		}
	}
}

// TestFetchCache is Task 6.2.3: the file lands at Dir/owner/name/<commit>/path
// by rename, so nothing else is left beside it, and a second Fetch of a cached
// file downloads nothing.
func TestFetchCache(t *testing.T) {
	f := newFakeHub(t)
	cl := f.client(t, "")

	local, err := cl.Fetch(context.Background(), "org/model", "main", "sub/config.json")
	if err != nil {
		t.Fatalf("Fetch: %v", err)
	}
	want := filepath.Join(cl.Dir, "org", "model", commit, "sub", "config.json")
	if local != want {
		t.Errorf("local = %s, want %s", local, want)
	}
	if got := files(t, cl.Dir); len(got) != 1 || got[0] != "org/model/"+commit+"/sub/config.json" {
		t.Errorf("cache holds %v, want only the file", got)
	}

	again, err := cl.Fetch(context.Background(), "org/model", "main", "sub/config.json")
	if err != nil || again != local {
		t.Fatalf("second Fetch = %s, %v; want %s", again, err, local)
	}
	if f.gets != 1 {
		t.Errorf("body served %d times, want 1: the second Fetch re-downloaded", f.gets)
	}
}

// TestDefaultDir is the other half of Task 6.2.3: $LAYA_CACHE, else
// os.UserCacheDir()/laya (docs/API.md WithCacheDir).
func TestDefaultDir(t *testing.T) {
	t.Setenv("LAYA_CACHE", "/somewhere/else")
	if got, err := DefaultDir(); err != nil || got != "/somewhere/else" {
		t.Errorf("with LAYA_CACHE: %q, %v", got, err)
	}

	t.Setenv("LAYA_CACHE", "")
	base, err := os.UserCacheDir()
	if err != nil {
		t.Skipf("no user cache dir here: %v", err)
	}
	if got, err := DefaultDir(); err != nil || got != filepath.Join(base, "laya") {
		t.Errorf("without LAYA_CACHE: %q, %v; want %s", got, err, filepath.Join(base, "laya"))
	}
}

// TestFetchCancel is Task 6.2.5: a download cancelled, or cut off, halfway
// through its body leaves no file anywhere in the cache -- not the final one,
// and not a partial one a later run might trip over.
func TestFetchCancel(t *testing.T) {
	t.Run("cancelled", func(t *testing.T) {
		f := newFakeHub(t)
		started := make(chan struct{})
		f.serve = func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Length", "6")
			_, _ = w.Write([]byte("hel"))
			w.(http.Flusher).Flush()
			close(started)
			<-r.Context().Done()
		}
		cl := f.client(t, "")
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		done := make(chan error, 1)
		go func() {
			_, err := cl.Fetch(ctx, "org/model", "main", "a.bin")
			done <- err
		}()
		<-started
		// Halfway through, the cache path must not exist yet: a crash here
		// runs no cleanup, and the next run would serve a truncated file.
		// The server has flushed; wait until the client has a file open.
		var mid []string
		for deadline := time.Now().Add(5 * time.Second); len(mid) == 0 && time.Now().Before(deadline); {
			time.Sleep(time.Millisecond)
			mid = files(t, cl.Dir)
		}
		for _, p := range mid {
			if p == "org/model/"+commit+"/a.bin" {
				t.Errorf("mid-download, the cache path %s already exists", p)
			}
		}
		if len(mid) == 0 {
			t.Error("mid-download, no file was being written")
		}
		cancel()
		if err := <-done; !errors.Is(err, context.Canceled) {
			t.Fatalf("err = %v, want context.Canceled", err)
		}
		if got := files(t, cl.Dir); len(got) != 0 {
			t.Errorf("a cancelled download left files: %v", got)
		}
	})

	t.Run("connection dropped", func(t *testing.T) {
		f := newFakeHub(t)
		f.serve = func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Length", "6")
			_, _ = w.Write([]byte("hel"))
			w.(http.Flusher).Flush()
			panic(http.ErrAbortHandler)
		}
		cl := f.client(t, "")
		if _, err := cl.Fetch(context.Background(), "org/model", "main", "a.bin"); err == nil {
			t.Fatal("a truncated download succeeded")
		}
		if got := files(t, cl.Dir); len(got) != 0 {
			t.Errorf("a truncated download left files: %v", got)
		}
	})
}

// TestFetchRejectsUnsafeNames: repo, revision and path become URL and file
// path segments, and from 6.2.4 on the path comes from the Hub's own listing.
// Nothing that could leave the cache directory is requested, let alone
// written -- and neither is a commit id the server made up.
func TestFetchRejectsUnsafeNames(t *testing.T) {
	cases := []struct{ repo, rev, path string }{
		{"org/model", "main", "../x"},
		{"org/model", "main", "/etc/passwd"},
		{"org/model", "main", "a/../../b"},
		{"org/model", "main", "a//b"},
		{"org/model", "main", "./a"},
		{"org/model", "main", `a\b`},
		{"org/model", "main", ""},
		{"../x", "main", "a"},
		{"a/b/c", "main", "a"},
		{"model", "main", "a"},
		{"org/..", "main", "a"},
		{"org/model", "", "a"},
		{"org/model", "..", "a"},
		{"org/model", "a/../b", "a"},
	}
	f := newFakeHub(t)
	for _, c := range cases {
		cl := f.client(t, "")
		if _, err := cl.Fetch(context.Background(), c.repo, c.rev, c.path); !errors.Is(err, ErrInvalidPath) {
			t.Errorf("Fetch(%q, %q, %q) = %v, want ErrInvalidPath", c.repo, c.rev, c.path, err)
		}
	}
	if got := f.seen(); len(got) != 0 {
		t.Errorf("invalid names reached the server: %+v", got)
	}

	f.header.Set("X-Repo-Commit", "../../evil")
	cl := f.client(t, "")
	if _, err := cl.Fetch(context.Background(), "org/model", "main", "a"); !errors.Is(err, ErrInvalidPath) {
		t.Errorf("commit ../../evil: err = %v, want ErrInvalidPath", err)
	}
	if got := files(t, filepath.Dir(cl.Dir)); len(got) != 0 {
		t.Errorf("a made-up commit id wrote %v", got)
	}
}

// TestFetchNotFound: a missing file is ErrNotFound, another failure names its
// status, and neither error repeats the token.
func TestFetchNotFound(t *testing.T) {
	const tok = "hf_secret_token"
	for _, c := range []struct {
		status int
		want   error
	}{{http.StatusNotFound, ErrNotFound}, {http.StatusInternalServerError, nil}} {
		f := newFakeHub(t)
		f.status = c.status
		f.header.Set("Etag", `W/"f-mY2VvLxuxB7KhsoOdQTlMTccuAQ"`)
		cl := f.client(t, tok)
		_, err := cl.Fetch(context.Background(), "org/model", "main", "a")
		switch {
		case err == nil:
			t.Errorf("status %d: no error", c.status)
		case c.want != nil && !errors.Is(err, c.want):
			t.Errorf("status %d: err = %v, want %v", c.status, err, c.want)
		case !strings.Contains(err.Error(), http.StatusText(c.status)) && c.want == nil:
			t.Errorf("status %d: error %q does not name the status", c.status, err)
		}
		if err != nil && strings.Contains(err.Error(), tok) {
			t.Errorf("status %d: error leaks the token: %v", c.status, err)
		}
		if got := files(t, cl.Dir); len(got) != 0 {
			t.Errorf("status %d left files: %v", c.status, got)
		}
	}
}
