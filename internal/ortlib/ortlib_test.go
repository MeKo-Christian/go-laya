package ortlib

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io/fs"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

const member = "onnxruntime-test-1.23.0/lib/libonnxruntime.so.1.23.0"

var libBytes = []byte("\x7fELF pretend shared library")

func sum(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

// entry is one tar member of a synthetic release archive.
type entry struct {
	name     string
	body     []byte
	typeflag byte
	link     string
}

func tgz(t *testing.T, entries ...entry) []byte {
	t.Helper()
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gz)
	for _, e := range entries {
		hdr := &tar.Header{Name: e.name, Mode: 0o755, Size: int64(len(e.body)), Typeflag: e.typeflag, Linkname: e.link}
		if e.typeflag == 0 {
			hdr.Typeflag = tar.TypeReg
		}
		if hdr.Typeflag != tar.TypeReg {
			hdr.Size = 0
		}
		if err := tw.WriteHeader(hdr); err != nil {
			t.Fatal(err)
		}
		if hdr.Typeflag == tar.TypeReg {
			if _, err := tw.Write(e.body); err != nil {
				t.Fatal(err)
			}
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// release describes archive the way the pinned table describes a real one.
func release(archive []byte) Release {
	return Release{
		Archive:       "onnxruntime-test-1.23.0.tgz",
		ArchiveSHA256: sum(archive),
		ArchiveSize:   int64(len(archive)),
		Member:        member,
		LibSHA256:     sum(libBytes),
		LibSize:       int64(len(libBytes)),
	}
}

// server serves body at /<name> and counts the requests it answered.
func server(t *testing.T, name string, body []byte) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		if r.URL.Path != "/"+name {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)
	return srv, &hits
}

// files lists every file under dir, so a test can assert nothing was left
// behind -- no library, no partial archive.
func files(t *testing.T, dir string) []string {
	t.Helper()
	var out []string
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() {
			rel, _ := filepath.Rel(dir, p)
			out = append(out, rel)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func TestDownload(t *testing.T) {
	archive := tgz(
		t,
		entry{name: "onnxruntime-test-1.23.0/lib/libonnxruntime.so", typeflag: tar.TypeSymlink, link: "libonnxruntime.so.1.23.0"},
		entry{name: member, body: libBytes},
		entry{name: "onnxruntime-test-1.23.0/include/onnxruntime_c_api.h", body: []byte("/* header */")},
	)
	rel := release(archive)
	srv, hits := server(t, rel.Archive, archive)
	c := &Client{BaseURL: srv.URL, Dir: t.TempDir()}

	lib, err := c.Download(context.Background(), rel)
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	got, err := os.ReadFile(lib)
	if err != nil || !bytes.Equal(got, libBytes) {
		t.Fatalf("library at %s = %q, %v; want the archive's member", lib, got, err)
	}
	if want := []string{filepath.Join("onnxruntime", "onnxruntime-test-1.23.0", "libonnxruntime.so.1.23.0")}; strings.Join(files(t, c.Dir), ",") != want[0] {
		t.Fatalf("cache holds %v, want exactly %v", files(t, c.Dir), want)
	}

	again, err := c.Download(context.Background(), rel)
	if err != nil || again != lib || hits.Load() != 1 {
		t.Fatalf("second Download = %q, %v after %d requests; want the cached %q and no request", again, err, hits.Load(), lib)
	}
	if cached, err := c.Cached(rel); err != nil || cached != lib {
		t.Fatalf("Cached = %q, %v; want %q", cached, err, lib)
	}
}

// TestDownloadRejects: every way the download or its member can be wrong
// fails with a named error and leaves nothing in the cache.
func TestDownloadRejects(t *testing.T) {
	good := tgz(t, entry{name: member, body: libBytes})

	// pinned is true where the served archive is itself the pinned one, so its
	// hash passes and the case is about what is inside it; otherwise the table
	// pins the good archive and the case is about the download itself.
	cases := []struct {
		name    string
		serve   []byte
		pinned  bool
		wantErr error
		naming  string
	}{
		{
			name:    "archive one byte off",
			serve:   append(append([]byte{}, good[:len(good)-1]...), good[len(good)-1]^1),
			wantErr: ErrHashMismatch, naming: "sha256",
		},
		{name: "archive longer than pinned", serve: append(append([]byte{}, good...), 0), wantErr: ErrHashMismatch, naming: "bytes"},
		{name: "archive shorter than pinned", serve: good[:len(good)-1], wantErr: ErrHashMismatch, naming: "bytes"},
		{
			name: "member missing", pinned: true,
			serve:   tgz(t, entry{name: "onnxruntime-test-1.23.0/lib/other.so", body: libBytes}),
			wantErr: ErrBadArchive, naming: member,
		},
		{
			name: "member is a symlink", pinned: true,
			serve:   tgz(t, entry{name: member, typeflag: tar.TypeSymlink, link: "/etc/passwd"}),
			wantErr: ErrBadArchive, naming: "not a regular file",
		},
		{
			name: "member of another size", pinned: true,
			serve:   tgz(t, entry{name: member, body: append(append([]byte{}, libBytes...), '!')}),
			wantErr: ErrBadArchive, naming: "bytes",
		},
		{
			name: "member with other content", pinned: true,
			serve:   tgz(t, entry{name: member, body: bytes.ToUpper(libBytes)}),
			wantErr: ErrHashMismatch, naming: member,
		},
		{name: "not a gzip", serve: []byte("<html>rate limited</html>"), pinned: true, wantErr: ErrBadArchive, naming: "gzip"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rel := release(good)
			if tc.pinned {
				rel = release(tc.serve)
			}
			srv, _ := server(t, rel.Archive, tc.serve)
			c := &Client{BaseURL: srv.URL, Dir: t.TempDir()}

			lib, err := c.Download(context.Background(), rel)
			if !errors.Is(err, tc.wantErr) || !strings.Contains(err.Error(), tc.naming) {
				t.Fatalf("Download = %q, %v; want %v naming %q", lib, err, tc.wantErr, tc.naming)
			}
			if left := files(t, c.Dir); len(left) != 0 {
				t.Fatalf("a failed download left %v behind", left)
			}
		})
	}
}

func TestDownloadHTTPError(t *testing.T) {
	archive := tgz(t, entry{name: member, body: libBytes})
	rel := release(archive)
	srv, _ := server(t, "somewhere-else.tgz", archive)
	c := &Client{BaseURL: srv.URL, Dir: t.TempDir()}
	if _, err := c.Download(context.Background(), rel); err == nil || !strings.Contains(err.Error(), "404") {
		t.Fatalf("Download = %v, want an error naming the 404", err)
	}
	if left := files(t, c.Dir); len(left) != 0 {
		t.Fatalf("left %v behind", left)
	}
}

// TestDownloadCancel: a download cancelled halfway returns the context's
// error and leaves no partial file.
func TestDownloadCancel(t *testing.T) {
	archive := tgz(t, entry{name: member, body: bytes.Repeat(libBytes, 4096)})
	rel := release(archive)
	ctx, cancel := context.WithCancel(context.Background())
	stalled := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "999999")
		_, _ = w.Write(archive[:len(archive)/2])
		w.(http.Flusher).Flush()
		close(stalled)
		<-r.Context().Done()
	}))
	t.Cleanup(srv.Close)
	c := &Client{BaseURL: srv.URL, Dir: t.TempDir()}

	done := make(chan error, 1)
	go func() { _, err := c.Download(ctx, rel); done <- err }()
	<-stalled
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("Download = %v, want context.Canceled", err)
	}
	if left := files(t, c.Dir); len(left) != 0 {
		t.Fatalf("a cancelled download left %v behind", left)
	}
}

// TestCached: the cache answers only with a library whose bytes are still the
// pinned ones, as a regular file.
func TestCached(t *testing.T) {
	archive := tgz(t, entry{name: member, body: libBytes})
	rel := release(archive)
	srv, _ := server(t, rel.Archive, archive)

	newCache := func(t *testing.T) (*Client, string) {
		t.Helper()
		c := &Client{BaseURL: srv.URL, Dir: t.TempDir()}
		lib, err := c.Download(context.Background(), rel)
		if err != nil {
			t.Fatal(err)
		}
		return c, lib
	}

	t.Run("empty", func(t *testing.T) {
		c := &Client{Dir: t.TempDir()}
		if _, err := c.Cached(rel); !errors.Is(err, ErrNotCached) {
			t.Fatalf("Cached = %v, want ErrNotCached", err)
		}
	})
	t.Run("tampered", func(t *testing.T) {
		c, lib := newCache(t)
		if err := os.WriteFile(lib, bytes.ToUpper(libBytes), 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := c.Cached(rel); !errors.Is(err, ErrHashMismatch) {
			t.Fatalf("Cached = %v, want ErrHashMismatch", err)
		}
		if _, err := c.Download(context.Background(), rel); !errors.Is(err, ErrHashMismatch) {
			t.Fatalf("Download over a tampered cache = %v, want ErrHashMismatch, not a silent re-download", err)
		}
	})
	t.Run("no cache directory", func(t *testing.T) {
		// Nothing to look in is nothing downloaded: resolution moves on to the
		// system's libraries rather than failing on a machine with no $HOME.
		t.Setenv("LAYA_CACHE", "")
		t.Setenv("XDG_CACHE_HOME", "")
		t.Setenv("HOME", "")
		if _, err := (&Client{}).Cached(rel); !errors.Is(err, ErrNotCached) {
			t.Fatalf("Cached = %v, want ErrNotCached", err)
		}
	})
	t.Run("unreadable", func(t *testing.T) {
		if os.Getuid() == 0 {
			t.Skip("root reads a mode-000 file anyway")
		}
		c, lib := newCache(t)
		if err := os.Chmod(lib, 0); err != nil {
			t.Fatal(err)
		}
		_, err := c.Cached(rel)
		if err == nil || errors.Is(err, ErrNotCached) || errors.Is(err, ErrHashMismatch) {
			t.Fatalf("Cached = %v, want the read error itself, not ErrNotCached or ErrHashMismatch", err)
		}
	})
	t.Run("symlink", func(t *testing.T) {
		c, lib := newCache(t)
		elsewhere := filepath.Join(t.TempDir(), "lib.so")
		if err := os.WriteFile(elsewhere, libBytes, 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.Remove(lib); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(elsewhere, lib); err != nil {
			t.Fatal(err)
		}
		if _, err := c.Cached(rel); !errors.Is(err, ErrHashMismatch) || !strings.Contains(err.Error(), "regular file") {
			t.Fatalf("Cached = %v, want ErrHashMismatch naming a non-regular file", err)
		}
	})
}

// TestPinned: the table covers exactly the platforms Microsoft publishes a
// CPU archive for that the ONNX backend builds on, all at Version.
func TestPinned(t *testing.T) {
	for _, p := range []string{"linux/amd64", "linux/arm64", "darwin/amd64", "darwin/arm64"} {
		goos, goarch, _ := strings.Cut(p, "/")
		rel, err := Pinned(goos, goarch)
		if err != nil {
			t.Errorf("Pinned(%s): %v", p, err)
			continue
		}
		if !strings.Contains(rel.Archive, "-"+Version+".tgz") || !strings.Contains(rel.Member, Version) ||
			len(rel.ArchiveSHA256) != 64 || len(rel.LibSHA256) != 64 || rel.ArchiveSize <= 0 || rel.LibSize <= 0 {
			t.Errorf("Pinned(%s) = %+v: not a complete %s entry", p, rel, Version)
		}
	}
	for _, p := range []string{"linux/loong64", "netbsd/amd64", "windows/amd64", "linux/386"} {
		goos, goarch, _ := strings.Cut(p, "/")
		if _, err := Pinned(goos, goarch); !errors.Is(err, ErrUnsupportedPlatform) || !strings.Contains(err.Error(), "LAYA_ORT_LIB") {
			t.Errorf("Pinned(%s) = %v, want ErrUnsupportedPlatform pointing at LAYA_ORT_LIB", p, err)
		}
	}
}

// TestPinnedLive re-downloads every pinned archive from GitHub and checks both
// hashes and sizes in the table. It needs the network, so it only runs with
// LAYA_ORT_NET=1; CI never does.
func TestPinnedLive(t *testing.T) {
	if os.Getenv("LAYA_ORT_NET") != "1" {
		t.Skip("set LAYA_ORT_NET=1 to download the pinned archives from GitHub")
	}
	for _, p := range []string{"linux/amd64", "linux/arm64", "darwin/amd64", "darwin/arm64"} {
		t.Run(strings.ReplaceAll(p, "/", "-"), func(t *testing.T) {
			goos, goarch, _ := strings.Cut(p, "/")
			rel, err := Pinned(goos, goarch)
			if err != nil {
				t.Fatal(err)
			}
			c := &Client{Dir: t.TempDir()}
			lib, err := c.Download(context.Background(), rel)
			if err != nil {
				t.Fatalf("Download(%s): %v", rel.Archive, err)
			}
			t.Logf("%s -> %s", rel.Archive, lib)
		})
	}
}
