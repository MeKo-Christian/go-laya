// Package ortlib downloads the pinned ONNX Runtime shared library into the
// laya cache and verifies it (PLAN.md Task 6.6.1, decided: a verified download
// rather than bring-your-own only).
//
// The pin is ONNX Runtime Version, the release every recorded number in this
// repo was measured on. Each platform's release archive is checked against its
// sha256 before it is parsed at all, and only the one library member is
// extracted, as a regular file of the pinned size and sha256. The cached
// library is re-hashed every time it is resolved, so a file that changed in the
// cache is an error rather than something to load.
//
// Nothing here runs on its own: the ONNX backend only ever reads the cache
// (Cached), and a download happens when cmd/laya-ort asks for one.
package ortlib

import (
	"archive/tar"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"
	pathpkg "path"
	"path/filepath"
	"strings"

	"github.com/MeKo-Christian/go-laya/internal/hub"
)

// Version is the pinned ONNX Runtime release.
const Version = "1.23.0"

// DefaultBaseURL is where Microsoft publishes the release archives.
const DefaultBaseURL = "https://github.com/microsoft/onnxruntime/releases/download/v" + Version

var (
	// ErrUnsupportedPlatform is Pinned's error for a platform Microsoft
	// publishes no CPU archive for.
	ErrUnsupportedPlatform = errors.New("ortlib: no pinned ONNX Runtime download for this platform")

	// ErrHashMismatch is the error for an archive or library whose bytes are
	// not the pinned ones.
	ErrHashMismatch = errors.New("ortlib: content does not match the pinned hash")

	// ErrBadArchive is the error for an archive that matched its hash but does
	// not hold the library as the table describes it.
	ErrBadArchive = errors.New("ortlib: archive does not hold the pinned library")

	// ErrNotCached is Cached's error when the library has not been downloaded.
	ErrNotCached = errors.New("ortlib: ONNX Runtime not downloaded")
)

// Release is one platform's pinned archive and the library inside it.
type Release struct {
	Archive       string // asset name, fetched from BaseURL/Archive
	ArchiveSHA256 string // as GitHub publishes it for the asset
	ArchiveSize   int64
	Member        string // the library's path inside the archive, exactly
	LibSHA256     string
	LibSize       int64
}

// pinned is the table for Version. The archive hashes are the digests GitHub
// publishes for the v1.23.0 release assets; the library hashes and sizes were
// taken from those verified archives, and TestPinnedLive (LAYA_ORT_NET=1)
// re-derives all of it. The linux/amd64 library is byte-identical to the one
// every test and benchmark in this repo ran against.
var pinned = map[string]Release{
	"linux/amd64": {
		Archive:       "onnxruntime-linux-x64-1.23.0.tgz",
		ArchiveSHA256: "b6deea7f2e22c10c043019f294a0ea4d2a6c0ae52a009c34847640db75ec5580",
		ArchiveSize:   8257032,
		Member:        "onnxruntime-linux-x64-1.23.0/lib/libonnxruntime.so.1.23.0",
		LibSHA256:     "98b0253652d36c706cd9b873f3e8dc74e107c26cf9694672fb4d88da1c00f250",
		LibSize:       22207288,
	},
	"linux/arm64": {
		Archive:       "onnxruntime-linux-aarch64-1.23.0.tgz",
		ArchiveSHA256: "0b9f47d140411d938e47915824d8daaa424df95a88b5f1fc843172a75168f7a0",
		ArchiveSize:   7216713,
		Member:        "onnxruntime-linux-aarch64-1.23.0/lib/libonnxruntime.so.1.23.0",
		LibSHA256:     "cb068adc50115db2cca077b385a15b35cc07a14ef3bb71aa5d7c66f1982f5264",
		LibSize:       18562312,
	},
	"darwin/amd64": {
		Archive:       "onnxruntime-osx-x86_64-1.23.0.tgz",
		ArchiveSHA256: "a8e43edcaa349cbfc51578a7fc61ea2b88793ccf077b4bc65aca58999d20cf0f",
		ArchiveSize:   11621905,
		Member:        "./onnxruntime-osx-x86_64-1.23.0/lib/libonnxruntime.1.23.0.dylib",
		LibSHA256:     "091d265e49da84ac8eafd6ff76b67688555192a272d784a252d550a858797d6f",
		LibSize:       39582416,
	},
	"darwin/arm64": {
		Archive:       "onnxruntime-osx-arm64-1.23.0.tgz",
		ArchiveSHA256: "8182db0ebb5caa21036a3c78178f17fabb98a7916bdab454467c8f4cf34bcfdf",
		ArchiveSize:   9962096,
		Member:        "./onnxruntime-osx-arm64-1.23.0/lib/libonnxruntime.1.23.0.dylib",
		LibSHA256:     "d3859aecdb70ea099f5b5f4185fe16f0527c6680b18731e6e96fc971ec767cca",
		LibSize:       35011904,
	},
}

// Pinned returns the pinned release for goos/goarch.
func Pinned(goos, goarch string) (Release, error) {
	rel, ok := pinned[goos+"/"+goarch]
	if !ok {
		return Release{}, fmt.Errorf("%w (%s/%s): install ONNX Runtime %s or later yourself and set LAYA_ORT_LIB",
			ErrUnsupportedPlatform, goos, goarch, Version)
	}
	return rel, nil
}

// Client downloads into one cache directory.
type Client struct {
	BaseURL string       // DefaultBaseURL when empty
	Dir     string       // the cache root; hub.DefaultDir() when empty
	HTTP    *http.Client // http.DefaultClient when nil
}

// Cached returns the cached library for rel, after checking it is a regular
// file with the pinned size and sha256. A library that was never downloaded
// is ErrNotCached, and so is one with no cache directory to be in (no
// $LAYA_CACHE and no user cache directory); one that is there but differs is
// ErrHashMismatch. Any other error, such as an unreadable cache, is returned
// as it is.
func (c *Client) Cached(rel Release) (string, error) {
	lib, err := c.path(rel)
	if err != nil {
		return "", fmt.Errorf("%w: %w", ErrNotCached, err)
	}
	fi, err := os.Lstat(lib)
	if errors.Is(err, fs.ErrNotExist) {
		return "", fmt.Errorf("%w: %s", ErrNotCached, lib)
	}
	if err != nil {
		return "", err
	}
	if !fi.Mode().IsRegular() {
		return "", fmt.Errorf("%w: %s is not a regular file", ErrHashMismatch, lib)
	}
	// #nosec G304 -- lib is built from the cache root and the pinned table, never from remote input.
	f, err := os.Open(lib)
	if err != nil {
		return "", err
	}
	defer f.Close()
	if err := verify(f, rel.LibSize, rel.LibSHA256, lib); err != nil {
		return "", err
	}
	return lib, nil
}

// Download returns the cached library for rel, downloading and verifying it
// first unless it is already there. A cached library that fails verification
// is an error, not a reason to download again: something changed a file only
// this package writes, and that is worth a look before anything loads it.
func (c *Client) Download(ctx context.Context, rel Release) (string, error) {
	lib, err := c.Cached(rel)
	if err == nil || !errors.Is(err, ErrNotCached) {
		return lib, err
	}
	if lib, err = c.path(rel); err != nil {
		return "", err
	}
	if err := os.MkdirAll(filepath.Dir(lib), 0o750); err != nil {
		return "", err
	}

	archive, err := c.fetch(ctx, rel, filepath.Dir(lib))
	if err != nil {
		return "", err
	}
	defer func() { _ = os.Remove(archive.Name()) }()
	defer archive.Close()

	if err := extract(archive, rel, lib); err != nil {
		return "", err
	}
	return lib, nil
}

// path is where rel's library lives in the cache:
// Dir/onnxruntime/<archive name without .tgz>/<library file name>.
func (c *Client) path(rel Release) (string, error) {
	dir := c.Dir
	if dir == "" {
		var err error
		if dir, err = hub.DefaultDir(); err != nil {
			return "", err
		}
	}
	return filepath.Join(dir, "onnxruntime", strings.TrimSuffix(rel.Archive, ".tgz"), pathpkg.Base(rel.Member)), nil
}

// fetch downloads rel's archive into a temporary file in dir and returns it
// rewound, only once its size and sha256 are the pinned ones.
func (c *Client) fetch(ctx context.Context, rel Release, dir string) (*os.File, error) {
	base := c.BaseURL
	if base == "" {
		base = DefaultBaseURL
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimSuffix(base, "/")+"/"+rel.Archive, nil)
	if err != nil {
		return nil, err
	}
	client := c.HTTP
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ortlib: %s: %w", rel.Archive, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ortlib: %s: HTTP %s", rel.Archive, resp.Status)
	}

	tmp, err := os.CreateTemp(dir, ".partial-*.tgz")
	if err != nil {
		return nil, err
	}
	fail := func(err error) error {
		_ = tmp.Close()
		_ = os.Remove(tmp.Name())
		return err
	}
	// One byte past the pinned size is enough to tell a longer body apart.
	if _, err := io.Copy(tmp, io.LimitReader(resp.Body, rel.ArchiveSize+1)); err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			err = ctxErr
		}
		return nil, fail(fmt.Errorf("ortlib: %s: %w", rel.Archive, err))
	}
	if _, err := tmp.Seek(0, io.SeekStart); err != nil {
		return nil, fail(err)
	}
	if err := verify(tmp, rel.ArchiveSize, rel.ArchiveSHA256, rel.Archive); err != nil {
		return nil, fail(err)
	}
	if _, err := tmp.Seek(0, io.SeekStart); err != nil {
		return nil, fail(err)
	}
	return tmp, nil
}

// extract writes rel.Member from the verified archive to lib, by rename after
// its size and sha256 matched, so lib never exists half-written.
func extract(archive io.Reader, rel Release, lib string) error {
	gz, err := gzip.NewReader(archive)
	if err != nil {
		return fmt.Errorf("%w: %s: gzip: %w", ErrBadArchive, rel.Archive, err)
	}
	defer gz.Close()
	tr := tar.NewReader(gz)
	for {
		hdr, err := tr.Next()
		if errors.Is(err, io.EOF) {
			return fmt.Errorf("%w: %s has no %s", ErrBadArchive, rel.Archive, rel.Member)
		}
		if err != nil {
			return fmt.Errorf("%w: %s: %w", ErrBadArchive, rel.Archive, err)
		}
		if hdr.Name != rel.Member {
			continue
		}
		if hdr.Typeflag != tar.TypeReg {
			return fmt.Errorf("%w: %s is not a regular file", ErrBadArchive, rel.Member)
		}
		if hdr.Size != rel.LibSize {
			return fmt.Errorf("%w: %s is %d bytes, pinned %d", ErrBadArchive, rel.Member, hdr.Size, rel.LibSize)
		}
		return writeVerified(tr, rel, lib)
	}
}

func writeVerified(r io.Reader, rel Release, lib string) error {
	tmp, err := os.CreateTemp(filepath.Dir(lib), ".partial-*")
	if err != nil {
		return err
	}
	defer func() { _ = os.Remove(tmp.Name()) }() // a no-op once renamed
	h := sha256.New()
	_, err = io.Copy(io.MultiWriter(tmp, h), io.LimitReader(r, rel.LibSize))
	if closeErr := tmp.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != rel.LibSHA256 {
		return fmt.Errorf("%w: %s: sha256 %s, pinned %s", ErrHashMismatch, rel.Member, got, rel.LibSHA256)
	}
	return os.Rename(tmp.Name(), lib)
}

// verify reads all of r and requires exactly size bytes hashing to want.
func verify(r io.Reader, size int64, want, name string) error {
	h := sha256.New()
	n, err := io.Copy(h, io.LimitReader(r, size+1))
	if err != nil {
		return err
	}
	if n != size {
		return fmt.Errorf("%w: %s is %d bytes, pinned %d", ErrHashMismatch, name, n, size)
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != want {
		return fmt.Errorf("%w: %s: sha256 %s, pinned %s", ErrHashMismatch, name, got, want)
	}
	return nil
}
