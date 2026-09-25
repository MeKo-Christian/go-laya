// Package hub downloads checkpoint files from the Hugging Face Hub into a local
// cache (PLAN.md Task 6.2), replacing agent.py:122-128's snapshot_download.
//
// Every file is verified before it is cached (R7): the Hub's X-Linked-Etag is
// the git blob sha1 of a regular file and the sha256 of an LFS file, and a
// download that matches neither never reaches its cache path. A response with
// no usable hash fails rather than skipping the check.
package hub

import (
	"context"
	"crypto/sha1" //nolint:gosec // git's blob id is sha1; it is the Hub's hash, not our choice
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

var (
	// ErrInvalidPath is Fetch's error for a repo, revision, path or commit id
	// that could not safely become a URL and a cache path.
	ErrInvalidPath = errors.New("hub: invalid repo, revision or path")

	// ErrUnverifiable is Fetch's error when the Hub sends no content hash.
	ErrUnverifiable = errors.New("hub: response carries no content hash")

	// ErrHashMismatch is Fetch's error for a download that does not match the
	// hash the Hub announced for it.
	ErrHashMismatch = errors.New("hub: downloaded content does not match its hash")

	// ErrNotFound is Fetch's error for a repo, revision or file the Hub does
	// not have.
	ErrNotFound = errors.New("hub: not found")
)

// DefaultEndpoint is the Hub Fetch talks to when Client.Endpoint is empty.
const DefaultEndpoint = "https://huggingface.co"

// maxRedirects bounds the hops of one download: the Hub answers with at most a
// relative redirect or one to its CDN.
const maxRedirects = 5

var (
	nameRE   = regexp.MustCompile(`^[A-Za-z0-9._-]+$`)
	commitRE = regexp.MustCompile(`^[0-9a-f]{40}$`)
	hashRE   = regexp.MustCompile(`^(?:[0-9a-f]{40}|[0-9a-f]{64})$`)
)

// Client downloads from one Hub endpoint into one cache directory.
type Client struct {
	Endpoint string       // DefaultEndpoint when empty
	Token    string       // sent as a bearer token, and only to Endpoint's host
	Dir      string       // the cache root; DefaultDir() when empty
	HTTP     *http.Client // http.DefaultClient when nil; Fetch follows redirects itself
}

// DefaultDir is the cache root: $LAYA_CACHE, else os.UserCacheDir()/laya.
func DefaultDir() (string, error) {
	if dir := os.Getenv("LAYA_CACHE"); dir != "" {
		return dir, nil
	}
	base, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("hub: no cache directory: %w", err)
	}
	return filepath.Join(base, "laya"), nil
}

// Fetch returns the local path of repo's file at rev, downloading it into
// Dir/owner/name/<commit>/path first unless it is already there. The file only
// ever appears there by rename after its hash matched, so a cancelled or
// failed download leaves nothing behind.
func (c *Client) Fetch(ctx context.Context, repo, rev, path string) (string, error) {
	owner, name, ok := splitRepo(repo)
	if !ok || !validSegments(rev) || !validSegments(path) || strings.Contains(path, `\`) {
		return "", fmt.Errorf("%w: repo %q, revision %q, path %q", ErrInvalidPath, repo, rev, path)
	}
	dir := c.Dir
	if dir == "" {
		var err error
		if dir, err = DefaultDir(); err != nil {
			return "", err
		}
	}
	endpoint := c.Endpoint
	if endpoint == "" {
		endpoint = DefaultEndpoint
	}
	// The revision is escaped whole, as huggingface_hub quotes it, so
	// "refs/pr/1" stays one segment; the path is escaped segment by segment.
	segs := strings.Split(path, "/")
	for i, seg := range segs {
		segs[i] = url.PathEscape(seg)
	}
	resolve, err := url.Parse(strings.TrimRight(endpoint, "/") + "/" + owner + "/" + name +
		"/resolve/" + url.PathEscape(rev) + "/" + strings.Join(segs, "/"))
	if err != nil {
		return "", fmt.Errorf("hub: endpoint %q: %w", endpoint, err)
	}
	hc := noRedirects(c.HTTP)

	commit, want, err := c.head(ctx, hc, resolve.String())
	if err != nil {
		return "", err
	}
	local := filepath.Join(dir, owner, name, commit, filepath.FromSlash(path))
	// Only a regular file is a hit: Stat would follow a symlink planted here
	// to unverified bytes outside the cache, and accept a directory. Anything
	// else is downloaded again; the rename replaces a symlink itself, never
	// its target, and fails on a directory.
	if fi, err := os.Lstat(local); err == nil && fi.Mode().IsRegular() {
		return local, nil
	}

	body, err := c.get(ctx, hc, resolve, resolve.Host)
	if err != nil {
		return "", err
	}
	defer body.Close()
	if err := store(ctx, body, local, want); err != nil {
		return "", fmt.Errorf("%s %s: %w", repo, path, err)
	}
	return local, nil
}

// head reads the commit and the content hash from the resolve URL itself: the
// CDN an LFS file redirects to carries neither, and must not be trusted for
// the hash anyway.
func (c *Client) head(ctx context.Context, hc *http.Client, u string) (commit, want string, err error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodHead, u, http.NoBody)
	if err != nil {
		return "", "", fmt.Errorf("hub: %w", err)
	}
	c.authorize(req)
	resp, err := hc.Do(req)
	if err != nil {
		return "", "", fmt.Errorf("hub: HEAD %s: %w", u, err)
	}
	_ = resp.Body.Close()
	if err := statusErr(resp, u); err != nil {
		return "", "", err
	}

	commit = resp.Header.Get("X-Repo-Commit")
	if !commitRE.MatchString(commit) {
		return "", "", fmt.Errorf("%w: %s: commit id %q", ErrInvalidPath, u, commit)
	}
	tag := resp.Header.Get("X-Linked-Etag")
	if tag == "" {
		tag = resp.Header.Get("Etag")
	}
	// A weak tag is the server's cache validator, not a content hash; the
	// Hub's 404 page carries one.
	want = strings.ToLower(strings.Trim(tag, `"`))
	if strings.HasPrefix(tag, "W/") || !hashRE.MatchString(want) {
		return "", "", fmt.Errorf("%w: %s: etag %q", ErrUnverifiable, u, tag)
	}
	return commit, want, nil
}

// get follows the download's redirects by hand, so that the token goes only to
// the Hub's own host and never to a presigned CDN URL.
func (c *Client) get(ctx context.Context, hc *http.Client, u *url.URL, host string) (io.ReadCloser, error) {
	auth := true
	for range maxRedirects + 1 {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), http.NoBody)
		if err != nil {
			return nil, fmt.Errorf("hub: %w", err)
		}
		if auth {
			c.authorize(req)
		}
		resp, err := hc.Do(req)
		if err != nil {
			return nil, fmt.Errorf("hub: GET %s: %w", u.Redacted(), err)
		}
		if resp.StatusCode < 300 || resp.StatusCode >= 400 {
			if err := statusErr(resp, u.Redacted()); err != nil {
				_ = resp.Body.Close()
				return nil, err
			}
			return resp.Body, nil
		}
		next, err := resp.Location()
		_ = resp.Body.Close()
		if err != nil {
			return nil, fmt.Errorf("hub: GET %s: redirect: %w", u.Redacted(), err)
		}
		if next.Host != host {
			auth = false
		}
		u = next
	}
	return nil, fmt.Errorf("hub: GET %s: more than %d redirects", u.Redacted(), maxRedirects)
}

func (c *Client) authorize(req *http.Request) {
	if c.Token != "" {
		req.Header.Set("Authorization", "Bearer "+c.Token)
	}
}

// store writes body to a temporary file beside local, checks it against want
// and renames it into place. Every failure removes the temporary file.
func store(ctx context.Context, body io.Reader, local, want string) (err error) {
	if err := os.MkdirAll(filepath.Dir(local), 0o750); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(local), ".partial-*")
	if err != nil {
		return err
	}
	defer func() {
		if err != nil {
			_ = tmp.Close()
			_ = os.Remove(tmp.Name())
		}
	}()

	sum := sha256.New()
	n, err := io.Copy(io.MultiWriter(tmp, sum), body)
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return err
	}
	if len(want) == 40 {
		// A regular file's etag is its git blob id, which hashes a header
		// carrying the length -- only known now.
		sum = sha1.New() //nolint:gosec // see the import
		fmt.Fprintf(sum, "blob %d\x00", n)
		if err := rehash(tmp, sum); err != nil {
			return err
		}
	}
	if got := hex.EncodeToString(sum.Sum(nil)); got != want {
		return fmt.Errorf("%w: got %s, want %s", ErrHashMismatch, got, want)
	}

	if err := tmp.Sync(); err != nil {
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), local)
}

func rehash(f *os.File, h hash.Hash) error {
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return err
	}
	_, err := io.Copy(h, f)
	return err
}

// statusErr accepts a success or a redirect; u names the request in the error
// and must not carry credentials.
func statusErr(resp *http.Response, u string) error {
	switch {
	case resp.StatusCode == http.StatusNotFound:
		return fmt.Errorf("%w: %s", ErrNotFound, u)
	case resp.StatusCode >= 200 && resp.StatusCode < 400:
		return nil
	default:
		return fmt.Errorf("hub: %s %s", u, resp.Status)
	}
}

// noRedirects is hc, or the default client, with redirects left to the caller.
func noRedirects(hc *http.Client) *http.Client {
	if hc == nil {
		hc = http.DefaultClient
	}
	cp := *hc
	cp.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	return &cp
}

func splitRepo(repo string) (owner, name string, ok bool) {
	owner, name, ok = strings.Cut(repo, "/")
	return owner, name, ok && validName(owner) && validName(name)
}

func validName(s string) bool {
	return nameRE.MatchString(s) && s != "." && s != ".."
}

// validSegments reports whether s is a relative slash path with no empty, "."
// or ".." segment -- one that cannot climb out of the directory it joins.
func validSegments(s string) bool {
	if s == "" {
		return false
	}
	for seg := range strings.SplitSeq(s, "/") {
		if seg == "" || seg == "." || seg == ".." {
			return false
		}
	}
	return true
}
