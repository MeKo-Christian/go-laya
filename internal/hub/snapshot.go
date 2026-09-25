package hub

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/url"
	"os"
	pathpkg "path"
	"path/filepath"
	"regexp"
	"strings"
)

// maxListing bounds the listing Snapshot reads: the bundle repo's is 7 KB.
const maxListing = 16 << 20

// listing is the part of GET /api/models/{repo}/revision/{rev} Snapshot uses.
type listing struct {
	SHA      string `json:"sha"`
	Siblings []struct {
		Name string `json:"rfilename"`
	} `json:"siblings"`
}

// Snapshot downloads every file of repo at rev that matches one of allow into
// Dir/owner/name/<commit>/ and returns that directory; an empty allow takes
// every file. It is snapshot_download(repo, allow_patterns=allow), which
// agent.py:122-128 calls with f"{subfolder}/*": the patterns are fnmatch's, so
// "*" crosses "/", and one ending in "/" matches everything beneath it.
//
// Every file is fetched at the commit the listing names, not at rev, and must
// be served from that commit. Only a complete snapshot records rev's commit
// and the listing, which is what lets an Offline client resolve it later.
func (c *Client) Snapshot(ctx context.Context, repo, rev string, allow []string) (string, error) {
	owner, name, ok := splitRepo(repo)
	if !ok || !validPath(rev) {
		return "", fmt.Errorf("%w: repo %q, revision %q", ErrInvalidPath, repo, rev)
	}
	dir, err := c.dir()
	if err != nil {
		return "", err
	}
	base := filepath.Join(dir, owner, name)
	if c.Offline {
		return offlineSnapshot(dir, owner, name, rev, allow)
	}

	commit, names, err := c.list(ctx, owner, name, rev)
	if err != nil {
		return "", err
	}
	want, err := filter(repo, names, allow)
	if err != nil {
		return "", err
	}
	for _, path := range want {
		if _, err := c.fetch(ctx, dir, owner, name, commit, path, commit); err != nil {
			return "", err
		}
	}

	raw, err := json.Marshal(names)
	if err != nil {
		return "", fmt.Errorf("hub: %w", err)
	}
	if err := writeAtomic(dir, repo+"/listings/"+commit+".json", raw); err != nil {
		return "", fmt.Errorf("hub: %s: %w", repo, err)
	}
	if rev != commit {
		if err := writeAtomic(dir, repo+"/refs/"+rev, []byte(commit)); err != nil {
			return "", fmt.Errorf("hub: %s: %w", repo, err)
		}
	}
	return filepath.Join(base, commit), nil
}

// list returns the commit rev resolves to and every file the Hub lists there.
// The names are remote input that becomes cache paths, so one that could
// climb out of the cache fails the whole listing.
func (c *Client) list(ctx context.Context, owner, name, rev string) (string, []string, error) {
	u, err := c.url("api/models/" + owner + "/" + name + "/revision/" + url.PathEscape(rev))
	if err != nil {
		return "", nil, err
	}
	body, err := c.get(ctx, noRedirects(c.HTTP), u, u.Host)
	if err != nil {
		return "", nil, err
	}
	defer body.Close()
	raw, err := io.ReadAll(io.LimitReader(body, maxListing+1))
	if err != nil {
		return "", nil, fmt.Errorf("hub: %s: %w", u.Redacted(), err)
	}
	if len(raw) > maxListing {
		return "", nil, fmt.Errorf("hub: %s: listing larger than %d bytes", u.Redacted(), maxListing)
	}
	var l listing
	if err := json.Unmarshal(raw, &l); err != nil {
		return "", nil, fmt.Errorf("hub: %s: %w", u.Redacted(), err)
	}
	if !commitRE.MatchString(l.SHA) {
		return "", nil, fmt.Errorf("%w: %s: commit id %q", ErrInvalidPath, u.Redacted(), l.SHA)
	}
	names := make([]string, len(l.Siblings))
	for i, s := range l.Siblings {
		names[i] = s.Name
	}
	if err := checkNames(names); err != nil {
		return "", nil, fmt.Errorf("%s: %w", u.Redacted(), err)
	}
	return l.SHA, names, nil
}

// offlineSnapshot answers Snapshot from the cache alone: rev's recorded
// commit, that commit's recorded listing, and every matching file present as
// a regular file.
func offlineSnapshot(dir, owner, name, rev string, allow []string) (string, error) {
	repo, base := owner+"/"+name, filepath.Join(dir, owner, name)
	commit, err := cachedCommit(dir, owner, name, rev)
	if err != nil {
		return "", err
	}
	raw, err := readCached(dir, repo+"/listings/"+commit+".json")
	if errors.Is(err, fs.ErrNotExist) {
		return "", fmt.Errorf("%w: %s at %s: no listing", ErrNotCached, repo, rev)
	}
	if err != nil {
		return "", fmt.Errorf("hub: %s: %w", repo, err)
	}
	var names []string
	if err := json.Unmarshal(raw, &names); err != nil {
		return "", fmt.Errorf("hub: %s: cached listing: %w", repo, err)
	}
	if err := checkNames(names); err != nil {
		return "", fmt.Errorf("hub: %s: cached listing: %w", repo, err)
	}
	want, err := filter(repo, names, allow)
	if err != nil {
		return "", err
	}
	for _, path := range want {
		if !cached(dir, repo+"/"+commit+"/"+path) {
			return "", fmt.Errorf("%w: %s %s at %s", ErrNotCached, repo, path, rev)
		}
	}
	return filepath.Join(base, commit), nil
}

// cachedCommit is the commit an earlier Snapshot resolved rev to; a commit id
// is its own.
func cachedCommit(dir, owner, name, rev string) (string, error) {
	if commitRE.MatchString(rev) {
		return rev, nil
	}
	raw, err := readCached(dir, owner+"/"+name+"/refs/"+rev)
	if errors.Is(err, fs.ErrNotExist) {
		return "", fmt.Errorf("%w: %s/%s at %s", ErrNotCached, owner, name, rev)
	}
	if err != nil {
		return "", fmt.Errorf("hub: %s/%s: %w", owner, name, err)
	}
	commit := strings.TrimSpace(string(raw))
	if !commitRE.MatchString(commit) {
		return "", fmt.Errorf("%w: %s/%s ref %s: commit id %q", ErrInvalidPath, owner, name, rev, commit)
	}
	return commit, nil
}

func checkNames(names []string) error {
	for _, n := range names {
		if !validPath(n) || strings.HasPrefix(n, "/") {
			return fmt.Errorf("%w: listed file %q", ErrInvalidPath, n)
		}
	}
	return nil
}

// filter keeps the names matching any of allow, all of them when allow is
// empty, as huggingface_hub's filter_repo_objects does. Matching nothing is
// ErrNotFound: upstream downloads nothing and fails later on the missing
// subfolder.
func filter(repo string, names, allow []string) ([]string, error) {
	if len(allow) == 0 {
		if len(names) == 0 {
			return nil, fmt.Errorf("%w: %s lists no files", ErrNotFound, repo)
		}
		return names, nil
	}
	var out []string
	for _, n := range names {
		for _, p := range allow {
			if strings.HasSuffix(p, "/") {
				p += "*"
			}
			if match(p, n) {
				out = append(out, n)
				break
			}
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("%w: no file of %s matches %q", ErrNotFound, repo, allow)
	}
	return out, nil
}

// match is Python's fnmatch.fnmatchcase: "*" matches any run, "/" included,
// "?" any one character, "[seq]" and "[!seq]" a set; an unclosed "[" is
// literal and there is no escape character.
func match(pattern, name string) bool {
	re, err := regexp.Compile(translate(pattern))
	return err == nil && re.MatchString(name)
}

// translate follows fnmatch.translate.
func translate(pat string) string {
	var b strings.Builder
	b.WriteString(`(?s)^`)
	for i := 0; i < len(pat); {
		c := pat[i]
		i++
		switch c {
		case '*':
			for i < len(pat) && pat[i] == '*' {
				i++
			}
			b.WriteString(`.*`)
		case '?':
			b.WriteString(`.`)
		case '[':
			j := i
			if j < len(pat) && pat[j] == '!' {
				j++
			}
			if j < len(pat) && pat[j] == ']' {
				j++
			}
			for j < len(pat) && pat[j] != ']' {
				j++
			}
			if j >= len(pat) {
				b.WriteString(`\[`)
				continue
			}
			set := strings.ReplaceAll(pat[i:j], `\`, `\\`)
			i = j + 1
			switch {
			case strings.HasPrefix(set, "!"):
				set = "^" + set[1:]
			case strings.HasPrefix(set, "^"):
				set = `\` + set
			}
			b.WriteString("[" + set + "]")
		default:
			b.WriteString(regexp.QuoteMeta(string(c)))
		}
	}
	b.WriteString(`$`)
	return b.String()
}

// readCached reads rel, a slash path below the cache root, only if cached
// accepts it; anything else is fs.ErrNotExist.
func readCached(root, rel string) ([]byte, error) {
	if !cached(root, rel) {
		return nil, fs.ErrNotExist
	}
	return os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
}

// writeAtomic writes data to rel, a slash path below the cache root, by
// rename, so a reader sees all of it or nothing.
func writeAtomic(root, rel string, data []byte) (err error) {
	if err := cacheDir(root, pathpkg.Dir(rel)); err != nil {
		return err
	}
	path := filepath.Join(root, filepath.FromSlash(rel))
	tmp, err := os.CreateTemp(filepath.Dir(path), ".partial-*")
	if err != nil {
		return err
	}
	defer func() {
		if err != nil {
			_ = tmp.Close()
			_ = os.Remove(tmp.Name())
		}
	}()
	if _, err := tmp.Write(data); err != nil {
		return err
	}
	if err := tmp.Sync(); err != nil {
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path)
}
