package evalset

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"hash/fnv"
	"io"
	"math/rand/v2"
	"net/http"
	"net/url"
	"slices"
	"strconv"
	"time"
)

// maxPage caps one rows API response. A page is at most 100 short texts, so
// one that needs more than this is not the page that was asked for.
const maxPage = 8 << 20

// maxAttempts is how often one request is tried before Fetch gives up. With
// the backoff doubling from 5 s, the longest wait is 80 s.
const maxAttempts = 6

// SampleOffsets draws n distinct row offsets out of numRows, ascending. The
// stream is a PCG seeded with seed and the dataset's name, so two datasets of
// one size do not draw the same rows and the draw needs nothing but the
// manifest to repeat. With n >= numRows every row is drawn.
func SampleOffsets(name string, numRows, n int, seed uint64) []int {
	h := fnv.New64a()
	_, _ = h.Write([]byte(name))
	r := rand.New(rand.NewPCG(seed, h.Sum64())) // #nosec G404 -- a reproducible sample, not a secret
	offsets := r.Perm(numRows)[:min(n, numRows)]
	slices.Sort(offsets)
	return offsets
}

// Fetcher reads rows from the datasets-server rows API.
type Fetcher struct {
	HTTP     *http.Client
	Base     string        // RowsAPI, or a test server
	PageSize int           // rows per request; 0 is the API's maximum, 100
	Pace     time.Duration // the least time between two requests
	last     time.Time
}

// page is the part of a rows API response Fetch reads.
type page struct {
	Features []struct {
		Name string `json:"name"`
		Type struct {
			Kind  string   `json:"_type"` //nolint:tagliatelle // the API's own key
			Names []string `json:"names"`
		} `json:"type"`
	} `json:"features"`
	Rows         []pageRow `json:"rows"`
	NumRowsTotal int       `json:"num_rows_total"`
	Partial      bool      `json:"partial"`
}

// pageRow is one row of a page.
type pageRow struct {
	RowIdx    int             `json:"row_idx"`
	Row       json.RawMessage `json:"row"`
	Truncated []string        `json:"truncated_cells"`
}

// Fetch draws n rows of s with SampleOffsets(s.Name, rows, n, seed). It
// learns the row count from one request, then reads only the pages that hold
// a drawn row; a row a page truncates is read again on its own. It refuses a
// dataset whose label names are not s.ClassNames in order, an index the
// server reports as partial, a row count that moves, and a row that is still
// truncated or missing. The Entry it returns has everything but the SHA256,
// which belongs to the file the caller writes.
func (f *Fetcher) Fetch(ctx context.Context, s Spec, n int, seed uint64) (Entry, []Row, error) {
	numRows, err := f.probe(ctx, s)
	if err != nil {
		return Entry{}, nil, err
	}
	e := Entry{
		Name: s.Name, Dataset: s.Dataset, Config: s.Config, Split: s.Split,
		NumRows: numRows, File: s.Name + ".jsonl",
		Offsets: SampleOffsets(s.Name, numRows, n, seed),
	}
	rows, err := f.rows(ctx, s, numRows, e.Offsets)
	if err != nil {
		return Entry{}, nil, err
	}
	return e, rows, nil
}

// Refetch reads the rows a manifest entry records again, at its offsets,
// with Fetch's checks. The dataset must still have e.NumRows rows, or the
// offsets would name other ones.
func (f *Fetcher) Refetch(ctx context.Context, s Spec, e Entry) ([]Row, error) {
	if e.Dataset != s.Dataset || e.Config != s.Config || e.Split != s.Split {
		return nil, fmt.Errorf("evalset: %s: the manifest records %s %s/%s, the spec is %s %s/%s",
			s.Name, e.Dataset, e.Config, e.Split, s.Dataset, s.Config, s.Split)
	}
	numRows, err := f.probe(ctx, s)
	if err != nil {
		return nil, err
	}
	if numRows != e.NumRows {
		return nil, fmt.Errorf("evalset: %s: the row count moved from the manifest's %d to %d", s.Name, e.NumRows, numRows)
	}
	return f.rows(ctx, s, numRows, e.Offsets)
}

// probe learns s's row count and checks its features.
func (f *Fetcher) probe(ctx context.Context, s Spec) (int, error) {
	p, err := f.get(ctx, s, 0, 1)
	if err != nil {
		return 0, err
	}
	if err := checkFeatures(s, p); err != nil {
		return 0, err
	}
	return p.NumRowsTotal, nil
}

// rows reads the rows at offsets, ascending, page by page.
func (f *Fetcher) rows(ctx context.Context, s Spec, numRows int, offsets []int) ([]Row, error) {
	size := f.PageSize
	if size <= 0 {
		size = 100
	}
	rows := make([]Row, 0, len(offsets))
	var p *page
	for _, off := range offsets {
		if off < 0 || off >= numRows {
			return nil, fmt.Errorf("evalset: %s: offset %d outside %d rows", s.Name, off, numRows)
		}
		start := off - off%size
		if p == nil || len(p.Rows) == 0 || p.Rows[0].RowIdx != start {
			var err error
			if p, err = f.page(ctx, s, start, min(size, numRows-start), numRows); err != nil {
				return nil, err
			}
		}
		raw, err := f.row(ctx, s, p, off, numRows)
		if err != nil {
			return nil, err
		}
		r, err := s.Extract(off, raw)
		if err != nil {
			return nil, err
		}
		rows = append(rows, r)
	}
	return rows, nil
}

// page reads length rows from offset, holding the row count to numRows.
func (f *Fetcher) page(ctx context.Context, s Spec, offset, length, numRows int) (*page, error) {
	p, err := f.get(ctx, s, offset, length)
	if err != nil {
		return nil, err
	}
	if p.NumRowsTotal != numRows {
		return nil, fmt.Errorf("evalset: %s: the row count moved from %d to %d", s.Name, numRows, p.NumRowsTotal)
	}
	if p.Partial {
		return nil, fmt.Errorf("evalset: %s: the server's index went partial", s.Name)
	}
	return p, nil
}

// row returns row off out of p, reading it on its own if p truncated it.
func (f *Fetcher) row(ctx context.Context, s Spec, p *page, off, numRows int) (json.RawMessage, error) {
	for range 2 {
		i := slices.IndexFunc(p.Rows, func(r pageRow) bool { return r.RowIdx == off })
		if i < 0 {
			return nil, fmt.Errorf("evalset: %s: the page read for row %d has no row %d", s.Name, off, off)
		}
		if len(p.Rows[i].Truncated) == 0 {
			return p.Rows[i].Row, nil
		}
		single, err := f.page(ctx, s, off, 1, numRows)
		if err != nil {
			return nil, err
		}
		p = single
	}
	return nil, fmt.Errorf("evalset: %s row %d: truncated even on its own", s.Name, off)
}

func checkFeatures(s Spec, p *page) error {
	if p.Partial {
		return fmt.Errorf("evalset: %s: the server's index is partial, so offsets are not the dataset's", s.Name)
	}
	if p.NumRowsTotal <= 0 {
		return fmt.Errorf("evalset: %s: the server reports %d rows", s.Name, p.NumRowsTotal)
	}
	for _, feat := range p.Features {
		if feat.Name != s.LabelField {
			continue
		}
		if feat.Type.Kind != "ClassLabel" || !slices.Equal(feat.Type.Names, s.ClassNames) {
			return fmt.Errorf("evalset: %s: label class names are %q (%s), want %q",
				s.Name, feat.Type.Names, feat.Type.Kind, s.ClassNames)
		}
		return nil
	}
	return fmt.Errorf("evalset: %s: no feature %q", s.Name, s.LabelField)
}

// get fetches length rows from offset, retrying a 429, a 5xx or a failed
// connection, at most one request per Pace.
func (f *Fetcher) get(ctx context.Context, s Spec, offset, length int) (*page, error) {
	q := url.Values{
		"dataset": {s.Dataset}, "config": {s.Config}, "split": {s.Split},
		"offset": {strconv.Itoa(offset)}, "length": {strconv.Itoa(length)},
	}
	u := f.Base + "?" + q.Encode()
	var lastErr error
	for attempt := range maxAttempts {
		if err := sleep(ctx, time.Until(f.last.Add(f.Pace))); err != nil {
			return nil, err
		}
		f.last = time.Now()
		p, r, err := f.try(ctx, u)
		if err == nil {
			return p, nil
		}
		lastErr = err
		if !r.again || attempt == maxAttempts-1 {
			break
		}
		wait := r.after
		if wait < 0 {
			wait = time.Duration(5<<attempt) * time.Second
		}
		if err := sleep(ctx, wait); err != nil {
			return nil, err
		}
	}
	return nil, fmt.Errorf("evalset: %s rows %d+%d: %w", s.Name, offset, length, lastErr)
}

func sleep(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return ctx.Err()
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// retry says whether a failed request is worth repeating, and after how long
// the server asked for (after < 0: it did not say).
type retry struct {
	again bool
	after time.Duration
}

var (
	final   = retry{}
	backoff = retry{again: true, after: -1}
)

// try is one request.
func (f *Fetcher) try(ctx context.Context, u string) (*page, retry, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, final, err
	}
	resp, err := f.HTTP.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return nil, final, err
		}
		return nil, backoff, err
	}
	defer func() { _ = resp.Body.Close() }()
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxPage+1))
	if err != nil {
		return nil, backoff, err
	}
	switch {
	case resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500:
		r := backoff
		if secs, err := strconv.Atoi(resp.Header.Get("Retry-After")); err == nil {
			r.after = time.Duration(min(max(secs, 0), 60)) * time.Second
		}
		return nil, r, fmt.Errorf("HTTP %s", resp.Status)
	case resp.StatusCode != http.StatusOK:
		return nil, final, fmt.Errorf("HTTP %s: %.200s", resp.Status, body)
	case len(body) > maxPage:
		return nil, final, errors.New("response larger than 8 MiB")
	}
	var p page
	if err := json.Unmarshal(body, &p); err != nil {
		return nil, final, err
	}
	return &p, final, nil
}
