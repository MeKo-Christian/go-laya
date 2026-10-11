package evalset

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
)

func TestSampleOffsets(t *testing.T) {
	a := SampleOffsets("sst2", 872, 200, 20261011)
	if len(a) != 200 || !slices.IsSorted(a) || a[0] < 0 || a[len(a)-1] >= 872 {
		t.Fatalf("SampleOffsets = %d offsets, sorted %v, range [%d, %d]", len(a), slices.IsSorted(a), a[0], a[len(a)-1])
	}
	if len(slices.Compact(slices.Clone(a))) != len(a) {
		t.Error("SampleOffsets repeated an offset")
	}
	if !slices.Equal(a, SampleOffsets("sst2", 872, 200, 20261011)) {
		t.Error("SampleOffsets is not deterministic")
	}
	if slices.Equal(a, SampleOffsets("sst2", 872, 200, 1)) {
		t.Error("the seed does not change the sample")
	}
	if slices.Equal(a, SampleOffsets("emotion", 872, 200, 20261011)) {
		t.Error("two datasets of one size draw the same offsets")
	}
	if got := SampleOffsets("sst2", 5, 200, 1); !slices.Equal(got, []int{0, 1, 2, 3, 4}) {
		t.Errorf("n > rows: %v, want every row", got)
	}
}

// rowsServer serves a fake rows API over the 10 rows "row 0" .. "row 9",
// labelled i % 2, and counts the requests. mutate edits each response.
func rowsServer(t *testing.T, names []string, mutate func(offset, length int, page map[string]any)) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		q := r.URL.Query()
		off, err1 := strconv.Atoi(q.Get("offset"))
		length, err2 := strconv.Atoi(q.Get("length"))
		if q.Get("dataset") != "stanfordnlp/sst2" || q.Get("config") != "default" ||
			q.Get("split") != "validation" || err1 != nil || err2 != nil || length < 1 || length > 100 {
			http.Error(w, "bad query "+r.URL.RawQuery, http.StatusBadRequest)
			return
		}
		var rows []any
		for i := off; i < min(off+length, 10); i++ {
			rows = append(rows, map[string]any{
				"row_idx":         i,
				"row":             map[string]any{"idx": i, "sentence": fmt.Sprintf("row %d", i), "label": i % 2},
				"truncated_cells": []string{},
			})
		}
		page := map[string]any{
			"features": []any{
				map[string]any{"name": "sentence", "type": map[string]any{"_type": "Value"}},
				map[string]any{"name": "label", "type": map[string]any{"_type": "ClassLabel", "names": names}},
			},
			"rows":           rows,
			"num_rows_total": 10,
			"partial":        false,
		}
		if mutate != nil {
			mutate(off, length, page)
		}
		_ = json.NewEncoder(w).Encode(page)
	}))
	t.Cleanup(srv.Close)
	return srv, &calls
}

func TestFetch(t *testing.T) {
	srv, calls := rowsServer(t, []string{"negative", "positive"}, nil)
	f := &Fetcher{HTTP: srv.Client(), Base: srv.URL, PageSize: 3}
	spec, _ := SpecByName("sst2")

	e, rows, err := f.Fetch(context.Background(), spec, 4, 3)
	if err != nil {
		t.Fatalf("Fetch: %v", err)
	}
	want := SampleOffsets("sst2", 10, 4, 3)
	if e.NumRows != 10 || !slices.Equal(e.Offsets, want) || e.Dataset != "stanfordnlp/sst2" ||
		e.Config != "default" || e.Split != "validation" || e.Name != "sst2" || e.File != "sst2.jsonl" {
		t.Errorf("entry = %+v, want offsets %v of 10", e, want)
	}
	if len(rows) != len(want) {
		t.Fatalf("%d rows, want %d", len(rows), len(want))
	}
	for i, r := range rows {
		if r != (Row{Offset: want[i], Text: fmt.Sprintf("row %d", want[i]), Label: want[i] % 2}) {
			t.Errorf("row %d = %+v", i, r)
		}
	}
	pages := map[int]bool{}
	for _, off := range want {
		pages[off/3] = true
	}
	if n := int(calls.Load()); n != 1+len(pages) {
		t.Errorf("%d requests, want 1 probe + %d pages", n, len(pages))
	}
}

// A row the page truncates is fetched again on its own.
func TestFetchRefetchesTruncated(t *testing.T) {
	srv, _ := rowsServer(t, []string{"negative", "positive"}, func(_, length int, p map[string]any) {
		if length > 1 {
			for _, r := range p["rows"].([]any) {
				r.(map[string]any)["truncated_cells"] = []string{"sentence"}
				r.(map[string]any)["row"].(map[string]any)["sentence"] = "ro"
			}
		}
	})
	f := &Fetcher{HTTP: srv.Client(), Base: srv.URL, PageSize: 5}
	spec, _ := SpecByName("sst2")
	_, rows, err := f.Fetch(context.Background(), spec, 3, 3)
	if err != nil {
		t.Fatalf("Fetch: %v", err)
	}
	for _, r := range rows {
		if r.Text != fmt.Sprintf("row %d", r.Offset) {
			t.Errorf("row %d = %q, the truncated text", r.Offset, r.Text)
		}
	}
}

func TestFetchRefuses(t *testing.T) {
	tests := []struct {
		name   string
		names  []string
		mutate func(int, int, map[string]any)
		want   string
	}{
		{"reordered class names", []string{"positive", "negative"}, nil, "class names"},
		{"partial index", []string{"negative", "positive"}, func(_, _ int, p map[string]any) { p["partial"] = true }, "partial"},
		{"truncated text", []string{"negative", "positive"}, func(_, _ int, p map[string]any) {
			for _, r := range p["rows"].([]any) {
				r.(map[string]any)["truncated_cells"] = []string{"sentence"}
			}
		}, "truncated"},
		{"missing row", []string{"negative", "positive"}, func(off, _ int, p map[string]any) {
			if off > 0 {
				p["rows"] = []any{}
			}
		}, "no row"},
		{"row count moves", []string{"negative", "positive"}, func(off, _ int, p map[string]any) {
			if off > 0 {
				p["num_rows_total"] = 11
			}
		}, "row count"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv, _ := rowsServer(t, tt.names, tt.mutate)
			f := &Fetcher{HTTP: srv.Client(), Base: srv.URL, PageSize: 3}
			spec, _ := SpecByName("sst2")
			_, _, err := f.Fetch(context.Background(), spec, 4, 3)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Fetch: err = %v, want one mentioning %q", err, tt.want)
			}
		})
	}
}

func TestFetchRetries(t *testing.T) {
	var fails atomic.Int32
	fails.Store(2)
	inner, _ := rowsServer(t, []string{"negative", "positive"}, nil)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if fails.Add(-1) >= 0 {
			w.Header().Set("Retry-After", "0")
			http.Error(w, "slow down", http.StatusTooManyRequests)
			return
		}
		resp, err := http.Get(inner.URL + "?" + r.URL.RawQuery)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		defer resp.Body.Close()
		w.WriteHeader(resp.StatusCode)
		_, _ = io.Copy(w, resp.Body)
	}))
	t.Cleanup(srv.Close)

	f := &Fetcher{HTTP: srv.Client(), Base: srv.URL}
	spec, _ := SpecByName("sst2")
	if _, _, err := f.Fetch(context.Background(), spec, 2, 1); err != nil {
		t.Fatalf("Fetch after two 429s: %v", err)
	}
}
