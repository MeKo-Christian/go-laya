package evalset

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// subsetFixture is two rows as WriteSubset lays them out.
const subsetFixture = `{"offset":3,"text":"first row","label":1}
{"offset":17,"text":"second \"row\"","label":0}
`

// writeFixture puts subsetFixture into dir and returns the manifest entry that
// describes it.
func writeFixture(t *testing.T, dir string) Entry {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, "sst2.jsonl"), []byte(subsetFixture), 0o600); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256([]byte(subsetFixture))
	return Entry{
		Name: "sst2", Dataset: "stanfordnlp/sst2", Config: "default", Split: "validation",
		NumRows: 872, Offsets: []int{3, 17}, File: "sst2.jsonl", SHA256: hex.EncodeToString(sum[:]),
	}
}

func TestReadSubset(t *testing.T) {
	dir := t.TempDir()
	e := writeFixture(t, dir)

	rows, err := ReadSubset(dir, e)
	if err != nil {
		t.Fatalf("ReadSubset: %v", err)
	}
	want := []Row{{Offset: 3, Text: "first row", Label: 1}, {Offset: 17, Text: `second "row"`, Label: 0}}
	if !slices.Equal(rows, want) {
		t.Errorf("rows = %+v, want %+v", rows, want)
	}
}

// TestReadSubsetChecksSHA is the guard against evaluating on data nobody
// committed to: one changed byte is an error, before anything is parsed.
func TestReadSubsetChecksSHA(t *testing.T) {
	dir := t.TempDir()
	e := writeFixture(t, dir)
	tampered := []byte(subsetFixture)
	tampered[len(`{"offset":3,"text":"first row","label":`)] = '0' // label 1 -> 0
	if err := os.WriteFile(filepath.Join(dir, e.File), tampered, 0o600); err != nil {
		t.Fatal(err)
	}

	_, err := ReadSubset(dir, e)
	if !errors.Is(err, ErrChecksum) {
		t.Fatalf("ReadSubset on a tampered file: err = %v, want ErrChecksum", err)
	}
}

func TestReadSubsetRejects(t *testing.T) {
	tests := []struct {
		name string
		edit func(*Entry)
	}{
		{"file outside the directory", func(e *Entry) { e.File = "../sst2.jsonl" }},
		{"absolute file", func(e *Entry) { e.File = "/etc/passwd" }},
		{"offsets disagree", func(e *Entry) { e.Offsets = []int{3, 18} }},
		{"fewer offsets", func(e *Entry) { e.Offsets = []int{3} }},
		{"no sha", func(e *Entry) { e.SHA256 = "" }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			e := writeFixture(t, dir)
			tt.edit(&e)
			if rows, err := ReadSubset(dir, e); err == nil {
				t.Fatalf("ReadSubset = %+v, want an error", rows)
			}
		})
	}
}

func TestManifestRoundTrip(t *testing.T) {
	dir := t.TempDir()
	m := Manifest{
		Source: RowsAPI, Seed: 7, Fetched: "2026-10-11",
		Datasets: []Entry{writeFixture(t, dir)},
	}
	path := filepath.Join(dir, "manifest.json")
	if err := WriteManifest(path, m); err != nil {
		t.Fatalf("WriteManifest: %v", err)
	}
	got, err := LoadManifest(path)
	if err != nil {
		t.Fatalf("LoadManifest: %v", err)
	}
	if got.Seed != 7 || got.Fetched != "2026-10-11" || len(got.Datasets) != 1 ||
		got.Datasets[0].SHA256 != m.Datasets[0].SHA256 || !slices.Equal(got.Datasets[0].Offsets, []int{3, 17}) {
		t.Errorf("LoadManifest = %+v, want %+v", got, m)
	}
	e, ok := got.Entry("sst2")
	if !ok || e.File != "sst2.jsonl" {
		t.Errorf("Entry(sst2) = %+v, %v", e, ok)
	}
	if _, ok := got.Entry("nope"); ok {
		t.Error("Entry(nope) found something")
	}
}

func TestLoadManifestRejectsUnknownFields(t *testing.T) {
	path := filepath.Join(t.TempDir(), "manifest.json")
	if err := os.WriteFile(path, []byte(`{"seed":1,"sha":"x"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadManifest(path); err == nil {
		t.Fatal("LoadManifest accepted an unknown field")
	}
}

func TestWriteSubset(t *testing.T) {
	dir := t.TempDir()
	rows := []Row{{Offset: 3, Text: "first row", Label: 1}, {Offset: 17, Text: `second "row"`, Label: 0}}
	sum, err := WriteSubset(filepath.Join(dir, "sst2.jsonl"), rows)
	if err != nil {
		t.Fatalf("WriteSubset: %v", err)
	}
	b, err := os.ReadFile(filepath.Join(dir, "sst2.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if string(b) != subsetFixture {
		t.Errorf("file =\n%s\nwant\n%s", b, subsetFixture)
	}
	want := sha256.Sum256(b)
	if sum != hex.EncodeToString(want[:]) {
		t.Errorf("sha = %s, want %x", sum, want)
	}
}

// TestEntryPath is the one check every reader and writer of a subset file
// goes through: the manifest names a file inside the subset directory, never
// a path out of it.
func TestEntryPath(t *testing.T) {
	dir := t.TempDir()
	if got, err := (Entry{Name: "sst2", File: "sst2.jsonl"}).Path(dir); err != nil || got != filepath.Join(dir, "sst2.jsonl") {
		t.Errorf("Path(sst2.jsonl) = %q, %v", got, err)
	}
	for _, file := range []string{"", ".", "..", "../x", "a/b", "/etc/passwd"} {
		if got, err := (Entry{Name: "sst2", File: file}).Path(dir); err == nil {
			t.Errorf("Path(%q) = %q, want an error", file, got)
		}
	}
}

// TestLoadManifestRejectsPaths: a manifest naming a file outside the subset
// directory is refused when it is read, before laya-evalfetch fetches or
// writes anything for it.
func TestLoadManifestRejectsPaths(t *testing.T) {
	for _, file := range []string{"../x.jsonl", "a/b.jsonl", "/tmp/x.jsonl", ""} {
		t.Run(file, func(t *testing.T) {
			dir := t.TempDir()
			e := writeFixture(t, dir)
			e.File = file
			path := filepath.Join(dir, "manifest.json")
			if err := WriteManifest(path, Manifest{Source: RowsAPI, Seed: 1, Datasets: []Entry{e}}); err != nil {
				t.Fatal(err)
			}
			if m, err := LoadManifest(path); err == nil {
				t.Fatalf("LoadManifest = %+v, want an error for file %q", m, file)
			}
		})
	}
}
