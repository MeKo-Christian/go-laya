// Package evalset holds the public-dataset evaluation behind PLAN.md B.5.3:
// fixed, seeded subsets of four Hugging Face datasets, the question each row
// becomes, and the calibration metrics that compare an int8 export against
// fp32 on them.
//
// The subsets themselves are not checked in. cmd/laya-evalfetch draws them
// through the datasets-server rows API into build/eval/ and writes a committed
// manifest (docs/benchmarks/eval-manifest.json): dataset, split, row offsets,
// seed, fetch date and each file's sha256. ReadSubset verifies the sha before
// it parses a byte, so a result is only ever computed on the data the manifest
// names.
package evalset

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
)

// RowsAPI is the datasets-server endpoint the subsets are fetched from.
const RowsAPI = "https://datasets-server.huggingface.co/rows"

// ErrChecksum is a subset file whose sha256 is not the manifest's.
var ErrChecksum = errors.New("evalset: subset does not match the manifest's sha256")

// Manifest records how every subset was drawn and what it hashes to.
type Manifest struct {
	Source   string  `json:"source"`
	Seed     uint64  `json:"seed"`
	Fetched  string  `json:"fetched"`
	Datasets []Entry `json:"datasets"`
}

// Entry is one subset: where its rows came from and the file holding them.
type Entry struct {
	Name    string `json:"name"`
	Dataset string `json:"dataset"`
	Config  string `json:"config"`
	Split   string `json:"split"`
	NumRows int    `json:"num_rows"`
	Offsets []int  `json:"offsets"`
	File    string `json:"file"`
	SHA256  string `json:"sha256"`
}

// Row is one dataset row as a subset file stores it, one JSON object a line.
type Row struct {
	Offset int    `json:"offset"`
	Text   string `json:"text"`
	Label  int    `json:"label"`
}

// Entry returns the subset called name.
func (m *Manifest) Entry(name string) (Entry, bool) {
	i := slices.IndexFunc(m.Datasets, func(e Entry) bool { return e.Name == name })
	if i < 0 {
		return Entry{}, false
	}
	return m.Datasets[i], true
}

// LoadManifest reads a manifest, refusing fields it does not know.
func LoadManifest(path string) (*Manifest, error) {
	// #nosec G304 -- a developer-supplied path to the committed manifest.
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("evalset: manifest: %w", err)
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	var m Manifest
	if err := dec.Decode(&m); err != nil {
		return nil, fmt.Errorf("evalset: manifest %s: %w", path, err)
	}
	return &m, nil
}

// WriteManifest writes m as indented JSON.
func WriteManifest(path string, m Manifest) error {
	b, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return fmt.Errorf("evalset: manifest: %w", err)
	}
	if err := os.WriteFile(path, append(b, '\n'), 0o600); err != nil {
		return fmt.Errorf("evalset: manifest: %w", err)
	}
	return nil
}

// WriteSubset writes rows to path, one JSON object a line, and returns the
// file's sha256 in hex.
func WriteSubset(path string, rows []Row) (string, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	for _, r := range rows {
		if err := enc.Encode(r); err != nil {
			return "", fmt.Errorf("evalset: subset row %d: %w", r.Offset, err)
		}
	}
	if err := os.WriteFile(path, buf.Bytes(), 0o600); err != nil {
		return "", fmt.Errorf("evalset: subset: %w", err)
	}
	sum := sha256.Sum256(buf.Bytes())
	return hex.EncodeToString(sum[:]), nil
}

// ReadSubset reads e's file from dir. The file's sha256 must be e.SHA256
// before anything is parsed (ErrChecksum otherwise), and its rows must be
// e.Offsets, in order.
func ReadSubset(dir string, e Entry) ([]Row, error) {
	if e.File == "" || filepath.Base(e.File) != e.File || e.File == "." || e.File == ".." {
		return nil, fmt.Errorf("evalset: %s: file %q is not a plain name inside the subset directory", e.Name, e.File)
	}
	if e.SHA256 == "" {
		return nil, fmt.Errorf("evalset: %s: the manifest records no sha256", e.Name)
	}
	path := filepath.Join(dir, e.File)
	// #nosec G304 G703 -- e.File is checked above to be a plain name in dir.
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("evalset: %s: %w", e.Name, err)
	}
	sum := sha256.Sum256(b)
	if got := hex.EncodeToString(sum[:]); got != e.SHA256 {
		return nil, fmt.Errorf("%w: %s is %s, the manifest says %s", ErrChecksum, path, got, e.SHA256)
	}

	var rows []Row
	sc := bufio.NewScanner(bytes.NewReader(b))
	sc.Buffer(nil, len(b)+1)
	for sc.Scan() {
		dec := json.NewDecoder(bytes.NewReader(sc.Bytes()))
		dec.DisallowUnknownFields()
		var r Row
		if err := dec.Decode(&r); err != nil {
			return nil, fmt.Errorf("evalset: %s line %d: %w", path, len(rows)+1, err)
		}
		rows = append(rows, r)
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("evalset: %s: %w", path, err)
	}
	offsets := make([]int, len(rows))
	for i, r := range rows {
		offsets[i] = r.Offset
	}
	if !slices.Equal(offsets, e.Offsets) {
		return nil, fmt.Errorf("evalset: %s holds rows %v, the manifest says %v", path, offsets, e.Offsets)
	}
	return rows, nil
}
