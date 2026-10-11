// Command laya-evalfetch draws the fixed, seeded evaluation subsets of PLAN.md
// B.5.3 from the Hugging Face datasets-server rows API: about 200 rows each of
// AG News, DAIR Emotion, SST-2 and Yelp reviews.
//
//	go run ./cmd/laya-evalfetch -out build/eval -manifest docs/benchmarks/eval-manifest.json
//
// The subsets go to -out, one JSONL file per dataset, and are not checked in.
// The manifest is: it records each dataset's id, config, split, row offsets,
// the seed, the fetch date and each file's sha256, and the int8 calibration
// harness (internal/evalset) refuses a subset whose sha differs. The offsets
// follow from the seed and the dataset's row count alone, so a rerun with the
// same seed draws the same rows; the sha then says whether the rows still hold
// the same text and labels.
package main

import (
	"context"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"time"

	"github.com/MeKo-Christian/go-laya/internal/evalset"
)

func main() {
	out := flag.String("out", "build/eval", "directory the subsets are written to")
	manifest := flag.String("manifest", "docs/benchmarks/eval-manifest.json", "the manifest to write")
	n := flag.Int("n", 200, "rows per dataset")
	seed := flag.Uint64("seed", 20261011, "the sampling seed")
	flag.Parse()
	if flag.NArg() != 0 {
		flag.Usage()
		os.Exit(2)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	if err := run(ctx, *out, *manifest, *n, *seed); err != nil {
		fmt.Fprintln(os.Stderr, "laya-evalfetch:", err)
		stop()
		os.Exit(1) //nolint:gocritic // stop() ran above; nothing else is deferred
	}
}

func run(ctx context.Context, out, manifest string, n int, seed uint64) error {
	if err := os.MkdirAll(out, 0o750); err != nil {
		return fmt.Errorf("subset directory: %w", err)
	}
	// One request a second: the server answers a burst with 429s and does not
	// say how long to wait.
	f := &evalset.Fetcher{HTTP: &http.Client{Timeout: time.Minute}, Base: evalset.RowsAPI, Pace: time.Second}
	m := evalset.Manifest{
		Source:  evalset.RowsAPI,
		Seed:    seed,
		Fetched: time.Now().UTC().Format(time.DateOnly),
	}
	for _, s := range evalset.Specs {
		e, rows, err := f.Fetch(ctx, s, n, seed)
		if err != nil {
			return fmt.Errorf("fetch: %w", err)
		}
		if e.SHA256, err = evalset.WriteSubset(filepath.Join(out, e.File), rows); err != nil {
			return fmt.Errorf("write: %w", err)
		}
		fmt.Fprintf(os.Stderr, "%s: %d of %d rows of %s %s/%s, sha256 %s\n",
			s.Name, len(rows), e.NumRows, s.Dataset, s.Config, s.Split, e.SHA256)
		m.Datasets = append(m.Datasets, e)
	}
	if err := evalset.WriteManifest(manifest, m); err != nil {
		return fmt.Errorf("write: %w", err)
	}
	return nil
}
