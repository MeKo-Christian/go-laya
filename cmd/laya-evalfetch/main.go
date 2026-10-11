// Command laya-evalfetch fetches the fixed, seeded evaluation subsets of PLAN.md
// B.5.3 from the Hugging Face datasets-server rows API: about 200 rows each of
// AG News, DAIR Emotion, SST-2 and Yelp reviews.
//
//	go run ./cmd/laya-evalfetch -out build/eval
//
// The subsets go to -out, one JSONL file per dataset, and are not checked in.
// The manifest is (-manifest, docs/benchmarks/eval-manifest.json): it records
// each dataset's id, config, split, row count and row offsets, the seed, the
// fetch date and each file's sha256, and the int8 calibration harness
// (internal/evalset) refuses a subset whose sha differs.
//
// With a manifest present, the tool reads the rows it records again, at its
// offsets, and fails if a file's sha256 differs: the rows on the Hub changed,
// and the committed numbers no longer describe them. It never rewrites the
// manifest then. -update draws a new sample with -seed and -n and records it,
// which makes a reviewed diff of the manifest.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"time"

	"github.com/MeKo-Christian/go-laya/internal/evalset"
)

// options are the command's flags.
type options struct {
	out, manifest string
	update        bool
	n             int
	seed          uint64
}

// parseFlags reads the command line. -n must be at least 1: a negative
// sample used to panic in evalset.SampleOffsets, and zero rows is no
// evaluation set.
func parseFlags(args []string, stderr io.Writer) (options, error) {
	var o options
	fs := flag.NewFlagSet("laya-evalfetch", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&o.out, "out", "build/eval", "directory the subsets are written to")
	fs.StringVar(&o.manifest, "manifest", "docs/benchmarks/eval-manifest.json", "the manifest")
	fs.BoolVar(&o.update, "update", false, "draw a new sample and rewrite the manifest")
	fs.IntVar(&o.n, "n", 200, "rows per dataset, with -update (at least 1)")
	fs.Uint64Var(&o.seed, "seed", 20261011, "the sampling seed, with -update")
	if err := fs.Parse(args); err != nil {
		return o, fmt.Errorf("flags: %w", err)
	}
	switch {
	case fs.NArg() != 0:
		err := fmt.Errorf("unexpected arguments %q", fs.Args())
		fmt.Fprintln(stderr, err)
		fs.Usage()
		return o, err
	case o.n < 1:
		err := fmt.Errorf("-n %d: draw at least one row per dataset", o.n)
		fmt.Fprintln(stderr, err)
		fs.Usage()
		return o, err
	}
	return o, nil
}

func main() {
	o, err := parseFlags(os.Args[1:], os.Stderr)
	if errors.Is(err, flag.ErrHelp) {
		os.Exit(0)
	}
	if err != nil {
		os.Exit(2)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	// One request a second: the server answers a burst with 429s and does not
	// say how long to wait.
	f := &evalset.Fetcher{HTTP: &http.Client{Timeout: time.Minute}, Base: evalset.RowsAPI, Pace: time.Second}
	err = os.MkdirAll(o.out, 0o750)
	if err == nil {
		if _, statErr := os.Stat(o.manifest); statErr == nil && !o.update {
			err = verify(ctx, f, o.out, o.manifest)
		} else {
			err = draw(ctx, f, o.out, o.manifest, o.n, o.seed)
		}
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "laya-evalfetch:", err)
		stop()
		os.Exit(1) //nolint:gocritic // stop() ran above; nothing else is deferred
	}
}

// verify fetches the rows the manifest records and holds each file to its
// sha256.
func verify(ctx context.Context, f *evalset.Fetcher, out, manifest string) error {
	m, err := evalset.LoadManifest(manifest)
	if err != nil {
		return fmt.Errorf("manifest: %w", err)
	}
	for _, s := range evalset.Specs {
		e, ok := m.Entry(s.Name)
		if !ok {
			return fmt.Errorf("%s records no %s; -update draws every dataset again", manifest, s.Name)
		}
		path, err := e.Path(out)
		if err != nil {
			return fmt.Errorf("manifest: %w", err)
		}
		rows, err := f.Refetch(ctx, s, e)
		if err != nil {
			return fmt.Errorf("fetch: %w", err)
		}
		sum, err := evalset.WriteSubset(path, rows)
		if err != nil {
			return fmt.Errorf("write: %w", err)
		}
		if sum != e.SHA256 {
			return fmt.Errorf("%s: the rows changed on the Hub: sha256 %s, the manifest records %s "+
				"(-update draws and records a new sample)", s.Name, sum, e.SHA256)
		}
		fmt.Fprintf(os.Stderr, "%s: %d rows match the manifest, sha256 %s\n", s.Name, len(rows), sum)
	}
	return nil
}

// draw samples every dataset anew and writes the manifest.
func draw(ctx context.Context, f *evalset.Fetcher, out, manifest string, n int, seed uint64) error {
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
		path, err := e.Path(out)
		if err != nil {
			return fmt.Errorf("manifest: %w", err)
		}
		if e.SHA256, err = evalset.WriteSubset(path, rows); err != nil {
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
