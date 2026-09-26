// Command laya-ort downloads the pinned ONNX Runtime shared library into the
// laya cache, verifies it, and prints its path (PLAN.md Task 6.6.1).
//
//	go run github.com/MeKo-Christian/go-laya/cmd/laya-ort
//
// The cache is $LAYA_CACHE, else the user cache directory + /laya, and that is
// the only way to choose it: the ONNX backend looks in the same place, so it
// finds the library on its own once it is downloaded. LAYA_ORT_LIB and
// ORT_LIBRARY_PATH still take precedence. Running it again only re-verifies
// the cached copy.
package main

import (
	"context"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"time"

	"github.com/MeKo-Christian/go-laya/internal/ortlib"
)

func main() {
	flag.Usage = func() {
		fmt.Fprintln(flag.CommandLine.Output(), "usage: laya-ort\n\n"+
			"Downloads the pinned ONNX Runtime into $LAYA_CACHE (else the user cache dir + /laya),\n"+
			"where the ONNX backend finds it, and prints its path.")
	}
	flag.Parse()
	if flag.NArg() != 0 {
		flag.Usage()
		os.Exit(2)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	lib, err := download(ctx)
	if err != nil {
		fmt.Fprintln(os.Stderr, "laya-ort:", err)
		stop()
		os.Exit(1) //nolint:gocritic // stop() ran above; nothing else is deferred
	}
	fmt.Fprintln(os.Stdout, lib)
}

func download(ctx context.Context) (string, error) {
	rel, err := ortlib.Pinned(runtime.GOOS, runtime.GOARCH)
	if err != nil {
		return "", fmt.Errorf("pinned release: %w", err)
	}
	c := &ortlib.Client{HTTP: &http.Client{Timeout: 10 * time.Minute}}
	lib, err := c.Download(ctx, rel)
	if err != nil {
		return "", fmt.Errorf("download %s: %w", rel.Archive, err)
	}
	return lib, nil
}
