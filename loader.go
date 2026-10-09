package laya

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"github.com/MeKo-Christian/go-laya/backend"
	"github.com/MeKo-Christian/go-laya/internal/backend/onnx"
	"github.com/MeKo-Christian/go-laya/internal/checkpoint"
	"github.com/MeKo-Christian/go-laya/internal/hub"
	"github.com/MeKo-Christian/go-laya/tokenizer"
)

// pinnedRevision is the bundle repo's commit every golden vector describes
// (D17). The Hub's main has moved past it, so following main would load a
// checkpoint nothing here was verified against.
const pinnedRevision = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"

// defaultLoader is what NewRouter loads with when no WithLoader is given: the
// Go side of router.py:174-178, which builds an Agent from a spec's repo and
// subfolder.
//
// It differs from upstream in where the weights come from (D24). The Hub
// checkpoints ship safetensors, which only torch can run, so the forward pass
// is a local ONNX export of the same checkpoint and only the config and the
// tokenizer are downloaded.
type defaultLoader struct {
	hub      *hub.Client
	onnxDir  string // "" is onnx/ under the laya cache, resolved per load
	device   string
	revision string       // "" is D17's pin for the bundle repo, main for any other
	cacheDir string       // "" is hub.DefaultDir(), resolved per load
	logger   *slog.Logger // nil is slog.Default()

	// snapshot and open are the network and the runtime. They are fields so
	// that tests can stand in for both; CI has neither.
	snapshot func(ctx context.Context, repo, rev string, allow []string) (string, error)
	open     func(path string, opts onnx.Options) (backend.Backend, error)
}

// loaderSettings configure the default loader. The Router's options set them
// for every checkpoint it loads, Open's for the one it opens.
type loaderSettings struct {
	onnxDir  string // "" is $LAYA_ONNX_DIR, else onnx/ under the laya cache
	device   string
	token    string       // "" is $HF_TOKEN
	revision string       // "" is D17's pin for the bundle repo, main for any other
	cacheDir string       // "" is hub.DefaultDir()
	logger   *slog.Logger // nil is slog.Default()
}

func newDefaultLoader(cfg loaderSettings) *defaultLoader {
	token := cfg.token
	if token == "" {
		token = os.Getenv("HF_TOKEN")
	}
	l := &defaultLoader{
		hub:      &hub.Client{Token: token, Dir: cfg.cacheDir, Offline: offlineFromEnv()},
		onnxDir:  cfg.onnxDir,
		device:   cfg.device,
		revision: cfg.revision,
		cacheDir: cfg.cacheDir,
		logger:   cfg.logger,
	}
	if l.onnxDir == "" {
		l.onnxDir = os.Getenv("LAYA_ONNX_DIR")
	}
	l.snapshot = l.hub.Snapshot
	l.open = func(path string, opts onnx.Options) (backend.Backend, error) {
		return onnx.Open(path, opts)
	}
	return l
}

// offlineFromEnv is huggingface_hub's HF_HUB_OFFLINE,
// `_is_true(HF_HUB_OFFLINE or TRANSFORMERS_OFFLINE)` (constants.py:194), with
// LAYA_OFFLINE as a third switch of this port's own. Offline, a snapshot is
// answered from the cache alone and fails with hub.ErrNotCached otherwise.
func offlineFromEnv() bool {
	hf := os.Getenv("HF_HUB_OFFLINE")
	if hf == "" {
		hf = os.Getenv("TRANSFORMERS_OFFLINE")
	}
	return envTrue(hf) || envTrue(os.Getenv("LAYA_OFFLINE"))
}

// envTrue is huggingface_hub's _is_true: ENV_VARS_TRUE_VALUES, compared
// upper-cased (constants.py:12-19).
func envTrue(v string) bool {
	switch strings.ToUpper(v) {
	case "1", "ON", "YES", "TRUE":
		return true
	}
	return false
}

// load builds the agent for the checkpoint the router calls name.
func (l *defaultLoader) load(ctx context.Context, name string, spec ModelSpec) (Predictor, error) {
	a, err := l.build(ctx, spec, func() (string, error) { return l.graph(name) })
	if err != nil {
		// Not a typed nil: the Router stores whatever comes back.
		return nil, err
	}
	return a, nil
}

// build loads the checkpoint spec locates and runs it on the export graph
// names. graph is asked for only once the checkpoint has been read, so a bad
// checkpoint is reported as such rather than as a missing export.
func (l *defaultLoader) build(ctx context.Context, spec ModelSpec, graph func() (string, error)) (*Agent, error) {
	cfg, tok, actWidth, err := l.readCheckpoint(ctx, spec)
	if err != nil {
		return nil, err
	}
	path, err := graph()
	if err != nil {
		return nil, err
	}
	be, err := l.open(path, onnx.Options{
		Device: l.device, ActWidth: actWidth, Logger: l.logger, CacheDir: l.cacheDir,
	})
	if err != nil {
		return nil, err
	}
	a, err := newAgent(be, cfg, tok)
	if err != nil {
		_ = be.Close()
		return nil, err
	}
	return a, nil
}

// buildOn loads the checkpoint spec locates and runs it on be, the caller's
// backend (WithBackend). Unlike build it never closes be: on failure the
// caller still owns it.
func (l *defaultLoader) buildOn(ctx context.Context, spec ModelSpec, be backend.Backend) (*Agent, error) {
	cfg, tok, actWidth, err := l.readCheckpoint(ctx, spec)
	if err != nil {
		return nil, err
	}
	return newAgent(shapeChecked{be, actWidth}, cfg, tok)
}

// shapeChecked holds a caller's backend to the output shapes the ONNX backend
// enforces on itself: per row, one logit per marker column and the config's
// act width. A wrong width would otherwise answer plausibly, an act
// probability of 1 from a row one wide, where it should fail.
type shapeChecked struct {
	backend.Backend
	actWidth int
}

func (b shapeChecked) Forward(ctx context.Context, in backend.Batch) (logits, act [][]float32, err error) {
	logits, act, err = b.Backend.Forward(ctx, in)
	if err != nil {
		return nil, nil, err
	}
	if len(logits) != len(in.MarkerPos) || len(act) != len(in.MarkerPos) {
		return nil, nil, fmt.Errorf("%w: %d logit rows and %d act rows for %d in the batch",
			ErrIncompatibleCheckpoint, len(logits), len(act), len(in.MarkerPos))
	}
	for r := range logits {
		if kmax := len(in.MarkerPos[r]); len(logits[r]) != kmax {
			return nil, nil, fmt.Errorf("%w: row %d has %d logits, %d marker columns",
				ErrIncompatibleCheckpoint, r, len(logits[r]), kmax)
		}
		if len(act[r]) != b.actWidth {
			return nil, nil, fmt.Errorf("%w: row %d has %d act logits, the config's act head %d",
				ErrIncompatibleCheckpoint, r, len(act[r]), b.actWidth)
		}
	}
	return logits, act, nil
}

// readCheckpoint reads the config and the tokenizer of the checkpoint spec
// locates, downloading them first if it is on the Hub, and the act_logits
// width the config derives.
func (l *defaultLoader) readCheckpoint(ctx context.Context, spec ModelSpec) (
	cfg *checkpoint.Config, tok *tokenizer.HF, actWidth int, err error,
) {
	dir, err := l.checkpointDir(ctx, spec)
	if err != nil {
		return nil, nil, 0, err
	}
	if cfg, err = checkpoint.LoadConfig(dir); err != nil {
		return nil, nil, 0, fmt.Errorf("laya: %w", err)
	}
	if actWidth, err = cfg.ActWidth(); err != nil {
		return nil, nil, 0, fmt.Errorf("laya: %w", err)
	}
	if tok, err = tokenizer.Open(filepath.Join(dir, "tokenizer")); err != nil {
		return nil, nil, 0, fmt.Errorf("%w: %w", ErrIncompatibleCheckpoint, err)
	}
	return cfg, tok, actWidth, nil
}

// checkpointDir is agent.py:115-135: a directory that exists is used as it is,
// a path-shaped id that does not is an error rather than a repo to download,
// and anything else is a Hub snapshot. The subfolder is joined afterwards in
// both cases and must exist.
func (l *defaultLoader) checkpointDir(ctx context.Context, spec ModelSpec) (string, error) {
	// Before anything is read or requested: joined as it is, "../other" would
	// load whatever checkpoint sits beside the one selected.
	if spec.Subfolder != "" && !filepath.IsLocal(spec.Subfolder) {
		return "", fmt.Errorf("%w: subfolder %q is not a path inside %q",
			ErrCheckpointNotFound, spec.Subfolder, spec.Repo)
	}
	dir := spec.Repo
	if _, err := os.Stat(dir); err != nil {
		if !errors.Is(err, fs.ErrNotExist) {
			return "", fmt.Errorf("laya: %w", err)
		}
		if pathShaped(spec.Repo) {
			return "", fmt.Errorf("%w: local path %q", ErrCheckpointNotFound, spec.Repo)
		}
		if dir, err = l.download(ctx, spec); err != nil {
			return "", err
		}
	}
	if spec.Subfolder == "" {
		return dir, nil
	}
	sub := filepath.Join(dir, spec.Subfolder)
	if info, err := os.Stat(sub); err != nil || !info.IsDir() {
		return "", fmt.Errorf("%w: subfolder %q not in %q", ErrCheckpointNotFound, spec.Subfolder, spec.Repo)
	}
	return sub, nil
}

// download snapshots the spec's config and tokenizer at the loader's
// revision, else the bundle repo at D17's pin and any other repo at main.
func (l *defaultLoader) download(ctx context.Context, spec ModelSpec) (string, error) {
	rev := l.revision
	switch {
	case rev != "":
	case spec.Repo == bundleRepo:
		rev = pinnedRevision
	default:
		rev = "main"
	}
	dir, err := l.snapshot(ctx, spec.Repo, rev, allowPatterns(spec.Subfolder))
	// A missing repo, and a subfolder the allow filter finds no file for, are
	// both hub.ErrNotFound; the local cases answer with ErrCheckpointNotFound,
	// so the Hub's do too.
	if errors.Is(err, hub.ErrNotFound) {
		return "", fmt.Errorf("%w: %w", ErrCheckpointNotFound, err)
	}
	return dir, err
}

// pathShaped is agent.py:117's test for an id that can only mean a local path.
func pathShaped(repo string) bool {
	return strings.HasPrefix(repo, "/") || strings.HasPrefix(repo, "./") ||
		strings.HasPrefix(repo, "../") || filepath.IsAbs(repo)
}

// allowPatterns is what a snapshot downloads: the config and the tokenizer of
// one checkpoint, nothing else (D24). Upstream fetches sub/* and, for the root
// checkpoint, the entire repo; the weights among them are what the local ONNX
// export replaces. Patterns are anchored, so the root's tokenizer/* cannot
// match another checkpoint's multilingual/tokenizer/.
func allowPatterns(sub string) []string {
	prefix := ""
	if sub != "" {
		prefix = sub + "/"
	}
	return []string{prefix + checkpoint.ConfigFile, prefix + "tokenizer/*"}
}

// graphDir is where the loader looks for exports.
func (l *defaultLoader) graphDir() (string, error) {
	if l.onnxDir != "" {
		return l.onnxDir, nil
	}
	base := l.cacheDir
	if base == "" {
		var err error
		if base, err = hub.DefaultDir(); err != nil {
			return "", fmt.Errorf("laya: %w", err)
		}
	}
	return filepath.Join(base, "onnx"), nil
}

// graph is the export for the checkpoint the router calls name, under the
// file name scripts/export_onnx.py gives it.
func (l *defaultLoader) graph(name string) (string, error) {
	dir, err := l.graphDir()
	if err != nil {
		return "", err
	}
	path := filepath.Join(dir, "laya-"+name+".onnx")
	if _, err := os.Stat(path); err != nil {
		if !errors.Is(err, fs.ErrNotExist) {
			// There may well be an export; it cannot be reached. Saying
			// "export it" would send the caller after the wrong problem.
			return "", fmt.Errorf("laya: ONNX export: %w", err)
		}
		return "", fmt.Errorf("%w: %s (export it with "+
			"`scripts/export_onnx.py --checkpoint %s --out %s`, or point WithONNXDir or "+
			"LAYA_ONNX_DIR at an existing export)", ErrNoGraph, path, name, dir)
	}
	return path, nil
}
