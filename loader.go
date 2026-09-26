package laya

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
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
	hub     *hub.Client
	onnxDir string // "" is onnx/ under the laya cache, resolved per load

	// snapshot and open are the network and the runtime. They are fields so
	// that tests can stand in for both; CI has neither.
	snapshot func(ctx context.Context, repo, rev string, allow []string) (string, error)
	open     func(path string, opts onnx.Options) (backend.Backend, error)
}

func newDefaultLoader(cfg routerConfig) *defaultLoader {
	l := &defaultLoader{
		hub:     &hub.Client{},
		onnxDir: cfg.onnxDir,
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

// load builds the agent for the checkpoint the router calls name.
func (l *defaultLoader) load(ctx context.Context, name string, spec ModelSpec) (Agent, error) {
	dir, err := l.checkpointDir(ctx, spec)
	if err != nil {
		return nil, err
	}
	cfg, err := checkpoint.LoadConfig(dir)
	if err != nil {
		return nil, fmt.Errorf("laya: %w", err)
	}
	actWidth, err := cfg.ActWidth()
	if err != nil {
		return nil, fmt.Errorf("laya: %w", err)
	}
	tok, err := tokenizer.Open(filepath.Join(dir, "tokenizer"))
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrIncompatibleCheckpoint, err)
	}

	graph, err := l.graph(name)
	if err != nil {
		return nil, err
	}
	be, err := l.open(graph, onnx.Options{ActWidth: actWidth})
	if err != nil {
		return nil, err
	}
	return &onnxAgent{backend: be, cfg: cfg, tok: tok}, nil
}

// checkpointDir is agent.py:115-135: a directory that exists is used as it is,
// a path-shaped id that does not is an error rather than a repo to download,
// and anything else is a Hub snapshot. The subfolder is joined afterwards in
// both cases and must exist.
func (l *defaultLoader) checkpointDir(ctx context.Context, spec ModelSpec) (string, error) {
	dir := spec.Repo
	if _, err := os.Stat(dir); err != nil {
		if !errors.Is(err, fs.ErrNotExist) {
			return "", fmt.Errorf("laya: %w", err)
		}
		if pathShaped(spec.Repo) {
			return "", fmt.Errorf("%w: local path %q", ErrModelNotFound, spec.Repo)
		}
		rev := "main"
		if spec.Repo == bundleRepo {
			rev = pinnedRevision
		}
		if dir, err = l.snapshot(ctx, spec.Repo, rev, allowPatterns(spec.Subfolder)); err != nil {
			return "", err
		}
	}
	if spec.Subfolder == "" {
		return dir, nil
	}
	sub := filepath.Join(dir, spec.Subfolder)
	if info, err := os.Stat(sub); err != nil || !info.IsDir() {
		return "", fmt.Errorf("%w: subfolder %q not in %q", ErrModelNotFound, spec.Subfolder, spec.Repo)
	}
	return sub, nil
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
	base, err := hub.DefaultDir()
	if err != nil {
		return "", fmt.Errorf("laya: %w", err)
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
		return "", fmt.Errorf("%w: %s (export it with "+
			"`scripts/export_onnx.py --checkpoint %s --out %s`, or point WithONNXDir or "+
			"LAYA_ONNX_DIR at an existing export)", ErrNoGraph, path, name, dir)
	}
	return path, nil
}

// onnxAgent is a loaded checkpoint: its runtime session, its config and its
// tokenizer. Today it is only closed; SystemOne arrives with PLAN Task 7.3.
type onnxAgent struct {
	backend backend.Backend
	cfg     *checkpoint.Config
	tok     *tokenizer.HF
}

// Close releases the runtime session.
func (a *onnxAgent) Close() error {
	if err := a.backend.Close(); err != nil {
		return fmt.Errorf("laya: close agent: %w", err)
	}
	return nil
}
