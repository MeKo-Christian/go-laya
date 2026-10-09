package laya

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"sync"

	"github.com/MeKo-Christian/go-laya/backend"
	"github.com/MeKo-Christian/go-laya/internal/calib"
	"github.com/MeKo-Christian/go-laya/internal/checkpoint"
	"github.com/MeKo-Christian/go-laya/tokenizer"
)

// Agent is one loaded checkpoint: its ONNX Runtime session, its config and
// its tokenizer. It is Python's laya.Agent (agent.py:99), built by Open as
// laya.load builds that one, and it is also what the Router's default loader
// caches, so it satisfies Predictor.
//
// SystemOne, Predict and SetLimits may be called from several goroutines at
// once.
type Agent struct {
	backend backend.Backend
	cfg     *checkpoint.Config
	tok     *tokenizer.HF
	temps   calib.Temperatures

	mu                 sync.RWMutex // guards the limits, which SetLimits changes
	maxLen, headMaxLen int
}

// newAgent reads what SystemOne needs from the checkpoint's config once:
// the sequence budgets (agent.py:256-257) and the temperatures
// (agent.py:194-195).
func newAgent(be backend.Backend, cfg *checkpoint.Config, tok *tokenizer.HF) (*Agent, error) {
	maxLen, err := cfg.MaxLen()
	if err != nil {
		return nil, fmt.Errorf("laya: %w", err)
	}
	headMaxLen, err := cfg.HeadMaxLen()
	if err != nil {
		return nil, fmt.Errorf("laya: %w", err)
	}
	byQType, byOptions, err := cfg.Temperatures()
	if err != nil {
		return nil, fmt.Errorf("laya: %w", err)
	}
	return &Agent{
		backend:    be,
		cfg:        cfg,
		tok:        tok,
		maxLen:     maxLen,
		headMaxLen: headMaxLen,
		temps:      calib.Temperatures{ByQType: byQType, ByOptions: byOptions},
	}, nil
}

// MaxLen is the token budget of a whole sequence, the config's max_len
// unless SetLimits or WithLimits changed it.
func (a *Agent) MaxLen() int {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.maxLen
}

// HeadMaxLen is the token budget of the instructions and options, the
// config's head_max_len unless SetLimits or WithLimits changed it.
func (a *Agent) HeadMaxLen() int {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.headMaxLen
}

// SetLimits replaces both budgets for every later SystemOne call. It is how
// Python's agent.cfg["max_len"] = ... and agent.cfg["head_max_len"] = ...
// are written here; a pass already running keeps the limits it started with.
// A value that is not positive is ErrInvalidLimits and changes nothing,
// where Python would silently truncate every sequence.
func (a *Agent) SetLimits(maxLen, headMaxLen int) error {
	if err := checkLimits(maxLen, headMaxLen); err != nil {
		return err
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	a.maxLen, a.headMaxLen = maxLen, headMaxLen
	return nil
}

// checkLimits applies the rule the config's own max_len and head_max_len
// are held to.
func checkLimits(maxLen, headMaxLen int) error {
	if maxLen <= 0 || headMaxLen <= 0 {
		return fmt.Errorf("%w: max_len %d, head_max_len %d", ErrInvalidLimits, maxLen, headMaxLen)
	}
	return nil
}

// Close releases the runtime session. A leaked one is hundreds of megabytes.
func (a *Agent) Close() error {
	if err := a.backend.Close(); err != nil {
		return fmt.Errorf("laya: close agent: %w", err)
	}
	return nil
}

// limits reads both budgets at once, for one SystemOne call.
func (a *Agent) limits() (maxLen, headMaxLen int) {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.maxLen, a.headMaxLen
}

// Option configures Open. The options are laya.load's keyword arguments
// (agent.py:351-352), plus WithGraph, which the local ONNX export needs (D24).
type Option func(*agentConfig) error

type agentConfig struct {
	loaderSettings

	subfolder          string
	graph              string
	maxLen, headMaxLen int // 0 keeps the config's
}

// WithSubfolder selects one checkpoint from a repo that bundles several, as
// laya.load's subfolder does: "multilingual" or "typed-decisions" in
// convaiinnovations/laya. Only that subfolder's config and tokenizer are
// downloaded.
func WithSubfolder(sub string) Option {
	return func(c *agentConfig) error {
		c.subfolder = sub
		return nil
	}
}

// WithDevice is the device the agent runs on: "cpu", "cuda", "cuda:N",
// "coreml", or "" / "auto" for the best one the ONNX Runtime library offers.
// An unusable device falls back to the CPU with a warning, and a name that is
// no device fails Open with ErrUnknownDevice.
func WithDevice(d string) Option {
	return func(c *agentConfig) error {
		c.device = d
		return nil
	}
}

// WithHFToken is the Hub token Open downloads with. Empty falls back to
// $HF_TOKEN. The token is sent only to the Hub's own host.
func WithHFToken(tok string) Option {
	return func(c *agentConfig) error {
		c.token = tok
		return nil
	}
}

// WithGraph is the ONNX export to run, a file `scripts/export_onnx.py`
// wrote. Without it, Open looks for laya-<name>.onnx in $LAYA_ONNX_DIR, else
// in onnx/ under the laya cache, where name is the checkpoint the repo and
// subfolder locate in DefaultModels or StandaloneModels. A checkpoint neither
// registry names, such as a local directory, needs WithGraph.
func WithGraph(path string) Option {
	return func(c *agentConfig) error {
		c.graph = path
		return nil
	}
}

// WithLimits sets the token budgets SetLimits sets, in place of the config's
// max_len and head_max_len. Python changes them by writing agent.cfg; the
// README raises head_max_len for questions with many options. A value that is
// not positive is ErrInvalidLimits.
func WithLimits(maxLen, headMaxLen int) Option {
	return func(c *agentConfig) error {
		if err := checkLimits(maxLen, headMaxLen); err != nil {
			return err
		}
		c.maxLen, c.headMaxLen = maxLen, headMaxLen
		return nil
	}
}

// Open loads one checkpoint, as laya.load does (agent.py:351-360). ref is a
// Hub repo id or a local directory; "" is convaiinnovations/laya, upstream's
// default. A Hub repo is downloaded as the Router's loader downloads it: only
// the config and the tokenizer, the bundle repo at the revision every golden
// vector was recorded against (D17), and the forward pass runs on a local
// ONNX export (D24).
//
// It fails with ErrCheckpointNotFound for a checkpoint that is not there,
// ErrNoGraph for an export that is not, and ErrIncompatibleCheckpoint for one
// this port cannot run. The caller closes the agent.
func Open(ctx context.Context, ref string, opts ...Option) (*Agent, error) {
	return openAgent(ctx, ref, opts, newDefaultLoader)
}

// openAgent is Open with the loader's constructor as a parameter, so tests
// can stand in for the network and the runtime.
func openAgent(ctx context.Context, ref string, opts []Option, newLoader func(loaderSettings) *defaultLoader) (*Agent, error) {
	var cfg agentConfig
	for _, opt := range opts {
		if err := opt(&cfg); err != nil {
			return nil, err
		}
	}
	if ref == "" {
		ref = bundleRepo
	}
	spec := ModelSpec{Repo: ref, Subfolder: cfg.subfolder}
	l := newLoader(cfg.loaderSettings)
	a, err := l.build(ctx, spec, func() (string, error) {
		if cfg.graph != "" {
			if _, err := os.Stat(cfg.graph); err != nil {
				if errors.Is(err, fs.ErrNotExist) {
					return "", fmt.Errorf("%w: %s", ErrNoGraph, cfg.graph)
				}
				return "", fmt.Errorf("laya: ONNX export: %w", err)
			}
			return cfg.graph, nil
		}
		name, ok := exportName(spec)
		if !ok {
			return "", fmt.Errorf("%w: %s is in no model registry, so its export has no known name; "+
				"pass WithGraph", ErrNoGraph, spec)
		}
		return l.graph(name)
	})
	if err != nil {
		return nil, err
	}
	if cfg.maxLen > 0 {
		a.maxLen, a.headMaxLen = cfg.maxLen, cfg.headMaxLen
	}
	return a, nil
}

// exportName is the name `scripts/export_onnx.py` gives spec's export: the
// checkpoint either registry locates at spec.
func exportName(spec ModelSpec) (string, bool) {
	for _, reg := range []map[string]ModelSpec{defaultModels, standaloneModels} {
		for name, s := range reg {
			if s == spec {
				return name, true
			}
		}
	}
	return "", false
}
