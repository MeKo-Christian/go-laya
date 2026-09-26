//go:build !android && !ios && ((darwin && (amd64 || arm64)) || (linux && (amd64 || arm64 || loong64)) || (netbsd && (amd64 || arm64)))

// The constraint lists the targets where the binding actually compiles with
// CGO_ENABLED=0, found by building this package for every linux, darwin,
// freebsd and netbsd target in `go tool dist list`. It is narrower than
// purego's Dlopen set: purego's fakecgo does not build on freebsd, linux/386 or
// linux/riscv64, for example, and android (which implies linux) has no Dlopen
// without cgo. backend_other.go covers everything else.

package onnx

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"sync"

	ort "github.com/shota3506/onnxruntime-purego/onnxruntime"

	"github.com/MeKo-Christian/go-laya/backend"
)

// Backend runs one exported DecisionModel graph through ONNX Runtime. It is
// safe for concurrent Forward calls; Close waits for those in flight.
type Backend struct {
	mu     sync.RWMutex
	rt     *ort.Runtime
	env    *ort.Env
	sess   *ort.Session
	device string
}

var _ backend.Backend = (*Backend)(nil)

// Open loads the ONNX Runtime library and creates a session over the graph at
// modelPath. The session is created from the path and never from a reader:
// the exports exceed protobuf's 2 GB limit and keep their weights in an
// .onnx.data sibling, which ORT resolves relative to the model path.
//
// The graph must declare exactly the five inputs and two outputs
// scripts/export_onnx.py writes, in any order; anything else is an error
// wrapping backend.ErrIncompatibleCheckpoint.
//
// opts.Device picks the execution provider; see Options.Device for when that
// falls back to the CPU and warns.
func Open(modelPath string, opts Options) (*Backend, error) {
	if _, err := parseDevice(opts.Device); err != nil {
		return nil, err
	}
	log := opts.Logger
	if log == nil {
		log = slog.Default()
	}

	// #nosec G703 -- stat only; the path is the caller's, and whatever
	// resolved it (a developer flag, internal/hub's validated cache path) owns
	// its trust. ORT reads the file below either way.
	if _, err := os.Stat(modelPath); err != nil {
		return nil, fmt.Errorf("onnx backend: model: %w", err)
	}

	lib := opts.Library
	if lib == "" {
		var err error
		if lib, err = findORTLibrary(); err != nil {
			return nil, fmt.Errorf("onnx backend: %w", err)
		}
	}

	rt, err := ort.NewRuntime(lib, APIVersion)
	if err != nil {
		return nil, fmt.Errorf("onnx backend: load %s (C API %d): %w", lib, APIVersion, err)
	}
	b := &Backend{rt: rt}

	if b.env, err = rt.NewEnv("laya", ort.LoggingLevelWarning); err != nil {
		return nil, errors.Join(fmt.Errorf("onnx backend: env: %w", err), b.Close())
	}
	if err := b.openSession(modelPath, opts, log); err != nil {
		return nil, errors.Join(err, b.Close())
	}

	if err := checkGraphIO(b.sess.InputNames(), b.sess.OutputNames()); err != nil {
		return nil, errors.Join(fmt.Errorf("onnx backend: %s: %w", modelPath, err), b.Close())
	}
	return b, nil
}

// availableProviders is GetAvailableProviders, swappable so a test can make
// the library advertise a provider it cannot actually build a session with.
var availableProviders = (*ort.Runtime).GetAvailableProviders

// Device reports the device the session runs on: "cpu", "cuda" or "coreml".
// After a fallback it is "cpu", whatever Options.Device asked for.
func (b *Backend) Device() string {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.device
}

// Forward runs one forward pass and returns logits (n×kmax) and act_logits
// (n×len(act_costs)+1).
//
// ctx is checked before and after the pass but cannot interrupt it: the
// binding's Session.Run takes a ctx and never reads it (it passes NULL
// RunOptions), so a pass that has started runs to completion. A ctx cancelled
// meanwhile still wins over the result.
func (b *Backend) Forward(ctx context.Context, in backend.Batch) (logits, act [][]float32, err error) {
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}

	b.mu.RLock()
	defer b.mu.RUnlock()
	if b.sess == nil {
		return nil, nil, ErrClosed
	}

	f, err := flatten(in)
	if err != nil {
		return nil, nil, err
	}

	// Every value is closed explicitly: the binding's finalizer path races
	// Runtime.Close (PLAN.md D5, TestValueCleanup).
	inputs := make(map[string]*ort.Value, len(graphInputs))
	defer func() {
		for _, v := range inputs {
			v.Close()
		}
	}()
	n, seq, k := int64(f.rows), int64(f.seq), int64(f.k)
	for name, mk := range map[string]func() (*ort.Value, error){
		"input_ids":      func() (*ort.Value, error) { return ort.NewTensorValue(b.rt, f.inputIDs, []int64{n, seq}) },
		"attention_mask": func() (*ort.Value, error) { return ort.NewTensorValue(b.rt, f.attention, []int64{n, seq}) },
		"marker_pos":     func() (*ort.Value, error) { return ort.NewTensorValue(b.rt, f.markerPos, []int64{n, k}) },
		"marker_mask":    func() (*ort.Value, error) { return ort.NewTensorValue(b.rt, f.markerMask, []int64{n, k}) },
		"qtype":          func() (*ort.Value, error) { return ort.NewTensorValue(b.rt, f.qtype, []int64{n}) },
	} {
		v, err := mk()
		if err != nil {
			return nil, nil, fmt.Errorf("onnx backend: %s: %w", name, err)
		}
		inputs[name] = v
	}

	outputs, err := b.sess.Run(ctx, inputs)
	defer func() {
		for _, v := range outputs {
			v.Close()
		}
	}()
	if err != nil {
		return nil, nil, fmt.Errorf("onnx backend: run: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}

	if logits, err = output(outputs, "logits", f.rows); err != nil {
		return nil, nil, err
	}
	if act, err = output(outputs, "act_logits", f.rows); err != nil {
		return nil, nil, err
	}
	return logits, act, nil
}

func output(outputs map[string]*ort.Value, name string, rows int) ([][]float32, error) {
	v, ok := outputs[name]
	if !ok {
		return nil, fmt.Errorf("onnx backend: run returned no %s", name)
	}
	data, shape, err := ort.GetTensorData[float32](v)
	if err != nil {
		return nil, fmt.Errorf("onnx backend: %s: %w", name, err)
	}
	return unflatten(name, data, shape, rows)
}

// Close releases the session, the environment and the runtime, in that order.
// It waits for Forward calls in flight and is safe to call more than once.
func (b *Backend) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.sess != nil {
		b.sess.Close()
		b.sess = nil
	}
	if b.env != nil {
		b.env.Close()
		b.env = nil
	}
	if b.rt != nil {
		err := b.rt.Close()
		b.rt = nil
		if err != nil {
			return fmt.Errorf("onnx backend: close runtime: %w", err)
		}
	}
	return nil
}

// openSession creates the session on the device opts asks for. It warns, once,
// only when an explicitly requested device is not what the session ends up on:
// either the library cannot provide it, or building the session with it
// failed and the CPU was tried instead (agent.py:156-165, 203-227).
func (b *Backend) openSession(modelPath string, opts Options, log *slog.Logger) error {
	available, err := availableProviders(b.rt)
	if err != nil {
		return fmt.Errorf("onnx backend: execution providers: %w", err)
	}
	choice, err := resolveDevice(opts.Device, available)
	if err != nil {
		return err
	}

	newSession := func(providers []string) (*ort.Session, error) {
		return b.rt.NewSession(b.env, modelPath, &ort.SessionOptions{
			IntraOpNumThreads:  opts.IntraOpThreads,
			ExecutionProviders: providers,
		})
	}

	sess, err := newSession(choice.providers)
	if err != nil && len(choice.providers) > 0 {
		choice.fallback = fmt.Sprintf("session on %s failed: %v", choice.device, err)
		choice.device = deviceCPU
		sess, err = newSession(nil)
	}
	if err != nil {
		return fmt.Errorf("onnx backend: session %s: %w", modelPath, err)
	}

	if choice.fallback != "" {
		log.Warn("onnx backend: requested device unavailable, running on CPU",
			"requested", choice.requested, "reason", choice.fallback)
	}
	b.sess, b.device = sess, choice.device
	return nil
}
