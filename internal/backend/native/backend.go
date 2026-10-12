package native

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"sync"

	"github.com/MeKo-Christian/go-laya/backend"
	"github.com/MeKo-Christian/go-laya/internal/head"
	"github.com/MeKo-Christian/go-laya/internal/modernbert"
	"github.com/MeKo-Christian/go-laya/internal/safetensors"
)

var (
	// ErrClosed is Forward's error on a Backend that was closed or never
	// opened.
	ErrClosed = errors.New("native backend: closed")

	// ErrBadBatch is Forward's error for a batch whose tensors are not the
	// rectangular, mutually consistent shapes the model takes, or whose
	// values the encoder or the head refuse.
	ErrBadBatch = errors.New("native backend: malformed batch")
)

// Options configures Open. It holds no tuning: the only tuning the kernels
// of github.com/cwbudde/algo-tensor (D35) have is tensor.SetWorkers, a
// process-wide setting that a per-backend option would misrepresent. They
// default to one worker per GOMAXPROCS as read at start-up (D34), so a
// process that wants fewer sets the GOMAXPROCS environment variable.
type Options struct {
	// Head, if not nil, is the head the caller expects the file to fix.
	// Open holds model.safetensors' head to it from the header of the file
	// it then decodes, before a weight is decoded, and fails with a
	// *HeadMismatchError otherwise. A check on a separate open would pass
	// a file that is replaced before the load.
	Head *HeadShape
}

// Backend runs a laya DecisionModel in pure Go: the ModernBERT or mmBERT
// encoder of internal/modernbert, then the decision head of internal/head,
// with weights decoded from the checkpoint's model.safetensors. It is safe
// for concurrent Forward calls; Close waits for those in flight.
type Backend struct {
	mu sync.RWMutex
	m  *model // nil once closed
}

// model is the loaded network. Neither half is written after Open: the
// encoder's and the head's Forward only read their weights and allocate
// every intermediate per call, so concurrent passes share it safely.
type model struct {
	enc  modernbert.Encoder
	head *head.Head
}

var _ backend.Backend = (*Backend)(nil)

// Open builds the model of the laya checkpoint in dir: the layer plan from
// dir/encoder/config.json (modernbert.ParseConfig, Config.Layers) and every
// weight from dir/model.safetensors, decoded to float32. The head has as
// many layers as the file has head.layers.N, and as many act logits as
// act_head.2 has rows. Upstream takes both from rl_agent_config.json
// (head_layers, len(act_costs)+1; common.py:137), which this does not read:
// the caller passes them as opts.Head, as it passes onnx.Options.ActWidth
// to the ONNX backend.
//
// Loading is strict, as upstream's load_state_dict(strict=True) is
// (agent.py:184): a missing or unexpected tensor, a shape other than the
// config implies, an attn_norm on layer 0 (nn.Identity upstream), an encoder
// layer count other than the config's num_hidden_layers, and a config the
// encoder does not implement are errors wrapping
// backend.ErrIncompatibleCheckpoint that name the tensor or config field.
// temperature must be there as upstream registers it, [3] in any dtype, and
// is never decoded. The layer count is compared before the plan is resolved,
// so a config cannot make Open allocate for more layers than the file has.
//
// With opts.Head set, the head is held to it from the header before
// encoder/config.json is read or a weight decoded; see Options.Head.
//
// ctx is checked between tensors; a cancellation returns ctx's error alone.
func Open(ctx context.Context, dir string, opts Options) (*Backend, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	path := filepath.Join(dir, weightsFile)
	f, err := safetensors.Open(path)
	if err != nil {
		return nil, fmt.Errorf("native backend: %w", err)
	}
	defer f.Close()
	// The header of the file load decodes, so the head checked is the head
	// built: a file replaced after a separate check could not be.
	if opts.Head != nil {
		if err := checkHead(f, path, *opts.Head); err != nil {
			return nil, err
		}
	}
	cfg, err := readConfig(dir)
	if err != nil {
		return nil, err
	}
	m, err := load(ctx, f, path, cfg)
	if err != nil {
		return nil, err
	}
	return &Backend{m: m}, nil
}

// Forward runs the encoder and the head on in and returns logits (n×kmax)
// and act_logits (n×n_act), as backend.Backend.Forward does.
//
// ctx is checked before the encoder, between the encoder and the head, and
// after the head. Neither half can be interrupted once it has started, so a
// cancellation is seen at the next check, and one seen there wins over the
// result.
func (b *Backend) Forward(ctx context.Context, in backend.Batch) (logits, act [][]float32, err error) {
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}

	b.mu.RLock()
	defer b.mu.RUnlock()
	if b.m == nil {
		return nil, nil, ErrClosed
	}
	if err := checkBatch(in); err != nil {
		return nil, nil, err
	}

	// checkBatch has checked the shapes, and Open the model, so what the
	// encoder and the head still refuse is the batch's contents: an id
	// outside the vocabulary, a mask value, a qtype, a marker position, a
	// row of padding only, or fewer than two marker columns. Those are the
	// caller's errors, as a ragged batch is.
	h, err := b.m.enc.Forward(in.InputIDs, in.AttentionMask)
	if err != nil {
		return nil, nil, fmt.Errorf("%w: encoder: %w", ErrBadBatch, err)
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	logits, act, err = b.m.head.Forward(h, in)
	if err != nil {
		return nil, nil, fmt.Errorf("%w: %w", ErrBadBatch, err)
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	return logits, act, nil
}

// Close drops the weights. It waits for Forward calls in flight and is safe
// to call more than once; Forward afterwards returns ErrClosed.
func (b *Backend) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.m = nil
	return nil
}
