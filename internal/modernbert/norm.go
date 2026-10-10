package modernbert

import (
	"errors"
	"fmt"

	"github.com/MeKo-Christian/go-laya/internal/runtime/tensor"
)

// NormEps is the epsilon of every ModernBERT LayerNorm. Both checkpoints'
// encoder config.json set norm_eps 1e-05: ModernBERT-large (english,
// models/laya/encoder) and mmBERT-base (models/laya/multilingual/encoder).
const NormEps float32 = 1e-5

// Norm is one of ModernBERT's LayerNorms: the embeddings norm, an encoder
// layer's attn_norm or mlp_norm, or the final norm. It has a weight and never a
// bias, because both checkpoints set norm_bias false.
//
// A Norm is either a LayerNorm (NewNorm) or the identity (IdentityNorm), which
// is how layer 0's attn_norm is modelled: upstream it is nn.Identity()
// (modeling_modernbert.py, ModernBertEncoderLayer). The two are kept apart on
// purpose. A LayerNorm whose weight is all ones still centres and scales, and
// a nil weight is a missing tensor rather than a request for the identity. The
// zero Norm is neither; its Forward returns an error, so a norm the loader
// never set fails loudly instead of passing its input through.
type Norm struct {
	weight   *tensor.Tensor
	identity bool
}

// NewNorm returns the bias-free LayerNorm with the given rank-1 weight over the
// hidden dimension. The Norm keeps weight; the caller must not modify it.
func NewNorm(weight *tensor.Tensor) (Norm, error) {
	if weight == nil {
		return Norm{}, errors.New("modernbert: norm weight is nil; use IdentityNorm for a missing norm")
	}
	if weight.Rank() != 1 {
		return Norm{}, fmt.Errorf("modernbert: norm weight has shape %v, want rank 1", weight.Shape())
	}
	return Norm{weight: weight}, nil
}

// IdentityNorm returns the Norm that is no norm: Forward returns a copy of its
// input.
func IdentityNorm() Norm {
	return Norm{identity: true}
}

// Forward normalizes the last dimension of x. It always returns a new tensor
// that the caller owns, the identity included. A weight whose length does not
// match the last dimension of x is an error.
func (n Norm) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	if x == nil {
		return nil, errors.New("modernbert: norm input is nil")
	}
	if n.identity {
		return x.Clone(), nil
	}
	if n.weight == nil {
		return nil, errors.New("modernbert: uninitialized norm; build it with NewNorm or IdentityNorm")
	}
	out, err := tensor.LayerNorm(x, n.weight, nil, NormEps)
	if err != nil {
		return nil, fmt.Errorf("modernbert: %w", err)
	}
	return out, nil
}
