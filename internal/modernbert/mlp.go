package modernbert

import (
	"errors"
	"fmt"
	"math"

	"github.com/MeKo-Christian/go-laya/internal/runtime/tensor"
)

// MLP is ModernBertMLP, the GeGLU feed-forward block of every encoder layer
// (modeling_modernbert.py):
//
//	input, gate = self.Wi(hidden_states).chunk(2, dim=-1)
//	return self.Wo(self.drop(self.act(input) * gate))
//
// Wi [2I, H] fuses two projections: its first I rows give the input, which is
// activated, and its last I rows the gate, which multiplies. Wo is [H, I].
// Neither has a bias, because both checkpoints set mlp_bias false: Wi is
// [5248, 1024] and Wo [1024, 2624] in ModernBERT-large, [2304, 768] and
// [768, 1152] in mmBERT-base. The dropout is inference's no-op and is not
// modelled.
//
// The zero MLP has no weights; its Forward returns an error rather than a
// plausible zero.
type MLP struct {
	wi, wo        *tensor.Tensor
	hidden, inter int64
}

// NewMLP returns the MLP with the fused Wi [2I, H] and Wo [H, I], in
// transformers' [out, in] layout. The MLP keeps both; the caller must not
// modify them. A Wi that is empty or has an odd number of rows, or a Wo that
// is not [H, I], is an error: Wo as [I, H] has the right element count and is
// the likely mistake.
func NewMLP(wi, wo *tensor.Tensor) (MLP, error) {
	if wi == nil || wo == nil {
		return MLP{}, errors.New("modernbert: MLP weight is nil")
	}
	if wi.Rank() != 2 || wi.Shape()[0]%2 != 0 || wi.ElemCount() == 0 {
		return MLP{}, fmt.Errorf("modernbert: MLP Wi has shape %v, want [2*intermediate, hidden]", wi.Shape())
	}
	hidden, inter := wi.Shape()[1], wi.Shape()[0]/2
	if wo.Rank() != 2 || wo.Shape()[0] != hidden || wo.Shape()[1] != inter {
		return MLP{}, fmt.Errorf("modernbert: MLP Wo has shape %v, want [%d %d] for Wi %v",
			wo.Shape(), hidden, inter, wi.Shape())
	}
	return MLP{wi: wi, wo: wo, hidden: hidden, inter: inter}, nil
}

// Forward applies the MLP to the last dimension of x, which must be the hidden
// size. It returns a new tensor of the shape of x that the caller owns.
func (m MLP) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	if m.wi == nil {
		return nil, errors.New("modernbert: uninitialized MLP; build it with NewMLP")
	}
	if x == nil {
		return nil, errors.New("modernbert: MLP input is nil")
	}
	shape := x.Shape()
	if len(shape) == 0 || shape[len(shape)-1] != m.hidden {
		return nil, fmt.Errorf("modernbert: MLP input has shape %v, want a last dimension of %d", shape, m.hidden)
	}

	fused, err := tensor.Linear(x, m.wi, nil) // [..., 2I]
	if err != nil {
		return nil, fmt.Errorf("modernbert: MLP Wi: %w", err)
	}

	shape[len(shape)-1] = m.inter
	h, err := tensor.Zeros(shape)
	if err != nil {
		return nil, fmt.Errorf("modernbert: %w", err)
	}
	// chunk(2, dim=-1): each row of fused is the input's I values, then the
	// gate's.
	inter := int(m.inter)
	f, out := fused.RawData(), h.RawData()
	for r := range len(out) / inter {
		in, gate := f[2*r*inter:(2*r+1)*inter], f[(2*r+1)*inter:(2*r+2)*inter]
		row := out[r*inter : (r+1)*inter]
		for j := range row {
			row[j] = gelu(in[j]) * gate[j]
		}
	}

	y, err := tensor.Linear(h, m.wo, nil)
	if err != nil {
		return nil, fmt.Errorf("modernbert: MLP Wo: %w", err)
	}
	return y, nil
}

// gelu is the exact GELU, 0.5*x*(1 + erf(x/sqrt 2)). The checkpoints'
// hidden_activation "gelu" is ACT2FN["gelu"], GELUActivation in
// transformers/activations.py, which calls nn.functional.gelu with its default
// approximate="none". The tanh approximation ("gelu_new",
// "gelu_pytorch_tanh") is off by up to 4.7e-4 near |x| = 2.7.
func gelu(x float32) float32 {
	v := float64(x)
	return float32(0.5 * v * (1 + math.Erf(v/math.Sqrt2)))
}
