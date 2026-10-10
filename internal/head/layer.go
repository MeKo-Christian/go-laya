package head

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/MeKo-Christian/go-laya/internal/runtime/tensor"
)

// LayerWeights are one nn.TransformerEncoderLayer's tensors, in torch's
// [out, in] layout, named after its state_dict (docs/ARCHITECTURE.md §1.3):
// head.layers.N.self_attn.in_proj_weight and so on.
type LayerWeights struct {
	// InProjWeight [3d, d] packs the query, key and value projections, in that
	// order (nn.MultiheadAttention with _qkv_same_embed_dim); InProjBias is
	// [3d].
	InProjWeight, InProjBias *tensor.Tensor
	// OutProjWeight [d, d] and OutProjBias [d] are self_attn.out_proj.
	OutProjWeight, OutProjBias *tensor.Tensor
	// Linear1Weight [4d, d], Linear1Bias [4d], Linear2Weight [d, 4d] and
	// Linear2Bias [d] are the feed-forward block, dim_feedforward 4d.
	Linear1Weight, Linear1Bias, Linear2Weight, Linear2Bias *tensor.Tensor
	// Norm1Weight, Norm1Bias, Norm2Weight and Norm2Bias, each [d], are the
	// LayerNorms before the attention and before the feed-forward block.
	Norm1Weight, Norm1Bias, Norm2Weight, Norm2Bias *tensor.Tensor
}

// Layer is the head's nn.TransformerEncoderLayer as DecisionModel builds it
// (common.py:97): norm_first=True, batch_first=True, a ReLU feed-forward block
// of width 4d, LayerNorms with weight and bias at eps 1e-5, and a multi-head
// self-attention with biased projections. In inference (torch/nn/modules/
// transformer.py, TransformerEncoderLayer.forward) it computes
//
//	x = x + self_attn(norm1(x), key_padding_mask=~attention_mask)
//	x = x + linear2(relu(linear1(norm2(x))))
//
// The dropouts are inference's no-ops and are not modelled.
//
// torch runs this through one of two paths. With an even head count, as the
// checkpoints' 16 and 12 heads are, eval() and no_grad() send it through the
// fused torch._transformer_encoder_layer_fwd; with an odd one it takes the
// Python path above. Both compute the same function, within float32 rounding
// (2e-6 apart on the fixture's encoder_layer case), and this is that function.
//
// The zero Layer has no weights; its Forward returns an error rather than a
// plausible zero.
type Layer struct {
	w        LayerWeights
	d, nhead int
	headDim  int
}

// LayerNormEps is the eps of every LayerNorm in the head:
// nn.TransformerEncoderLayer's default layer_norm_eps and nn.LayerNorm's
// default eps, neither of which DecisionModel overrides (common.py:97, 100).
const LayerNormEps float32 = 1e-5

// NewLayer returns the layer with the given weights split into nhead heads.
// The Layer keeps the tensors; the caller must not modify them. A nil
// tensor, a shape that does not fit the d InProjWeight fixes, or a head count
// that does not divide d is an error. DecisionModel's own head count is
// max(1, d // 64), which New derives; NewLayer takes it explicitly so that a
// layer of any head count can be checked against torch.
func NewLayer(w LayerWeights, nhead int) (Layer, error) {
	if w.InProjWeight == nil || w.InProjWeight.Rank() != 2 || w.InProjWeight.Shape()[1] == 0 {
		return Layer{}, fmt.Errorf("head: layer in_proj_weight is %s, want [3d, d]", shapeOf(w.InProjWeight))
	}
	d := w.InProjWeight.Shape()[1]
	if nhead <= 0 || d%int64(nhead) != 0 {
		return Layer{}, fmt.Errorf("head: %d heads do not divide d = %d", nhead, d)
	}
	ff := 4 * d
	for _, c := range []struct {
		name  string
		t     *tensor.Tensor
		shape []int64
	}{
		{"in_proj_weight", w.InProjWeight, []int64{3 * d, d}},
		{"in_proj_bias", w.InProjBias, []int64{3 * d}},
		{"out_proj.weight", w.OutProjWeight, []int64{d, d}},
		{"out_proj.bias", w.OutProjBias, []int64{d}},
		{"linear1.weight", w.Linear1Weight, []int64{ff, d}},
		{"linear1.bias", w.Linear1Bias, []int64{ff}},
		{"linear2.weight", w.Linear2Weight, []int64{d, ff}},
		{"linear2.bias", w.Linear2Bias, []int64{d}},
		{"norm1.weight", w.Norm1Weight, []int64{d}},
		{"norm1.bias", w.Norm1Bias, []int64{d}},
		{"norm2.weight", w.Norm2Weight, []int64{d}},
		{"norm2.bias", w.Norm2Bias, []int64{d}},
	} {
		if err := checkShape("layer "+c.name, c.t, c.shape); err != nil {
			return Layer{}, err
		}
	}
	return Layer{w: w, d: int(d), nhead: nhead, headDim: int(d) / nhead}, nil
}

// Forward applies the layer to x [B, S, d] with the collator's attention mask
// [B][S], nonzero over real tokens and 0 over padding; upstream passes its negation
// as src_key_padding_mask (common.py:111). It returns a new tensor [B, S, d]
// that the caller owns.
//
// The mask drops padded keys only: a padded query still attends to the real
// keys, as in torch on either path, and its row is computed but never read by
// the head. A row with no real token is an error. The collator cannot produce
// one, since every row holds [CLS], and torch's fast path returns NaN for it.
func (l Layer) Forward(x *tensor.Tensor, attention [][]int64) (*tensor.Tensor, error) {
	if l.w.InProjWeight == nil {
		return nil, errors.New("head: uninitialized layer; build it with NewLayer")
	}
	if x == nil {
		return nil, errors.New("head: layer input is nil")
	}
	shape := x.Shape()
	if len(shape) != 3 || shape[2] != int64(l.d) {
		return nil, fmt.Errorf("head: layer input has shape %v, want [batch, seq, %d]", shape, l.d)
	}
	if err := checkAttention(attention, int(shape[0]), int(shape[1])); err != nil {
		return nil, err
	}

	n1, err := tensor.LayerNorm(x, l.w.Norm1Weight, l.w.Norm1Bias, LayerNormEps)
	if err != nil {
		return nil, fmt.Errorf("head: norm1: %w", err)
	}
	sa, err := l.selfAttention(n1, attention)
	if err != nil {
		return nil, err
	}
	x1 := addInto(sa, x) // x + SA(norm1(x))

	n2, err := tensor.LayerNorm(x1, l.w.Norm2Weight, l.w.Norm2Bias, LayerNormEps)
	if err != nil {
		return nil, fmt.Errorf("head: norm2: %w", err)
	}
	hidden, err := tensor.Linear(n2, l.w.Linear1Weight, l.w.Linear1Bias)
	if err != nil {
		return nil, fmt.Errorf("head: linear1: %w", err)
	}
	// nn.TransformerEncoderLayer's default activation, F.relu: the head is
	// ReLU where the encoder body is GeGLU (docs/ARCHITECTURE.md §1.3).
	for i, v := range hidden.RawData() {
		hidden.RawData()[i] = max(v, 0)
	}
	ff, err := tensor.Linear(hidden, l.w.Linear2Weight, l.w.Linear2Bias)
	if err != nil {
		return nil, fmt.Errorf("head: linear2: %w", err)
	}
	return addInto(ff, x1), nil // x + FF(norm2(x))
}

// selfAttention is self_attn as _sa_block calls it, with x as query, key and
// value, the key_padding_mask and need_weights=False:
//
//	q, k, v = linear(x, in_proj_weight, in_proj_bias).chunk(3, -1)
//	split each into nhead heads of head_dim = d / nhead, head-major
//	softmax(q k^T / sqrt(head_dim), masking padded keys) v
//	out_proj(the heads concatenated)
//
// It returns a new tensor [B, S, d].
func (l Layer) selfAttention(x *tensor.Tensor, attention [][]int64) (*tensor.Tensor, error) {
	qkv, err := tensor.Linear(x, l.w.InProjWeight, l.w.InProjBias) // [B, S, 3d]
	if err != nil {
		return nil, fmt.Errorf("head: in_proj: %w", err)
	}
	shape := x.Shape()
	merged, err := tensor.Zeros(shape)
	if err != nil {
		return nil, fmt.Errorf("head: %w", err)
	}

	seq, d, hd := int(shape[1]), l.d, l.headDim
	scale := float32(1 / math.Sqrt(float64(hd)))
	q := make([]float32, seq*hd)
	k := make([]float32, seq*hd)
	vT := make([]float32, hd*seq)
	scores := make([]float32, seq*seq)
	out := make([]float32, seq*hd)
	src, dst := qkv.RawData(), merged.RawData()
	for b, mask := range attention {
		for head := range l.nhead {
			// Token s's row of qkv is q, then k, then v, each d wide; head
			// h owns columns h*hd to (h+1)*hd of each.
			for s := range seq {
				row := src[(b*seq+s)*3*d:]
				copy(q[s*hd:(s+1)*hd], row[head*hd:(head+1)*hd])
				copy(k[s*hd:(s+1)*hd], row[d+head*hd:d+(head+1)*hd])
				for j := range hd {
					vT[j*seq+s] = row[2*d+head*hd+j]
				}
			}
			tensor.MatMulTransB(scores, seq, q, k, seq, seq, hd)
			for qi := range seq {
				softmaxKeys(scores[qi*seq:(qi+1)*seq], mask, scale)
			}
			tensor.MatMulTransB(out, hd, scores, vT, seq, hd, seq)
			for s := range seq {
				copy(dst[(b*seq+s)*d+head*hd:], out[s*hd:(s+1)*hd])
			}
		}
	}

	y, err := tensor.Linear(merged, l.w.OutProjWeight, l.w.OutProjBias)
	if err != nil {
		return nil, fmt.Errorf("head: out_proj: %w", err)
	}
	return y, nil
}

// softmaxKeys turns one query's raw scores q·k into attention probabilities
// in place: scaled, padded keys dropped, normalized. checkAttention has made
// sure the row has a real key.
func softmaxKeys(row []float32, mask []int64, scale float32) {
	top := float32(math.Inf(-1))
	for k := range row {
		if mask[k] != 0 {
			row[k] *= scale
			top = max(top, row[k])
		}
	}
	var total float32
	for k := range row {
		if mask[k] == 0 {
			row[k] = 0
			continue
		}
		row[k] = float32(math.Exp(float64(row[k] - top)))
		total += row[k]
	}
	for k := range row {
		row[k] /= total
	}
}

// addInto adds x to y element-wise in place and returns y: the residual.
func addInto(y, x *tensor.Tensor) *tensor.Tensor {
	dst, src := y.RawData(), x.RawData()
	for i := range dst {
		dst[i] += src[i]
	}
	return y
}

// checkAttention requires a [batch][seq] mask with a real token in every row.
// Like upstream's ~attention_mask.bool() (common.py:111), any nonzero value
// marks a real token, so a value is never rejected for not being 1.
func checkAttention(attention [][]int64, batch, seq int) error {
	if len(attention) != batch {
		return fmt.Errorf("head: attention mask has %d rows for a batch of %d", len(attention), batch)
	}
	for b, row := range attention {
		if len(row) != seq {
			return fmt.Errorf("head: attention mask row %d has %d entries for a sequence of %d", b, len(row), seq)
		}
		if !slices.ContainsFunc(row, func(m int64) bool { return m != 0 }) {
			return fmt.Errorf("head: attention mask row %d has no real token", b)
		}
	}
	return nil
}

// checkShape requires t to be non-nil and of exactly the given shape.
func checkShape(name string, t *tensor.Tensor, want []int64) error {
	if t == nil || !slices.Equal(t.Shape(), want) {
		return fmt.Errorf("head: %s is %s, want %v", name, shapeOf(t), want)
	}
	return nil
}

func shapeOf(t *tensor.Tensor) string {
	if t == nil {
		return "nil"
	}
	return fmt.Sprint(t.Shape())
}
