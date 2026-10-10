package modernbert

import (
	"errors"
	"fmt"
	"math"

	"github.com/MeKo-Christian/go-laya/internal/runtime/tensor"
)

// LayerType is what config.layer_types decides for one encoder layer: the
// RoPE theta its queries and keys rotate at, and whether its attention is
// windowed. Both checkpoints make every third layer, from layer 0, full
// attention and the rest sliding, with local_attention 128; ModernBERT-large
// has theta 160000 on full layers and 10000 on sliding ones, mmBERT-base
// 160000 on both (docs/ARCHITECTURE.md §1.2).
type LayerType struct {
	// RopeTheta is rope_parameters[layer_type].rope_theta. Only rope_type
	// "default" exists in either checkpoint, whose attention scaling is 1.
	RopeTheta float64
	// Sliding is true for "sliding_attention".
	Sliding bool
	// Window is, for a sliding layer, the largest |q - k| a query may attend
	// across: config.sliding_window, which is local_attention // 2, so 64.
	// It is the distance the sdpa mask applies (masking_utils.py,
	// sliding_window_bidirectional_overlay: abs(q_idx - kv_idx) <=
	// sliding_window). ModernBertAttention stores sliding_window + 1 for flash
	// attention's bounds, but sdpa never reads it.
	Window int
}

// FullAttention returns the LayerType of a "full_attention" layer.
func FullAttention(ropeTheta float64) LayerType {
	return LayerType{RopeTheta: ropeTheta}
}

// SlidingAttention returns the LayerType of a "sliding_attention" layer for
// config.json's local_attention, the window's full width: a query sees the
// keys within local_attention // 2 of it on either side.
func SlidingAttention(ropeTheta float64, localAttention int) LayerType {
	return LayerType{RopeTheta: ropeTheta, Sliding: true, Window: localAttention / 2}
}

// allows reports whether query q may attend to key k in a row whose padding
// mask is pad: the key is a real token and, on a sliding layer, within the
// window. The padding mask masks keys only, so a padded query still attends.
func (l LayerType) allows(pad []int64, q, k int) bool {
	if pad[k] == 0 {
		return false
	}
	return !l.Sliding || absInt(q-k) <= l.Window
}

func absInt(n int) int {
	if n < 0 {
		return -n
	}
	return n
}

// Attention is ModernBertAttention, the self-attention of every encoder layer,
// as the reference runs it with attn_implementation "sdpa", padded
// (modeling_modernbert.py):
//
//	qkv = self.Wqkv(hidden_states).view(*input_shape, 3, -1, self.head_dim)
//	query, key, value = qkv.unbind(dim=-3)
//	query, key = apply_rotary_pos_emb(query, key, cos, sin)
//	sdpa(query, key, value, mask, scaling=self.head_dim**-0.5)
//	return self.Wo(attn_output.reshape(*input_shape, -1))
//
// Wqkv [3H, H] fuses the three projections: its rows are (3, heads, head_dim),
// q, k and v in that order and head-major within each. Wo is [H, H]. Neither
// has a bias, because both checkpoints set attention_bias false: Wqkv is
// [3072, 1024] with 16 heads in ModernBERT-large, [2304, 768] with 12 in
// mmBERT-base, head_dim 64 in both. The dropout is inference's no-op and is
// not modelled.
//
// The zero Attention has no weights; its Forward returns an error rather than
// a plausible zero.
type Attention struct {
	wqkv, wo      *tensor.Tensor
	hidden, heads int
	headDim       int
	layer         LayerType
}

// NewAttention returns the attention of a layer of the given type with the
// fused Wqkv [3H, H] and Wo [H, H], in transformers' [out, in] layout, split
// into heads heads. The Attention keeps both; the caller must not modify them.
// A Wqkv that is empty or whose rows are not 3H, a hidden size the heads do
// not divide, an odd head_dim (rotate_half needs halves), a Wo that is not
// [H, H], a theta that is not positive and finite, or a negative window is an
// error.
func NewAttention(wqkv, wo *tensor.Tensor, heads int, layer LayerType) (Attention, error) {
	if wqkv == nil || wo == nil {
		return Attention{}, errors.New("modernbert: attention weight is nil")
	}
	if wqkv.Rank() != 2 || wqkv.ElemCount() == 0 || wqkv.Shape()[0] != 3*wqkv.Shape()[1] {
		return Attention{}, fmt.Errorf("modernbert: attention Wqkv has shape %v, want [3*hidden, hidden]", wqkv.Shape())
	}
	hidden := int(wqkv.Shape()[1])
	if heads <= 0 || hidden%heads != 0 {
		return Attention{}, fmt.Errorf("modernbert: hidden size %d is not a multiple of %d heads", hidden, heads)
	}
	headDim := hidden / heads
	if headDim%2 != 0 {
		return Attention{}, fmt.Errorf("modernbert: head_dim %d is odd; rotate_half RoPE needs an even one", headDim)
	}
	if wo.Rank() != 2 || wo.Shape()[0] != int64(hidden) || wo.Shape()[1] != int64(hidden) {
		return Attention{}, fmt.Errorf("modernbert: attention Wo has shape %v, want [%d %d]", wo.Shape(), hidden, hidden)
	}
	if math.IsNaN(layer.RopeTheta) || layer.RopeTheta <= 0 || math.IsInf(layer.RopeTheta, 0) {
		return Attention{}, fmt.Errorf("modernbert: rope_theta %g, want a positive finite value", layer.RopeTheta)
	}
	if layer.Sliding && layer.Window < 0 {
		return Attention{}, fmt.Errorf("modernbert: sliding window %d is negative", layer.Window)
	}
	return Attention{wqkv: wqkv, wo: wo, hidden: hidden, heads: heads, headDim: headDim, layer: layer}, nil
}

// Forward applies the attention to x [B, S, H] with the tokenizer's padding
// mask [B][S], 1 over real tokens and 0 over padding, at positions 0..S-1. It
// returns a new tensor [B, S, H] that the caller owns.
//
// A query attends to the real keys its layer type allows (LayerType.allows).
// One that has none gets zeros: a padded query on a sliding layer whose
// padding reaches further than the window, or any query of a row that is all
// padding. That is what torch's sdpa returns for a fully masked row of a
// boolean mask, and what the reference therefore computes. (Eager attention
// would average every value instead, through its finite min-value mask.)
func (a Attention) Forward(x *tensor.Tensor, padding [][]int64) (*tensor.Tensor, error) {
	if a.wqkv == nil {
		return nil, errors.New("modernbert: uninitialized attention; build it with NewAttention")
	}
	if x == nil {
		return nil, errors.New("modernbert: attention input is nil")
	}
	shape := x.Shape()
	if len(shape) != 3 || shape[2] != int64(a.hidden) {
		return nil, fmt.Errorf("modernbert: attention input has shape %v, want [batch, seq, %d]", shape, a.hidden)
	}
	batch, seq := int(shape[0]), int(shape[1])
	if err := checkPadding(padding, batch, seq); err != nil {
		return nil, err
	}

	qkv, err := tensor.Linear(x, a.wqkv, nil) // [B, S, 3H]
	if err != nil {
		return nil, fmt.Errorf("modernbert: attention Wqkv: %w", err)
	}
	merged, err := tensor.Zeros(shape) // the heads' outputs, [B, S, H]
	if err != nil {
		return nil, fmt.Errorf("modernbert: %w", err)
	}

	h, d := a.hidden, a.headDim
	cos, sin := ropeTables(a.layer.RopeTheta, d, seq)
	scale := float32(1 / math.Sqrt(float64(d)))
	q := make([]float32, seq*d)
	k := make([]float32, seq*d)
	vT := make([]float32, d*seq)
	scores := make([]float32, seq*seq)
	out := make([]float32, seq*d)
	src, dst := qkv.RawData(), merged.RawData()
	for b, pad := range padding {
		for head := range a.heads {
			// Token s's row of qkv is (3, heads, head_dim): q at head*d, k
			// one H further, v two.
			for s := range seq {
				row := src[(b*seq+s)*3*h:]
				copy(q[s*d:(s+1)*d], row[head*d:(head+1)*d])
				copy(k[s*d:(s+1)*d], row[h+head*d:h+(head+1)*d])
				for j := range d {
					vT[j*seq+s] = row[2*h+head*d+j]
				}
			}
			rotate(q, cos, sin, d)
			rotate(k, cos, sin, d)

			tensor.MatMulTransB(scores, seq, q, k, seq, seq, d)
			for qi := range seq {
				softmaxRow(scores[qi*seq:(qi+1)*seq], a.layer, pad, qi, scale)
			}
			tensor.MatMulTransB(out, d, scores, vT, seq, d, seq)

			// transpose(1, 2).reshape(*input_shape, -1): head h's output
			// fills columns h*d to (h+1)*d of each token.
			for s := range seq {
				copy(dst[(b*seq+s)*h+head*d:], out[s*d:(s+1)*d])
			}
		}
	}

	y, err := tensor.Linear(merged, a.wo, nil)
	if err != nil {
		return nil, fmt.Errorf("modernbert: attention Wo: %w", err)
	}
	return y, nil
}

// softmaxRow turns query q's raw scores q·k into its attention probabilities
// in place: scaled, restricted to the keys the layer allows, and normalized.
// A row with no allowed key becomes zeros, as sdpa leaves it.
func softmaxRow(row []float32, layer LayerType, pad []int64, q int, scale float32) {
	top := float32(math.Inf(-1))
	for k := range row {
		if layer.allows(pad, q, k) {
			row[k] *= scale
			top = max(top, row[k])
		} else {
			row[k] = float32(math.Inf(-1))
		}
	}
	if math.IsInf(float64(top), -1) {
		clear(row)
		return
	}
	var total float32
	for k, v := range row {
		// exp(-Inf - top) is 0: a masked key drops out.
		row[k] = float32(math.Exp(float64(v - top)))
		total += row[k]
	}
	for k := range row {
		row[k] /= total
	}
}

// checkPadding requires a [batch][seq] mask of zeros and ones. transformers
// would read any non-zero value as a real token; a tokenizer never writes one,
// so anything else here is a caller's bug and an error.
func checkPadding(padding [][]int64, batch, seq int) error {
	if len(padding) != batch {
		return fmt.Errorf("modernbert: padding mask has %d rows for a batch of %d", len(padding), batch)
	}
	for b, row := range padding {
		if len(row) != seq {
			return fmt.Errorf("modernbert: padding mask row %d has %d entries for a sequence of %d", b, len(row), seq)
		}
		for s, m := range row {
			if m != 0 && m != 1 {
				return fmt.Errorf("modernbert: padding mask [%d][%d] = %d, want 0 or 1", b, s, m)
			}
		}
	}
	return nil
}

// ropeTables returns ModernBertRotaryEmbedding's cos and sin for one theta,
// each [seq, d] row-major, at positions 0..seq-1:
//
//	inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
//	freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
//	emb = torch.cat((freqs, freqs), dim=-1)
//	cos, sin = emb.cos() * attention_scaling, emb.sin() * attention_scaling
//
// with attention_scaling 1 for rope_type "default". torch computes all of it
// in float32, and so does this, step for step; only pow, cos and sin go
// through float64 and round back.
func ropeTables(theta float64, d, seq int) (cos, sin []float32) {
	half := d / 2
	invFreq := ropeInvFreq(theta, d)
	cos = make([]float32, seq*d)
	sin = make([]float32, seq*d)
	for s := range seq {
		for i, f := range invFreq {
			ang := float64(float32(s) * f)
			c, sn := float32(math.Cos(ang)), float32(math.Sin(ang))
			cos[s*d+i], cos[s*d+half+i] = c, c
			sin[s*d+i], sin[s*d+half+i] = sn, sn
		}
	}
	return cos, sin
}

// ropeInvFreq is compute_default_rope_parameters' inv_freq, theta^(-2i/d) for
// i < d/2, in float32 as torch computes it: the exponent a float32 quotient,
// the power of float32(theta) rounded to float32, then its float32 reciprocal.
// At head_dim 64 and both checkpoints' thetas this equals torch bit for bit
// (TestRopeMatchesTorch).
func ropeInvFreq(theta float64, d int) []float32 {
	invFreq := make([]float32, d/2)
	for i := range invFreq {
		exp := float32(2*i) / float32(d)
		invFreq[i] = 1 / float32(math.Pow(float64(float32(theta)), float64(exp)))
	}
	return invFreq
}

// rotate applies apply_rotary_pos_emb to x, [seq, d] row-major, in place:
//
//	x * cos + rotate_half(x) * sin,  rotate_half(x) = cat(-x[d/2:], x[:d/2])
//
// so element j pairs with j + d/2, not with its neighbour as in interleaved
// RoPE. The explicit float32 conversions keep each product rounded on its own,
// as torch rounds it, rather than fused into an FMA.
func rotate(x, cos, sin []float32, d int) {
	half := d / 2
	for off := 0; off < len(x); off += d {
		r := x[off : off+d]
		c, s := cos[off:off+d], sin[off:off+d]
		for j := range half {
			lo, hi := r[j], r[j+half]
			r[j] = float32(lo*c[j]) - float32(hi*s[j])
			r[j+half] = float32(hi*c[j+half]) + float32(lo*s[j+half])
		}
	}
}
