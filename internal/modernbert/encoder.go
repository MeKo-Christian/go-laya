package modernbert

import (
	"errors"
	"fmt"
	"slices"

	"github.com/MeKo-Christian/go-laya/internal/runtime/tensor"
)

// Layer is ModernBertEncoderLayer, one pre-norm encoder layer
// (modeling_modernbert.py:304-334):
//
//	attn_output, _ = self.attn(self.attn_norm(hidden_states), ...)
//	hidden_states = hidden_states + attn_output
//	hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
//
// Both residuals add the un-normed input of their sub-block. attn_norm is the
// identity on layer 0 only (Encoder enforces where it may be); mlp_norm is
// always a LayerNorm.
//
// The zero Layer has no blocks; its Forward returns an error.
type Layer struct {
	attnNorm Norm
	attn     Attention
	mlpNorm  Norm
	mlp      MLP
}

// NewLayer returns the encoder layer made of the given blocks. The attention
// decides the layer type. Blocks the zero value of their type, an identity
// mlp_norm, or blocks whose hidden sizes disagree are an error.
func NewLayer(attnNorm Norm, attn Attention, mlpNorm Norm, mlp MLP) (Layer, error) {
	if attn.wqkv == nil {
		return Layer{}, errors.New("modernbert: layer has an uninitialized attention")
	}
	if mlp.wi == nil {
		return Layer{}, errors.New("modernbert: layer has an uninitialized MLP")
	}
	hidden := int64(attn.hidden)
	if mlp.hidden != hidden {
		return Layer{}, fmt.Errorf("modernbert: layer MLP has hidden size %d, its attention %d", mlp.hidden, hidden)
	}
	if err := checkNorm(attnNorm, "attn_norm", hidden, true); err != nil {
		return Layer{}, err
	}
	if err := checkNorm(mlpNorm, "mlp_norm", hidden, false); err != nil {
		return Layer{}, err
	}
	return Layer{attnNorm: attnNorm, attn: attn, mlpNorm: mlpNorm, mlp: mlp}, nil
}

// checkNorm requires n to be a LayerNorm over hidden, or, where identity
// allows it, the identity.
func checkNorm(n Norm, what string, hidden int64, identity bool) error {
	switch {
	case n.identity && identity:
		return nil
	case n.identity:
		return fmt.Errorf("modernbert: %s is the identity; it is a LayerNorm upstream", what)
	case n.weight == nil:
		return fmt.Errorf("modernbert: %s is uninitialized", what)
	case n.weight.Rank() != 1 || n.weight.Shape()[0] != hidden:
		return fmt.Errorf("modernbert: %s weight has shape %v, want [%d]", what, n.weight.Shape(), hidden)
	}
	return nil
}

// Forward applies the layer to x [B, S, H] with the padding mask [B][S], as
// Attention.Forward takes it. It returns a new tensor [B, S, H] that the
// caller owns.
func (l Layer) Forward(x *tensor.Tensor, padding [][]int64) (*tensor.Tensor, error) {
	if l.attn.wqkv == nil {
		return nil, errors.New("modernbert: uninitialized layer; build it with NewLayer")
	}
	normed, err := l.attnNorm.Forward(x)
	if err != nil {
		return nil, err
	}
	h, err := l.attn.Forward(normed, padding)
	if err != nil {
		return nil, err
	}
	addInto(h, x)
	if normed, err = l.mlpNorm.Forward(h); err != nil {
		return nil, err
	}
	y, err := l.mlp.Forward(normed)
	if err != nil {
		return nil, err
	}
	addInto(y, h)
	return y, nil
}

// addInto adds src to dst in place, element by element: a residual, which
// torch adds in float32 as here.
func addInto(dst, src *tensor.Tensor) {
	d := dst.RawData()
	for i, v := range src.RawData() {
		d[i] += v
	}
}

// Encoder is ModernBertModel as the reference runs it, at inference
// (modeling_modernbert.py:412-478): the token embeddings, the embeddings norm,
// the layers in layer_types order, then final_norm. Its output is
// last_hidden_state.
//
//	hidden_states = self.drop(self.norm(self.tok_embeddings(input_ids)))
//	for encoder_layer in self.layers:
//	    hidden_states = encoder_layer(hidden_states,
//	        attention_mask=attention_mask_mapping[encoder_layer.attention_type],
//	        position_embeddings=position_embeddings[encoder_layer.attention_type])
//	hidden_states = self.final_norm(hidden_states)
//
// Each layer's type picks both its mask and its RoPE theta; here its Attention
// carries the type (LayerType), and positions are 0..S-1, as transformers
// defaults position_ids. The dropout is inference's no-op and is not
// modelled. There is no positional embedding tensor: RoPE is the only
// position signal.
//
// The zero Encoder has no weights; its Forward returns an error.
type Encoder struct {
	embeddings *tensor.Tensor // tok_embeddings [V, H]
	embNorm    Norm
	layers     []Layer
	finalNorm  Norm
	vocab      int64
	hidden     int64
}

// NewEncoder returns the encoder with the token embeddings [V, H], the
// embeddings norm, the layers and final_norm, where types is the layer plan
// (Config.Layers) that each layer's attention must have been built with, in
// order. The Encoder keeps the weights; the caller must not modify them.
//
// These are errors: embeddings that are not a non-empty [V, H]; a norm that is
// not a LayerNorm over H; a layer count that is not the plan's; a layer whose
// attention has another type than the plan says, or another hidden size; and
// an attn_norm that is the identity anywhere but on layer 0 or a LayerNorm on
// layer 0, where upstream it is nn.Identity() (modeling_modernbert.py:309-312).
func NewEncoder(embeddings *tensor.Tensor, embNorm Norm, layers []Layer, finalNorm Norm, types []LayerType) (Encoder, error) {
	if embeddings == nil || embeddings.Rank() != 2 || embeddings.ElemCount() == 0 {
		var shape []int64
		if embeddings != nil {
			shape = embeddings.Shape()
		}
		return Encoder{}, fmt.Errorf("modernbert: token embeddings have shape %v, want [vocab, hidden]", shape)
	}
	vocab, hidden := embeddings.Shape()[0], embeddings.Shape()[1]
	if err := checkNorm(embNorm, "embeddings norm", hidden, false); err != nil {
		return Encoder{}, err
	}
	if err := checkNorm(finalNorm, "final norm", hidden, false); err != nil {
		return Encoder{}, err
	}
	if len(layers) != len(types) {
		return Encoder{}, fmt.Errorf("modernbert: %d layers for %d layer types", len(layers), len(types))
	}
	for i, l := range layers {
		if l.attn.wqkv == nil {
			return Encoder{}, fmt.Errorf("modernbert: layer %d is uninitialized; build it with NewLayer", i)
		}
		if int64(l.attn.hidden) != hidden {
			return Encoder{}, fmt.Errorf("modernbert: layer %d has hidden size %d, the embeddings %d", i, l.attn.hidden, hidden)
		}
		if l.attn.layer != types[i] {
			return Encoder{}, fmt.Errorf("modernbert: layer %d's attention is %+v, the layer types say %+v", i, l.attn.layer, types[i])
		}
		if l.attnNorm.identity != (i == 0) {
			return Encoder{}, fmt.Errorf("modernbert: layer %d's attn_norm identity is %v; only layer 0's is nn.Identity",
				i, l.attnNorm.identity)
		}
	}
	return Encoder{
		embeddings: embeddings,
		embNorm:    embNorm,
		layers:     slices.Clone(layers),
		finalNorm:  finalNorm,
		vocab:      vocab,
		hidden:     hidden,
	}, nil
}

// Forward runs the encoder on a right- or left-padded batch: the token ids
// [B][S] and the tokenizer's attention mask [B][S], 1 over real tokens and 0
// over padding, the types of backend.Batch. It returns last_hidden_state
// [B, S, H], a new tensor the caller owns.
//
// Every position gets an output, padded ones included, as in torch: a padded
// query attends to the real keys its layer allows, and one with none left on
// a sliding layer gets zeros from the attention (Attention.Forward) but still
// carries its residual stream through the MLPs and the norms. An empty batch
// or sequence, ragged rows, an id outside [0, V), and a mask of another shape
// or with values other than 0 and 1 are errors.
func (e Encoder) Forward(inputIDs, attentionMask [][]int64) (*tensor.Tensor, error) {
	if e.embeddings == nil {
		return nil, errors.New("modernbert: uninitialized encoder; build it with NewEncoder")
	}
	h, err := e.embed(inputIDs)
	if err != nil {
		return nil, err
	}
	if err := checkPadding(attentionMask, len(inputIDs), len(inputIDs[0])); err != nil {
		return nil, err
	}
	if h, err = e.embNorm.Forward(h); err != nil {
		return nil, err
	}
	for i, l := range e.layers {
		if h, err = l.Forward(h, attentionMask); err != nil {
			return nil, fmt.Errorf("modernbert: layer %d: %w", i, err)
		}
	}
	return e.finalNorm.Forward(h)
}

// embed looks up each token's row of tok_embeddings, giving [B, S, H].
func (e Encoder) embed(inputIDs [][]int64) (*tensor.Tensor, error) {
	if len(inputIDs) == 0 || len(inputIDs[0]) == 0 {
		return nil, errors.New("modernbert: empty batch or sequence")
	}
	batch, seq, hidden := len(inputIDs), len(inputIDs[0]), int(e.hidden)
	out, err := tensor.Zeros([]int64{int64(batch), int64(seq), e.hidden})
	if err != nil {
		return nil, fmt.Errorf("modernbert: %w", err)
	}
	table, dst := e.embeddings.RawData(), out.RawData()
	for b, row := range inputIDs {
		if len(row) != seq {
			return nil, fmt.Errorf("modernbert: input ids row %d has %d tokens, row 0 %d", b, len(row), seq)
		}
		for s, id := range row {
			if id < 0 || id >= e.vocab {
				return nil, fmt.Errorf("modernbert: input id [%d][%d] = %d is outside the vocabulary [0, %d)", b, s, id, e.vocab)
			}
			copy(dst[(b*seq+s)*hidden:(b*seq+s+1)*hidden], table[int(id)*hidden:(int(id)+1)*hidden])
		}
	}
	return out, nil
}
