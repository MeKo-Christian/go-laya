package native

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/MeKo-Christian/go-laya/backend"
	"github.com/MeKo-Christian/go-laya/internal/head"
	"github.com/MeKo-Christian/go-laya/internal/modernbert"
	"github.com/MeKo-Christian/go-laya/internal/runtime/tensor"
)

// The tiny checkpoint's dimensions. Two encoder heads keep head_dim even for
// RoPE; the decision head gets max(1, 8 // 64) = 1. A window of 2 (local
// attention 4) is far shorter than tinyBatch's 10 tokens, so the sliding
// layers see less than the full ones.
const (
	tinyHidden     = 8
	tinyHeads      = 2
	tinyInter      = 6
	tinyVocab      = 16
	tinyLocal      = 4
	tinyHeadLayers = 2
	tinyAct        = 2
	// Not ModernBertConfig's defaults (160000 and 10000), so a loader that
	// dropped rope_parameters would give other outputs.
	tinyFullTheta  = 1000.0
	tinySlideTheta = 50.0
)

// variant is how assemble departs from the checkpoint's own model, for
// tests that show the batch can tell the departure apart.
type variant struct {
	// allFull makes every layer full attention, the layer plan ignored.
	allFull bool
	// defaultThetas uses ModernBertConfig's default thetas, 160000 and
	// 10000, rope_parameters ignored.
	defaultThetas bool
}

// tinyLayerTypes is the encoder's plan: both layer types, and a full layer
// after two sliding ones, as the checkpoints have.
var tinyLayerTypes = []string{"full_attention", "sliding_attention", "sliding_attention", "full_attention"}

// stTensor is one tensor of a synthetic safetensors file. data holds the
// values of an F32 tensor; raw, when set, is written as it is, under dtype.
type stTensor struct {
	dtype string
	shape []int64
	data  []float32
	raw   []byte
}

// synth is a laya checkpoint small enough to write in a test: every tensor
// DecisionModel's state_dict has, as F32 except temperature, and the encoder's
// config.json.
type synth struct {
	tensors map[string]stTensor
	config  map[string]any
}

// newSynth returns the tiny checkpoint with weights drawn from seed.
func newSynth(seed uint64) *synth {
	rng := rand.New(rand.NewPCG(seed, 0))
	s := &synth{tensors: map[string]stTensor{}, config: map[string]any{
		"model_type":          "modernbert",
		"dtype":               "float32",
		"torch_dtype":         nil,
		"attention_bias":      false,
		"mlp_bias":            false,
		"norm_bias":           false,
		"hidden_size":         tinyHidden,
		"num_hidden_layers":   len(tinyLayerTypes),
		"num_attention_heads": tinyHeads,
		"intermediate_size":   tinyInter,
		"vocab_size":          tinyVocab,
		"local_attention":     tinyLocal,
		"layer_types":         tinyLayerTypes,
		"norm_eps":            1e-5,
		"hidden_activation":   "gelu",
		"rope_parameters": map[string]any{
			"full_attention":    map[string]any{"rope_theta": tinyFullTheta, "rope_type": "default"},
			"sliding_attention": map[string]any{"rope_theta": tinySlideTheta, "rope_type": "default"},
		},
	}}
	// linear draws a weight with the scale nn.Linear's init has; norm one
	// near 1, as a trained LayerNorm's is.
	linear := func(name string, shape ...int64) {
		s.set(name, shape, rng, 1/math.Sqrt(float64(shape[len(shape)-1])), 0)
	}
	norm := func(name string) { s.set(name, []int64{tinyHidden}, rng, 0.1, 1) }
	bias := func(name string, n int64) { s.set(name, []int64{n}, rng, 0.1, 0) }

	const d, ff = tinyHidden, 4 * tinyHidden
	s.set("encoder.embeddings.tok_embeddings.weight", []int64{tinyVocab, d}, rng, 1, 0)
	norm("encoder.embeddings.norm.weight")
	for i := range tinyLayerTypes {
		l := func(rest string) string { return fmt.Sprintf("encoder.layers.%d.%s", i, rest) }
		if i > 0 {
			norm(l("attn_norm.weight"))
		}
		linear(l("attn.Wqkv.weight"), 3*d, d)
		linear(l("attn.Wo.weight"), d, d)
		norm(l("mlp_norm.weight"))
		linear(l("mlp.Wi.weight"), 2*tinyInter, d)
		linear(l("mlp.Wo.weight"), d, tinyInter)
	}
	norm("encoder.final_norm.weight")

	s.set("type_emb.weight", []int64{3, d}, rng, 1, 0)
	for i := range tinyHeadLayers {
		l := func(rest string) string { return fmt.Sprintf("head.layers.%d.%s", i, rest) }
		linear(l("self_attn.in_proj_weight"), 3*d, d)
		bias(l("self_attn.in_proj_bias"), 3*d)
		linear(l("self_attn.out_proj.weight"), d, d)
		bias(l("self_attn.out_proj.bias"), d)
		linear(l("linear1.weight"), ff, d)
		bias(l("linear1.bias"), ff)
		linear(l("linear2.weight"), d, ff)
		bias(l("linear2.bias"), d)
		for _, n := range []string{"norm1", "norm2"} {
			norm(l(n + ".weight"))
			bias(l(n+".bias"), d)
		}
	}
	norm("scorer.0.weight")
	bias("scorer.0.bias", d)
	linear("scorer.1.weight", d, d)
	bias("scorer.1.bias", d)
	linear("scorer.3.weight", 1, d)
	bias("scorer.3.bias", 1)
	linear("act_head.0.weight", head.ActHidden, d+4)
	bias("act_head.0.bias", head.ActHidden)
	linear("act_head.2.weight", tinyAct, head.ActHidden)
	bias("act_head.2.bias", tinyAct)
	s.tensors["temperature"] = stTensor{dtype: "F32", shape: []int64{3}, data: []float32{1.6, 1.25, 1.98}}
	return s
}

// set draws an F32 tensor of the given shape as offset + scale·N(0, 1).
func (s *synth) set(name string, shape []int64, rng *rand.Rand, scale, offset float64) {
	n := int64(1)
	for _, d := range shape {
		n *= d
	}
	data := make([]float32, n)
	for i := range data {
		data[i] = float32(offset + scale*rng.NormFloat64())
	}
	s.tensors[name] = stTensor{dtype: "F32", shape: shape, data: data}
}

// zeros sets an F32 tensor of zeros, for tensors a test adds.
func (s *synth) zeros(name string, shape ...int64) {
	n := int64(1)
	for _, d := range shape {
		n *= d
	}
	s.tensors[name] = stTensor{dtype: "F32", shape: shape, data: make([]float32, n)}
}

// write puts the checkpoint into a new directory as model.safetensors and
// encoder/config.json, and returns the directory.
func (s *synth) write(t testing.TB) string {
	t.Helper()
	dir := t.TempDir()
	if err := os.Mkdir(filepath.Join(dir, "encoder"), 0o755); err != nil {
		t.Fatal(err)
	}
	cfg, err := json.Marshal(s.config)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "encoder", "config.json"), cfg, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), s.encode(t), 0o600); err != nil {
		t.Fatal(err)
	}
	return dir
}

// encode lays the tensors out as a safetensors file, in name order, the way
// internal/safetensors' own tests build theirs.
func (s *synth) encode(t testing.TB) []byte {
	t.Helper()
	names := make([]string, 0, len(s.tensors))
	for name := range s.tensors {
		names = append(names, name)
	}
	slices.Sort(names)

	header := make(map[string]any, len(names))
	var data []byte
	for _, name := range names {
		st := s.tensors[name]
		raw := st.raw
		if raw == nil {
			for _, v := range st.data {
				raw = binary.LittleEndian.AppendUint32(raw, math.Float32bits(v))
			}
		}
		header[name] = map[string]any{
			"dtype": st.dtype, "shape": st.shape,
			"data_offsets": []int{len(data), len(data) + len(raw)},
		}
		data = append(data, raw...)
	}
	raw, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	b := binary.LittleEndian.AppendUint64(nil, uint64(len(raw)))
	b = append(b, raw...)
	return append(b, data...)
}

// tensor returns the named tensor as the test's own reference mapping reads
// it.
func (s *synth) tensor(t testing.TB, name string) *tensor.Tensor {
	t.Helper()
	st, ok := s.tensors[name]
	if !ok {
		t.Fatalf("synthetic checkpoint has no %s", name)
	}
	x, err := tensor.New(st.data, st.shape)
	if err != nil {
		t.Fatal(err)
	}
	return x
}

// assemble builds the encoder and the head straight from the tensors, by the
// mapping docs/ARCHITECTURE.md §1.2-1.3 gives, independently of Open, as v
// departs from it.
func (s *synth) assemble(t testing.TB, v variant) (modernbert.Encoder, *head.Head) {
	t.Helper()
	norm := func(name string) modernbert.Norm {
		n, err := modernbert.NewNorm(s.tensor(t, name))
		if err != nil {
			t.Fatal(err)
		}
		return n
	}
	types := make([]modernbert.LayerType, len(tinyLayerTypes))
	layers := make([]modernbert.Layer, len(tinyLayerTypes))
	for i, typ := range tinyLayerTypes {
		l := func(rest string) string { return fmt.Sprintf("encoder.layers.%d.%s", i, rest) }
		full, slide := tinyFullTheta, tinySlideTheta
		if v.defaultThetas {
			full, slide = 160000, 10000
		}
		types[i] = modernbert.FullAttention(full)
		if typ == "sliding_attention" && !v.allFull {
			types[i] = modernbert.SlidingAttention(slide, tinyLocal)
		}
		attnNorm := modernbert.IdentityNorm()
		if i > 0 {
			attnNorm = norm(l("attn_norm.weight"))
		}
		attn, err := modernbert.NewAttention(s.tensor(t, l("attn.Wqkv.weight")), s.tensor(t, l("attn.Wo.weight")), tinyHeads, types[i])
		if err != nil {
			t.Fatal(err)
		}
		mlp, err := modernbert.NewMLP(s.tensor(t, l("mlp.Wi.weight")), s.tensor(t, l("mlp.Wo.weight")))
		if err != nil {
			t.Fatal(err)
		}
		if layers[i], err = modernbert.NewLayer(attnNorm, attn, norm(l("mlp_norm.weight")), mlp); err != nil {
			t.Fatal(err)
		}
	}
	enc, err := modernbert.NewEncoder(s.tensor(t, "encoder.embeddings.tok_embeddings.weight"),
		norm("encoder.embeddings.norm.weight"), layers, norm("encoder.final_norm.weight"), types)
	if err != nil {
		t.Fatal(err)
	}

	w := head.Weights{
		TypeEmb:             s.tensor(t, "type_emb.weight"),
		ScorerNormWeight:    s.tensor(t, "scorer.0.weight"),
		ScorerNormBias:      s.tensor(t, "scorer.0.bias"),
		ScorerLinear1Weight: s.tensor(t, "scorer.1.weight"),
		ScorerLinear1Bias:   s.tensor(t, "scorer.1.bias"),
		ScorerLinear2Weight: s.tensor(t, "scorer.3.weight"),
		ScorerLinear2Bias:   s.tensor(t, "scorer.3.bias"),
		ActLinear1Weight:    s.tensor(t, "act_head.0.weight"),
		ActLinear1Bias:      s.tensor(t, "act_head.0.bias"),
		ActLinear2Weight:    s.tensor(t, "act_head.2.weight"),
		ActLinear2Bias:      s.tensor(t, "act_head.2.bias"),
	}
	for i := range tinyHeadLayers {
		l := func(rest string) *tensor.Tensor { return s.tensor(t, fmt.Sprintf("head.layers.%d.%s", i, rest)) }
		w.Layers = append(w.Layers, head.LayerWeights{
			InProjWeight: l("self_attn.in_proj_weight"), InProjBias: l("self_attn.in_proj_bias"),
			OutProjWeight: l("self_attn.out_proj.weight"), OutProjBias: l("self_attn.out_proj.bias"),
			Linear1Weight: l("linear1.weight"), Linear1Bias: l("linear1.bias"),
			Linear2Weight: l("linear2.weight"), Linear2Bias: l("linear2.bias"),
			Norm1Weight: l("norm1.weight"), Norm1Bias: l("norm1.bias"),
			Norm2Weight: l("norm2.weight"), Norm2Bias: l("norm2.bias"),
		})
	}
	h, err := head.New(w)
	if err != nil {
		t.Fatal(err)
	}
	return enc, h
}

// tinyBatch is two questions of 10 tokens: a choice question with three
// options and a padded noul question with two, so kmax 3 has a fill column.
func tinyBatch() backend.Batch {
	return backend.Batch{
		InputIDs:      [][]int64{{1, 5, 7, 2, 9, 3, 11, 4, 6, 2}, {1, 8, 3, 2, 13, 2, 15, 0, 0, 0}},
		AttentionMask: [][]int64{{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 0, 0, 0}},
		MarkerPos:     [][]int64{{3, 6, 9}, {3, 5, 0}},
		MarkerMask:    [][]bool{{true, true, true}, {true, true, false}},
		QType:         []int64{0, 2},
	}
}
