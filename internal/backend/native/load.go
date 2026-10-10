package native

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"github.com/MeKo-Christian/go-laya/backend"
	"github.com/MeKo-Christian/go-laya/internal/head"
	"github.com/MeKo-Christian/go-laya/internal/modernbert"
	"github.com/MeKo-Christian/go-laya/internal/runtime/tensor"
	"github.com/MeKo-Christian/go-laya/internal/safetensors"
)

const (
	// weightsFile is the checkpoint's state_dict (agent.py:149).
	weightsFile = "model.safetensors"

	// maxConfigSize caps how much of encoder/config.json is read, as
	// internal/checkpoint caps rl_agent_config.json. The shipped ones are
	// under 2 KiB.
	maxConfigSize = 1 << 20

	// temperature is the buffer DecisionModel registers as torch.ones(3)
	// (common.py:102). Inference never reads it; calibration comes from
	// rl_agent_config.json.
	temperature = "temperature"

	// The prefixes of the encoder's and the head's numbered layers.
	encoderLayers = "encoder.layers."
	headLayers    = "head.layers."
)

// ModernBertConfig's defaults for the keys modernbert.Config does not hold
// (configuration_modernbert.py:79-88).
const (
	defaultVocabSize        = 50368
	defaultIntermediateSize = 1152
	defaultNormEps          = 1e-5
	defaultHiddenActivation = "gelu"
	modelTypeModernBERT     = "modernbert"
)

// encoderConfig is encoder/config.json as far as the loader reads it: the
// layer plan modernbert.Config resolves, and the sizes that fix the shapes of
// tok_embeddings and the MLPs.
type encoderConfig struct {
	modernbert.Config

	// path is the file's, for errors.
	path string
	// plan is Config.Layers(), which load resolves only once the weights
	// file has bounded num_hidden_layers.
	plan         []modernbert.LayerType
	vocab, inter int64
}

// configError wraps err as an incompatible encoder/config.json at path.
func configError(path string, err error) error {
	return fmt.Errorf("native backend: %s: %w: %w", path, err, backend.ErrIncompatibleCheckpoint)
}

// readConfig reads and checks dir/encoder/config.json, all but its layer
// plan, which load resolves. Every failure wraps
// backend.ErrIncompatibleCheckpoint and names the file.
func readConfig(dir string) (encoderConfig, error) {
	path := filepath.Join(dir, "encoder", "config.json")
	fail := func(err error) error { return configError(path, err) }
	// #nosec G304 -- dir is the checkpoint directory the caller chose to
	// load; reading its config is the point.
	f, err := os.Open(path)
	if err != nil {
		return encoderConfig{}, fail(err)
	}
	defer f.Close()
	raw, err := io.ReadAll(io.LimitReader(f, maxConfigSize+1))
	if err != nil {
		return encoderConfig{}, fail(err)
	}
	if len(raw) > maxConfigSize {
		return encoderConfig{}, fail(fmt.Errorf("larger than %d bytes", maxConfigSize))
	}

	cfg := encoderConfig{path: path, vocab: defaultVocabSize, inter: defaultIntermediateSize}
	if cfg.Config, err = modernbert.ParseConfig(raw); err != nil {
		return encoderConfig{}, fail(err)
	}

	// The keys modernbert.Config does not hold, read by their exact names
	// as ParseConfig reads its own. ParseConfig has already required a JSON
	// object.
	var keys map[string]json.RawMessage
	if err := json.Unmarshal(raw, &keys); err != nil {
		return encoderConfig{}, fail(err)
	}
	if err := cfg.readSizes(keys); err != nil {
		return encoderConfig{}, fail(err)
	}
	if err := checkImplemented(keys); err != nil {
		return encoderConfig{}, fail(err)
	}
	return cfg, nil
}

// readSizes reads vocab_size and intermediate_size into cfg, in this order,
// and requires them, hidden_size and num_attention_heads to be positive.
func (cfg *encoderConfig) readSizes(keys map[string]json.RawMessage) error {
	for _, k := range []struct {
		name string
		dst  *int64
	}{
		{"vocab_size", &cfg.vocab},
		{"intermediate_size", &cfg.inter},
	} {
		if err := decodeKey(keys, k.name, k.dst); err != nil {
			return err
		}
	}
	for _, c := range []struct {
		name string
		v    int64
	}{
		{"hidden_size", int64(cfg.HiddenSize)},
		{"num_attention_heads", int64(cfg.NumAttentionHeads)},
		{"vocab_size", cfg.vocab},
		{"intermediate_size", cfg.inter},
	} {
		if c.v <= 0 {
			return fmt.Errorf("config %s %d, want a positive value", c.name, c.v)
		}
	}
	return nil
}

// decodeKey decodes the key name, when keys has it, into dst. null is an
// error: none of the keys it reads may be None.
func decodeKey(keys map[string]json.RawMessage, name string, dst any) error {
	v, ok := keys[name]
	if !ok {
		return nil
	}
	if string(v) == "null" {
		return fmt.Errorf("config %s is null", name)
	}
	if err := json.Unmarshal(v, dst); err != nil {
		return fmt.Errorf("config %s: %w", name, err)
	}
	return nil
}

// checkImplemented refuses a config whose model upstream would build
// differently from the encoder internal/modernbert implements, which would
// otherwise run and answer differently.
//
//   - model_type must be "modernbert": AutoConfig.from_pretrained dispatches
//     on it (common.py:133), and the encoder directory's name matches no
//     other.
//   - dtype and torch_dtype must be missing, null or "float32":
//     AutoModel.from_config builds the encoder in config.dtype
//     (modeling_utils.py:1407-1436), and this computes in float32 only.
//   - norm_eps must be 1e-5, which modernbert.Norm has built in.
//   - hidden_activation must be "gelu", the exact erf GELU of its MLP.
//   - attention_bias, mlp_bias and norm_bias must be missing, null or
//     false: with them set, upstream's strict load needs bias tensors the
//     encoder has no place for.
func checkImplemented(keys map[string]json.RawMessage) error {
	var modelType string
	if err := decodeKey(keys, "model_type", &modelType); err != nil {
		return err
	}
	if modelType != modelTypeModernBERT {
		return fmt.Errorf("config model_type %q; only %q is implemented", modelType, modelTypeModernBERT)
	}
	for _, name := range []string{"dtype", "torch_dtype"} {
		var dtype *string
		if err := json.Unmarshal(orNull(keys[name]), &dtype); err != nil {
			return fmt.Errorf("config %s: %w", name, err)
		}
		if dtype != nil && *dtype != "float32" {
			return fmt.Errorf("config %s %q; only float32 is implemented", name, *dtype)
		}
	}

	eps, act := defaultNormEps, defaultHiddenActivation
	if err := decodeKey(keys, "norm_eps", &eps); err != nil {
		return err
	}
	if err := decodeKey(keys, "hidden_activation", &act); err != nil {
		return err
	}
	if eps != defaultNormEps {
		return fmt.Errorf("config norm_eps %g; only %g is implemented", eps, defaultNormEps)
	}
	if act != defaultHiddenActivation {
		return fmt.Errorf("config hidden_activation %q; only %q is implemented", act, defaultHiddenActivation)
	}

	for _, name := range []string{"attention_bias", "mlp_bias", "norm_bias"} {
		var bias *bool
		if err := json.Unmarshal(orNull(keys[name]), &bias); err != nil {
			return fmt.Errorf("config %s: %w", name, err)
		}
		if bias != nil && *bias {
			return fmt.Errorf("config %s is true; the encoder has no biases", name)
		}
	}
	return nil
}

// orNull is v, or JSON null for a missing key.
func orNull(v json.RawMessage) json.RawMessage {
	if v == nil {
		return json.RawMessage("null")
	}
	return v
}

// load opens dir/model.safetensors, resolves cfg's layer plan, checks every
// tensor's name and shape against cfg, decodes them and builds the model.
func load(ctx context.Context, dir string, cfg encoderConfig) (*model, error) {
	path := filepath.Join(dir, weightsFile)
	f, err := safetensors.Open(path)
	if err != nil {
		return nil, fmt.Errorf("native backend: %w", err)
	}
	defer f.Close()

	// Config.Layers allocates per configured layer, and without layer_types
	// nothing but num_hidden_layers bounds that: a config.json asking for
	// 2^40 layers would exhaust memory. The file's own layer count, bounded
	// by its header, is compared first, so the plan is never longer than
	// the file.
	names := f.Names()
	nEnc, nHead := len(layerIndices(names, encoderLayers)), len(layerIndices(names, headLayers))
	if nEnc != cfg.NumHiddenLayers {
		return nil, fmt.Errorf("native backend: %s has %d encoder layers, but %s has num_hidden_layers %d: %w",
			path, nEnc, cfg.path, cfg.NumHiddenLayers, backend.ErrIncompatibleCheckpoint)
	}
	// checkNames lists every tensor the layers imply, a dozen per head
	// layer. A file that numbers more layers than it has tensors to fill
	// them is incompatible whatever else it holds, and refusing it here
	// keeps that list no longer than the file's own.
	if need := layerTensors(nEnc, nHead); need > len(names) {
		return nil, fmt.Errorf("native backend: %s numbers %d encoder layers and %d head layers, which alone need %d tensors, but it has %d: %w",
			path, nEnc, nHead, need, len(names), backend.ErrIncompatibleCheckpoint)
	}
	if cfg.plan, err = cfg.Layers(); err != nil {
		return nil, configError(cfg.path, err)
	}

	if err := checkNames(f, cfg, nHead); err != nil {
		return nil, fmt.Errorf("native backend: %s: %w: %w", path, err, backend.ErrIncompatibleCheckpoint)
	}

	weights := make(map[string]*tensor.Tensor, len(names))
	for _, name := range names {
		if name == temperature {
			continue
		}
		data, shape, err := f.Float32(ctx, name)
		if err != nil {
			return nil, err // ctx's error alone, or one naming the tensor
		}
		if weights[name], err = tensor.New(data, shape); err != nil {
			return nil, fmt.Errorf("native backend: %s: tensor %q: %w: %w", path, name, err, backend.ErrIncompatibleCheckpoint)
		}
	}

	m, err := build(cfg, weights, nHead)
	if err != nil {
		return nil, fmt.Errorf("native backend: %s: %w: %w", path, err, backend.ErrIncompatibleCheckpoint)
	}
	return m, nil
}

// layerTensors is how many tensors nEnc encoder layers and nHead head layers
// hold: layer 0 of the encoder has no attn_norm (expectedShapes).
func layerTensors(nEnc, nHead int) int {
	const perEncoder, perHead = 6, 12
	return max(perEncoder*nEnc-1, 0) + perHead*nHead
}

// checkNames holds the file's tensors to the ones the model has, before any
// is decoded: an encoder as cfg.plan describes it, nHead head layers. load
// has already matched the file's encoder layer count to cfg.plan.
func checkNames(f *safetensors.File, cfg encoderConfig, nHead int) error {
	names := f.Names()

	if _, ok := f.Info(encoderLayers + "0.attn_norm.weight"); ok {
		return fmt.Errorf("tensor %q is present, but layer 0's attn_norm is nn.Identity (modeling_modernbert.py:309-312)",
			encoderLayers+"0.attn_norm.weight")
	}

	var nAct int64
	if info, ok := f.Info("act_head.2.weight"); ok && len(info.Shape) > 0 {
		nAct = info.Shape[0]
	}
	want := expectedShapes(cfg, nHead, nAct)

	var missing, unexpected, wrong []string
	for _, name := range slices.Sorted(maps.Keys(want)) {
		info, ok := f.Info(name)
		switch {
		case !ok:
			missing = append(missing, name)
		case !slices.Equal(info.Shape, want[name]):
			wrong = append(wrong, fmt.Sprintf("%q has shape %v, want %v", name, info.Shape, want[name]))
		}
	}
	for _, name := range names {
		if _, ok := want[name]; !ok {
			unexpected = append(unexpected, name)
		}
	}
	slices.Sort(unexpected)

	var errs []error
	if len(missing) > 0 {
		errs = append(errs, fmt.Errorf("missing tensors %s", firstFew(quoted(missing))))
	}
	if len(unexpected) > 0 {
		errs = append(errs, fmt.Errorf("unexpected tensors %s", firstFew(quoted(unexpected))))
	}
	if len(wrong) > 0 {
		errs = append(errs, fmt.Errorf("tensors of the wrong shape: %s", firstFew(wrong)))
	}
	return errors.Join(errs...)
}

// maxListed is how many entries of each list a checkNames error names: a
// hostile header can hold a million names, and the error should not.
const maxListed = 10

// quoted is names, each %q-quoted.
func quoted(names []string) []string {
	out := make([]string, len(names))
	for i, n := range names {
		out[i] = strconv.Quote(n)
	}
	return out
}

// firstFew joins the first maxListed items and counts the rest.
func firstFew(items []string) string {
	if len(items) <= maxListed {
		return strings.Join(items, ", ")
	}
	return fmt.Sprintf("%s and %d more", strings.Join(items[:maxListed], ", "), len(items)-maxListed)
}

// layerIndices returns the layer numbers N that names hold under
// prefix + "N.". Only a canonical decimal counts, so "01" is no layer and its
// tensors are unexpected rather than layer 1's.
func layerIndices(names []string, prefix string) map[int]bool {
	out := map[int]bool{}
	for _, name := range names {
		rest, ok := strings.CutPrefix(name, prefix)
		if !ok {
			continue
		}
		num, _, _ := strings.Cut(rest, ".")
		if n, err := strconv.Atoi(num); err == nil && n >= 0 && strconv.Itoa(n) == num {
			out[n] = true
		}
	}
	return out
}

// expectedShapes is every tensor of DecisionModel's state_dict and its
// shape (docs/ARCHITECTURE.md §1.2-1.3), for an encoder as cfg describes it,
// nHead head layers and nAct act logits. Layer 0 has no attn_norm.
func expectedShapes(cfg encoderConfig, nHead int, nAct int64) map[string][]int64 {
	h, inter := int64(cfg.HiddenSize), cfg.inter
	want := map[string][]int64{
		"encoder.embeddings.tok_embeddings.weight": {cfg.vocab, h},
		"encoder.embeddings.norm.weight":           {h},
		"encoder.final_norm.weight":                {h},
	}
	for i := range cfg.plan {
		l := func(rest string) string { return fmt.Sprintf("%s%d.%s", encoderLayers, i, rest) }
		if i > 0 {
			want[l("attn_norm.weight")] = []int64{h}
		}
		want[l("attn.Wqkv.weight")] = []int64{3 * h, h}
		want[l("attn.Wo.weight")] = []int64{h, h}
		want[l("mlp_norm.weight")] = []int64{h}
		want[l("mlp.Wi.weight")] = []int64{2 * inter, h}
		want[l("mlp.Wo.weight")] = []int64{h, inter}
	}

	// DecisionModel's head is as wide as the encoder (common.py:95).
	d := h
	want["type_emb.weight"] = []int64{3, d}
	for i := range nHead {
		l := func(rest string) string { return fmt.Sprintf("%s%d.%s", headLayers, i, rest) }
		want[l("self_attn.in_proj_weight")] = []int64{3 * d, d}
		want[l("self_attn.in_proj_bias")] = []int64{3 * d}
		want[l("self_attn.out_proj.weight")] = []int64{d, d}
		want[l("self_attn.out_proj.bias")] = []int64{d}
		want[l("linear1.weight")] = []int64{4 * d, d}
		want[l("linear1.bias")] = []int64{4 * d}
		want[l("linear2.weight")] = []int64{d, 4 * d}
		want[l("linear2.bias")] = []int64{d}
		for _, n := range []string{"norm1", "norm2"} {
			want[l(n+".weight")] = []int64{d}
			want[l(n+".bias")] = []int64{d}
		}
	}
	// scorer = Sequential(LayerNorm(d), Linear(d, d), GELU(), Linear(d, 1))
	// (common.py:100): indices 0, 1 and 3; the GELU at 2 has no weights.
	want["scorer.0.weight"] = []int64{d}
	want["scorer.0.bias"] = []int64{d}
	want["scorer.1.weight"] = []int64{d, d}
	want["scorer.1.bias"] = []int64{d}
	want["scorer.3.weight"] = []int64{1, d}
	want["scorer.3.bias"] = []int64{1}
	// act_head = Sequential(Linear(d + 4, 256), GELU(), Linear(256, n_act))
	// (common.py:101): indices 0 and 2.
	want["act_head.0.weight"] = []int64{head.ActHidden, d + 4}
	want["act_head.0.bias"] = []int64{head.ActHidden}
	want["act_head.2.weight"] = []int64{nAct, head.ActHidden}
	want["act_head.2.bias"] = []int64{nAct}
	want[temperature] = []int64{3}
	return want
}

// build maps the decoded tensors, whose names and shapes checkNames has
// checked, onto the encoder and the head.
func build(cfg encoderConfig, w map[string]*tensor.Tensor, nHead int) (*model, error) {
	layers := make([]modernbert.Layer, len(cfg.plan))
	for i, typ := range cfg.plan {
		l := func(rest string) *tensor.Tensor { return w[fmt.Sprintf("%s%d.%s", encoderLayers, i, rest)] }
		attnNorm := modernbert.IdentityNorm()
		if i > 0 {
			var err error
			if attnNorm, err = modernbert.NewNorm(l("attn_norm.weight")); err != nil {
				return nil, fmt.Errorf("encoder layer %d attn_norm: %w", i, err)
			}
		}
		attn, err := modernbert.NewAttention(l("attn.Wqkv.weight"), l("attn.Wo.weight"), cfg.NumAttentionHeads, typ)
		if err != nil {
			return nil, fmt.Errorf("encoder layer %d: %w", i, err)
		}
		mlpNorm, err := modernbert.NewNorm(l("mlp_norm.weight"))
		if err != nil {
			return nil, fmt.Errorf("encoder layer %d mlp_norm: %w", i, err)
		}
		mlp, err := modernbert.NewMLP(l("mlp.Wi.weight"), l("mlp.Wo.weight"))
		if err != nil {
			return nil, fmt.Errorf("encoder layer %d: %w", i, err)
		}
		if layers[i], err = modernbert.NewLayer(attnNorm, attn, mlpNorm, mlp); err != nil {
			return nil, fmt.Errorf("encoder layer %d: %w", i, err)
		}
	}
	embNorm, err := modernbert.NewNorm(w["encoder.embeddings.norm.weight"])
	if err != nil {
		return nil, fmt.Errorf("embeddings norm: %w", err)
	}
	finalNorm, err := modernbert.NewNorm(w["encoder.final_norm.weight"])
	if err != nil {
		return nil, fmt.Errorf("final_norm: %w", err)
	}
	enc, err := modernbert.NewEncoder(w["encoder.embeddings.tok_embeddings.weight"], embNorm, layers, finalNorm, cfg.plan)
	if err != nil {
		return nil, err
	}

	hw := head.Weights{
		TypeEmb:             w["type_emb.weight"],
		ScorerNormWeight:    w["scorer.0.weight"],
		ScorerNormBias:      w["scorer.0.bias"],
		ScorerLinear1Weight: w["scorer.1.weight"],
		ScorerLinear1Bias:   w["scorer.1.bias"],
		ScorerLinear2Weight: w["scorer.3.weight"],
		ScorerLinear2Bias:   w["scorer.3.bias"],
		ActLinear1Weight:    w["act_head.0.weight"],
		ActLinear1Bias:      w["act_head.0.bias"],
		ActLinear2Weight:    w["act_head.2.weight"],
		ActLinear2Bias:      w["act_head.2.bias"],
	}
	for i := range nHead {
		l := func(rest string) *tensor.Tensor { return w[fmt.Sprintf("%s%d.%s", headLayers, i, rest)] }
		hw.Layers = append(hw.Layers, head.LayerWeights{
			InProjWeight:  l("self_attn.in_proj_weight"),
			InProjBias:    l("self_attn.in_proj_bias"),
			OutProjWeight: l("self_attn.out_proj.weight"),
			OutProjBias:   l("self_attn.out_proj.bias"),
			Linear1Weight: l("linear1.weight"),
			Linear1Bias:   l("linear1.bias"),
			Linear2Weight: l("linear2.weight"),
			Linear2Bias:   l("linear2.bias"),
			Norm1Weight:   l("norm1.weight"),
			Norm1Bias:     l("norm1.bias"),
			Norm2Weight:   l("norm2.weight"),
			Norm2Bias:     l("norm2.bias"),
		})
	}
	hd, err := head.New(hw)
	if err != nil {
		return nil, err
	}
	return &model{enc: enc, head: hd}, nil
}
