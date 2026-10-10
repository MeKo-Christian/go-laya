package modernbert

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
)

// The two attention layer types of ModernBertConfig.layer_types.
const (
	fullAttention    = "full_attention"
	slidingAttention = "sliding_attention"
)

// ModernBertConfig's defaults (configuration_modernbert.py:77-99,115) for the
// keys Config reads.
const (
	defaultHiddenSize             = 768
	defaultNumHiddenLayers        = 22
	defaultNumAttentionHeads      = 12
	defaultLocalAttention         = 128
	defaultGlobalAttnEveryNLayers = 3
	defaultGlobalRopeTheta        = 160000.0
	defaultLocalRopeTheta         = 10000.0
	defaultRopeType               = "default"
)

// Optional is a config.json value that tells a missing key from a null one,
// because transformers does: a missing key takes the default, while null is
// Python's None and is used as such.
type Optional[T any] struct {
	// Present is whether the key was in the JSON at all.
	Present bool
	// Value is the value, nil for null.
	Value *T
}

// UnmarshalJSON records that the key was present, and its value unless it is
// null.
func (o *Optional[T]) UnmarshalJSON(raw []byte) error {
	o.Present = true
	if isNull(raw) {
		o.Value = nil
		return nil
	}
	var v T
	if err := json.Unmarshal(raw, &v); err != nil {
		return err
	}
	o.Value = &v
	return nil
}

func isNull(raw json.RawMessage) bool {
	return bytes.Equal(bytes.TrimSpace(raw), []byte("null"))
}

// RopeParameters is one layer type's entry of config.json's rope_parameters.
type RopeParameters struct {
	// RopeType is the RoPE variant. When it is missing, the legacy key "type"
	// stands in for it, and failing that it is "default"
	// (standardize_rope_params). Only "default" is implemented.
	RopeType, Type Optional[string]
	// RopeTheta is the theta; when it is missing, the legacy key or the
	// default applies (convert_rope_params_to_dict's setdefault).
	RopeTheta Optional[float64]
}

// UnmarshalJSON reads the entry's keys by their exact names, as Python does;
// encoding/json alone would also match "Rope_Theta".
func (p *RopeParameters) UnmarshalJSON(raw []byte) error {
	keys, err := objectKeys(raw)
	if err != nil {
		return err
	}
	return decodeKeys(keys, map[string]any{
		"rope_type":  &p.RopeType,
		"type":       &p.Type,
		"rope_theta": &p.RopeTheta,
	})
}

// ropeType is the entry's resolved rope_type, nil for None.
func (p *RopeParameters) ropeType() *string {
	switch {
	case p.RopeType.Present:
		return p.RopeType.Value
	case p.Type.Present:
		return p.Type.Value
	}
	t := defaultRopeType
	return &t
}

// Config is the part of a ModernBERT config.json that decides the encoder's
// layer plan: how many layers, which are full and which sliding attention, the
// RoPE theta of each type, the window and the heads. Build it with
// ParseConfig, which applies ModernBertConfig's defaults; Layers resolves it
// the way the pinned transformers does.
type Config struct {
	HiddenSize        int
	NumHiddenLayers   int
	NumAttentionHeads int
	// LocalAttention is the sliding window's full width, local_attention.
	LocalAttention int
	// LayerTypes, when not nil, is the explicit layer plan. Both checkpoints
	// write one. When it is nil (missing or null), the plan follows
	// GlobalAttnEveryNLayers.
	LayerTypes             []string
	GlobalAttnEveryNLayers Optional[int]
	// RopeParameters holds the RoPE of each layer type. A nil entry is the
	// null transformers rejects.
	RopeParameters map[string]*RopeParameters
	// GlobalRopeTheta and LocalRopeTheta are the legacy theta keys of the
	// full and sliding layers, read where RopeParameters gives no theta.
	GlobalRopeTheta, LocalRopeTheta Optional[float64]
	// RopeScaling is the legacy rope_scaling, which transformers merges into
	// both types' RoPE. It is not implemented: anything but null is an error.
	RopeScaling json.RawMessage
	// HeadDim is head_dim, which ModernBertConfig does not define but its
	// RoPE reads when a config sets it. Only hidden / heads is implemented.
	HeadDim Optional[int]
}

// ParseConfig reads a ModernBERT config.json, starting from ModernBertConfig's
// defaults for the keys it leaves out. It reads keys by their exact names, as
// Python does, and ignores the keys Config does not hold. The input must be a
// JSON object. A null hidden_size, num_hidden_layers, num_attention_heads or
// local_attention is an error, as transformers' strict dataclass makes it.
func ParseConfig(raw []byte) (Config, error) {
	keys, err := objectKeys(raw)
	if err != nil {
		return Config{}, fmt.Errorf("modernbert: config: %w", err)
	}
	cfg := Config{
		HiddenSize:        defaultHiddenSize,
		NumHiddenLayers:   defaultNumHiddenLayers,
		NumAttentionHeads: defaultNumAttentionHeads,
		LocalAttention:    defaultLocalAttention,
	}
	for _, k := range []string{"hidden_size", "num_hidden_layers", "num_attention_heads", "local_attention"} {
		if v, ok := keys[k]; ok && isNull(v) {
			return Config{}, fmt.Errorf("modernbert: config %s is null, want an integer", k)
		}
	}
	err = decodeKeys(keys, map[string]any{
		"hidden_size":                &cfg.HiddenSize,
		"num_hidden_layers":          &cfg.NumHiddenLayers,
		"num_attention_heads":        &cfg.NumAttentionHeads,
		"local_attention":            &cfg.LocalAttention,
		"layer_types":                &cfg.LayerTypes,
		"global_attn_every_n_layers": &cfg.GlobalAttnEveryNLayers,
		"rope_parameters":            &cfg.RopeParameters,
		"global_rope_theta":          &cfg.GlobalRopeTheta,
		"local_rope_theta":           &cfg.LocalRopeTheta,
		"rope_scaling":               &cfg.RopeScaling,
		"head_dim":                   &cfg.HeadDim,
	})
	if err != nil {
		return Config{}, fmt.Errorf("modernbert: config: %w", err)
	}
	return cfg, nil
}

// objectKeys splits a JSON object into its keys, failing on anything else.
func objectKeys(raw []byte) (map[string]json.RawMessage, error) {
	var keys map[string]json.RawMessage
	if err := json.Unmarshal(raw, &keys); err != nil {
		return nil, err
	}
	if keys == nil {
		return nil, errors.New("want a JSON object, got null")
	}
	return keys, nil
}

// decodeKeys decodes each present key into its destination.
func decodeKeys(keys map[string]json.RawMessage, dst map[string]any) error {
	for k, d := range dst {
		if v, ok := keys[k]; ok {
			if err := json.Unmarshal(v, d); err != nil {
				return fmt.Errorf("%s: %w", k, err)
			}
		}
	}
	return nil
}

// Layers returns each layer's LayerType, in order, as the pinned transformers
// resolves them:
//
//   - The layer plan is LayerTypes when the config has one; otherwise layer i
//     is sliding unless i is a multiple of global_attn_every_n_layers, 3 by
//     default (configuration_modernbert.py:113-120). Its length must be
//     num_hidden_layers.
//   - A type's theta is rope_parameters[type].rope_theta; failing that,
//     global_rope_theta for full and local_rope_theta for sliding layers;
//     failing that, 160000 and 10000 (configuration_modernbert.py:124-152). A
//     top-level rope_theta is never read.
//   - A sliding layer's window is local_attention // 2.
//
// Layers returns an error where transformers raises, and also where it would
// go on with a theta of None or zero, a negative window, a RoPE other than
// "default" (rope_scaling, another rope_type, or the legacy "type" key naming
// one), or a head_dim other than hidden / heads: none of those is
// implemented, and an error is better than an answer that differs. The
// thetas are resolved for the types the plan uses, as
// ModernBertRotaryEmbedding builds only those.
func (c Config) Layers() ([]LayerType, error) {
	if c.NumHiddenLayers <= 0 {
		return nil, fmt.Errorf("modernbert: config num_hidden_layers %d, want a positive value", c.NumHiddenLayers)
	}
	if c.LocalAttention < 0 {
		return nil, fmt.Errorf("modernbert: config local_attention %d is negative", c.LocalAttention)
	}
	if err := c.checkRope(); err != nil {
		return nil, err
	}
	names := c.LayerTypes
	if names == nil {
		var err error
		if names, err = c.derivedLayerTypes(); err != nil {
			return nil, err
		}
	}
	if len(names) != c.NumHiddenLayers {
		return nil, fmt.Errorf("modernbert: config has %d layers but %d layer_types", c.NumHiddenLayers, len(names))
	}

	out := make([]LayerType, len(names))
	for i, name := range names {
		var legacy Optional[float64]
		var fallback float64
		switch name {
		case fullAttention:
			legacy, fallback = c.GlobalRopeTheta, defaultGlobalRopeTheta
		case slidingAttention:
			legacy, fallback = c.LocalRopeTheta, defaultLocalRopeTheta
		default:
			return nil, fmt.Errorf("modernbert: config layer_types[%d] is %q, want %q or %q",
				i, name, fullAttention, slidingAttention)
		}
		theta, err := c.ropeTheta(name, legacy, fallback)
		if err != nil {
			return nil, err
		}
		if name == fullAttention {
			out[i] = FullAttention(theta)
		} else {
			out[i] = SlidingAttention(theta, c.LocalAttention)
		}
	}
	return out, nil
}

// checkRope rejects what Config does not implement: rope_scaling, a RoPE
// other than "default" in any rope_parameters entry, a null entry, and a
// head_dim other than hidden / heads.
func (c Config) checkRope() error {
	if len(c.RopeScaling) > 0 && !isNull(c.RopeScaling) {
		return errors.New("modernbert: config sets rope_scaling, which is not implemented")
	}
	for name, p := range c.RopeParameters {
		if p == nil {
			return fmt.Errorf("modernbert: config rope_parameters[%q] is null", name)
		}
		if t := p.ropeType(); t == nil || *t != defaultRopeType {
			got := "null"
			if t != nil {
				got = fmt.Sprintf("%q", *t)
			}
			return fmt.Errorf("modernbert: config rope_parameters[%q] has rope_type %s; only %q is implemented",
				name, got, defaultRopeType)
		}
	}
	if c.NumAttentionHeads <= 0 {
		return fmt.Errorf("modernbert: config num_attention_heads %d, want a positive value", c.NumAttentionHeads)
	}
	if c.HeadDim.Value != nil && *c.HeadDim.Value*c.NumAttentionHeads != c.HiddenSize {
		return fmt.Errorf("modernbert: config head_dim %d is not hidden_size %d / %d heads",
			*c.HeadDim.Value, c.HiddenSize, c.NumAttentionHeads)
	}
	return nil
}

// derivedLayerTypes is ModernBertConfig.__post_init__'s plan for a config
// without layer_types: every global_attn_every_n_layers-th layer, from layer
// 0, full and the rest sliding.
func (c Config) derivedLayerTypes() ([]string, error) {
	every := defaultGlobalAttnEveryNLayers
	if c.GlobalAttnEveryNLayers.Present {
		if c.GlobalAttnEveryNLayers.Value == nil {
			return nil, errors.New("modernbert: config global_attn_every_n_layers is null and layer_types is missing")
		}
		every = *c.GlobalAttnEveryNLayers.Value
	}
	if every <= 0 {
		return nil, fmt.Errorf("modernbert: config global_attn_every_n_layers %d, want a positive value", every)
	}
	names := make([]string, c.NumHiddenLayers)
	for i := range names {
		names[i] = fullAttention
		if i%every != 0 {
			names[i] = slidingAttention
		}
	}
	return names, nil
}

// ropeTheta resolves one layer type's theta: its rope_parameters entry's, the
// legacy key's, or the default, in that order.
func (c Config) ropeTheta(name string, legacy Optional[float64], fallback float64) (float64, error) {
	theta := &fallback
	if legacy.Present {
		theta = legacy.Value
	}
	// checkRope has ruled out a nil entry.
	if p, ok := c.RopeParameters[name]; ok && p.RopeTheta.Present {
		theta = p.RopeTheta.Value
	}
	if theta == nil {
		return 0, fmt.Errorf("modernbert: config rope_theta of %s is null", name)
	}
	v := *theta
	if math.IsNaN(v) || v <= 0 || math.IsInf(v, 0) {
		return 0, fmt.Errorf("modernbert: config rope_theta of %s is %g, want a positive finite value", name, v)
	}
	return v, nil
}
