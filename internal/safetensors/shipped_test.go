package safetensors

import (
	"fmt"
	"math"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/MeKo-Christian/go-laya/internal/golden"
)

// arch is what docs/ARCHITECTURE.md §1.1-1.3 says one checkpoint holds.
type arch struct {
	d, layers, intermediate, vocab int64
	params                         int    // §1.1, temperature's 3 included
	temperature                    string // its dtype, §1.1
}

var shipped = map[string]arch{
	golden.English:        {d: 1024, layers: 28, intermediate: 2624, vocab: 50368, params: 421_293_830, temperature: "F32"},
	golden.Multilingual:   {d: 768, layers: 22, intermediate: 1152, vocab: 256000, params: 321_908_998, temperature: "F32"},
	golden.TypedDecisions: {d: 1024, layers: 28, intermediate: 2624, vocab: 50368, params: 421_293_830, temperature: "F16"},
}

// shapes spot-checks the encoder (§1.2) and head (§1.3) tensors by name and
// shape; a nil shape means the tensor must be absent.
func (a arch) shapes() map[string][]int64 {
	d, last := a.d, a.layers-1
	enc := func(layer int64, rest string) string { return fmt.Sprintf("encoder.layers.%d.%s", layer, rest) }
	want := map[string][]int64{
		"encoder.embeddings.tok_embeddings.weight": {a.vocab, d},
		"encoder.embeddings.norm.weight":           {d},
		enc(0, "attn_norm.weight"):                 nil, // layer 0's attn_norm is nn.Identity
		enc(1, "attn_norm.weight"):                 {d},
		enc(last, "attn_norm.weight"):              {d},
		enc(last, "attn.Wqkv.weight"):              {3 * d, d},
		enc(last, "attn.Wo.weight"):                {d, d},
		enc(last, "mlp_norm.weight"):               {d},
		enc(last, "mlp.Wi.weight"):                 {2 * a.intermediate, d},
		enc(last, "mlp.Wo.weight"):                 {d, a.intermediate},
		enc(0, "attn.Wqkv.weight"):                 {3 * d, d},
		enc(a.layers, "attn.Wqkv.weight"):          nil,
		"encoder.final_norm.weight":                {d},
		"type_emb.weight":                          {3, d},
		"head.layers.2.linear1.weight":             nil,
		"scorer.0.weight":                          {d},
		"scorer.0.bias":                            {d},
		"scorer.1.weight":                          {d, d},
		"scorer.1.bias":                            {d},
		"scorer.3.weight":                          {1, d},
		"scorer.3.bias":                            {1},
		"act_head.0.weight":                        {256, d + 4},
		"act_head.0.bias":                          {256},
		"act_head.2.weight":                        {2, 256},
		"act_head.2.bias":                          {2},
		"temperature":                              {3},
	}
	for _, l := range []int{0, 1} {
		h := func(rest string) string { return fmt.Sprintf("head.layers.%d.%s", l, rest) }
		want[h("self_attn.in_proj_weight")] = []int64{3 * d, d}
		want[h("self_attn.in_proj_bias")] = []int64{3 * d}
		want[h("self_attn.out_proj.weight")] = []int64{d, d}
		want[h("self_attn.out_proj.bias")] = []int64{d}
		want[h("linear1.weight")] = []int64{4 * d, d}
		want[h("linear1.bias")] = []int64{4 * d}
		want[h("linear2.weight")] = []int64{d, 4 * d}
		want[h("linear2.bias")] = []int64{d}
		for _, n := range []string{"norm1", "norm2"} {
			want[h(n+".weight")] = []int64{d}
			want[h(n+".bias")] = []int64{d}
		}
	}
	return want
}

// The three shipped checkpoints open, every tensor but temperature is F16
// and decodes to finite values, the names and shapes are the ones §1.2/§1.3
// describe, and temperature has the dtype §1.1 records for each checkpoint.
func TestShippedCheckpointsDecode(t *testing.T) {
	root := golden.SkipWithoutModels(t)
	for _, ck := range []string{golden.English, golden.Multilingual, golden.TypedDecisions} {
		t.Run(ck, func(t *testing.T) {
			a := shipped[ck]
			start := time.Now()
			f, err := Open(filepath.Join(golden.CheckpointDir(root, ck), "model.safetensors"))
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()

			for name, shape := range a.shapes() {
				info, ok := f.Info(name)
				if shape == nil {
					if ok {
						t.Errorf("%s is present, want it absent", name)
					}
					continue
				}
				if !ok || !slices.Equal(info.Shape, shape) {
					t.Errorf("%s: shape %v (present %v), want %v", name, info.Shape, ok, shape)
				}
			}
			if info, _ := f.Info(temperature); info.DType != a.temperature {
				t.Errorf("temperature dtype %s, want %s", info.DType, a.temperature)
			}
			// The encoder's 6 tensors per layer (5 in layer 0) plus
			// tok_embeddings, the embeddings norm and final_norm, and the
			// head's 36 (§1.3, temperature included).
			names := f.Names()
			if want := int(6*a.layers + 2 + 36); len(names) != want {
				t.Errorf("%d tensors, want %d", len(names), want)
			}

			var elems int
			for _, name := range names {
				if name == temperature {
					continue
				}
				if info, _ := f.Info(name); info.DType != "F16" {
					t.Errorf("%s: dtype %s, want F16", name, info.DType)
				}
				data, _, err := f.Float32(t.Context(), name)
				if err != nil {
					t.Fatal(err)
				}
				if i := slices.IndexFunc(data, func(v float32) bool {
					return math.IsNaN(float64(v)) || math.IsInf(float64(v), 0)
				}); i >= 0 {
					t.Errorf("%s[%d] = %v", name, i, data[i])
				}
				elems += len(data)
			}
			if elems+3 != a.params {
				t.Errorf("%d elements decoded besides temperature, want %d", elems, a.params-3)
			}
			t.Logf("%d tensors, %d elements decoded in %v", len(names)-1, elems, time.Since(start).Round(time.Millisecond))
		})
	}
}
