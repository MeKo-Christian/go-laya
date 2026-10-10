package head

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/MeKo-Christian/go-laya/backend"
	"github.com/MeKo-Christian/go-laya/internal/golden"
	"github.com/MeKo-Christian/go-laya/internal/runtime/tensor"
)

// headFixture is testdata/head.json, written by scripts/dump_head_ops.py from
// the real DecisionModel and nn.TransformerEncoderLayer of the pinned torch.
type headFixture struct {
	Header struct {
		Fixture  string            `json:"fixture"`
		Versions map[string]string `json:"versions"`
		Seed     *int64            `json:"seed"`
		Threads  *int              `json:"torch_threads"`
		Mode     struct {
			Device string `json:"device"`
			DType  string `json:"dtype"`
			Eval   bool   `json:"eval"`
			NoGrad bool   `json:"no_grad"`
		} `json:"mode"`
	} `json:"header"`
	Cases []json.RawMessage `json:"cases"`
}

// tensorRec is the dumper's tensor_rec: a row-major float32 buffer and its shape.
type tensorRec struct {
	DType string    `json:"dtype"`
	Shape []int64   `json:"shape"`
	Data  []float32 `json:"data"`
}

func (r tensorRec) tensor(tb testing.TB, what string) *tensor.Tensor {
	tb.Helper()

	if r.DType != "float32" {
		tb.Fatalf("%s: dtype %q, want float32", what, r.DType)
	}
	// A copy, so a test that edits the tensor cannot reach the record.
	t, err := tensor.New(append([]float32(nil), r.Data...), r.Shape)
	if err != nil {
		tb.Fatalf("%s: %v", what, err)
	}
	return t
}

// modelCase is a "decision_model" record: the whole head after the encoder.
type modelCase struct {
	Name          string `json:"name"`
	D             int    `json:"d"`
	NHead         int    `json:"nhead"`
	HeadLayers    int    `json:"head_layers"`
	NAct          int    `json:"n_act"`
	Path          string `json:"path"`
	FastPathCalls int    `json:"fast_path_calls"`
	Batch         struct {
		InputIDs      [][]int64 `json:"input_ids"`
		AttentionMask [][]int64 `json:"attention_mask"`
		MarkerPos     [][]int64 `json:"marker_pos"`
		MarkerMask    [][]bool  `json:"marker_mask"`
		QType         []int64   `json:"qtype"`
	} `json:"batch"`
	MarkerFill   int64                `json:"marker_fill"`
	H            tensorRec            `json:"h"`
	Weights      map[string]tensorRec `json:"weights"`
	HTyped       tensorRec            `json:"h_typed"`
	HLayers      []tensorRec          `json:"h_layers"`
	Gathered     tensorRec            `json:"gathered"`
	Logits       tensorRec            `json:"logits"`
	Probs        tensorRec            `json:"probs"`
	Feats        tensorRec            `json:"feats"`
	ActLogits    tensorRec            `json:"act_logits"`
	FillMinusOne map[string]bool      `json:"fill_minus_one"`
}

func (c modelCase) batch() backend.Batch {
	b := c.Batch
	return backend.Batch{
		InputIDs:      b.InputIDs,
		AttentionMask: b.AttentionMask,
		MarkerPos:     b.MarkerPos,
		MarkerMask:    b.MarkerMask,
		QType:         b.QType,
	}
}

// layerCase is an "encoder_layer" record: one nn.TransformerEncoderLayer.
type layerCase struct {
	Name          string               `json:"name"`
	D             int                  `json:"d"`
	NHead         int                  `json:"nhead"`
	FF            int                  `json:"dim_feedforward"`
	Path          string               `json:"path"`
	FastPathCalls int                  `json:"fast_path_calls"`
	PaddingMask   [][]int64            `json:"padding_mask"`
	Input         tensorRec            `json:"input"`
	Weights       map[string]tensorRec `json:"weights"`
	Output        tensorRec            `json:"output"`
	OutputSlow    tensorRec            `json:"output_slow"`
}

// topkCase is a "topk_error" record: what forward raised with kmax < 2.
type topkCase struct {
	Name  string `json:"name"`
	KMax  int    `json:"kmax"`
	Error string `json:"error"`
}

func loadHead(tb testing.TB) headFixture {
	tb.Helper()

	raw, err := os.ReadFile(filepath.Join("testdata", "head.json"))
	if err != nil {
		tb.Fatalf("read the fixture: %v", err)
	}
	var f headFixture
	if err := json.Unmarshal(raw, &f); err != nil {
		tb.Fatalf("decode testdata/head.json: %v", err)
	}
	if len(f.Cases) == 0 {
		tb.Fatal("testdata/head.json has no cases; an empty fixture is a broken checkout")
	}
	return f
}

// casesOf decodes the cases whose "op" is op into T, failing if there are
// none: a filter that silently selects nothing is a test that silently passes.
func casesOf[T any](tb testing.TB, f headFixture, op string) []T {
	tb.Helper()

	var out []T
	for i, raw := range f.Cases {
		var head struct {
			Op string `json:"op"`
		}
		if err := json.Unmarshal(raw, &head); err != nil {
			tb.Fatalf("case %d: %v", i, err)
		}
		if head.Op != op {
			continue
		}
		var c T
		if err := json.Unmarshal(raw, &c); err != nil {
			tb.Fatalf("case %d (%s): %v", i, op, err)
		}
		out = append(out, c)
	}
	if len(out) == 0 {
		tb.Fatalf("testdata/head.json has no %q cases", op)
	}
	return out
}

// layerWeights reads one TransformerEncoderLayer's tensors through get, named
// as its state_dict names them under prefix.
func layerWeights(get func(name string) *tensor.Tensor, prefix string) LayerWeights {
	return LayerWeights{
		InProjWeight:  get(prefix + "self_attn.in_proj_weight"),
		InProjBias:    get(prefix + "self_attn.in_proj_bias"),
		OutProjWeight: get(prefix + "self_attn.out_proj.weight"),
		OutProjBias:   get(prefix + "self_attn.out_proj.bias"),
		Linear1Weight: get(prefix + "linear1.weight"),
		Linear1Bias:   get(prefix + "linear1.bias"),
		Linear2Weight: get(prefix + "linear2.weight"),
		Linear2Bias:   get(prefix + "linear2.bias"),
		Norm1Weight:   get(prefix + "norm1.weight"),
		Norm1Bias:     get(prefix + "norm1.bias"),
		Norm2Weight:   get(prefix + "norm2.weight"),
		Norm2Bias:     get(prefix + "norm2.bias"),
	}
}

// modelWeights reads the DecisionModel head's tensors by their state_dict
// names (docs/ARCHITECTURE.md §1.3), failing on any name the fixture has that
// Weights does not take: a tensor left unread is a weight the port ignores.
func modelWeights(tb testing.TB, c modelCase) Weights {
	tb.Helper()

	used := map[string]bool{}
	w := c.Weights
	track := func(name string) *tensor.Tensor {
		used[name] = true
		return weight(tb, w, name)
	}
	out := Weights{
		TypeEmb:             track("type_emb.weight"),
		ScorerNormWeight:    track("scorer.0.weight"),
		ScorerNormBias:      track("scorer.0.bias"),
		ScorerLinear1Weight: track("scorer.1.weight"),
		ScorerLinear1Bias:   track("scorer.1.bias"),
		ScorerLinear2Weight: track("scorer.3.weight"),
		ScorerLinear2Bias:   track("scorer.3.bias"),
		ActLinear1Weight:    track("act_head.0.weight"),
		ActLinear1Bias:      track("act_head.0.bias"),
		ActLinear2Weight:    track("act_head.2.weight"),
		ActLinear2Bias:      track("act_head.2.bias"),
	}
	for i := range c.HeadLayers {
		out.Layers = append(out.Layers, layerWeights(track, fmt.Sprintf("head.layers.%d.", i)))
	}
	for name := range w {
		if !used[name] {
			tb.Errorf("fixture weight %q is not read into Weights", name)
		}
	}
	return out
}

func weight(tb testing.TB, w map[string]tensorRec, name string) *tensor.Tensor {
	tb.Helper()

	rec, ok := w[name]
	if !ok {
		tb.Fatalf("fixture has no weight %q", name)
	}
	return rec.tensor(tb, name)
}

// requirementPins reads the name==version lines of scripts/requirements-ref.txt,
// the reference environment PLAN.md R6 calls the contract.
func requirementPins(tb testing.TB) map[string]string {
	tb.Helper()

	path := filepath.Join(golden.Root(tb), "scripts", "requirements-ref.txt")
	raw, err := os.ReadFile(path)
	if err != nil {
		tb.Fatalf("read %s: %v", path, err)
	}
	pins := map[string]string{}
	sc := bufio.NewScanner(bytes.NewReader(raw))
	for sc.Scan() {
		if name, version, ok := strings.Cut(strings.TrimSpace(sc.Text()), "=="); ok {
			pins[name] = version
		}
	}
	return pins
}

// TestHeadProvenance asserts the fixture came from the pinned environment, run
// as agent.py runs the model on the CPU (eval, no_grad, float32). The
// tolerances below are claims about float32 torch on one version; a
// regeneration elsewhere must fail here rather than quietly move the oracle.
func TestHeadProvenance(t *testing.T) {
	h := loadHead(t).Header
	if h.Fixture != "head" {
		t.Errorf("header fixture = %q, want \"head\"", h.Fixture)
	}
	if h.Seed == nil {
		t.Error("header records no seed")
	}
	if h.Threads == nil || *h.Threads != 1 {
		t.Errorf("header torch_threads = %v, want 1", h.Threads)
	}
	if m := h.Mode; m.Device != "cpu" || m.DType != "float32" || !m.Eval || !m.NoGrad {
		t.Errorf("header mode = %+v, want cpu float32 eval no_grad (agent.py:196-215)", m)
	}

	pins := requirementPins(t)
	for _, name := range []string{"torch", "transformers", "numpy"} {
		want, ok := pins[name]
		if !ok {
			t.Fatalf("scripts/requirements-ref.txt pins no %s", name)
		}
		if got := h.Versions[name]; got != want {
			t.Errorf("fixture %s = %q, requirements-ref.txt pins %q: regenerate "+
				"testdata/head.json under .venv-ref, as a reviewed diff (R6)", name, got, want)
		}
	}
}
