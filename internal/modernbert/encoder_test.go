package modernbert

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/MeKo-Christian/go-laya/internal/runtime/tensor"
)

// encCase is the "encoder" record of testdata/ops.json: the real
// ModernBertModel of the dumper's tiny encoder config, with every parameter
// drawn at random, run on a padded batch. Config is the config as the dumper
// passed it; LayerTypes, RopeThetas and SlidingWindow are what transformers
// resolved from it. HiddenStates are the outputs of the embeddings and of
// each layer, Output is last_hidden_state.
type encCase struct {
	Name          string               `json:"name"`
	ModuleClass   string               `json:"module_class"`
	Config        json.RawMessage      `json:"config"`
	LayerTypes    []string             `json:"layer_types"`
	RopeThetas    map[string]float64   `json:"rope_thetas"`
	SlidingWindow int                  `json:"sliding_window"`
	InputIDs      [][]int64            `json:"input_ids"`
	Padding       [][]int64            `json:"attention_mask"`
	Weights       map[string]tensorRec `json:"weights"`
	HiddenStates  []tensorRec          `json:"hidden_states"`
	Output        tensorRec            `json:"output"`
}

// encTolerance bounds |go - torch| for one element of a hidden state, the
// encoder's output included.
//
// The blocks' own tolerances do not compose into a useful bound: each norm
// divides by a row's standard deviation, and four layers feed one another's
// rounding forward. So the bound is empirical, as attnTolerance's is, and
// pinned against two measurements. torch's own float32 output is 9.8e-7 from
// the same model run in float64 (dump_modernbert_ops.py prints it); Go runs
// the same arithmetic in float32, so |go - torch| should be of that order, and
// it is: 1.3e-6 on the output and up to 2.4e-6 on the hidden states, which
// grow to |h| ~ 8 through the residual stream, where a float32 ulp is 9.5e-7;
// hence the relative term. The worst stage uses 26 % of the bound (the test
// logs each stage), which leaves room for another CPU's kernels.
//
// The failure modes the test is for sit far above it: on this fixture the
// wrong assemblies TestEncoderMatchesTorch builds miss torch by 85000 times
// the bound and more (the test logs each).
func encTolerance(want float32) float64 {
	return 8e-6 + 8*float32Eps*math.Abs(float64(want))
}

func loadEncCase(t *testing.T) encCase {
	t.Helper()

	raws := casesOf(t, loadOps(t), "encoder")
	if len(raws) != 1 {
		t.Fatalf("testdata/ops.json has %d encoder cases, want 1", len(raws))
	}
	var c encCase
	if err := json.Unmarshal(raws[0], &c); err != nil {
		t.Fatalf("decode the encoder case: %v", err)
	}
	if c.Name != "encoder_padded" || c.ModuleClass != "ModernBertModel" {
		t.Fatalf("encoder case %q of %q, want encoder_padded of ModernBertModel", c.Name, c.ModuleClass)
	}
	return c
}

// encBuild is an Encoder built from the record, with what it was built from.
type encBuild struct {
	enc   Encoder
	cfg   Config
	types []LayerType
}

// buildEncoder builds the Encoder from the record the way Task 8.10 will from
// a checkpoint: the layer plan from the config through Config.Layers, then
// each block from its weights. It fails unless the plan is the one
// transformers resolved and every recorded weight was used.
func buildEncoder(t *testing.T, c encCase) encBuild {
	t.Helper()

	cfg, err := ParseConfig(c.Config)
	if err != nil {
		t.Fatalf("ParseConfig: %v", err)
	}
	types, err := cfg.Layers()
	if err != nil {
		t.Fatalf("Layers: %v", err)
	}
	if len(types) != len(c.LayerTypes) {
		t.Fatalf("Layers() gives %d layers, transformers %d", len(types), len(c.LayerTypes))
	}
	for i, l := range types {
		name := fullAttention
		if l.Sliding {
			name = slidingAttention
		}
		if name != c.LayerTypes[i] || l.RopeTheta != c.RopeThetas[name] {
			t.Fatalf("layer %d: Layers() gives %s at theta %g, transformers %s at %g",
				i, name, l.RopeTheta, c.LayerTypes[i], c.RopeThetas[c.LayerTypes[i]])
		}
		if l.Sliding && l.Window != c.SlidingWindow {
			t.Fatalf("layer %d: window %d, transformers' sliding_window %d", i, l.Window, c.SlidingWindow)
		}
	}
	if !slices.Contains(c.LayerTypes, fullAttention) || !slices.Contains(c.LayerTypes, slidingAttention) {
		t.Fatalf("layer types %v: the fixture must have both", c.LayerTypes)
	}

	used := map[string]bool{}
	weight := func(name string) *tensor.Tensor {
		t.Helper()
		rec, ok := c.Weights[name]
		if !ok {
			t.Fatalf("the record has no %s", name)
		}
		used[name] = true
		return rec.tensor(t, name)
	}
	newNorm := func(name string) Norm {
		t.Helper()
		n, err := NewNorm(weight(name))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		return n
	}

	layers := make([]Layer, len(types))
	for i, lt := range types {
		p := fmt.Sprintf("layers.%d.", i)
		attnNorm := IdentityNorm()
		if _, ok := c.Weights[p+"attn_norm.weight"]; ok != (i > 0) {
			t.Fatalf("%sattn_norm.weight recorded: %v; only layer 0's norm is nn.Identity", p, ok)
		}
		if i > 0 {
			attnNorm = newNorm(p + "attn_norm.weight")
		}
		attn, err := NewAttention(weight(p+"attn.Wqkv.weight"), weight(p+"attn.Wo.weight"), cfg.NumAttentionHeads, lt)
		if err != nil {
			t.Fatalf("layer %d: NewAttention: %v", i, err)
		}
		mlp, err := NewMLP(weight(p+"mlp.Wi.weight"), weight(p+"mlp.Wo.weight"))
		if err != nil {
			t.Fatalf("layer %d: NewMLP: %v", i, err)
		}
		if layers[i], err = NewLayer(attnNorm, attn, newNorm(p+"mlp_norm.weight"), mlp); err != nil {
			t.Fatalf("layer %d: NewLayer: %v", i, err)
		}
	}
	enc, err := NewEncoder(weight("embeddings.tok_embeddings.weight"), newNorm("embeddings.norm.weight"),
		layers, newNorm("final_norm.weight"), types)
	if err != nil {
		t.Fatalf("NewEncoder: %v", err)
	}
	for name := range c.Weights {
		if !used[name] {
			t.Errorf("recorded weight %s was never used", name)
		}
	}
	return encBuild{enc: enc, cfg: cfg, types: types}
}

// compareStage checks got against torch's want, element by element, and logs
// the worst |diff|/tol.
func compareStage(t *testing.T, stage string, got *tensor.Tensor, want tensorRec) {
	t.Helper()

	if !slices.Equal(got.Shape(), want.Shape) {
		t.Fatalf("%s: shape %v, torch %v", stage, got.Shape(), want.Shape)
	}
	var worst, maxDiff float64
	bad := 0
	for i, g := range got.Data() {
		diff := math.Abs(float64(g) - float64(want.Data[i]))
		tol := encTolerance(want.Data[i])
		worst, maxDiff = math.Max(worst, diff/tol), math.Max(maxDiff, diff)
		if !(diff <= tol) { // NaN-safe
			if bad++; bad <= 5 {
				t.Errorf("%s[%d] = %.9g, torch %.9g (|diff| %.3g > tol %.3g)", stage, i, g, want.Data[i], diff, tol)
			}
		}
	}
	t.Logf("%s: max |diff| %.2e, worst |diff|/tol %.3f", stage, maxDiff, worst)
}

// missBy returns the largest |got - torch| / tol over the output.
func missBy(got *tensor.Tensor, want tensorRec) float64 {
	var worst float64
	for i, g := range got.Data() {
		r := math.Abs(float64(g)-float64(want.Data[i])) / encTolerance(want.Data[i])
		if math.IsNaN(r) {
			return math.Inf(1)
		}
		worst = math.Max(worst, r)
	}
	return worst
}

// TestEncoderMatchesTorch runs the recorded ModernBertModel in Go: the layer
// plan from its config, the blocks from its weights, the input ids and padding
// mask through Encoder.Forward, against torch's last_hidden_state. Then it
// runs the stages one by one against torch's hidden state after each, so a
// failure names the stage, and checks that the fixture tells the right
// assembly from the plausible wrong ones: each must miss torch by ten times
// the tolerance somewhere.
//
// Padded positions are compared like real ones. torch computes them, and they
// are not zeros: a padded query attends to the real keys its layer allows, and
// one with none left on a sliding layer gets sdpa's zeros from the attention
// but still carries the residual stream and the MLP. The record has such
// queries, and the test requires them.
func TestEncoderMatchesTorch(t *testing.T) {
	c := loadEncCase(t)
	b := buildEncoder(t, c)
	e := b.enc

	got, err := e.Forward(c.InputIDs, c.Padding)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	compareStage(t, "last_hidden_state", got, c.Output)

	// Stage by stage, each fed the previous Go stage's output.
	h, err := e.embed(c.InputIDs)
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	if h, err = e.embNorm.Forward(h); err != nil {
		t.Fatalf("embeddings norm: %v", err)
	}
	if len(c.HiddenStates) != len(e.layers)+1 {
		t.Fatalf("%d hidden states recorded for %d layers", len(c.HiddenStates), len(e.layers))
	}
	compareStage(t, "embeddings", h, c.HiddenStates[0])
	for i, l := range e.layers {
		if h, err = l.Forward(h, c.Padding); err != nil {
			t.Fatalf("layer %d: %v", i, err)
		}
		compareStage(t, fmt.Sprintf("layers.%d (%s)", i, c.LayerTypes[i]), h, c.HiddenStates[i+1])
	}
	if h, err = e.finalNorm.Forward(h); err != nil {
		t.Fatalf("final norm: %v", err)
	}
	if !slices.Equal(h.Data(), got.Data()) {
		t.Error("Forward differs from its stages run one by one")
	}

	checkPaddedPositions(t, c, b.types)

	// The wrong assemblies. Each is the right Encoder with one part changed.
	hidden := c.Weights["final_norm.weight"].Shape[0]
	ones := make([]float32, hidden)
	for i := range ones {
		ones[i] = 1
	}
	onesNorm := norm(t, &tensorRec{DType: "float32", Shape: []int64{hidden}, Data: ones})
	full, sliding := c.RopeThetas[fullAttention], c.RopeThetas[slidingAttention]
	for _, v := range []struct {
		name   string
		mutate func(e *Encoder)
	}{
		{"layer 0 with layer 1's attn_norm", func(e *Encoder) { e.layers[0].attnNorm = e.layers[1].attnNorm }},
		{"layer 0 with an attn_norm of ones", func(e *Encoder) { e.layers[0].attnNorm = onesNorm }},
		{"layer types shifted by one", func(e *Encoder) {
			for i := range e.layers {
				e.layers[i].attn.layer = SlidingAttention(sliding, 2*c.SlidingWindow)
				if (i+1)%3 == 0 {
					e.layers[i].attn.layer = FullAttention(full)
				}
			}
		}},
		{"embeddings norm skipped", func(e *Encoder) { e.embNorm = IdentityNorm() }},
		{"final norm skipped", func(e *Encoder) { e.finalNorm = IdentityNorm() }},
	} {
		wrong := e
		wrong.layers = slices.Clone(e.layers)
		v.mutate(&wrong)
		y, err := wrong.Forward(c.InputIDs, c.Padding)
		if err != nil {
			t.Fatalf("%s: %v", v.name, err)
		}
		checkMisses(t, v.name, y, c.Output)
	}
	for _, aroundAttn := range []bool{true, false} {
		name := "MLP residual from the normed input"
		if aroundAttn {
			name = "attention residual from the normed input"
		}
		checkMisses(t, name, normedResidualForward(t, e, c.InputIDs, c.Padding, aroundAttn), c.Output)
	}
}

// checkMisses requires a wrong assembly's output y to miss torch by ten times
// the tolerance somewhere, so that the fixture can tell it from the module.
func checkMisses(t *testing.T, name string, y *tensor.Tensor, want tensorRec) {
	t.Helper()

	miss := missBy(y, want)
	t.Logf("%s misses torch by up to %.0f x tol", name, miss)
	if miss < 10 {
		t.Errorf("%s misses torch by only %.1f x tol; the fixture cannot tell it from the module", name, miss)
	}
}

// normedResidualForward is Encoder.Forward with one residual of every layer
// taken from its sub-block's normed input rather than the un-normed one, a
// test-only copy of Layer.Forward's wrong twin.
func normedResidualForward(t *testing.T, e Encoder, ids, mask [][]int64, aroundAttn bool) *tensor.Tensor {
	t.Helper()

	must := func(y *tensor.Tensor, err error) *tensor.Tensor {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		return y
	}
	h := must(e.embNorm.Forward(must(e.embed(ids))))
	for _, l := range e.layers {
		normed := must(l.attnNorm.Forward(h))
		a := must(l.attn.Forward(normed, mask))
		if aroundAttn {
			addInto(a, normed)
		} else {
			addInto(a, h)
		}
		normed = must(l.mlpNorm.Forward(a))
		y := must(l.mlp.Forward(normed))
		if aroundAttn {
			addInto(y, a)
		} else {
			addInto(y, normed)
		}
		h = y
	}
	return must(e.finalNorm.Forward(h))
}

// checkPaddedPositions requires the record to have padded queries with no key
// on a sliding layer, and torch's outputs there to be ordinary values rather
// than zeros, so that comparing them, as TestEncoderMatchesTorch does, pins
// what torch computes there.
func checkPaddedPositions(t *testing.T, c encCase, types []LayerType) {
	t.Helper()

	hidden := int(c.Output.Shape[2])
	padded, empty := 0, 0
	for b, pad := range c.Padding {
		for q, m := range pad {
			if m != 0 {
				continue
			}
			padded++
			row := c.Output.Data[(b*len(pad)+q)*hidden : (b*len(pad)+q+1)*hidden]
			if !slices.ContainsFunc(row, func(v float32) bool { return v != 0 }) {
				t.Errorf("torch's output at padded [%d][%d] is all zeros", b, q)
			}
			for _, l := range types {
				if !l.Sliding {
					continue
				}
				has := false
				for k := range pad {
					has = has || l.allows(pad, q, k)
				}
				if !has {
					empty++
				}
			}
		}
	}
	if padded == 0 || empty == 0 {
		t.Fatalf("%d padded positions, %d (query, sliding layer) pairs with no key; the record must have both",
			padded, empty)
	}
	t.Logf("%d padded positions compared; %d (query, sliding layer) pairs among them have no key", padded, empty)
}

// TestEncoderRealDimensions runs three layers at ModernBERT-large's shapes
// (hidden 1024, 16 heads, intermediate 2624), full then sliding twice, on
// synthetic weights. The blocks are checked at these shapes on their own
// (TestAttentionRealDimensions, TestMLPRealDimensions); here the assembly is,
// through what must hold without a reference: the shape, finite values, the
// final norm's statistics, and real positions that depend on neither the
// padded tokens nor the other rows of the batch, bit for bit.
func TestEncoderRealDimensions(t *testing.T) {
	const (
		hidden, heads, inter = 1024, 16, 2624
		vocab, batch, seq    = 32, 2, 6
	)
	start := time.Now()
	rng := rand.New(rand.NewPCG(85, hidden))
	normal := func(shape []int64, scale float64) *tensor.Tensor {
		n := int64(1)
		for _, d := range shape {
			n *= d
		}
		data := make([]float32, n)
		for i := range data {
			data[i] = float32(rng.NormFloat64() * scale)
		}
		return mustTensor(t, data, shape)
	}
	newNorm := func() Norm {
		w := make([]float32, hidden)
		for i := range w {
			w[i] = float32(0.5 + rng.Float64())
		}
		return norm(t, &tensorRec{DType: "float32", Shape: []int64{hidden}, Data: w})
	}

	types := []LayerType{FullAttention(160000), SlidingAttention(10000, 128), SlidingAttention(10000, 128)}
	layers := make([]Layer, len(types))
	for i, lt := range types {
		attnNorm := IdentityNorm()
		if i > 0 {
			attnNorm = newNorm()
		}
		attn, err := NewAttention(normal([]int64{3 * hidden, hidden}, 1/math.Sqrt(hidden)),
			normal([]int64{hidden, hidden}, 1/math.Sqrt(hidden)), heads, lt)
		if err != nil {
			t.Fatalf("NewAttention: %v", err)
		}
		mlp, err := NewMLP(normal([]int64{2 * inter, hidden}, 1/math.Sqrt(hidden)),
			normal([]int64{hidden, inter}, 1/math.Sqrt(inter)))
		if err != nil {
			t.Fatalf("NewMLP: %v", err)
		}
		if layers[i], err = NewLayer(attnNorm, attn, newNorm(), mlp); err != nil {
			t.Fatalf("NewLayer: %v", err)
		}
	}
	finalNorm := newNorm()
	e, err := NewEncoder(normal([]int64{vocab, hidden}, 1), newNorm(), layers, finalNorm, types)
	if err != nil {
		t.Fatalf("NewEncoder: %v", err)
	}

	ids := [][]int64{{4, 17, 9, 31, 2, 5}, {4, 23, 6, 0, 0, 0}}
	mask := [][]int64{{1, 1, 1, 1, 1, 1}, {1, 1, 1, 0, 0, 0}}
	got, err := e.Forward(ids, mask)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if want := []int64{batch, seq, hidden}; !slices.Equal(got.Shape(), want) {
		t.Fatalf("shape %v, want %v", got.Shape(), want)
	}
	y := got.Data()
	w := finalNorm.weight.Data()
	for r := range batch * seq {
		// final_norm's output divided by its weight is centred and of unit
		// variance (less eps/var).
		var mean, sq float64
		for j := range hidden {
			v := float64(y[r*hidden+j])
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("row %d: %g", r, v)
			}
			v /= float64(w[j])
			mean += v
			sq += v * v
		}
		mean /= hidden
		if variance := sq/hidden - mean*mean; math.Abs(mean) > 1e-4 || math.Abs(variance-1) > 1e-3 {
			t.Errorf("row %d after final_norm: mean %.2e, variance %.6f; want 0 and 1", r, mean, variance)
		}
	}

	// Real positions never see a padded key, so other padded ids leave them
	// bit for bit, while the padded positions themselves move.
	other := [][]int64{ids[0], {4, 23, 6, 11, 12, 13}}
	moved, err := e.Forward(other, mask)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	row := func(t *tensor.Tensor, b, from, to int) []float32 {
		return t.Data()[(b*seq+from)*hidden : (b*seq+to)*hidden]
	}
	if !slices.Equal(row(moved, 1, 0, 3), row(got, 1, 0, 3)) || !slices.Equal(row(moved, 0, 0, seq), row(got, 0, 0, seq)) {
		t.Error("the padded tokens' ids moved a real position")
	}
	if slices.Equal(row(moved, 1, 3, seq), row(got, 1, 3, seq)) {
		t.Error("the padded tokens' ids did not move the padded positions")
	}
	alone, err := e.Forward(ids[1:], mask[1:])
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if !slices.Equal(alone.Data(), row(got, 1, 0, seq)) {
		t.Error("row 1 on its own differs from row 1 in the batch")
	}
	t.Logf("3 layers at hidden %d, %d heads, intermediate %d: %v", hidden, heads, inter, time.Since(start))
}

// blocks are an encoder's parts, before NewEncoder.
type blocks struct {
	emb     *tensor.Tensor
	embNorm Norm
	layers  []Layer
	types   []LayerType
}

// tinyBlocks returns the blocks of a 2-layer encoder at hidden h with types
// full and sliding, for the error tests.
func tinyBlocks(t *testing.T, h int64) blocks {
	t.Helper()

	fill := func(shape ...int64) *tensor.Tensor {
		n := int64(1)
		for _, d := range shape {
			n *= d
		}
		data := make([]float32, n)
		for i := range data {
			data[i] = float32(i%7) / 7
		}
		return mustTensor(t, data, shape)
	}
	ln := func() Norm {
		n, err := NewNorm(fill(h))
		if err != nil {
			t.Fatal(err)
		}
		return n
	}
	types := []LayerType{FullAttention(160000), SlidingAttention(10000, 4)}
	layers := make([]Layer, 0, len(types))
	for i, lt := range types {
		attn, err := NewAttention(fill(3*h, h), fill(h, h), 2, lt)
		if err != nil {
			t.Fatal(err)
		}
		mlp, err := NewMLP(fill(6, h), fill(h, 3))
		if err != nil {
			t.Fatal(err)
		}
		attnNorm := IdentityNorm()
		if i > 0 {
			attnNorm = ln()
		}
		l, err := NewLayer(attnNorm, attn, ln(), mlp)
		if err != nil {
			t.Fatal(err)
		}
		layers = append(layers, l)
	}
	return blocks{emb: fill(10, h), embNorm: ln(), layers: layers, types: types}
}

func TestEncoderErrors(t *testing.T) {
	const h = 4
	tiny := tinyBlocks(t, h)
	emb, embNorm, layers, types := tiny.emb, tiny.embNorm, tiny.layers, tiny.types
	finalNorm := embNorm
	e, err := NewEncoder(emb, embNorm, layers, finalNorm, types)
	if err != nil {
		t.Fatalf("NewEncoder: %v", err)
	}
	// 0 and V-1, the vocabulary's ends, are valid ids.
	if _, err := e.Forward([][]int64{{0, 9, 3}}, [][]int64{{1, 1, 0}}); err != nil {
		t.Fatalf("Forward on a valid batch: %v", err)
	}

	for _, tc := range []struct {
		name       string
		ids, mask  [][]int64
		wantSubstr string
	}{
		{"id below 0", [][]int64{{1, -1}}, [][]int64{{1, 1}}, "-1"},
		{"id at the vocabulary size", [][]int64{{1, 10}}, [][]int64{{1, 1}}, "10"},
		{"ragged ids", [][]int64{{1, 2}, {1}}, [][]int64{{1, 1}, {1, 1}}, "row 1"},
		{"empty batch", nil, nil, "empty"},
		{"empty sequence", [][]int64{{}}, [][]int64{{}}, "empty"},
		{"mask rows", [][]int64{{1, 2}}, [][]int64{{1, 1}, {1, 1}}, "rows"},
		{"ragged mask", [][]int64{{1, 2}, {3, 4}}, [][]int64{{1, 1}, {1}}, "row 1"},
		{"mask value", [][]int64{{1, 2}}, [][]int64{{1, 2}}, "want 0 or 1"},
	} {
		if _, err := e.Forward(tc.ids, tc.mask); err == nil || !strings.Contains(err.Error(), tc.wantSubstr) {
			t.Errorf("%s: Forward error = %v, want one mentioning %q", tc.name, err, tc.wantSubstr)
		}
	}
	if _, err := (Encoder{}).Forward([][]int64{{1}}, [][]int64{{1}}); err == nil {
		t.Error("the zero Encoder ran")
	}

	wrongNorm, err := NewNorm(mustTensor(t, []float32{1, 1, 1}, []int64{3}))
	if err != nil {
		t.Fatal(err)
	}
	swapped := slices.Clone(layers)
	swapped[0], swapped[1] = swapped[1], swapped[0]
	layer0WithNorm := slices.Clone(layers)
	layer0WithNorm[0].attnNorm = embNorm
	layer1Identity := slices.Clone(layers)
	layer1Identity[1].attnNorm = IdentityNorm()
	wide := tinyBlocks(t, 2*h).layers
	for _, tc := range []struct {
		name               string
		emb                *tensor.Tensor
		embNorm, finalNorm Norm
		layers             []Layer
		types              []LayerType
		wantSubstr         string
	}{
		{"nil embeddings", nil, embNorm, finalNorm, layers, types, "embeddings"},
		{"rank-1 embeddings", mustTensor(t, make([]float32, h), []int64{h}), embNorm, finalNorm, layers, types, "embeddings"},
		{"identity embeddings norm", emb, IdentityNorm(), finalNorm, layers, types, "embeddings norm"},
		{"zero final norm", emb, embNorm, Norm{}, layers, types, "final norm"},
		{"final norm width", emb, embNorm, wrongNorm, layers, types, "final norm"},
		{"fewer layers than types", emb, embNorm, finalNorm, layers[:1], types, "2 layer types"},
		{"more layers than types", emb, embNorm, finalNorm, layers, types[:1], "1 layer types"},
		{"layer types out of order", emb, embNorm, finalNorm, swapped, types, "layer 0"},
		{"layer 0 with an attn_norm", emb, embNorm, finalNorm, layer0WithNorm, types, "layer 0"},
		{"layer 1 without one", emb, embNorm, finalNorm, layer1Identity, types, "layer 1"},
		{"layers of another width", emb, embNorm, finalNorm, wide, types, "hidden"},
	} {
		if _, err := NewEncoder(tc.emb, tc.embNorm, tc.layers, tc.finalNorm, tc.types); err == nil ||
			!strings.Contains(err.Error(), tc.wantSubstr) {
			t.Errorf("%s: NewEncoder error = %v, want one mentioning %q", tc.name, err, tc.wantSubstr)
		}
	}
}

func TestLayerErrors(t *testing.T) {
	tiny := tinyBlocks(t, 4)
	ln, l := tiny.embNorm, tiny.layers[1]
	wide := tinyBlocks(t, 8).layers
	wrongNorm, err := NewNorm(mustTensor(t, []float32{1, 1, 1}, []int64{3}))
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name              string
		attnNorm, mlpNorm Norm
		attn              Attention
		mlp               MLP
		wantSubstr        string
	}{
		{"zero attention", l.attnNorm, l.mlpNorm, Attention{}, l.mlp, "attention"},
		{"zero MLP", l.attnNorm, l.mlpNorm, l.attn, MLP{}, "MLP"},
		{"MLP of another width", l.attnNorm, l.mlpNorm, l.attn, wide[1].mlp, "hidden"},
		{"zero attn_norm", Norm{}, l.mlpNorm, l.attn, l.mlp, "attn_norm"},
		{"attn_norm width", wrongNorm, l.mlpNorm, l.attn, l.mlp, "attn_norm"},
		{"identity mlp_norm", l.attnNorm, IdentityNorm(), l.attn, l.mlp, "mlp_norm"},
		{"mlp_norm width", l.attnNorm, wrongNorm, l.attn, l.mlp, "mlp_norm"},
	} {
		if _, err := NewLayer(tc.attnNorm, tc.attn, tc.mlpNorm, tc.mlp); err == nil ||
			!strings.Contains(err.Error(), tc.wantSubstr) {
			t.Errorf("%s: NewLayer error = %v, want one mentioning %q", tc.name, err, tc.wantSubstr)
		}
	}
	if _, err := NewLayer(ln, l.attn, ln, l.mlp); err != nil {
		t.Errorf("NewLayer on valid blocks: %v", err)
	}
	if _, err := (Layer{}).Forward(mustTensor(t, make([]float32, 4), []int64{1, 1, 4}), [][]int64{{1}}); err == nil {
		t.Error("the zero Layer ran")
	}
}
