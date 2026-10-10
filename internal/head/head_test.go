package head

import (
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/MeKo-Christian/go-laya/backend"
	"github.com/MeKo-Christian/go-laya/internal/runtime/tensor"
)

// tolerance bounds |go - torch| for an element torch computed as want.
//
// Both sides run in float32 and round in different places: tensor.Linear sums
// in dotF32's lanes where torch's GEMM blocks its own way, the softmax and
// GELU exponentials here go through float64 math.Exp and math.Erf where torch
// uses SLEEF's vectorised float32 ones, and torch's fast path fuses the
// attention differently again (its output and the slow path's differ by 2e-6
// on the encoder_layer case). Each value goes through some ten float32
// stages -- two LayerNorms, a 3d-, d- and 4d-term dot product per layer, the
// residuals -- with sums of at most 32 terms at d=8, so a few ulps per stage
// relative to the operands' magnitude, which reaches 15 or so on the padded
// rows. 1e-5 relative is ~84 ulps; the absolute 1e-5 is for values that
// cancel to ~0. The bound is empirical rather than proven: the tests log the
// worst |diff|/tol per stage, which on this fixture is at most 9 % of it (the
// fast-path layer output; 6 % for the head's layers, 3 % for act_logits).
//
// What the test is for sits orders of magnitude above it: each wrong reading
// of the head the PR's mutations tried (type_emb at [CLS] only, post-norm,
// GELU in the FFN, -inf for the fill, the unclamped entropy denominator, the
// pooled marker, no key-padding mask, q and k swapped) moves an output by
// 1e-2 or more.
func tolerance(want float32) float64 {
	return 1e-5 * (1 + math.Abs(float64(want)))
}

// closeTo compares got against want element-wise within tolerance and
// returns the worst |diff|/tol, failing the test on any element beyond it.
func closeTo(t *testing.T, what string, got []float32, want tensorRec) float64 {
	t.Helper()

	if len(got) != len(want.Data) {
		t.Fatalf("%s: %d values, want %d (shape %v)", what, len(got), len(want.Data), want.Shape)
	}
	var worst float64
	for i, w := range want.Data {
		diff := math.Abs(float64(got[i]) - float64(w))
		tol := tolerance(w)
		worst = math.Max(worst, diff/tol)
		if !(diff <= tol) { // NaN-safe: a NaN fails here
			t.Errorf("%s[%d] = %.9g, want %.9g (|diff| %.3g > tol %.3g)", what, i, got[i], w, diff, tol)
		}
	}
	return worst
}

func flat(rows [][]float32) []float32 {
	var out []float32
	for _, r := range rows {
		out = append(out, r...)
	}
	return out
}

func newFromCase(t *testing.T, c modelCase) *Head {
	t.Helper()

	h, err := New(modelWeights(t, c))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return h
}

// TestHeadMatchesTorch runs the recorded DecisionModel head through the port
// and compares every stage the record holds against what torch computed:
// invariant #35 (type_emb at every position), #36 (the manual loop over
// norm_first ReLU layers with the key-padding mask), the gather, #37 (logits
// with -1e4 over the fill) and #38 (the features and act_logits).
func TestHeadMatchesTorch(t *testing.T) {
	f := loadHead(t)
	for _, c := range casesOf[modelCase](t, f, "decision_model") {
		t.Run(c.Name, func(t *testing.T) {
			h := newFromCase(t, c)
			if h.NHead() != c.NHead || c.NHead != max(1, c.D/64) {
				t.Fatalf("nhead %d, fixture %d, want max(1, d//64) = %d", h.NHead(), c.NHead, max(1, c.D/64))
			}
			// The d=8 head has one head, which keeps torch off the fast path
			// ("num_head is odd"); TestLayerMatchesTorch covers the fast one.
			if c.Path != "slow" || c.FastPathCalls != 0 {
				t.Errorf("fixture path %q (%d fast calls), want slow for one head", c.Path, c.FastPathCalls)
			}

			x := c.H.tensor(t, "h")
			var tr trace
			if _, _, err := h.forward(x, c.batch(), &tr); err != nil {
				t.Fatalf("forward: %v", err)
			}
			if !slices.Equal(x.Data(), c.H.Data) {
				t.Error("forward modified the caller's hidden state")
			}

			worst := map[string]float64{}
			worst["h_typed"] = closeTo(t, "h_typed", tr.typed.Data(), c.HTyped)
			if len(tr.layers) != len(c.HLayers) {
				t.Fatalf("%d layer outputs, want %d", len(tr.layers), len(c.HLayers))
			}
			for i, l := range tr.layers {
				worst["h_layers"] = math.Max(worst["h_layers"], closeTo(t, "h_layers", l.Data(), c.HLayers[i]))
			}
			worst["gathered"] = closeTo(t, "gathered", tr.gathered.Data(), c.Gathered)
			worst["logits"] = closeTo(t, "logits", flat(tr.logits), c.Logits)
			worst["probs"] = closeTo(t, "probs", flat(tr.probs), c.Probs)
			worst["feats"] = closeTo(t, "feats", flat(tr.feats), c.Feats)
			worst["act_logits"] = closeTo(t, "act_logits", flat(tr.act), c.ActLogits)
			for _, k := range []string{"h_typed", "h_layers", "gathered", "logits", "probs", "feats", "act_logits"} {
				t.Logf("%-10s worst |diff|/tol %.4f", k, worst[k])
			}

			// #37: the fill is exactly -1e4, not merely close to it.
			for b, row := range c.Batch.MarkerMask {
				for j, isMarker := range row {
					if !isMarker && tr.logits[b][j] != -1e4 {
						t.Errorf("logits[%d][%d] = %g over the fill, want exactly -1e4", b, j, tr.logits[b][j])
					}
				}
			}

			// Forward is forward's outputs, nothing else.
			logits, act, err := h.Forward(c.H.tensor(t, "h"), c.batch())
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
			if !equalRows(logits, tr.logits) || !equalRows(act, tr.act) {
				t.Error("Forward's outputs differ from forward's trace")
			}
			if len(act) != len(c.Batch.QType) || len(act[0]) != c.NAct {
				t.Errorf("act is %dx%d, want %dx%d", len(act), len(act[0]), len(c.Batch.QType), c.NAct)
			}
		})
	}
}

func equalRows(a, b [][]float32) bool {
	return slices.EqualFunc(a, b, slices.Equal)
}

// TestLayerMatchesTorch runs the recorded two-head TransformerEncoderLayer,
// the head split with nhead > 1 the d=8 head cannot show, against the output
// of torch's fast path, which is the path the checkpoints' 16 and 12 heads
// take. The slow path's output is within the same bound.
func TestLayerMatchesTorch(t *testing.T) {
	f := loadHead(t)
	for _, c := range casesOf[layerCase](t, f, "encoder_layer") {
		t.Run(c.Name, func(t *testing.T) {
			if c.Path != "fast" || c.FastPathCalls != 1 {
				t.Errorf("fixture path %q (%d fast calls), want the fast path", c.Path, c.FastPathCalls)
			}
			l, err := NewLayer(layerWeights(t, c.Weights, ""), c.NHead)
			if err != nil {
				t.Fatalf("NewLayer: %v", err)
			}
			y, err := l.Forward(c.Input.tensor(t, "input"), c.PaddingMask)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
			t.Logf("fast path: worst |diff|/tol %.4f", closeTo(t, "output", y.Data(), c.Output))
			t.Logf("slow path: worst |diff|/tol %.4f", closeTo(t, "output_slow", y.Data(), c.OutputSlow))
		})
	}
}

// TestTopKNeedsTwoMarkers: with fewer than two marker columns over the whole
// batch, upstream's p.topk(2, -1) (common.py:122) raises, which the fixture
// records for kmax 1 and 0. A batch of single-option choice questions gets
// there: the Go validation rejects only an empty option list (ErrNoOptions).
// The port errors where Python raises.
func TestTopKNeedsTwoMarkers(t *testing.T) {
	f := loadHead(t)
	m := casesOf[modelCase](t, f, "decision_model")[0]
	h := newFromCase(t, m)
	for _, c := range casesOf[topkCase](t, f, "topk_error") {
		t.Run(c.Name, func(t *testing.T) {
			if !strings.Contains(c.Error, "selected index k out of range") {
				t.Fatalf("fixture error %q is not topk's", c.Error)
			}
			const seq = 4
			x, err := tensor.Zeros([]int64{1, seq, int64(m.D)})
			if err != nil {
				t.Fatal(err)
			}
			b := backend.Batch{
				InputIDs:      [][]int64{{5, 6, 7, 8}},
				AttentionMask: [][]int64{{1, 1, 1, 1}},
				MarkerPos:     [][]int64{make([]int64, c.KMax)},
				MarkerMask:    [][]bool{make([]bool, c.KMax)},
				QType:         []int64{0},
			}
			for j := range c.KMax {
				b.MarkerPos[0][j], b.MarkerMask[0][j] = 2, true
			}
			_, _, err = h.Forward(x, b)
			if err == nil || !strings.Contains(err.Error(), "topk(2)") {
				t.Fatalf("kmax %d: error %v; torch raises %q", c.KMax, err, c.Error)
			}
			t.Logf("kmax %d: %v", c.KMax, err)
		})
	}
}

// TestMarkerFillMinusOne: marker_pos.clamp(min=0) (common.py:114) reads row 0
// for any negative fill, and the fill's logit is replaced by -1e4, so a fill
// of -1 gives the outputs a fill of 0 gives, bit for bit. The fixture records
// that torch agrees.
func TestMarkerFillMinusOne(t *testing.T) {
	c := casesOf[modelCase](t, loadHead(t), "decision_model")[0]
	for k, v := range c.FillMinusOne {
		if !v {
			t.Errorf("fixture: torch with a fill of -1: %s is false", k)
		}
	}
	if c.MarkerFill != 0 {
		t.Fatalf("fixture marker fill %d, want the collator's 0", c.MarkerFill)
	}
	h := newFromCase(t, c)
	b := c.batch()
	logits0, act0, err := h.Forward(c.H.tensor(t, "h"), b)
	if err != nil {
		t.Fatal(err)
	}

	neg := slices.Clone(b.MarkerPos)
	fills := 0
	for i, row := range neg {
		neg[i] = slices.Clone(row)
		for j := range row {
			if !b.MarkerMask[i][j] {
				neg[i][j] = -1
				fills++
			}
		}
	}
	if fills == 0 {
		t.Fatal("the fixture batch has no marker fill")
	}
	b.MarkerPos = neg
	logits1, act1, err := h.Forward(c.H.tensor(t, "h"), b)
	if err != nil {
		t.Fatal(err)
	}
	if !equalRows(logits0, logits1) || !equalRows(act0, act1) {
		t.Errorf("a fill of -1 moved the outputs:\nlogits %v\n  vs   %v\nact %v\n vs %v", logits0, logits1, act0, act1)
	}
}

// TestPaddedRowsDoNotReachOutputs: the key-padding mask drops every padded
// key, and the head reads only the marker rows and h[:, 0], so whatever sits
// in a padded row of h leaves logits and act unchanged, bit for bit.
func TestPaddedRowsDoNotReachOutputs(t *testing.T) {
	c := casesOf[modelCase](t, loadHead(t), "decision_model")[0]
	h := newFromCase(t, c)
	b := c.batch()
	logits0, act0, err := h.Forward(c.H.tensor(t, "h"), b)
	if err != nil {
		t.Fatal(err)
	}

	x := c.H.tensor(t, "h")
	data, seq, d := x.RawData(), len(b.AttentionMask[0]), c.D
	changed := 0
	for i, row := range b.AttentionMask {
		for s, m := range row {
			if m == 0 {
				for j := range d {
					data[(i*seq+s)*d+j] = 1e3 * float32(j+1)
				}
				changed++
			}
		}
	}
	if changed == 0 {
		t.Fatal("the fixture batch has no padding")
	}
	logits1, act1, err := h.Forward(x, b)
	if err != nil {
		t.Fatal(err)
	}
	if !equalRows(logits0, logits1) || !equalRows(act0, act1) {
		t.Errorf("changing %d padded rows moved the outputs", changed)
	}
}

// TestNHead pins nhead = max(1, d // 64) (common.py:96): the checkpoints'
// 1024 and 768 give 16 and 12 heads of 64.
func TestNHead(t *testing.T) {
	for d, want := range map[int]int{8: 1, 63: 1, 64: 1, 127: 1, 128: 2, 768: 12, 1024: 16} {
		if got := nheadFor(d); got != want {
			t.Errorf("nheadFor(%d) = %d, want %d", d, got, want)
		}
	}
}

// synthetic draws a head with seeded random weights at width d.
func synthetic(t *testing.T, rng *rand.Rand, d, layers, nAct int) Weights {
	t.Helper()

	mat := func(rows, cols int) *tensor.Tensor {
		data := make([]float32, rows*cols)
		s := 1 / math.Sqrt(float64(cols))
		for i := range data {
			data[i] = float32(rng.NormFloat64() * s)
		}
		x, err := tensor.New(data, []int64{int64(rows), int64(cols)})
		if err != nil {
			t.Fatal(err)
		}
		return x
	}
	vec := func(n int, mean float64) *tensor.Tensor {
		data := make([]float32, n)
		for i := range data {
			data[i] = float32(mean + 0.1*rng.NormFloat64())
		}
		x, err := tensor.New(data, []int64{int64(n)})
		if err != nil {
			t.Fatal(err)
		}
		return x
	}
	w := Weights{
		TypeEmb:             mat(3, d),
		ScorerNormWeight:    vec(d, 1),
		ScorerNormBias:      vec(d, 0),
		ScorerLinear1Weight: mat(d, d),
		ScorerLinear1Bias:   vec(d, 0),
		ScorerLinear2Weight: mat(1, d),
		ScorerLinear2Bias:   vec(1, 0),
		ActLinear1Weight:    mat(ActHidden, d+4),
		ActLinear1Bias:      vec(ActHidden, 0),
		ActLinear2Weight:    mat(nAct, ActHidden),
		ActLinear2Bias:      vec(nAct, 0),
	}
	for range layers {
		w.Layers = append(w.Layers, LayerWeights{
			InProjWeight: mat(3*d, d), InProjBias: vec(3*d, 0),
			OutProjWeight: mat(d, d), OutProjBias: vec(d, 0),
			Linear1Weight: mat(4*d, d), Linear1Bias: vec(4*d, 0),
			Linear2Weight: mat(d, 4*d), Linear2Bias: vec(d, 0),
			Norm1Weight: vec(d, 1), Norm1Bias: vec(d, 0),
			Norm2Weight: vec(d, 1), Norm2Bias: vec(d, 0),
		})
	}
	return w
}

// TestHeadRealDimensions runs the head at the checkpoints' widths with
// synthetic weights: ModernBERT-large's d=1024 (16 heads) and mmBERT-base's
// d=768 (12 heads), two layers, n_act 2. The values are the fixture's
// business; this checks the shapes, the fill, finiteness and that padding
// stays out at full width.
func TestHeadRealDimensions(t *testing.T) {
	for _, tc := range []struct {
		name     string
		d, heads int
	}{
		{"modernbert-large", 1024, 16},
		{"mmbert-base", 768, 12},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// ~0.5 s for both widths, most of it drawing 25M weights, so it
			// runs under -short too.
			start := time.Now()
			rng := rand.New(rand.NewPCG(86, uint64(tc.d)))
			h, err := New(synthetic(t, rng, tc.d, 2, 2))
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			if h.NHead() != tc.heads {
				t.Fatalf("nhead %d, want %d", h.NHead(), tc.heads)
			}

			const batch, seq = 2, 12
			data := make([]float32, batch*seq*tc.d)
			for i := range data {
				data[i] = float32(rng.NormFloat64())
			}
			x, err := tensor.New(data, []int64{batch, seq, int64(tc.d)})
			if err != nil {
				t.Fatal(err)
			}
			b := backend.Batch{
				InputIDs:      [][]int64{make([]int64, seq), make([]int64, seq)},
				AttentionMask: [][]int64{ones(seq, seq), ones(seq, 7)},
				MarkerPos:     [][]int64{{2, 4, 6, 9}, {3, 5, 0, 0}},
				MarkerMask:    [][]bool{{true, true, true, true}, {true, true, false, false}},
				QType:         []int64{0, 2},
			}
			logits, act, err := h.Forward(x, b)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
			if len(logits) != batch || len(act) != batch {
				t.Fatalf("%d logits rows and %d act rows, want %d", len(logits), len(act), batch)
			}
			for i := range batch {
				if len(logits[i]) != 4 || len(act[i]) != 2 {
					t.Fatalf("row %d: logits %d wide, act %d; want 4 (kmax) and 2", i, len(logits[i]), len(act[i]))
				}
				for j, v := range logits[i] {
					if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
						t.Errorf("logits[%d][%d] = %g", i, j, v)
					}
					if !b.MarkerMask[i][j] && v != -1e4 {
						t.Errorf("logits[%d][%d] = %g over the fill, want -1e4", i, j, v)
					}
				}
				for j, v := range act[i] {
					if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
						t.Errorf("act[%d][%d] = %g", i, j, v)
					}
				}
			}

			// Padding stays out at full width too.
			for s := 7; s < seq; s++ {
				for j := range tc.d {
					data[(seq+s)*tc.d+j] = 100
				}
			}
			logits2, act2, err := h.Forward(x, b)
			if err != nil {
				t.Fatal(err)
			}
			if !equalRows(logits, logits2) || !equalRows(act, act2) {
				t.Error("changing padded rows moved the outputs")
			}
			t.Logf("two forward passes in %v", time.Since(start))
		})
	}
}

func ones(n, tokens int) []int64 {
	out := make([]int64, n)
	for i := range tokens {
		out[i] = 1
	}
	return out
}

// TestNewRejectsBadShapes: every tensor's shape is checked against d, which
// type_emb fixes, before the head is usable.
func TestNewRejectsBadShapes(t *testing.T) {
	const d = 8
	rng := rand.New(rand.NewPCG(1, 2))
	good := func() Weights { return synthetic(t, rng, d, 2, 2) }
	if _, err := New(good()); err != nil {
		t.Fatalf("New(good): %v", err)
	}
	if _, err := New(Weights{}); err == nil {
		t.Error("New(Weights{}): no error")
	}
	bad, _ := tensor.Zeros([]int64{d + 1, d})
	badVec, _ := tensor.Zeros([]int64{d + 1})
	for name, mutate := range map[string]func(*Weights){
		"type_emb rows":       func(w *Weights) { w.TypeEmb, _ = tensor.Zeros([]int64{2, d}) },
		"type_emb nil":        func(w *Weights) { w.TypeEmb = nil },
		"in_proj [d,d]":       func(w *Weights) { w.Layers[0].InProjWeight = bad },
		"in_proj_bias":        func(w *Weights) { w.Layers[1].InProjBias = badVec },
		"out_proj":            func(w *Weights) { w.Layers[0].OutProjWeight = bad },
		"out_proj_bias nil":   func(w *Weights) { w.Layers[0].OutProjBias = nil },
		"linear1 transposed":  func(w *Weights) { w.Layers[0].Linear1Weight = w.Layers[0].Linear2Weight },
		"linear2 transposed":  func(w *Weights) { w.Layers[1].Linear2Weight = w.Layers[1].Linear1Weight },
		"linear1_bias":        func(w *Weights) { w.Layers[0].Linear1Bias = badVec },
		"norm1 weight":        func(w *Weights) { w.Layers[0].Norm1Weight = badVec },
		"norm2 bias":          func(w *Weights) { w.Layers[1].Norm2Bias = badVec },
		"scorer norm":         func(w *Weights) { w.ScorerNormWeight = badVec },
		"scorer linear1":      func(w *Weights) { w.ScorerLinear1Weight = bad },
		"scorer linear2 [2]":  func(w *Weights) { w.ScorerLinear2Weight, _ = tensor.Zeros([]int64{2, d}) },
		"scorer linear2 bias": func(w *Weights) { w.ScorerLinear2Bias = badVec },
		"act_head.0 width":    func(w *Weights) { w.ActLinear1Weight, _ = tensor.Zeros([]int64{ActHidden, d}) },
		"act_head.0 hidden":   func(w *Weights) { w.ActLinear1Weight, _ = tensor.Zeros([]int64{128, d + 4}) },
		"act_head.2 in":       func(w *Weights) { w.ActLinear2Weight, _ = tensor.Zeros([]int64{2, 128}) },
		"act_head.2 empty":    func(w *Weights) { w.ActLinear2Weight, _ = tensor.Zeros([]int64{0, ActHidden}) },
		"act_head.2 bias":     func(w *Weights) { w.ActLinear2Bias = badVec },
	} {
		w := good()
		mutate(&w)
		if _, err := New(w); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
	if _, err := NewLayer(good().Layers[0], 3); err == nil {
		t.Error("NewLayer with 3 heads over d=8: no error")
	}
	if _, err := NewLayer(good().Layers[0], 0); err == nil {
		t.Error("NewLayer with 0 heads: no error")
	}
	// d = 200 gives max(1, 200 // 64) = 3 heads, which upstream's
	// nn.MultiheadAttention refuses even with head_layers 0.
	if _, err := New(synthetic(t, rng, 200, 0, 2)); err == nil || !strings.Contains(err.Error(), "do not divide") {
		t.Errorf("d = 200 with no layers: error %v, want one about the heads", err)
	}
}

// TestForwardErrors: a batch that does not fit h, or that upstream would
// reject (an embedding index out of range, a gather out of bounds), is an
// error rather than a panic or a plausible answer.
func TestForwardErrors(t *testing.T) {
	c := casesOf[modelCase](t, loadHead(t), "decision_model")[0]
	h := newFromCase(t, c)
	if _, _, err := h.Forward(c.H.tensor(t, "h"), c.batch()); err != nil {
		t.Fatalf("the fixture batch: %v", err)
	}
	seq := len(c.Batch.AttentionMask[0])
	// Each case names a substring of the error it must get, so a batch
	// rejected for some other reason does not pass.
	for _, tc := range []struct {
		name, want string
		mutate     func(*backend.Batch)
	}{
		{"fewer mask rows", "attention mask has 2 rows", func(b *backend.Batch) { b.AttentionMask = b.AttentionMask[:2] }},
		{"short mask row", "attention mask row 1 has", func(b *backend.Batch) { b.AttentionMask[1] = b.AttentionMask[1][:seq-1] }},
		{"mask value 2", "want 0 or 1", func(b *backend.Batch) { b.AttentionMask[0][0] = 2 }},
		{"all-padding row", "row 2 has no real token", func(b *backend.Batch) { b.AttentionMask[2] = make([]int64, seq) }},
		{"input_ids rows", "input_ids rows", func(b *backend.Batch) { b.InputIDs = b.InputIDs[:1] }},
		{"input_ids width", "input_ids row 0", func(b *backend.Batch) { b.InputIDs[0] = b.InputIDs[0][:2] }},
		{"qtype count", "2 qtypes", func(b *backend.Batch) { b.QType = b.QType[:2] }},
		{"qtype 3", "qtype[0] = 3", func(b *backend.Batch) { b.QType[0] = 3 }},
		{"qtype -1", "qtype[1] = -1", func(b *backend.Batch) { b.QType[1] = -1 }},
		{"marker rows", "2 marker_pos and 3 marker_mask", func(b *backend.Batch) { b.MarkerPos = b.MarkerPos[:2] }},
		{"ragged marker_pos", "marker row 1", func(b *backend.Batch) { b.MarkerPos[1] = b.MarkerPos[1][:3] }},
		{"marker_mask rows", "3 marker_pos and 2 marker_mask", func(b *backend.Batch) { b.MarkerMask = b.MarkerMask[:2] }},
		{"ragged marker_mask", "marker row 2", func(b *backend.Batch) { b.MarkerMask[2] = b.MarkerMask[2][:1] }},
		{"marker past row", "marker_pos[0][1]", func(b *backend.Batch) { b.MarkerPos[0][1] = int64(seq) }},
		{"fill past row", "marker_pos[2][3]", func(b *backend.Batch) { b.MarkerPos[2][3] = int64(seq) + 5 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b := cloneBatch(c.batch())
			tc.mutate(&b)
			_, _, err := h.Forward(c.H.tensor(t, "h"), b)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error %v, want one containing %q", err, tc.want)
			}
		})
	}

	b := c.batch()
	for name, shape := range map[string][]int64{
		"rank 2":      {3, int64(seq) * int64(c.D)},
		"batch":       {2, int64(seq), int64(c.D)},
		"seq":         {3, int64(seq) + 1, int64(c.D)},
		"hidden size": {3, int64(seq), int64(c.D) + 1},
	} {
		x, err := tensor.Zeros(shape)
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err := h.Forward(x, b); err == nil {
			t.Errorf("h %s %v: no error", name, shape)
		}
	}
	if _, _, err := h.Forward(nil, b); err == nil {
		t.Error("nil h: no error")
	}
	var zero Head
	if _, _, err := zero.Forward(c.H.tensor(t, "h"), b); err == nil {
		t.Error("zero Head: no error")
	}
	var zeroLayer Layer
	if _, err := zeroLayer.Forward(c.H.tensor(t, "h"), b.AttentionMask); err == nil {
		t.Error("zero Layer: no error")
	}
}

func cloneBatch(b backend.Batch) backend.Batch {
	clone := func(rows [][]int64) [][]int64 {
		out := make([][]int64, len(rows))
		for i, r := range rows {
			out[i] = slices.Clone(r)
		}
		return out
	}
	mask := make([][]bool, len(b.MarkerMask))
	for i, r := range b.MarkerMask {
		mask[i] = slices.Clone(r)
	}
	return backend.Batch{
		InputIDs:      clone(b.InputIDs),
		AttentionMask: clone(b.AttentionMask),
		MarkerPos:     clone(b.MarkerPos),
		MarkerMask:    mask,
		QType:         slices.Clone(b.QType),
	}
}
