package modernbert

import (
	"encoding/json"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/runtime/tensor"
)

// attnCase is one "attention" record of testdata/ops.json: the real
// ModernBertAttention of the dumper's attention model, with the seeded random
// Wqkv [3H, H] and Wo [H, H] it ran with, the padding mask, the 4-D mask
// transformers built from it for sdpa, and the cos/sin of the real
// ModernBertRotaryEmbedding at the layer type's theta.
type attnCase struct {
	Name        string `json:"name"`
	Module      string `json:"module"`
	ModuleClass string `json:"module_class"`
	LayerType   string `json:"layer_type"`
	Config      struct {
		HiddenSize          int64    `json:"hidden_size"`
		NumAttentionHeads   int      `json:"num_attention_heads"`
		HeadDim             int      `json:"head_dim"`
		AttnImplementation  string   `json:"attn_implementation"`
		AttentionBias       *bool    `json:"attention_bias"`
		RopeType            string   `json:"rope_type"`
		RopeTheta           float64  `json:"rope_theta"`
		LocalAttention      int      `json:"local_attention"`
		SlidingWindow       int      `json:"sliding_window"`
		ModuleSlidingWindow *int     `json:"module_sliding_window"`
		Scaling             *float64 `json:"scaling"`
	} `json:"config"`
	Padding [][]int64 `json:"padding_mask"`
	Mask    [][][]int `json:"mask"`
	Input   tensorRec `json:"input"`
	Wqkv    tensorRec `json:"wqkv"`
	Wo      tensorRec `json:"wo"`
	Cos     tensorRec `json:"cos"`
	Sin     tensorRec `json:"sin"`
	Output  tensorRec `json:"output"`
}

const (
	fullAttentionName    = "full_attention"
	slidingAttentionName = "sliding_attention"
)

// layer returns the LayerType the record's config describes, built the way
// the encoder will build it from config.json.
func (c attnCase) layer(tb testing.TB) LayerType {
	tb.Helper()

	switch c.LayerType {
	case fullAttentionName:
		return FullAttention(c.Config.RopeTheta)
	case slidingAttentionName:
		return SlidingAttention(c.Config.RopeTheta, c.Config.LocalAttention)
	}
	tb.Fatalf("%s: layer type %q", c.Name, c.LayerType)
	return LayerType{}
}

// attnVariant is ModernBertAttention read correctly or in one of the plausible
// wrong ways; as with mlpVariant, the wrong ones exist so that the test can
// check the fixture tells each apart from the right one.
type attnVariant int

const (
	attnExact        attnVariant = iota
	attnQKSwap                   // q and k swapped
	attnSplitOrder               // Wqkv rows read as (heads, 3, head_dim)
	attnInterleaved              // interleaved-pair RoPE (ops.RoPE) for rotate_half
	attnOtherTheta               // the other layer type's theta
	attnWindowWider              // the window one wider than the mask's
	attnWindowNarrow             // the window one narrower
	attnNoPadding                // the padding mask dropped
	attnScaleByD                 // scores scaled by 1/head_dim, not 1/sqrt(head_dim)
)

var attnVariantNames = map[attnVariant]string{
	attnQKSwap:       "q and k swapped",
	attnSplitOrder:   "(heads, 3, head_dim) split",
	attnInterleaved:  "interleaved-pair RoPE",
	attnOtherTheta:   "the other layer type's theta",
	attnWindowWider:  "window one wider",
	attnWindowNarrow: "window one narrower",
	attnNoPadding:    "padding mask dropped",
	attnScaleByD:     "scale 1/head_dim",
}

// attnRefInput is what attnRef needs: the record's tensors as flat float32
// buffers, the dimensions, and the layer's theta and window (negative for full
// attention). otherTheta is the other layer type's, for attnOtherTheta.
type attnRefInput struct {
	x, wqkv, wo        []float32
	padding            [][]int64
	batch, seq, hidden int
	heads              int
	theta, otherTheta  float64
	window             int
}

// attnRef computes ModernBertAttention in float64, read as v, with the sdpa
// convention that a query with no allowed key yields zeros. It also returns,
// per output element, the magnitude attnTolerance scales with:
//
//	M_o = sum_c |Wo_oc| * (A_c + |o_c|)
//	A_c = sum_k p_k * (|v_kc - o_c| * (E_k + 1) + Va_kc + |v_kc|)
//	E_k = scale * sum_j (Qa_j*|k_j| + |q_j|*Ka_j + |q_j*k_j|)
//
// for the attention output o of the query's head, its probabilities p over
// keys k, and the rotated q and k. Qa, Ka and Va are the sums of |x_i*W_ri|
// behind each projected element (over both elements of a RoPE pair for q and
// k). Each term is how far one float32 rounding per stage can move y_o: E_k
// for a score, which moves o_c by p_k*|v_kc - o_c| through the softmax, the
// projection of v, the sum over keys and the sum in Wo.
func attnRef(in attnRefInput, v attnVariant) (y, mag []float64) {
	r := attnRefState{attnRefInput: in, d: in.hidden / in.heads}
	if v == attnOtherTheta {
		r.theta = in.otherTheta
	}
	if r.window >= 0 && v == attnWindowWider {
		r.window++
	}
	if r.window >= 0 && v == attnWindowNarrow {
		r.window--
	}
	r.scale = 1 / math.Sqrt(float64(r.d))
	if v == attnScaleByD {
		r.scale = 1 / float64(r.d)
	}
	r.noPadding = v == attnNoPadding

	r.project(v == attnSplitOrder)
	if v == attnQKSwap {
		n := r.batch * r.seq * r.hidden
		for i := range n {
			r.proj[i], r.proj[n+i] = r.proj[n+i], r.proj[i]
			r.pAbs[i], r.pAbs[n+i] = r.pAbs[n+i], r.pAbs[i]
		}
	}
	r.rotate(v == attnInterleaved)
	return r.attend()
}

// attnRefState is attnRef's working state: the input with the variant's
// theta, window and scale applied, and the projections q, k, v with their
// magnitudes, at proj[idx(t, b, s, head, j)] for t = 0, 1, 2.
type attnRefState struct {
	attnRefInput

	d          int
	scale      float64
	noPadding  bool
	proj, pAbs []float64
}

func (r *attnRefState) idx(t, b, s, head, j int) int {
	return (((t*r.batch+b)*r.seq+s)*r.heads+head)*r.d + j
}

// project computes x * Wqkv^T and the sums of |x_i*W_ri|, reading each row of
// Wqkv as (t, head, j) in (3, heads, head_dim) order, or as (head, t, j) if
// splitWrong.
func (r *attnRefState) project(splitWrong bool) {
	h, d := r.hidden, r.d
	r.proj = make([]float64, 3*r.batch*r.seq*h)
	r.pAbs = make([]float64, len(r.proj))
	for b := range r.batch {
		for s := range r.seq {
			xr := r.x[(b*r.seq+s)*h : (b*r.seq+s+1)*h]
			for row := range 3 * h {
				var sum, abs float64
				for i, xv := range xr {
					p := float64(xv) * float64(r.wqkv[row*h+i])
					sum += p
					abs += math.Abs(p)
				}
				t, head, j := row/h, (row%h)/d, row%d
				if splitWrong {
					head, t = row/(3*d), (row%(3*d))/d
				}
				r.proj[r.idx(t, b, s, head, j)], r.pAbs[r.idx(t, b, s, head, j)] = sum, abs
			}
		}
	}
}

// rotate applies RoPE to q and k in place: rotate_half pairs j with j +- d/2
// at frequency j mod d/2, the interleaved form pairs 2i with 2i+1 at
// frequency i. A rotated element's magnitude is its pair's sum.
func (r *attnRefState) rotate(interleaved bool) {
	d, half := r.d, r.d/2
	rot := make([]float64, d)
	rotAbs := make([]float64, d)
	for t := range 2 {
		for b := range r.batch {
			for s := range r.seq {
				for head := range r.heads {
					base := r.idx(t, b, s, head, 0)
					x, xa := r.proj[base:base+d], r.pAbs[base:base+d]
					for j := range d {
						freq, partner, sign := j%half, (j+half)%d, 1.0
						if interleaved {
							freq, partner = j/2, j^1
						}
						if (interleaved && j%2 == 0) || (!interleaved && j < half) {
							sign = -1
						}
						ang := float64(s) / math.Pow(r.theta, float64(2*freq)/float64(d))
						rot[j] = x[j]*math.Cos(ang) + sign*x[partner]*math.Sin(ang)
						rotAbs[j] = xa[j] + xa[partner]
					}
					copy(x, rot)
					copy(xa, rotAbs)
				}
			}
		}
	}
}

// attend runs the masked softmax attention over the rotated projections and
// Wo, returning y and its magnitudes (see attnRef).
func (r *attnRefState) attend() (y, mag []float64) {
	h, d := r.hidden, r.d
	y = make([]float64, r.batch*r.seq*h)
	mag = make([]float64, len(y))
	o := make([]float64, h)
	oMag := make([]float64, h)
	for b := range r.batch {
		for q := range r.seq {
			for head := range r.heads {
				p, e := r.probs(b, q, head)
				for j := range d {
					var sum, m float64
					for k, pk := range p {
						sum += pk * r.proj[r.idx(2, b, k, head, j)]
					}
					for k, pk := range p {
						vv := r.proj[r.idx(2, b, k, head, j)]
						m += pk * (math.Abs(vv-sum)*(e[k]+1) + r.pAbs[r.idx(2, b, k, head, j)] + math.Abs(vv))
					}
					o[head*d+j], oMag[head*d+j] = sum, m
				}
			}
			for oi := range h {
				var sum, m float64
				for c := range h {
					w := float64(r.wo[oi*h+c])
					sum += w * o[c]
					m += math.Abs(w) * (oMag[c] + math.Abs(o[c]))
				}
				y[(b*r.seq+q)*h+oi], mag[(b*r.seq+q)*h+oi] = sum, m
			}
		}
	}
	return y, mag
}

// probs returns query q's attention probabilities over the keys of its row
// and head, all zero if it may attend to none, and each score's error
// magnitude E_k (see attnRef).
func (r *attnRefState) probs(b, q, head int) (p, e []float64) {
	p = make([]float64, r.seq)
	e = make([]float64, r.seq)
	allowed := make([]bool, r.seq)
	top := math.Inf(-1)
	qv, qa := r.proj[r.idx(0, b, q, head, 0):], r.pAbs[r.idx(0, b, q, head, 0):]
	for k := range r.seq {
		allowed[k] = (r.noPadding || r.padding[b][k] != 0) && (r.window < 0 || absInt(q-k) <= r.window)
		if !allowed[k] {
			continue
		}
		kv, ka := r.proj[r.idx(1, b, k, head, 0):], r.pAbs[r.idx(1, b, k, head, 0):]
		var score, err float64
		for j := range r.d {
			score += qv[j] * kv[j]
			err += qa[j]*math.Abs(kv[j]) + math.Abs(qv[j])*ka[j] + math.Abs(qv[j]*kv[j])
		}
		p[k], e[k] = score*r.scale, err*r.scale
		top = math.Max(top, p[k])
	}
	var total float64
	for k := range p {
		if allowed[k] {
			p[k] = math.Exp(p[k] - top)
			total += p[k]
		} else {
			p[k] = 0
		}
	}
	for k := range p {
		if total > 0 {
			p[k] /= total
		}
	}
	return p, e
}

// attnTolerance bounds |go - torch| for an output element of magnitude m (see
// attnRef).
//
// Both sides run in float32 and round in different places: torch's sdpa
// kernel scales q and k and runs its softmax its own way, tensor.Linear and
// MatMulTransB use dotF32's lanes, and the exponentials here go through
// float64 math.Exp. m is what one float32 rounding per stage can move y_o by,
// so the bound is empirical rather than proven, as mlpTolerance's: every sum is
// at most 12 terms long here, and the worst case seen uses 3 % of the bound
// (the test logs it). The factor 4 leaves room for another CPU's kernels, and
// the 1e-6 for outputs that cancel to ~0. The bound runs from 1e-6 on the rows
// with no key to 5e-5 on the largest outputs.
//
// The failure modes the test is for sit far above it, and
// TestAttentionMatchesTorch asserts that they do: on this fixture the scale
// 1/head_dim misses torch by 39000 times the bound, the other wrong readings
// by 140000 times and more (the test logs each).
func attnTolerance(m float64) float64 {
	return 1e-6 + 4*float32Eps*m
}

// TestAttentionMatchesTorch runs each recorded ModernBertAttention against
// what the pinned transformers module returned, stage by stage: the window
// and padding mask against the 4-D mask transformers built, the RoPE tables
// against ModernBertRotaryEmbedding's cos/sin, then the output. It also checks
// the fixture is strong enough to catch the wrong readings of the module: each
// must miss the recorded output by ten times the tolerance somewhere. The case
// list is fixed, as in TestMLPMatchesTorch, so a regeneration cannot drop the
// case that guards one.
func TestAttentionMatchesTorch(t *testing.T) {
	cases, theta := loadAttnCases(t)

	separation := map[attnVariant]float64{}
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			cfg := c.Config
			if c.ModuleClass != "ModernBertAttention" || cfg.AttnImplementation != "sdpa" ||
				cfg.AttentionBias == nil || *cfg.AttentionBias || cfg.RopeType != "default" {
				t.Fatalf("%s is a %s, %q, bias %v, rope %q; want ModernBertAttention, sdpa, no bias, default",
					c.Module, c.ModuleClass, cfg.AttnImplementation, cfg.AttentionBias, cfg.RopeType)
			}
			hidden, heads, d := cfg.HiddenSize, cfg.NumAttentionHeads, cfg.HeadDim
			if int64(heads*d) != hidden {
				t.Fatalf("hidden %d is not %d heads x head_dim %d", hidden, heads, d)
			}
			if cfg.Scaling == nil || *cfg.Scaling != 1/math.Sqrt(float64(d)) {
				t.Fatalf("scaling %v, want head_dim^-0.5 = %g", cfg.Scaling, 1/math.Sqrt(float64(d)))
			}
			if !slices.Equal(c.Wqkv.Shape, []int64{3 * hidden, hidden}) ||
				!slices.Equal(c.Wo.Shape, []int64{hidden, hidden}) {
				t.Fatalf("Wqkv %v, Wo %v; want [%d %d], [%d %d]",
					c.Wqkv.Shape, c.Wo.Shape, 3*hidden, hidden, hidden, hidden)
			}
			shape := c.Input.Shape
			if len(shape) != 3 || shape[2] != hidden {
				t.Fatalf("input shape %v, want [B, S, %d]", shape, hidden)
			}
			batch, seq := int(shape[0]), int(shape[1])
			layer := c.layer(t)

			checkMask(t, c, layer, batch, seq)
			checkRope(t, c, layer, seq, d)

			a, err := NewAttention(c.Wqkv.tensor(t, "wqkv"), c.Wo.tensor(t, "wo"), heads, layer)
			if err != nil {
				t.Fatalf("NewAttention: %v", err)
			}
			got, err := a.Forward(c.Input.tensor(t, "input"), c.Padding)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
			want := c.Output.tensor(t, "output")
			if !slices.Equal(got.Shape(), want.Shape()) {
				t.Fatalf("shape %v, want %v", got.Shape(), want.Shape())
			}

			in := attnRefInput{
				x: c.Input.Data, wqkv: c.Wqkv.Data, wo: c.Wo.Data, padding: c.Padding,
				batch: batch, seq: seq, hidden: int(hidden), heads: heads,
				theta: cfg.RopeTheta, window: -1,
			}
			if layer.Sliding {
				in.window = layer.Window
				in.otherTheta = theta[fullAttentionName]
			} else {
				in.otherTheta = theta[slidingAttentionName]
			}
			_, mag := attnRef(in, attnExact)
			var maxDiff, worst, minTol, maxTol float64
			minTol = math.Inf(1)
			g, e := got.Data(), want.Data()
			for i := range e {
				diff := math.Abs(float64(g[i]) - float64(e[i]))
				tol := attnTolerance(mag[i])
				minTol, maxTol = math.Min(minTol, tol), math.Max(maxTol, tol)
				maxDiff, worst = math.Max(maxDiff, diff), math.Max(worst, diff/tol)
				if !(diff <= tol) { // NaN-safe: a NaN output fails here
					t.Errorf("[%d] = %.9g, want %.9g (|diff| %.3g > tol %.3g)", i, g[i], e[i], diff, tol)
				}
			}
			t.Logf("%s: max |diff| %.2e, worst |diff|/tol %.3f, tol %.1e to %.1e",
				c.Module, maxDiff, worst, minTol, maxTol)

			checkEmptyRows(t, c, g, int(hidden))

			for v := range attnVariantNames {
				ref, _ := attnRef(in, v)
				for i := range e {
					sep := math.Abs(ref[i]-float64(e[i])) / attnTolerance(mag[i])
					if math.IsNaN(sep) {
						t.Fatalf("%s: the reference for %s is NaN at [%d]", c.Name, attnVariantNames[v], i)
					}
					separation[v] = math.Max(separation[v], sep)
				}
			}
		})
	}

	for v, name := range attnVariantNames {
		t.Logf("%s misses torch by up to %.0f x tol", name, separation[v])
		if !(separation[v] >= 10) {
			t.Errorf("the fixture cannot tell %s apart: it misses torch by at most %.1f x tol, want >= 10",
				name, separation[v])
		}
	}
}

// loadAttnCases decodes the fixture's attention cases and returns them with
// each layer type's rope_theta. It fails unless the cases are exactly the
// expected ones, one per layer type with different thetas: the other type's
// theta is what attnOtherTheta reads, and a swap only shows if they differ.
func loadAttnCases(t *testing.T) ([]attnCase, map[string]float64) {
	t.Helper()

	wantCases := []string{"attn_full_padded", "attn_sliding_padded"}
	raws := casesOf(t, loadOps(t), "attention")
	cases := make([]attnCase, 0, len(raws))
	names := make([]string, 0, len(raws))
	theta := map[string]float64{}
	for _, raw := range raws {
		var c attnCase
		if err := json.Unmarshal(raw, &c); err != nil {
			t.Fatalf("decode case: %v", err)
		}
		if _, seen := theta[c.LayerType]; seen {
			t.Fatalf("two %s cases; want one per layer type", c.LayerType)
		}
		cases = append(cases, c)
		names = append(names, c.Name)
		theta[c.LayerType] = c.Config.RopeTheta
	}
	slices.Sort(names)
	if !slices.Equal(names, wantCases) {
		t.Fatalf("attention cases = %v, want %v", names, wantCases)
	}
	full, sliding := theta[fullAttentionName], theta[slidingAttentionName]
	if len(theta) != 2 || full <= 0 || sliding <= 0 {
		t.Fatalf("rope_theta by layer type = %v; want one full and one sliding case", theta)
	}
	if full == sliding {
		t.Fatalf("both layer types have rope_theta %g; the fixture cannot catch a swap", full)
	}
	return cases, theta
}

// checkMask compares the keys the Go side lets each query see with the 4-D
// mask transformers built for sdpa, bit by bit, and checks that the sliding
// case pins the window from both sides: in an unpadded row, distance
// sliding_window is allowed and sliding_window + 1 is not. That is the
// distance the sdpa path applies (masking_utils.py,
// sliding_window_bidirectional_overlay: abs(q_idx - kv_idx) <= sliding_window);
// ModernBertAttention's own sliding_window, one wider, is never read by it.
func checkMask(t *testing.T, c attnCase, layer LayerType, batch, seq int) {
	t.Helper()

	if len(c.Mask) != batch || len(c.Padding) != batch {
		t.Fatalf("mask has %d rows and padding %d, want %d", len(c.Mask), len(c.Padding), batch)
	}
	for b := range batch {
		for q := range seq {
			for k := range seq {
				want := c.Mask[b][q][k] != 0
				if got := layer.allows(c.Padding[b], q, k); got != want {
					t.Errorf("batch %d query %d key %d: allowed %v, transformers' mask says %v", b, q, k, got, want)
				}
			}
		}
	}

	if !layer.Sliding {
		return
	}
	w := c.Config.SlidingWindow
	if w != c.Config.LocalAttention/2 || layer.Window != w {
		t.Fatalf("sliding_window %d, local_attention %d, Go window %d; want local_attention/2 throughout",
			w, c.Config.LocalAttention, layer.Window)
	}
	if m := c.Config.ModuleSlidingWindow; m == nil || *m != w+1 {
		t.Errorf("module_sliding_window %v, want %d: the +1 this test documents as unused moved", m, w+1)
	}
	if w+1 >= seq {
		t.Fatalf("window %d does not bite inside %d tokens", w, seq)
	}
	row := -1
	for b, pad := range c.Padding {
		if !slices.Contains(pad, 0) {
			row = b
		}
	}
	if row < 0 {
		t.Fatal("no unpadded row: the window's edge is not pinned without padding in the way")
	}
	if c.Mask[row][0][w] == 0 || c.Mask[row][0][w+1] != 0 {
		t.Errorf("row %d: mask at distance %d = %d and %d = %d; want 1 and 0",
			row, w, c.Mask[row][0][w], w+1, c.Mask[row][0][w+1])
	}
}

// checkRope compares the Go RoPE tables with ModernBertRotaryEmbedding's
// cos/sin for the layer type's theta. Both compute inv_freq, the angle and its
// cosine in float32 the same way; only the last step differs, a float32
// cos/sin in torch against float64 math.Cos rounded to float32 here, which is
// one ulp at most (6e-8 below 1).
func checkRope(t *testing.T, c attnCase, layer LayerType, seq, d int) {
	t.Helper()

	want := []int64{int64(seq), int64(d)}
	if !slices.Equal(c.Cos.Shape, want) || !slices.Equal(c.Sin.Shape, want) {
		t.Fatalf("cos %v, sin %v; want %v", c.Cos.Shape, c.Sin.Shape, want)
	}
	cos, sin := ropeTables(layer.RopeTheta, d, seq)
	var worst float64
	for i := range c.Cos.Data {
		for _, pair := range [][2]float32{{cos[i], c.Cos.Data[i]}, {sin[i], c.Sin.Data[i]}} {
			diff := math.Abs(float64(pair[0]) - float64(pair[1]))
			worst = math.Max(worst, diff)
			if !(diff <= float32Eps) {
				t.Errorf("RoPE table [%d]: %.9g, torch %.9g", i, pair[0], pair[1])
			}
		}
	}
	t.Logf("RoPE tables at theta %g: max |diff| %.1e", layer.RopeTheta, worst)
}

// checkEmptyRows asserts the sdpa behaviour for a query with no allowed key,
// which a padded query on a sliding layer meets once the padding is longer than
// the window: torch's sdpa returns zeros there (not NaN, and not eager's
// average over every key from its finite min-value mask), so the attention
// output is zero, and Wo, without bias, keeps it zero. The Go output must be
// exactly zero too. The sliding case must contain such a row, or the
// behaviour goes untested.
func checkEmptyRows(t *testing.T, c attnCase, got []float32, hidden int) {
	t.Helper()

	empty := 0
	for b, rows := range c.Mask {
		for q, row := range rows {
			if slices.Contains(row, 1) {
				continue
			}
			empty++
			off := ((b*len(rows) + q) * hidden)
			for i := range hidden {
				if c.Output.Data[off+i] != 0 {
					t.Fatalf("fixture: batch %d query %d has no key but torch output %g", b, q, c.Output.Data[off+i])
				}
				if got[off+i] != 0 {
					t.Errorf("batch %d query %d has no key: output[%d] = %g, want exactly 0", b, q, i, got[off+i])
				}
			}
		}
	}
	if c.LayerType == slidingAttentionName && empty == 0 {
		t.Error("the sliding case has no query without keys; the dumper's padded row no longer reaches past the window")
	}
	t.Logf("%d queries with no key", empty)
}

// TestAttentionRealDimensions runs the attention at both checkpoints' sizes on
// synthetic weights, which no fixture covers: ModernBERT-large's attn.Wqkv
// [3072, 1024] and attn.Wo [1024, 1024] with 16 heads, and mmBERT-base's
// hidden 768 with 12 heads, head_dim 64 in both (docs/ARCHITECTURE.md §1.2),
// one as a sliding layer and one as a full one. Every output element is
// checked against the float64 reference, so a head split, RoPE pairing or
// merge that only goes wrong past head_dim 4 fails here. The window and
// padding are TestAttentionMatchesTorch's: a sequence long enough for a window
// of 64 to bite costs ~30 s under -race, nearly all of it in the reference's
// projections, so this one stays at 5 tokens with a padded row.
func TestAttentionRealDimensions(t *testing.T) {
	for _, tc := range []struct {
		name          string
		hidden, heads int
		layer         LayerType
	}{
		{"modernbert-large", 1024, 16, SlidingAttention(10000, 128)},
		{"mmbert-base", 768, 12, FullAttention(160000)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const batch, seq = 2, 5
			h := tc.hidden
			rng := rand.New(rand.NewPCG(84, uint64(h)))
			normal := func(n int, scale float64) []float32 {
				out := make([]float32, n)
				for i := range out {
					out[i] = float32(rng.NormFloat64() * scale)
				}
				return out
			}
			wqkvData := normal(3*h*h, 1/math.Sqrt(float64(h)))
			woData := normal(h*h, 1/math.Sqrt(float64(h)))
			xData := normal(batch*seq*h, 1)
			padding := [][]int64{{1, 1, 1, 1, 1}, {1, 1, 1, 0, 0}}

			a, err := NewAttention(
				mustTensor(t, wqkvData, []int64{int64(3 * h), int64(h)}),
				mustTensor(t, woData, []int64{int64(h), int64(h)}),
				tc.heads, tc.layer,
			)
			if err != nil {
				t.Fatalf("NewAttention: %v", err)
			}
			got, err := a.Forward(mustTensor(t, xData, []int64{batch, seq, int64(h)}), padding)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
			if want := []int64{batch, seq, int64(h)}; !slices.Equal(got.Shape(), want) {
				t.Fatalf("shape %v, want %v", got.Shape(), want)
			}

			// The reference is float64, so only go's rounding counts. The
			// bound is attnTolerance's, empirical as there: the projections
			// sum 768 and 1024 terms, but terms of random sign keep the
			// partial sums well below the magnitudes m adds up, so the bound
			// is loose here (the test logs how loose). An indexing slip
			// moves outputs by O(0.1), above it.
			window := -1
			if tc.layer.Sliding {
				window = tc.layer.Window
			}
			ref, mag := attnRef(attnRefInput{
				x: xData, wqkv: wqkvData, wo: woData, padding: padding,
				batch: batch, seq: seq, hidden: h, heads: tc.heads,
				theta: tc.layer.RopeTheta, window: window,
			}, attnExact)
			var worst, maxTol, maxRef float64
			for i, g := range got.Data() {
				diff := math.Abs(float64(g) - ref[i])
				tol := attnTolerance(mag[i])
				worst, maxTol = math.Max(worst, diff/tol), math.Max(maxTol, tol)
				maxRef = math.Max(maxRef, math.Abs(ref[i]))
				if !(diff <= tol) { // NaN-safe
					t.Fatalf("[%d] = %.9g, want %.9g (|diff| %.3g > tol %.3g)", i, g, ref[i], diff, tol)
				}
			}
			t.Logf("worst |diff|/tol %.2e, tol up to %.1e, |y| up to %.2f", worst, maxTol, maxRef)
		})
	}
}

// TestLayerTypes pins how config.json's attention settings become a
// LayerType: both checkpoints say local_attention 128, which is a window of
// ±64 (config.sliding_window = local_attention // 2).
func TestLayerTypes(t *testing.T) {
	if l := SlidingAttention(10000, 128); !l.Sliding || l.Window != 64 || l.RopeTheta != 10000 {
		t.Errorf("SlidingAttention(10000, 128) = %+v, want sliding, window 64, theta 10000", l)
	}
	if l := SlidingAttention(10000, 7); l.Window != 3 {
		t.Errorf("SlidingAttention(_, 7).Window = %d, want 7 // 2 = 3", l.Window)
	}
	if l := FullAttention(160000); l.Sliding || l.RopeTheta != 160000 {
		t.Errorf("FullAttention(160000) = %+v, want full, theta 160000", l)
	}

	pad := []int64{1, 1, 1, 1, 1, 1, 0, 0}
	sliding := SlidingAttention(1, 4) // window 2
	for _, tc := range []struct {
		q, k int
		want bool
	}{
		{3, 1, true}, {3, 5, true}, {3, 0, false}, {5, 6, false}, {7, 5, true}, {7, 4, false},
	} {
		if got := sliding.allows(pad, tc.q, tc.k); got != tc.want {
			t.Errorf("window 2: allows(q %d, k %d) = %v, want %v", tc.q, tc.k, got, tc.want)
		}
	}
	full := FullAttention(1)
	if !full.allows(pad, 7, 0) || full.allows(pad, 0, 6) {
		t.Error("full attention: want every real key and no padded one")
	}
}

// TestRopeTables pins rotate_half's layout: cos/sin repeat the d/2
// frequencies inv_freq[i] = theta^(-2i/d) over both halves (emb = cat(freqs,
// freqs)), not pairwise as the interleaved form does.
func TestRopeTables(t *testing.T) {
	cos, sin := ropeTables(100, 4, 3)
	// inv_freq = [1, 0.1]; at position 2 the angles are [2, 0.2, 2, 0.2].
	want := []float64{2, 0.2, 2, 0.2}
	for j, ang := range want {
		if c := float64(cos[2*4+j]); math.Abs(c-math.Cos(ang)) > float32Eps {
			t.Errorf("cos[2][%d] = %.9g, want cos(%g) = %.9g", j, c, ang, math.Cos(ang))
		}
		if s := float64(sin[2*4+j]); math.Abs(s-math.Sin(ang)) > float32Eps {
			t.Errorf("sin[2][%d] = %.9g, want sin(%g) = %.9g", j, s, ang, math.Sin(ang))
		}
	}
	for j := range 4 {
		if cos[j] != 1 || sin[j] != 0 {
			t.Errorf("position 0: cos %g, sin %g at %d; want 1, 0", cos[j], sin[j], j)
		}
	}
}

// ropeCase is one "rope" record of testdata/ops.json: the real
// ModernBertRotaryEmbedding at the checkpoints' head_dim 64 and thetas, at a
// few positions out to 8191.
type ropeCase struct {
	Name        string    `json:"name"`
	ModuleClass string    `json:"module_class"`
	LayerType   string    `json:"layer_type"`
	RopeType    string    `json:"rope_type"`
	RopeTheta   float64   `json:"rope_theta"`
	HeadDim     int       `json:"head_dim"`
	Scaling     *float64  `json:"attention_scaling"`
	Positions   []int     `json:"positions"`
	InvFreq     tensorRec `json:"inv_freq"`
	Cos         tensorRec `json:"cos"`
	Sin         tensorRec `json:"sin"`
}

// TestRopeMatchesTorch checks the RoPE tables at the real head_dim, which the
// attention cases' head_dim 4 cannot: inv_freq must equal torch's bit for bit,
// because an error in it grows with the position (at 8191 a one-ulp slip in
// the lowest frequency moves the angle by ~5e-4). The angle is then the same
// float32 product on both sides, and cos/sin may differ only in torch's
// float32 cos/sin against math.Cos rounded to float32: one ulp.
func TestRopeMatchesTorch(t *testing.T) {
	f := loadOps(t)
	wantCases := []string{"rope_full_d64", "rope_sliding_d64"}
	raws := casesOf(t, f, "rope")
	names := make([]string, 0, len(raws))
	for _, raw := range raws {
		var c ropeCase
		if err := json.Unmarshal(raw, &c); err != nil {
			t.Fatalf("decode case: %v", err)
		}
		names = append(names, c.Name)
		t.Run(c.Name, func(t *testing.T) {
			if c.ModuleClass != "ModernBertRotaryEmbedding" || c.RopeType != "default" ||
				c.Scaling == nil || *c.Scaling != 1 {
				t.Fatalf("%s %s, rope_type %q, scaling %v; want default RoPE with scaling 1",
					c.ModuleClass, c.LayerType, c.RopeType, c.Scaling)
			}
			d := c.HeadDim
			if d != 64 || len(c.Positions) == 0 {
				t.Fatalf("head_dim %d, positions %v; want 64 and some", d, c.Positions)
			}
			if !slices.Equal(c.InvFreq.Shape, []int64{int64(d / 2)}) {
				t.Fatalf("inv_freq shape %v, want [%d]", c.InvFreq.Shape, d/2)
			}
			for i, got := range ropeInvFreq(c.RopeTheta, d) {
				if want := c.InvFreq.Data[i]; got != want {
					t.Errorf("inv_freq[%d] = %.9g (%#08x), torch %.9g (%#08x)",
						i, got, math.Float32bits(got), want, math.Float32bits(want))
				}
			}

			last := slices.Max(c.Positions)
			cos, sin := ropeTables(c.RopeTheta, d, last+1)
			var ulps int
			for p, pos := range c.Positions {
				for j := range d {
					for _, pair := range [][2]float32{
						{cos[pos*d+j], c.Cos.Data[p*d+j]},
						{sin[pos*d+j], c.Sin.Data[p*d+j]},
					} {
						got, want := pair[0], pair[1]
						if got == want {
							continue
						}
						ulps++
						if !(math.Abs(float64(got-want)) <= float64(max(ulp(got), ulp(want)))) {
							t.Errorf("position %d [%d]: %.9g, torch %.9g: more than one ulp", pos, j, got, want)
						}
					}
				}
			}
			t.Logf("theta %g: inv_freq bit-exact, %d of %d cos/sin values one ulp off",
				c.RopeTheta, ulps, 2*len(c.Positions)*d)
		})
	}
	slices.Sort(names)
	if !slices.Equal(names, wantCases) {
		t.Errorf("rope cases = %v, want %v", names, wantCases)
	}
}

// ulp returns the gap from |x| to the next float32 away from zero.
func ulp(x float32) float32 {
	a := float32(math.Abs(float64(x)))
	return math.Nextafter32(a, float32(math.Inf(1))) - a
}

func TestAttentionErrors(t *testing.T) {
	// H = 4, 2 heads (head_dim 2): Wqkv [12, 4], Wo [4, 4].
	wqkv := mustTensor(t, make([]float32, 48), []int64{12, 4})
	wo := mustTensor(t, make([]float32, 16), []int64{4, 4})
	full := FullAttention(10000)
	for _, tc := range []struct {
		name      string
		wqkv, wo  *tensor.Tensor
		heads     int
		layerType LayerType
	}{
		{"nil Wqkv", nil, wo, 2, full},
		{"nil Wo", wqkv, nil, 2, full},
		{"Wqkv rank 1", mustTensor(t, make([]float32, 48), []int64{48}), wo, 2, full},
		{"Wqkv rows not 3H", mustTensor(t, make([]float32, 32), []int64{8, 4}), wo, 2, full},
		{"hidden 0", mustTensor(t, nil, []int64{0, 0}), mustTensor(t, nil, []int64{0, 0}), 2, full},
		{"heads 0", wqkv, wo, 0, full},
		{"heads negative", wqkv, wo, -2, full},
		{"H not divisible by heads", wqkv, wo, 3, full},
		// 4 heads of head_dim 1: rotate_half has no halves.
		{"odd head_dim", wqkv, wo, 4, full},
		{"Wo rank 3", wqkv, mustTensor(t, make([]float32, 16), []int64{1, 4, 4}), 2, full},
		{"Wo not [H, H]", wqkv, mustTensor(t, make([]float32, 12), []int64{4, 3}), 2, full},
		{"theta 0", wqkv, wo, 2, FullAttention(0)},
		{"theta NaN", wqkv, wo, 2, FullAttention(math.NaN())},
		{"theta +Inf", wqkv, wo, 2, FullAttention(math.Inf(1))},
		{"negative window", wqkv, wo, 2, SlidingAttention(10000, -2)},
	} {
		if _, err := NewAttention(tc.wqkv, tc.wo, tc.heads, tc.layerType); err == nil {
			t.Errorf("NewAttention(%s): no error", tc.name)
		}
	}

	a, err := NewAttention(wqkv, wo, 2, SlidingAttention(10000, 4))
	if err != nil {
		t.Fatalf("NewAttention: %v", err)
	}
	x := mustTensor(t, make([]float32, 2*3*4), []int64{2, 3, 4})
	pad := [][]int64{{1, 1, 1}, {1, 1, 0}}
	for _, tc := range []struct {
		name    string
		x       *tensor.Tensor
		padding [][]int64
	}{
		{"nil input", nil, pad},
		{"input rank 2", mustTensor(t, make([]float32, 12), []int64{3, 4}), pad[:1]},
		{"input hidden 5", mustTensor(t, make([]float32, 30), []int64{2, 3, 5}), pad},
		{"padding rows", x, pad[:1]},
		{"padding row length", x, [][]int64{{1, 1, 1}, {1, 1}}},
		{"padding value 2", x, [][]int64{{1, 1, 1}, {1, 2, 0}}},
		{"padding value -1", x, [][]int64{{1, 1, 1}, {1, -1, 0}}},
	} {
		if _, err := a.Forward(tc.x, tc.padding); err == nil {
			t.Errorf("Forward(%s): no error", tc.name)
		}
	}
	if y, err := a.Forward(mustTensor(t, nil, []int64{0, 3, 4}), nil); err != nil {
		t.Errorf("empty batch: %v", err)
	} else if !slices.Equal(y.Shape(), []int64{0, 3, 4}) {
		t.Errorf("empty batch: shape %v, want [0 3 4]", y.Shape())
	}

	var zero Attention
	if _, err := zero.Forward(x, pad); err == nil || !strings.Contains(err.Error(), "uninitialized") {
		t.Errorf("zero Attention Forward: err = %v, want an uninitialized-attention error", err)
	}
}
