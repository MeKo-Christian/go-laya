// Package head is the laya decision head of the pure-Go native backend
// (PLAN.md Task 8.6, D8): everything DecisionModel.forward does after the
// encoder (original/laya/common.py:108-126), from the encoder's last hidden
// state to the option logits and the act logits.
//
// It is tested against the real DecisionModel and nn.TransformerEncoderLayer
// of the pinned torch through testdata/head.json, which
// scripts/dump_head_ops.py writes. It does not load weights and is not a
// backend.Backend: internal/backend/native (Task 8.10) does both.
package head

import (
	"errors"
	"fmt"
	"math"

	"github.com/MeKo-Christian/go-laya/backend"
	"github.com/MeKo-Christian/go-laya/internal/calib"
	"github.com/MeKo-Christian/go-laya/internal/runtime/tensor"
)

const (
	// MaskFill is the logit every marker column outside a row's real markers
	// gets: -1e4, not -inf (common.py:117, invariant #37).
	MaskFill float32 = -1e4

	// ActHidden is the act head's hidden width, nn.Linear(d + 4, 256)
	// (common.py:101).
	ActHidden = 256

	// numFeats is the act features' count: top1, top1 - top2, the
	// normalized entropy and k / 255 (common.py:123).
	numFeats = 4

	// numQTypes is type_emb's row count, nn.Embedding(3, d) (common.py:99):
	// choice, score and noul.
	numQTypes = 3
)

// Weights are the decision head's tensors in torch's [out, in] layout, named
// after DecisionModel's state_dict (docs/ARCHITECTURE.md §1.3). temperature is
// not among them: inference never reads it.
type Weights struct {
	// TypeEmb is type_emb.weight [3, d]; it fixes d.
	TypeEmb *tensor.Tensor
	// Layers are head.layers.N, in order. Upstream builds head_layers of them
	// (config.json, 2 on every shipped checkpoint); zero is no head encoder.
	Layers []LayerWeights
	// ScorerNormWeight and ScorerNormBias [d] are scorer.0, a LayerNorm.
	ScorerNormWeight, ScorerNormBias *tensor.Tensor
	// ScorerLinear1Weight [d, d] and ScorerLinear1Bias [d] are scorer.1.
	ScorerLinear1Weight, ScorerLinear1Bias *tensor.Tensor
	// ScorerLinear2Weight [1, d] and ScorerLinear2Bias [1] are scorer.3.
	ScorerLinear2Weight, ScorerLinear2Bias *tensor.Tensor
	// ActLinear1Weight [256, d+4] and ActLinear1Bias [256] are act_head.0.
	ActLinear1Weight, ActLinear1Bias *tensor.Tensor
	// ActLinear2Weight [n_act, 256] and ActLinear2Bias [n_act] are
	// act_head.2; n_act is len(act_costs) + 1, 2 on every shipped checkpoint.
	ActLinear2Weight, ActLinear2Bias *tensor.Tensor
}

// Head is DecisionModel's head after the encoder. In inference, which on the
// CPU is eval() under no_grad() in float32 (agent.py:196-215, 268-277), it
// computes, for the encoder's last hidden state h [B, S, d]:
//
//	h = h + type_emb(qtype)[:, None, :]                      # every position (#35)
//	for layer in head.layers: h = layer(h, ~attention_mask)  # no final norm (#36)
//	m = gather(h, 1, marker_pos.clamp(min=0))
//	logits = scorer(m).squeeze(-1).masked_fill(~marker_mask, -1e4)  # (#37)
//	p = softmax(logits); k = clamp(marker_mask.sum(-1), min=2)
//	ent = -(p * log(p.clamp_min(1e-9))).sum(-1) / log(k)
//	feats = [top1, top1 - top2, ent, k / 255]                # (#38)
//	act_logits = act_head(cat([h[:, 0], feats]))
//
// The scorer is LayerNorm, Linear(d, d), GELU, Linear(d, 1); the act head
// Linear(d + 4, 256), GELU, Linear(256, n_act). Both GELUs are nn.GELU(),
// the exact erf form. The dropouts are inference's no-ops and are not
// modelled.
//
// The zero Head has no weights; its Forward returns an error.
type Head struct {
	w      Weights
	layers []Layer
	d      int
	nhead  int
	nAct   int
}

// nheadFor is DecisionModel's head count for width d, max(1, d // 64)
// (common.py:96): 16 for ModernBERT-large's 1024, 12 for mmBERT-base's 768.
func nheadFor(d int) int {
	return max(1, d/64)
}

// New returns the head with the given weights, its head count derived from d
// as upstream derives it. The Head keeps the tensors; the caller must not
// modify them. Every tensor's shape is checked against the d that TypeEmb
// fixes and the n_act that ActLinear2Weight fixes.
func New(w Weights) (*Head, error) {
	if w.TypeEmb == nil || w.TypeEmb.Rank() != 2 || w.TypeEmb.Shape()[0] != numQTypes || w.TypeEmb.Shape()[1] == 0 {
		return nil, fmt.Errorf("head: type_emb.weight is %s, want [3, d]", shapeOf(w.TypeEmb))
	}
	d := w.TypeEmb.Shape()[1]
	if w.ActLinear2Weight == nil || w.ActLinear2Weight.Rank() != 2 || w.ActLinear2Weight.Shape()[0] == 0 {
		return nil, fmt.Errorf("head: act_head.2.weight is %s, want [n_act, %d]", shapeOf(w.ActLinear2Weight), ActHidden)
	}
	nAct := w.ActLinear2Weight.Shape()[0]
	for _, c := range []struct {
		name  string
		t     *tensor.Tensor
		shape []int64
	}{
		{"scorer.0.weight", w.ScorerNormWeight, []int64{d}},
		{"scorer.0.bias", w.ScorerNormBias, []int64{d}},
		{"scorer.1.weight", w.ScorerLinear1Weight, []int64{d, d}},
		{"scorer.1.bias", w.ScorerLinear1Bias, []int64{d}},
		{"scorer.3.weight", w.ScorerLinear2Weight, []int64{1, d}},
		{"scorer.3.bias", w.ScorerLinear2Bias, []int64{1}},
		{"act_head.0.weight", w.ActLinear1Weight, []int64{ActHidden, d + numFeats}},
		{"act_head.0.bias", w.ActLinear1Bias, []int64{ActHidden}},
		{"act_head.2.weight", w.ActLinear2Weight, []int64{nAct, ActHidden}},
		{"act_head.2.bias", w.ActLinear2Bias, []int64{nAct}},
	} {
		if err := checkShape(c.name, c.t, c.shape); err != nil {
			return nil, err
		}
	}

	// Upstream builds the layer whatever head_layers is (common.py:97), and
	// nn.MultiheadAttention refuses a d its heads do not divide.
	nhead := nheadFor(int(d))
	if d%int64(nhead) != 0 {
		return nil, fmt.Errorf("head: %d heads do not divide d = %d", nhead, d)
	}
	layers := make([]Layer, len(w.Layers))
	for i, lw := range w.Layers {
		l, err := NewLayer(lw, nhead)
		if err != nil {
			return nil, fmt.Errorf("%w (head.layers.%d)", err, i)
		}
		if l.d != int(d) {
			return nil, fmt.Errorf("head: head.layers.%d has d = %d, type_emb says %d", i, l.d, d)
		}
		layers[i] = l
	}
	return &Head{w: w, layers: layers, d: int(d), nhead: nhead, nAct: int(nAct)}, nil
}

// NHead returns the attention head count of the head's layers.
func (h *Head) NHead() int {
	return h.nhead
}

// Forward runs the head on the encoder's last hidden state x [B, S, d] for
// the batch the encoder ran on. It returns, per row, the kmax option logits
// and the n_act act logits, as backend.Backend.Forward does.
//
// A batch that does not fit x, a qtype outside 0..2 (type_emb's IndexError
// upstream), a marker position at or past S (gather's out-of-bounds
// RuntimeError) and fewer than two marker columns are errors. The last is
// what upstream's p.topk(2, -1) (common.py:122) raises on ("selected index k
// out of range", for kmax 1 and 0 alike); a batch of single-option choice
// questions gets there. A negative marker position is clamped to 0, as
// upstream clamps it (common.py:114), so a fill of -1 reads [CLS] exactly as
// the collator's fill of 0 does, and its logit is MaskFill either way.
func (h *Head) Forward(x *tensor.Tensor, b backend.Batch) (logits, act [][]float32, err error) {
	return h.forward(x, b, nil)
}

// trace is what forward computes, stage by stage, for the tests to compare
// against the fixture's intermediates. Forward passes none, so it copies no
// hidden state it does not need.
type trace struct {
	typed                     *tensor.Tensor   // h after type_emb, [B, S, d]
	layers                    []*tensor.Tensor // h after each layer
	gathered                  *tensor.Tensor   // the marker rows, [B, kmax, d]
	logits, probs, feats, act [][]float32
}

// forward is Forward, recording each stage into tr unless it is nil.
func (h *Head) forward(x *tensor.Tensor, b backend.Batch, tr *trace) (logits, act [][]float32, err error) {
	if h == nil || h.w.TypeEmb == nil {
		return nil, nil, errors.New("head: uninitialized head; build it with New")
	}
	batch, seq, kmax, err := h.checkInput(x, b)
	if err != nil {
		return nil, nil, err
	}

	// #35: the type embedding goes to every position, not just [CLS]. x is
	// the caller's; hs is a copy.
	hs := x.Clone()
	data, emb := hs.RawData(), h.w.TypeEmb.RawData()
	for i, qt := range b.QType {
		row := emb[int(qt)*h.d : (int(qt)+1)*h.d]
		for s := range seq {
			tensor.Axpy(data[(i*seq+s)*h.d:(i*seq+s+1)*h.d], 1, row)
		}
	}
	if tr != nil {
		tr.typed = hs.Clone()
	}

	// #36: the layers one by one, as the manual loop runs them, so no final
	// norm.
	for i, l := range h.layers {
		hs, err = l.Forward(hs, b.AttentionMask)
		if err != nil {
			return nil, nil, fmt.Errorf("%w (head.layers.%d)", err, i)
		}
		if tr != nil {
			tr.layers = append(tr.layers, hs.Clone())
		}
	}

	gathered, err := h.gather(hs, b.MarkerPos, seq, kmax)
	if err != nil {
		return nil, nil, err
	}
	logits, err = h.score(gathered, b.MarkerMask, batch, kmax)
	if err != nil {
		return nil, nil, err
	}

	probs := make([][]float32, batch)
	feats := make([][]float32, batch)
	pooled := make([]float32, 0, batch*(h.d+numFeats))
	for i := range batch {
		probs[i], feats[i] = features(logits[i], b.MarkerMask[i])
		// pooled = h[:, 0] after the head layers (#38).
		pooled = append(pooled, hs.RawData()[i*seq*h.d:(i*seq+1)*h.d]...)
		pooled = append(pooled, feats[i]...)
	}
	act, err = h.actHead(pooled, batch)
	if err != nil {
		return nil, nil, err
	}
	if tr != nil {
		tr.gathered, tr.logits, tr.probs, tr.feats, tr.act = gathered, logits, probs, feats, act
	}
	return logits, act, nil
}

// checkInput holds x and the batch to each other and to what upstream
// accepts, returning B, S and kmax.
func (h *Head) checkInput(x *tensor.Tensor, b backend.Batch) (batch, seq, kmax int, err error) {
	if x == nil {
		return 0, 0, 0, errors.New("head: hidden state is nil")
	}
	shape := x.Shape()
	if len(shape) != 3 || shape[0] == 0 || shape[1] == 0 || shape[2] != int64(h.d) {
		return 0, 0, 0, fmt.Errorf("head: hidden state has shape %v, want [batch, seq, %d]", shape, h.d)
	}
	batch, seq = int(shape[0]), int(shape[1])

	if len(b.InputIDs) != batch {
		return 0, 0, 0, fmt.Errorf("head: batch has %d input_ids rows for a hidden state of %d", len(b.InputIDs), batch)
	}
	for i, row := range b.InputIDs {
		if len(row) != seq {
			return 0, 0, 0, fmt.Errorf("head: input_ids row %d has %d tokens for a hidden state of %d", i, len(row), seq)
		}
	}
	if err := checkAttention(b.AttentionMask, batch, seq); err != nil {
		return 0, 0, 0, err
	}
	if len(b.QType) != batch {
		return 0, 0, 0, fmt.Errorf("head: batch has %d qtypes for %d rows", len(b.QType), batch)
	}
	for i, qt := range b.QType {
		if qt < 0 || qt >= numQTypes {
			return 0, 0, 0, fmt.Errorf("head: qtype[%d] = %d, want 0 (choice), 1 (score) or 2 (noul)", i, qt)
		}
	}

	kmax, err = checkMarkers(b.MarkerPos, b.MarkerMask, batch, seq)
	if err != nil {
		return 0, 0, 0, err
	}
	return batch, seq, kmax, nil
}

// checkMarkers requires marker_pos and marker_mask to be [batch][kmax] with
// kmax >= 2 and every position before seq, and returns kmax. A negative
// position is upstream's to clamp, not an error.
func checkMarkers(pos [][]int64, mask [][]bool, batch, seq int) (int, error) {
	if len(pos) != batch || len(mask) != batch {
		return 0, fmt.Errorf("head: batch has %d marker_pos and %d marker_mask rows for %d rows",
			len(pos), len(mask), batch)
	}
	kmax := len(pos[0])
	for i := range batch {
		if len(pos[i]) != kmax || len(mask[i]) != kmax {
			return 0, fmt.Errorf("head: marker row %d has %d positions and %d mask entries, want %d",
				i, len(pos[i]), len(mask[i]), kmax)
		}
		for j, p := range pos[i] {
			if p >= int64(seq) {
				return 0, fmt.Errorf("head: marker_pos[%d][%d] = %d is out of bounds for %d tokens", i, j, p, seq)
			}
		}
	}
	if kmax < 2 {
		return 0, fmt.Errorf("head: %d marker columns; the act features' topk(2) needs at least 2 "+
			"(common.py:122 raises \"selected index k out of range\")", kmax)
	}
	return kmax, nil
}

// gather reads the marker rows of hs [B, S, d] into [B, kmax, d], each
// position clamped at 0 from below (common.py:114).
func (h *Head) gather(hs *tensor.Tensor, pos [][]int64, seq, kmax int) (*tensor.Tensor, error) {
	m, err := tensor.Zeros([]int64{int64(len(pos)), int64(kmax), int64(h.d)})
	if err != nil {
		return nil, fmt.Errorf("head: %w", err)
	}
	src, dst := hs.RawData(), m.RawData()
	for i, row := range pos {
		for j, p := range row {
			s := int(max(p, 0))
			copy(dst[(i*kmax+j)*h.d:(i*kmax+j+1)*h.d], src[(i*seq+s)*h.d:(i*seq+s+1)*h.d])
		}
	}
	return m, nil
}

// score runs the scorer over the gathered rows and fills the columns outside
// each row's markers with MaskFill (#37), in float32 as upstream's .float().
func (h *Head) score(m *tensor.Tensor, mask [][]bool, batch, kmax int) ([][]float32, error) {
	n, err := tensor.LayerNorm(m, h.w.ScorerNormWeight, h.w.ScorerNormBias, LayerNormEps)
	if err != nil {
		return nil, fmt.Errorf("head: scorer.0: %w", err)
	}
	y, err := tensor.Linear(n, h.w.ScorerLinear1Weight, h.w.ScorerLinear1Bias)
	if err != nil {
		return nil, fmt.Errorf("head: scorer.1: %w", err)
	}
	geluInPlace(y.RawData())
	s, err := tensor.Linear(y, h.w.ScorerLinear2Weight, h.w.ScorerLinear2Bias) // [B, kmax, 1]
	if err != nil {
		return nil, fmt.Errorf("head: scorer.3: %w", err)
	}
	out := make([][]float32, batch)
	for i := range batch {
		out[i] = make([]float32, kmax)
		copy(out[i], s.RawData()[i*kmax:(i+1)*kmax])
		for j, isMarker := range mask[i] {
			if !isMarker {
				out[i][j] = MaskFill
			}
		}
	}
	return out, nil
}

// features computes one row's act features (#38) from its masked logits,
// in float32 as torch does (common.py:119-123):
//
//	p = softmax(logits)
//	k = clamp(marker_mask.sum(), min=2)
//	ent = -(p * log(p.clamp_min(1e-9))).sum() / log(k)
//	[top1, top1 - top2, ent, k / 255]
//
// The softmax is calib.ActSoftmax, which reproduces torch's float32 CPU
// softmax over the last dimension (the kernel agent.py:295 runs on the act
// logits is the same one). The logs go through float64 and round back, where
// torch uses SLEEF's float32 log; the tests bound the difference. A fill
// column's logit is -1e4, so its p is exactly 0 and adds 0 to the entropy.
func features(logits []float32, mask []bool) (p, feats []float32) {
	p = calib.ActSoftmax(logits)

	count := 0
	for _, isMarker := range mask {
		if isMarker {
			count++
		}
	}
	k := float32(max(count, 2))

	var sum float32
	for _, v := range p {
		sum += v * log32(max(v, 1e-9))
	}
	ent := -sum / log32(k)

	// topk(2): the two largest, in descending order, a NaN ranking above
	// every number as in torch's topk.
	top1, top2 := float32(math.Inf(-1)), float32(math.Inf(-1))
	for _, v := range p {
		if v > top1 || v != v {
			top1, top2 = v, top1
		} else if v > top2 {
			top2 = v
		}
	}
	return p, []float32{top1, top1 - top2, ent, k / 255}
}

// actHead runs act_head over pooled [B, d+4], the pooled hidden state and the
// features of each row.
func (h *Head) actHead(pooled []float32, batch int) ([][]float32, error) {
	in, err := tensor.New(pooled, []int64{int64(batch), int64(h.d + numFeats)})
	if err != nil {
		return nil, fmt.Errorf("head: %w", err)
	}
	y, err := tensor.Linear(in, h.w.ActLinear1Weight, h.w.ActLinear1Bias)
	if err != nil {
		return nil, fmt.Errorf("head: act_head.0: %w", err)
	}
	geluInPlace(y.RawData())
	a, err := tensor.Linear(y, h.w.ActLinear2Weight, h.w.ActLinear2Bias)
	if err != nil {
		return nil, fmt.Errorf("head: act_head.2: %w", err)
	}
	out := make([][]float32, batch)
	for i := range batch {
		out[i] = append([]float32(nil), a.RawData()[i*h.nAct:(i+1)*h.nAct]...)
	}
	return out, nil
}

// geluInPlace applies nn.GELU() with its default approximate="none", the
// exact 0.5*x*(1 + erf(x/sqrt 2)), to every element. It is the function
// internal/modernbert's gelu computes for transformers' GELUActivation, which
// calls the same nn.functional.gelu; that one is unexported, so it is
// repeated here.
func geluInPlace(xs []float32) {
	for i, x := range xs {
		v := float64(x)
		xs[i] = float32(0.5 * v * (1 + math.Erf(v/math.Sqrt2)))
	}
}

// log32 is the natural log of a float32, rounded back to float32.
func log32(x float32) float32 {
	return float32(math.Log(float64(x)))
}
