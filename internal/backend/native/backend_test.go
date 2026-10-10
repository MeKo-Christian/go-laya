package native

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/MeKo-Christian/go-laya/backend"
	"github.com/MeKo-Christian/go-laya/internal/head"
	"github.com/MeKo-Christian/go-laya/internal/runtime/tensor"
)

// open writes s and opens it, closing the Backend when the test ends.
func open(t *testing.T, s *synth) *Backend {
	t.Helper()
	b, err := Open(t.Context(), s.write(t), Options{})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() {
		if err := b.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
	})
	return b
}

func TestOpenForwardShapes(t *testing.T) {
	b := open(t, newSynth(1))
	logits, act, err := b.Forward(t.Context(), tinyBatch())
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if len(logits) != 2 || len(act) != 2 {
		t.Fatalf("%d logits rows and %d act rows, want 2 and 2", len(logits), len(act))
	}
	for i := range 2 {
		if len(logits[i]) != 3 || len(act[i]) != tinyAct {
			t.Fatalf("row %d: %d logits and %d act logits, want 3 and %d", i, len(logits[i]), len(act[i]), tinyAct)
		}
		for _, v := range append(slices.Clone(logits[i]), act[i]...) {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("row %d: non-finite output %v %v", i, logits[i], act[i])
			}
		}
	}
	if logits[1][2] != head.MaskFill {
		t.Errorf("row 1's fill column = %v, want %v", logits[1][2], head.MaskFill)
	}
}

// TestForwardMatchesDirectAssembly holds Open's name-to-weight mapping to the
// test's own: an encoder and a head built straight from the same tensors must
// give bit-identical outputs. A Wqkv or in_proj read in another order, a norm
// moved to another layer, a layer plan or rope_parameters ignored all change
// the result.
func TestForwardMatchesDirectAssembly(t *testing.T) {
	s := newSynth(2)
	b := open(t, s)
	batch := tinyBatch()
	gotLogits, gotAct, err := b.Forward(t.Context(), batch)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	forward := func(v variant) (logits, act [][]float32) {
		enc, h := s.assemble(t, v)
		x, err := enc.Forward(batch.InputIDs, batch.AttentionMask)
		if err != nil {
			t.Fatal(err)
		}
		if logits, act, err = h.Forward(x, batch); err != nil {
			t.Fatal(err)
		}
		return logits, act
	}
	wantLogits, wantAct := forward(variant{})
	for i := range wantLogits {
		if !slices.Equal(gotLogits[i], wantLogits[i]) || !slices.Equal(gotAct[i], wantAct[i]) {
			t.Errorf("row %d: Open's model gives %v %v, the direct assembly %v %v",
				i, gotLogits[i], gotAct[i], wantLogits[i], wantAct[i])
		}
	}

	// The batch must be able to tell the layer plan and the thetas apart,
	// or the test above could not catch either being ignored.
	if fullLogits, _ := forward(variant{allFull: true}); slices.Equal(fullLogits[0], wantLogits[0]) {
		t.Error("an all-full-attention encoder gives the same logits; the batch does not exercise the sliding window")
	}
	if defLogits, _ := forward(variant{defaultThetas: true}); slices.Equal(defLogits[0], wantLogits[0]) {
		t.Error("the default thetas give the same logits; the batch does not exercise rope_parameters")
	}
}

// TestOpenStrict: every way a checkpoint can disagree with the model it
// claims to be is an error wrapping backend.ErrIncompatibleCheckpoint that
// names the tensor or config field at fault.
func TestOpenStrict(t *testing.T) {
	for _, c := range []struct {
		name   string
		mutate func(*synth)
		want   []string
	}{
		{"missing tensor", func(s *synth) {
			delete(s.tensors, "encoder.layers.2.mlp.Wo.weight")
		}, []string{"missing", `"encoder.layers.2.mlp.Wo.weight"`}},
		{"two missing tensors", func(s *synth) {
			delete(s.tensors, "scorer.3.bias")
			delete(s.tensors, "head.layers.1.norm2.weight")
		}, []string{"missing", `"scorer.3.bias"`, `"head.layers.1.norm2.weight"`}},
		{"missing temperature", func(s *synth) {
			delete(s.tensors, "temperature")
		}, []string{"missing", `"temperature"`}},
		{"unexpected bias", func(s *synth) {
			s.zeros("encoder.layers.1.attn.Wqkv.bias", 3*tinyHidden)
		}, []string{"unexpected", `"encoder.layers.1.attn.Wqkv.bias"`}},
		{"unexpected decoder", func(s *synth) {
			s.zeros("decoder.weight", tinyVocab, tinyHidden)
		}, []string{"unexpected", `"decoder.weight"`}},
		{"wrong shape", func(s *synth) {
			s.zeros("head.layers.0.linear1.weight", 2*tinyHidden, tinyHidden)
		}, []string{"shape", `"head.layers.0.linear1.weight"`}},
		{"vocabulary other than config's", func(s *synth) {
			s.zeros("encoder.embeddings.tok_embeddings.weight", tinyVocab+1, tinyHidden)
		}, []string{"shape", `"encoder.embeddings.tok_embeddings.weight"`}},
		{"transposed Wi", func(s *synth) {
			s.zeros("encoder.layers.3.mlp.Wi.weight", tinyHidden, 2*tinyInter)
		}, []string{"shape", `"encoder.layers.3.mlp.Wi.weight"`}},
		{"temperature of another shape", func(s *synth) {
			s.zeros("temperature", 2)
		}, []string{"shape", `"temperature"`}},
		{"attn_norm at layer 0", func(s *synth) {
			s.zeros("encoder.layers.0.attn_norm.weight", tinyHidden)
		}, []string{`"encoder.layers.0.attn_norm.weight"`, "nn.Identity"}},
		{"fewer encoder layers than the plan", func(s *synth) {
			for name := range s.tensors {
				if strings.HasPrefix(name, "encoder.layers.3.") {
					delete(s.tensors, name)
				}
			}
		}, []string{"3 encoder layers", "num_hidden_layers", "4"}},
		{"more encoder layers than the plan", func(s *synth) {
			s.config["num_hidden_layers"] = 3
			s.config["layer_types"] = tinyLayerTypes[:3]
		}, []string{"4 encoder layers", "num_hidden_layers", "3"}},
		// The plan transformers would derive is 2^40 layers long; the file's
		// count must refuse it before anything is that large.
		{"a plan far beyond the file", func(s *synth) {
			delete(s.config, "layer_types")
			s.config["num_hidden_layers"] = 1 << 40
		}, []string{"4 encoder layers", "num_hidden_layers 1099511627776"}},
		// Each numbered layer implies a dozen tensors; a file that numbers
		// far more layers than it has tensors must be refused before the
		// loader lists them all.
		{"head layers numbered beyond the file's tensors", func(s *synth) {
			for n := 2; n < 100; n++ {
				s.zeros(fmt.Sprintf("head.layers.%d.norm1.bias", n), tinyHidden)
			}
		}, []string{"100 head layers", "1223 tensors", "has 160"}},
		{"encoder layer numbered past a gap", func(s *synth) {
			for name, st := range s.tensors {
				if rest, ok := strings.CutPrefix(name, "encoder.layers.3."); ok {
					delete(s.tensors, name)
					s.tensors["encoder.layers.4."+rest] = st
				}
			}
		}, []string{"missing", `"encoder.layers.3.mlp.Wo.weight"`, "unexpected", `"encoder.layers.4.mlp.Wo.weight"`}},
		// "03" is no layer number, so the file has three layers, not four
		// with one misspelt.
		{"encoder layer numbered 03", func(s *synth) {
			for name, st := range s.tensors {
				if rest, ok := strings.CutPrefix(name, "encoder.layers.3."); ok {
					delete(s.tensors, name)
					s.tensors["encoder.layers.03."+rest] = st
				}
			}
		}, []string{"3 encoder layers", "num_hidden_layers 4"}},
		{"many unexpected tensors", func(s *synth) {
			for n := range 25 {
				s.zeros(fmt.Sprintf("decoder.%02d.weight", n), 1)
			}
		}, []string{"unexpected", `"decoder.00.weight"`, `"decoder.09.weight"`, "and 15 more"}},
		{"head layer numbered past a gap", func(s *synth) {
			for name, st := range s.tensors {
				if rest, ok := strings.CutPrefix(name, "head.layers.1."); ok {
					delete(s.tensors, name)
					s.tensors["head.layers.2."+rest] = st
				}
			}
		}, []string{"missing", `"head.layers.1.linear1.weight"`, "unexpected", `"head.layers.2.linear1.weight"`}},
		{"bad layer plan", func(s *synth) {
			s.config["layer_types"] = []string{"full_attention", "bogus", "sliding_attention", "full_attention"}
		}, []string{"layer_types", "bogus"}},
		{"another norm eps", func(s *synth) {
			s.config["norm_eps"] = 1e-6
		}, []string{"norm_eps"}},
		{"another activation", func(s *synth) {
			s.config["hidden_activation"] = "gelu_pytorch_tanh"
		}, []string{"hidden_activation", "gelu_pytorch_tanh"}},
		{"a null intermediate size", func(s *synth) {
			s.config["intermediate_size"] = nil
		}, []string{"intermediate_size"}},
		{"another model_type", func(s *synth) {
			s.config["model_type"] = "bert"
		}, []string{"model_type", `"bert"`}},
		{"no model_type", func(s *synth) {
			delete(s.config, "model_type")
		}, []string{"model_type"}},
		{"a float16 encoder", func(s *synth) {
			s.config["dtype"] = "float16"
		}, []string{"config dtype", "float16"}},
		{"a bfloat16 torch_dtype", func(s *synth) {
			s.config["torch_dtype"] = "bfloat16"
		}, []string{"torch_dtype", "bfloat16"}},
		{"attention biases the file lacks", func(s *synth) {
			s.config["attention_bias"] = true
		}, []string{"attention_bias"}},
		{"norm biases the file lacks", func(s *synth) {
			s.config["norm_bias"] = true
		}, []string{"norm_bias"}},
	} {
		t.Run(c.name, func(t *testing.T) {
			s := newSynth(3)
			c.mutate(s)
			b, err := Open(t.Context(), s.write(t), Options{})
			if err == nil {
				b.Close()
				t.Fatal("Open succeeded")
			}
			t.Log(err)
			if !errors.Is(err, backend.ErrIncompatibleCheckpoint) {
				t.Errorf("Open = %v, want it to wrap ErrIncompatibleCheckpoint", err)
			}
			for _, w := range c.want {
				if !strings.Contains(err.Error(), w) {
					t.Errorf("Open = %v, want it to mention %s", err, w)
				}
			}
		})
	}
}

// TestOpenMissingFiles: without either file the directory is no checkpoint.
func TestOpenMissingFiles(t *testing.T) {
	for _, file := range []string{"model.safetensors", filepath.Join("encoder", "config.json")} {
		t.Run(file, func(t *testing.T) {
			dir := newSynth(4).write(t)
			if err := os.Remove(filepath.Join(dir, file)); err != nil {
				t.Fatal(err)
			}
			_, err := Open(t.Context(), dir, Options{})
			if !errors.Is(err, backend.ErrIncompatibleCheckpoint) || !strings.Contains(err.Error(), filepath.Base(file)) {
				t.Fatalf("Open = %v, want ErrIncompatibleCheckpoint naming %s", err, filepath.Base(file))
			}
		})
	}
}

// TestTemperatureAnyDType: the shipped checkpoints store temperature as F32,
// F32 and F16 (docs/ARCHITECTURE.md §1.1). Inference never reads it, so any
// dtype loads, and the model is the same whichever it is.
func TestTemperatureAnyDType(t *testing.T) {
	var want [][]float32
	for _, c := range []struct {
		dtype string
		raw   []byte
	}{
		{"F32", nil},
		{"F16", binary.LittleEndian.AppendUint16(binary.LittleEndian.AppendUint16(binary.LittleEndian.AppendUint16(nil, 0x3e66), 0x3d00), 0x3fec)},
		{"BF16", binary.LittleEndian.AppendUint16(binary.LittleEndian.AppendUint16(binary.LittleEndian.AppendUint16(nil, 0x3fcd), 0x3fa0), 0x3ffd)},
	} {
		t.Run(c.dtype, func(t *testing.T) {
			s := newSynth(5)
			if c.raw != nil {
				s.tensors["temperature"] = stTensor{dtype: c.dtype, shape: []int64{3}, raw: c.raw}
			}
			logits, _, err := open(t, s).Forward(t.Context(), tinyBatch())
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
			if want == nil {
				want = logits
			}
			for i := range want {
				if !slices.Equal(logits[i], want[i]) {
					t.Errorf("row %d: %v with temperature %s, %v with F32", i, logits[i], c.dtype, want[i])
				}
			}
		})
	}
}

// TestForwardConcurrent: Forward must be safe for concurrent calls
// (backend.Backend). Under -race, eight goroutines run their own batches at
// once, and each result must equal the same batch run alone.
//
// It runs twice: at the default single kernel worker, and with
// tensor.SetWorkers(4) on batches long enough that tensor.Linear crosses its
// parallel threshold of 2^18 multiply-adds (nn_ops.go) on Wqkv (1440 tokens
// x 24 x 8) and the head's linear1, so the concurrent calls also fan out
// through the kernels' own goroutines (runtime.go, parallelFor).
func TestForwardConcurrent(t *testing.T) {
	t.Run("one worker", func(t *testing.T) {
		forwardConcurrently(t, func(i int) backend.Batch {
			b := tinyBatch()
			b.QType = []int64{int64(i % 3), int64((i + 1) % 3)}
			b.InputIDs[0][1] = int64(i)
			return b
		})
	})
	t.Run("parallel kernels", func(t *testing.T) {
		prev := tensor.Workers()
		tensor.SetWorkers(4)
		t.Cleanup(func() { tensor.SetWorkers(prev) })
		forwardConcurrently(t, longBatch)
	})
}

// longBatch is four questions of 360 tokens, the last one padded, whose ids
// depend on i.
func longBatch(i int) backend.Batch {
	const rows, seq = 4, 360
	var b backend.Batch
	for r := range rows {
		ids, mask := make([]int64, seq), make([]int64, seq)
		for s := range seq {
			ids[s] = int64((s*7 + r*3 + i) % tinyVocab)
			if r < rows-1 || s < seq-60 {
				mask[s] = 1
			}
		}
		b.InputIDs = append(b.InputIDs, ids)
		b.AttentionMask = append(b.AttentionMask, mask)
		b.MarkerPos = append(b.MarkerPos, []int64{10, 150, 290})
		b.MarkerMask = append(b.MarkerMask, []bool{true, true, r%2 == 0})
		b.QType = append(b.QType, int64((r+i)%3))
	}
	return b
}

// forwardConcurrently runs batch(i) for eight i, first one by one and then
// all at once on the same Backend, and requires the same results.
func forwardConcurrently(t *testing.T, batch func(i int) backend.Batch) {
	t.Helper()
	b := open(t, newSynth(6))
	const n = 8
	batches := make([]backend.Batch, n)
	want := make([][2][][]float32, n)
	for i := range batches {
		batches[i] = batch(i)
		logits, act, err := b.Forward(t.Context(), batches[i])
		if err != nil {
			t.Fatal(err)
		}
		want[i] = [2][][]float32{logits, act}
	}

	var wg sync.WaitGroup
	errs := make([]error, n)
	for i := range n {
		wg.Go(func() {
			logits, act, err := b.Forward(context.Background(), batches[i])
			if err != nil {
				errs[i] = err
				return
			}
			for r := range logits {
				if !slices.Equal(logits[r], want[i][0][r]) || !slices.Equal(act[r], want[i][1][r]) {
					errs[i] = fmt.Errorf("batch %d row %d: concurrent %v %v, serial %v %v",
						i, r, logits[r], act[r], want[i][0][r], want[i][1][r])
					return
				}
			}
		})
	}
	wg.Wait()
	if err := errors.Join(errs...); err != nil {
		t.Fatal(err)
	}
}

func TestClose(t *testing.T) {
	b, err := Open(t.Context(), newSynth(7).write(t), Options{})
	if err != nil {
		t.Fatal(err)
	}
	for i := range 2 {
		if err := b.Close(); err != nil {
			t.Fatalf("Close #%d: %v", i+1, err)
		}
	}
	if _, _, err := b.Forward(t.Context(), tinyBatch()); !errors.Is(err, ErrClosed) {
		t.Fatalf("Forward after Close = %v, want ErrClosed", err)
	}

	var zero Backend
	if _, _, err := zero.Forward(t.Context(), tinyBatch()); !errors.Is(err, ErrClosed) {
		t.Fatalf("Forward on a zero Backend = %v, want ErrClosed", err)
	}
	if err := zero.Close(); err != nil {
		t.Fatalf("Close on a zero Backend: %v", err)
	}
}

// cancelAfter is a context whose Err is nil for its first n calls and
// context.Canceled from then on, so a test can cancel between two of
// Forward's checks.
type cancelAfter struct {
	mu sync.Mutex
	n  int
}

func (*cancelAfter) Deadline() (time.Time, bool) { return time.Time{}, false }
func (*cancelAfter) Done() <-chan struct{}       { return nil }
func (*cancelAfter) Value(any) any               { return nil }

func (c *cancelAfter) Err() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.n <= 0 {
		return context.Canceled
	}
	c.n--
	return nil
}

// TestForwardContext: Forward checks ctx before the encoder, between the
// encoder and the head, and after the head, and a cancellation seen at any of
// them is returned alone.
func TestForwardContext(t *testing.T) {
	b := open(t, newSynth(8))
	for n := range 3 {
		ctx := &cancelAfter{n: n}
		logits, act, err := b.Forward(ctx, tinyBatch())
		if !errors.Is(err, context.Canceled) || logits != nil || act != nil {
			t.Errorf("cancelled after %d checks: Forward = %v, %v, %v; want only context.Canceled", n, logits, act, err)
		}
	}
	if _, _, err := b.Forward(&cancelAfter{n: 100}, tinyBatch()); err != nil {
		t.Fatalf("Forward with a live context: %v", err)
	}
}

// TestOpenContext: a cancelled Open returns the context's error, not a
// verdict on the checkpoint.
func TestOpenContext(t *testing.T) {
	dir := newSynth(9).write(t)
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	b, err := Open(ctx, dir, Options{})
	if !errors.Is(err, context.Canceled) || errors.Is(err, backend.ErrIncompatibleCheckpoint) || b != nil {
		t.Fatalf("Open(cancelled) = %v, %v; want context.Canceled alone", b, err)
	}
}

// TestForwardBadBatch mirrors internal/backend/onnx's flatten: the batch must
// be n×L token tensors, n×K marker tensors and n qtypes, all positive.
func TestForwardBadBatch(t *testing.T) {
	b := open(t, newSynth(10))
	for _, c := range []struct {
		name   string
		mutate func(*backend.Batch)
	}{
		{"no rows", func(b *backend.Batch) { *b = backend.Batch{} }},
		{"no tokens", func(b *backend.Batch) {
			b.InputIDs = [][]int64{{}, {}}
			b.AttentionMask = [][]int64{{}, {}}
		}},
		{"no markers", func(b *backend.Batch) {
			b.MarkerPos = [][]int64{{}, {}}
			b.MarkerMask = [][]bool{{}, {}}
		}},
		{"marker_pos missing", func(b *backend.Batch) { b.MarkerPos = nil }},
		{"qtype rows", func(b *backend.Batch) { b.QType = b.QType[:1] }},
		{"ragged input_ids", func(b *backend.Batch) { b.InputIDs[1] = b.InputIDs[1][:9] }},
		{"attention_mask rows", func(b *backend.Batch) { b.AttentionMask = b.AttentionMask[:1] }},
		{"ragged attention_mask", func(b *backend.Batch) { b.AttentionMask[1] = append(b.AttentionMask[1], 0) }},
		{"ragged marker_pos", func(b *backend.Batch) { b.MarkerPos[1] = b.MarkerPos[1][:2] }},
		{"marker_mask rows", func(b *backend.Batch) { b.MarkerMask = b.MarkerMask[:1] }},
		{"ragged marker_mask", func(b *backend.Batch) { b.MarkerMask[0] = b.MarkerMask[0][:2] }},
		// The shapes are right; the encoder or the head refuses the values.
		{"id outside the vocabulary", func(b *backend.Batch) { b.InputIDs[0][4] = tinyVocab }},
		{"attention_mask value 2", func(b *backend.Batch) { b.AttentionMask[0][4] = 2 }},
		{"qtype 3", func(b *backend.Batch) { b.QType[1] = 3 }},
		{"marker past the sequence", func(b *backend.Batch) { b.MarkerPos[0][2] = 10 }},
	} {
		t.Run(c.name, func(t *testing.T) {
			batch := tinyBatch()
			c.mutate(&batch)
			if _, _, err := b.Forward(t.Context(), batch); !errors.Is(err, ErrBadBatch) {
				t.Fatalf("Forward = %v, want ErrBadBatch", err)
			}
		})
	}
}

// TestOpenNoHeadLayers: head_layers 0 is an empty nn.ModuleList upstream, a
// model that runs, so a file without head.layers.N opens and answers.
func TestOpenNoHeadLayers(t *testing.T) {
	s := newSynth(11)
	for name := range s.tensors {
		if strings.HasPrefix(name, "head.layers.") {
			delete(s.tensors, name)
		}
	}
	logits, act, err := open(t, s).Forward(t.Context(), tinyBatch())
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if len(logits) != 2 || len(act) != 2 || len(logits[0]) != 3 || len(act[0]) != tinyAct {
		t.Fatalf("Forward = %v, %v; want 2x3 logits and 2x%d act logits", logits, act, tinyAct)
	}
}

// TestF16Weights: the shipped checkpoints store their weights as F16. A
// checkpoint with every weight in F16 must give exactly what the same values
// stored as F32 give, so Open decodes either dtype onto the same model.
func TestF16Weights(t *testing.T) {
	s16, s32 := newSynth(12), newSynth(12)
	for name, st := range s32.tensors {
		if name == "temperature" {
			continue
		}
		var raw []byte
		for i, v := range st.data {
			h, f := toF16(t, v)
			st.data[i] = f
			raw = binary.LittleEndian.AppendUint16(raw, h)
		}
		s16.tensors[name] = stTensor{dtype: "F16", shape: st.shape, raw: raw}
	}
	batch := tinyBatch()
	l16, a16, err := open(t, s16).Forward(t.Context(), batch)
	if err != nil {
		t.Fatalf("F16: %v", err)
	}
	l32, a32, err := open(t, s32).Forward(t.Context(), batch)
	if err != nil {
		t.Fatalf("F32: %v", err)
	}
	for i := range l32 {
		if !slices.Equal(l16[i], l32[i]) || !slices.Equal(a16[i], a32[i]) {
			t.Errorf("row %d: F16 gives %v %v, F32 %v %v", i, l16[i], a16[i], l32[i], a32[i])
		}
	}
}

// toF16 truncates v to binary16 and returns its bits and its value, which
// float32 holds exactly. Below the smallest normal it is 0; the test values
// never reach the largest.
func toF16(t *testing.T, v float32) (uint16, float32) {
	t.Helper()
	bits := math.Float32bits(v)
	exp := int(bits>>23&0xff) - 127
	if exp < -14 {
		return 0, 0
	}
	if exp > 15 {
		t.Fatalf("%g is beyond binary16", v)
	}
	h := uint16(bits>>31)<<15 | uint16(exp+15)<<10 | uint16(bits>>13&0x3ff)
	return h, math.Float32frombits(bits &^ 0x1fff)
}
