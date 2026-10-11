package native

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"runtime"
	"runtime/debug"
	"testing"

	"golang.org/x/sys/cpu"

	"github.com/MeKo-Christian/go-laya/backend"
	"github.com/MeKo-Christian/go-laya/internal/golden"
	"github.com/MeKo-Christian/go-laya/internal/runtime/tensor"
)

// digestCases are the inputs TestForwardDigest hashes: Task 8.9's reference
// cell, the batch BenchmarkForward times, and a small batch of three
// questions with padding, a dead marker and every qtype. At 160 tokens its
// attention products (160·160·64 multiply-adds) are past the 2^20 at which
// a parallel MatMulTransB splits them.
var digestCases = []struct {
	name  string
	batch func() backend.Batch
}{
	{benchRefShape.String(), func() backend.Batch { return newBenchBatch(benchRefShape, benchIDSpan) }},
	{"b3_s160_k4_padded", digestPaddedBatch},
}

// digestPaddedBatch is newBenchBatch at 3×160 with four options, the second
// row padded after 100 tokens, the third with its last option dead, and the
// rows asking choice, score and noul in turn.
func digestPaddedBatch() backend.Batch {
	in := newBenchBatch(benchShape{batch: 3, seq: 160, k: 4}, benchIDSpan)
	clear(in.AttentionMask[1][100:])
	in.MarkerMask[2][3] = false
	in.QType = []int64{0, 1, 2}
	return in
}

// forwardDigests are the sha256 sums forwardDigest gave on the kernels of
// internal/runtime/tensor before Backlog B.19, per checkpoint and case. They
// hold on amd64 with AVX2 and FMA, in a default build: arm64's NEON block, an
// amd64 without AVX2 and a purego or GOAMD64=v3 build add in another order,
// so there the test logs the digests and compares the worker counts only.
var forwardDigests = map[string]string{
	golden.English + "/b1_s512_k4":               "8aa73e3f5db46c8d913cff8f6c3a0f884bc4fd990b6f0585dffcd689be02f770",
	golden.English + "/b3_s160_k4_padded":        "410626e0cc79ff207e422086bbd661f3cf14349e868d4888c7d121ffeca5dc33",
	golden.Multilingual + "/b1_s512_k4":          "70d34d63503dca1058c0512488bff412918e81d4301d8d7e22bbb2f841304ba6",
	golden.Multilingual + "/b3_s160_k4_padded":   "7dcfef7f3ee647447bdb97531a69842e9a38d8cad1a34b47b54de6d5072b2e32",
	golden.TypedDecisions + "/b1_s512_k4":        "1cbc677b8f45e341f7fc7f72a99fafd1b33ed207a6fc05b917dfd44b5476db26",
	golden.TypedDecisions + "/b3_s160_k4_padded": "a5dadd84bcb0ea7b8438951c77e2b14c1ddb36f79b175a39f0da9ab4c0a28412",
}

// TestForwardDigest holds the native backend's output to the bit, where
// TestForwardGolden allows a tolerance against PyTorch: a change of kernels
// must change no output bit, and a change of the worker count neither.
// Every checkpoint runs every digestCases batch at one kernel worker and at
// the default; the sha256 over the raw float32 bits of both runs must agree
// with each other and with forwardDigests.
//
// One checkpoint is loaded at a time and closed before the next.
func TestForwardDigest(t *testing.T) {
	root := golden.SkipWithoutModels(t)

	def := tensor.Workers()
	t.Cleanup(func() { tensor.SetWorkers(def) })
	if def < 2 {
		t.Logf("the default is %d kernel worker: workers=1 and the default run the same split", def)
	}
	pinned := cpu.X86.HasAVX2 && cpu.X86.HasFMA // false off amd64

	for _, ck := range benchCheckpoints {
		t.Run(ck, func(t *testing.T) {
			be, err := Open(t.Context(), golden.CheckpointDir(root, ck), Options{})
			if err != nil {
				t.Fatalf("Open: %v", err)
			}
			defer func() {
				if err := be.Close(); err != nil {
					t.Errorf("Close: %v", err)
				}
				runtime.GC()
				debug.FreeOSMemory()
			}()

			for _, c := range digestCases {
				in := c.batch()
				sums := make([]string, 0, 2)
				for _, w := range []int{1, def} {
					tensor.SetWorkers(w)
					logits, act, err := be.Forward(t.Context(), in)
					if err != nil {
						t.Fatalf("%s at %d workers: Forward: %v", c.name, w, err)
					}
					sums = append(sums, forwardDigest(logits, act))
					t.Logf("%s %s workers=%d sha256=%s", ck, c.name, w, sums[len(sums)-1])
				}
				tensor.SetWorkers(def)

				if sums[0] != sums[1] {
					t.Errorf("%s: %d workers give %s, one worker %s", c.name, def, sums[1], sums[0])
				}
				if want := forwardDigests[ck+"/"+c.name]; pinned && sums[0] != want {
					t.Errorf("%s: sha256 %s, want %s as on the kernels before B.19", c.name, sums[0], want)
				}
			}
		})
	}
}

// forwardDigest is the sha256 over Forward's output in a fixed order: every
// row of logits, then of act, each as its length and then its values'
// float32 bits, all little-endian.
func forwardDigest(logits, act [][]float32) string {
	h := sha256.New()
	var buf []byte
	for _, rows := range [][][]float32{logits, act} {
		for _, row := range rows {
			buf = binary.LittleEndian.AppendUint32(buf[:0], uint32(len(row)))
			for _, v := range row {
				buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(v))
			}
			h.Write(buf)
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}
