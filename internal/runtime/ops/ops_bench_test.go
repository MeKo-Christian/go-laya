package ops

import (
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/runtime/tensor"
)

func BenchmarkMatMulFlowLM(b *testing.B) {
	a := mustTensor(b, seqData(1*16*64), []int64{1, 16, 64})
	w := mustTensor(b, seqData(1*64*64), []int64{1, 64, 64})
	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		_, err := tensor.MatMul(a, w)
		if err != nil {
			b.Fatalf("matmul: %v", err)
		}
	}
}

func BenchmarkLayerNormFlowLM(b *testing.B) {
	x := mustTensor(b, seqData(1*64*1024), []int64{1, 64, 1024})
	w := mustTensor(b, seqData(1024), []int64{1024})
	bias := mustTensor(b, seqData(1024), []int64{1024})
	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		_, err := tensor.LayerNorm(x, w, bias, 1e-5)
		if err != nil {
			b.Fatalf("layernorm: %v", err)
		}
	}
}

func BenchmarkAttentionFlowLM(b *testing.B) {
	q := mustTensor(b, seqData(1*8*32*64), []int64{1, 8, 32, 64})
	k := mustTensor(b, seqData(1*8*32*64), []int64{1, 8, 32, 64})
	v := mustTensor(b, seqData(1*8*32*64), []int64{1, 8, 32, 64})
	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		_, err := Attention(q, k, v, true, 0)
		if err != nil {
			b.Fatalf("attention: %v", err)
		}
	}
}

func seqData(n int) []float32 {
	out := make([]float32, n)
	for i := range n {
		out[i] = float32((i%17)-8) / 17
	}

	return out
}

func mustTensor(tb testing.TB, data []float32, shape []int64) *tensor.Tensor {
	tb.Helper()

	t, err := tensor.New(data, shape)
	if err != nil {
		tb.Fatalf("new tensor: %v", err)
	}

	return t
}
