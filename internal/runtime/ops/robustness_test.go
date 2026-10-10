package ops

import (
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/runtime/tensor"
)

// The tests in this file cover the robustness fixes made after the lift from
// go-pocket-tts (NOTICE, "4. Modified"). They live in their own file so the
// lifted test files stay diffable against the source.

// RoPE on an empty tensor has nothing to rotate: it returns an empty tensor of
// the same shape rather than dividing by seq*dim.
func TestRoPEEmpty(t *testing.T) {
	cases := []struct {
		name              string
		xShape, trigShape []int64
	}{
		{"empty sequence", []int64{1, 0, 2}, []int64{0, 1}},
		{"zero width", []int64{1, 2, 0}, []int64{2, 0}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			x, err := tensor.Zeros(tc.xShape)
			if err != nil {
				t.Fatal(err)
			}

			trig, err := tensor.Zeros(tc.trigShape)
			if err != nil {
				t.Fatal(err)
			}

			got, err := RoPE(x, trig, trig, 0)
			if err != nil {
				t.Fatalf("RoPE: %v", err)
			}

			shape := got.Shape()
			if len(shape) != len(tc.xShape) {
				t.Fatalf("shape = %v, want %v", shape, tc.xShape)
			}

			for i := range shape {
				if shape[i] != tc.xShape[i] {
					t.Fatalf("shape = %v, want %v", shape, tc.xShape)
				}
			}

			if n := len(got.RawData()); n != 0 {
				t.Fatalf("len(data) = %d, want 0", n)
			}
		})
	}
}
