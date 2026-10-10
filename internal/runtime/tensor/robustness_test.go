package tensor

import (
	"math"
	"testing"
)

// The tests in this file cover the robustness fixes made after the lift from
// go-pocket-tts (NOTICE, "4. Modified"). They live in their own file so the
// lifted test files stay diffable against the source.

// overflowShapes are shapes whose element count does not fit in an int64.
// Multiplied without a check, the first wraps to 0 and the second to a
// negative count.
var overflowShapes = [][]int64{
	{4294967296, 4294967296},
	{3, math.MaxInt64 / 2},
}

func TestShapeElemCountRejectsOverflow(t *testing.T) {
	for _, shape := range overflowShapes {
		got, err := shapeElemCount(shape)
		if err == nil {
			t.Errorf("shapeElemCount(%v) = %d, nil; want an overflow error", shape, got)
		}
	}
}

func TestConstructorsRejectOverflowingShapes(t *testing.T) {
	for _, shape := range overflowShapes {
		if _, err := Zeros(shape); err == nil {
			t.Errorf("Zeros(%v) succeeded; want an error", shape)
		}

		if _, err := Full(shape, 1); err == nil {
			t.Errorf("Full(%v) succeeded; want an error", shape)
		}

		// A wrapped count of 0 would let New accept empty data for a huge shape.
		if _, err := New(nil, shape); err == nil {
			t.Errorf("New(nil, %v) succeeded; want an error", shape)
		}
	}
}

// Linear with a zero input width is well-defined, as in torch: every output
// is the empty sum 0 plus the optional bias. The row count must come from the
// shape, because len(data)/in divides by zero.
func TestLinearZeroInputWidth(t *testing.T) {
	x, err := Zeros([]int64{2, 0})
	if err != nil {
		t.Fatal(err)
	}

	w, err := Zeros([]int64{3, 0})
	if err != nil {
		t.Fatal(err)
	}

	bias, err := New([]float32{1, 2, 3}, []int64{3})
	if err != nil {
		t.Fatal(err)
	}

	got, err := Linear(x, w, nil)
	if err != nil {
		t.Fatalf("Linear without bias: %v", err)
	}

	if shape := got.Shape(); !equalI64(shape, []int64{2, 3}) {
		t.Fatalf("shape = %v, want [2 3]", shape)
	}

	if data := got.Data(); !equalF32(data, make([]float32, 6), 0) {
		t.Fatalf("data = %v, want zeros", data)
	}

	got, err = Linear(x, w, bias)
	if err != nil {
		t.Fatalf("Linear with bias: %v", err)
	}

	want := []float32{1, 2, 3, 1, 2, 3}
	if data := got.Data(); !equalF32(data, want, 0) {
		t.Fatalf("data = %v, want %v", data, want)
	}
}

func TestLinearZeroRows(t *testing.T) {
	x, err := Zeros([]int64{0, 4})
	if err != nil {
		t.Fatal(err)
	}

	w, err := Zeros([]int64{3, 4})
	if err != nil {
		t.Fatal(err)
	}

	got, err := Linear(x, w, nil)
	if err != nil {
		t.Fatalf("Linear: %v", err)
	}

	if shape := got.Shape(); !equalI64(shape, []int64{0, 3}) {
		t.Fatalf("shape = %v, want [0 3]", shape)
	}
}

// MatMul derives its row loop from the shape, so a zero inner dimension
// already yields zeros; this pins that down next to Linear.
func TestMatMulZeroInnerDim(t *testing.T) {
	a, err := Zeros([]int64{2, 0})
	if err != nil {
		t.Fatal(err)
	}

	b, err := Zeros([]int64{0, 3})
	if err != nil {
		t.Fatal(err)
	}

	got, err := MatMul(a, b)
	if err != nil {
		t.Fatalf("MatMul: %v", err)
	}

	if shape := got.Shape(); !equalI64(shape, []int64{2, 3}) {
		t.Fatalf("shape = %v, want [2 3]", shape)
	}

	if data := got.Data(); !equalF32(data, make([]float32, 6), 0) {
		t.Fatalf("data = %v, want zeros", data)
	}
}
