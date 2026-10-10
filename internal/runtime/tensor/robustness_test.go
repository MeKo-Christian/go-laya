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
