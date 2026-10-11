package tensor

import (
	"testing"
)

// FromOwned was added after the lift from go-pocket-tts (PLAN.md B.16), so
// its tests live in their own file, as robustness_test.go's do.

// TestFromOwnedAliasesData: the tensor is the caller's slice, not a copy of
// it. That is the whole point of FromOwned: the native loader hands over each
// decoded weight, and a copy would hold it twice while load runs.
func TestFromOwnedAliasesData(t *testing.T) {
	data := []float32{1, 2, 3, 4, 5, 6}

	x, err := FromOwned(data, []int64{2, 3})
	if err != nil {
		t.Fatalf("FromOwned: %v", err)
	}

	raw := x.RawData()
	if len(raw) != len(data) {
		t.Fatalf("RawData has %d elements, want the %d passed in", len(raw), len(data))
	}

	if &raw[0] != &data[0] {
		t.Fatalf("RawData is at %p, want the slice passed in, at %p", &raw[0], &data[0])
	}

	if shape := x.Shape(); !equalI64(shape, []int64{2, 3}) {
		t.Fatalf("shape = %v, want [2 3]", shape)
	}
}

// TestFromOwnedCopiesShape: only the data is taken over. The shape is a few
// int64s, so a caller that reuses its shape slice cannot reshape the tensor
// behind its back.
func TestFromOwnedCopiesShape(t *testing.T) {
	shape := []int64{2, 3}

	x, err := FromOwned(make([]float32, 6), shape)
	if err != nil {
		t.Fatalf("FromOwned: %v", err)
	}

	shape[0], shape[1] = 3, 2

	if got := x.Shape(); !equalI64(got, []int64{2, 3}) {
		t.Fatalf("shape = %v after the caller changed its slice, want [2 3]", got)
	}
}

// TestFromOwnedRejectsWhatNewRejects: FromOwned validates exactly as New
// does, with the same error, so a caller switching from one to the other
// gains no new way to build a tensor whose data does not fill its shape.
func TestFromOwnedRejectsWhatNewRejects(t *testing.T) {
	cases := []struct {
		name  string
		data  []float32
		shape []int64
	}{
		{"short", make([]float32, 5), []int64{2, 3}},
		{"long", make([]float32, 7), []int64{2, 3}},
		{"scalar without data", nil, []int64{}},
		{"zero-sized with data", make([]float32, 1), []int64{0, 3}},
		// The product is 6, so only the sign check can reject it.
		{"negative", make([]float32, 6), []int64{-2, -3}},
		{"overflow to 0", nil, overflowShapes[0]},
		{"overflow to negative", nil, overflowShapes[1]},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, want := New(tc.data, tc.shape)
			if want == nil {
				t.Fatalf("New(%d elements, %v) succeeded; the case must be one New rejects", len(tc.data), tc.shape)
			}

			x, err := FromOwned(tc.data, tc.shape)
			if err == nil {
				t.Fatalf("FromOwned(%d elements, %v) = %v elements, nil; want %q",
					len(tc.data), tc.shape, x.ElemCount(), want)
			}

			if err.Error() != want.Error() {
				t.Errorf("FromOwned error = %q, want New's %q", err, want)
			}
		})
	}
}

// TestFromOwnedAcceptsWhatNewAccepts: a zero-sized tensor and a scalar are as
// valid here as in New.
func TestFromOwnedAcceptsWhatNewAccepts(t *testing.T) {
	cases := []struct {
		name  string
		data  []float32
		shape []int64
	}{
		{"zero-sized", nil, []int64{0, 3}},
		{"scalar", []float32{7}, []int64{}},
		{"vector", []float32{1, 2}, []int64{2}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			x, err := FromOwned(tc.data, tc.shape)
			if err != nil {
				t.Fatalf("FromOwned(%v, %v): %v", tc.data, tc.shape, err)
			}

			if !equalI64(x.Shape(), tc.shape) || !equalF32(x.RawData(), tc.data, 0) {
				t.Errorf("FromOwned = %v %v, want %v %v", x.Shape(), x.RawData(), tc.shape, tc.data)
			}
		})
	}
}
