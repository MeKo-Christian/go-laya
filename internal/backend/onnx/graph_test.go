package onnx

import (
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/MeKo-Christian/go-laya/backend"
	"github.com/MeKo-Christian/go-laya/internal/onnxheader"
)

// TestCheckGraphIO pins what Open accepts as a DecisionModel graph: exactly
// the five inputs and two outputs, in any order. Anything else is
// ErrIncompatibleCheckpoint, and the message names the offending tensor
// rather than dumping both lists. It is pure, so unlike the rest of Open it
// runs in CI and on every platform.
func TestCheckGraphIO(t *testing.T) {
	without := func(names []string, drop string) []string {
		return slices.DeleteFunc(slices.Clone(names), func(n string) bool { return n == drop })
	}
	reversed := func(names []string) []string {
		r := slices.Clone(names)
		slices.Reverse(r)
		return r
	}

	type tc struct {
		name            string
		inputs, outputs []string
		want            []string // substrings of the error; nil means no error
	}
	tests := make([]tc, 0, 6+len(graphInputs)+len(graphOutputs))
	tests = append(
		tests,
		tc{name: "exact", inputs: graphInputs, outputs: graphOutputs},
		tc{name: "any order", inputs: reversed(graphInputs), outputs: reversed(graphOutputs)},
		tc{
			name: "extra input", inputs: append(slices.Clone(graphInputs), "token_type_ids"), outputs: graphOutputs,
			want: []string{"unexpected input", "token_type_ids"},
		},
		tc{
			name: "extra output", inputs: graphInputs, outputs: append(slices.Clone(graphOutputs), "hidden"),
			want: []string{"unexpected output", "hidden"},
		},
		tc{
			name: "duplicate input", inputs: append(slices.Clone(graphInputs), "qtype"), outputs: graphOutputs,
			want: []string{"duplicate input", "qtype"},
		},
		tc{
			name: "swapped", inputs: graphOutputs, outputs: graphInputs,
			want: []string{"missing input", "input_ids", "unexpected input", "logits"},
		},
	)
	for _, in := range graphInputs {
		tests = append(tests, tc{
			name: "no " + in, inputs: without(graphInputs, in), outputs: graphOutputs,
			want: []string{"missing input", in},
		})
	}
	for _, out := range graphOutputs {
		tests = append(tests, tc{
			name: "no " + out, inputs: graphInputs, outputs: without(graphOutputs, out),
			want: []string{"missing output", out},
		})
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := checkGraphIO(tt.inputs, tt.outputs)
			if tt.want == nil {
				if err != nil {
					t.Fatalf("checkGraphIO = %v, want nil", err)
				}
				return
			}
			if !errors.Is(err, backend.ErrIncompatibleCheckpoint) {
				t.Fatalf("checkGraphIO = %v, want ErrIncompatibleCheckpoint", err)
			}
			for _, w := range tt.want {
				if !strings.Contains(err.Error(), w) {
					t.Errorf("error %q does not mention %q", err, w)
				}
			}
			// Naming what was wrong means not naming what was right.
			if tt.name != "swapped" && strings.Contains(err.Error(), "attention_mask") && tt.name != "no attention_mask" {
				t.Errorf("error %q names a tensor that is fine", err)
			}
		})
	}
}

// TestCheckHeadWidth pins the static half of 6.4.5: a graph that declares
// act_logits with a fixed width must declare the width the config derives
// (len(act_costs)+1), and one that leaves it symbolic or undeclared is left
// to Forward's check. It returns the declared width, anyWidth if none.
func TestCheckHeadWidth(t *testing.T) {
	batch := onnxheader.Dim{Value: -1, Param: "batch"}
	act := func(dims ...onnxheader.Dim) []onnxheader.Tensor {
		return []onnxheader.Tensor{
			{Name: "logits", ElemType: 1, Dims: []onnxheader.Dim{batch, {Value: -1, Param: "k"}}},
			{Name: "act_logits", ElemType: 1, Dims: dims},
		}
	}
	tests := []struct {
		name    string
		outputs []onnxheader.Tensor
		want    int // the config's width; 0 means unchecked
		got     int
		err     []string // substrings of the error; nil means no error
	}{
		{name: "export", outputs: act(batch, onnxheader.Dim{Value: 2}), want: 2, got: 2},
		{name: "unchecked", outputs: act(batch, onnxheader.Dim{Value: 2}), got: 2},
		{name: "symbolic", outputs: act(batch, onnxheader.Dim{Value: -1, Param: "n"}), want: 2, got: anyWidth},
		{name: "undeclared", outputs: act(), want: 2, got: anyWidth},
		{
			name: "wider than the config", outputs: act(batch, onnxheader.Dim{Value: 3}), want: 2,
			err: []string{"act_logits", "[batch 3]", "2 wide"},
		},
		{
			name: "narrower than the config", outputs: act(batch, onnxheader.Dim{Value: 2}), want: 4,
			err: []string{"act_logits", "[batch 2]", "4 wide"},
		},
		{
			name: "rank 3", outputs: act(batch, onnxheader.Dim{Value: 2}, onnxheader.Dim{Value: 1}), want: 2,
			err: []string{"act_logits", "[batch 2 1]", "2-D"},
		},
		{name: "no act_logits", outputs: act()[:1], want: 2, got: anyWidth},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := checkHeadWidth(tt.outputs, tt.want)
			if tt.err == nil {
				if err != nil || got != tt.got {
					t.Fatalf("checkHeadWidth = %d, %v; want %d, nil", got, err, tt.got)
				}
				return
			}
			if !errors.Is(err, backend.ErrIncompatibleCheckpoint) {
				t.Fatalf("checkHeadWidth = %d, %v; want ErrIncompatibleCheckpoint", got, err)
			}
			for _, w := range tt.err {
				if !strings.Contains(err.Error(), w) {
					t.Errorf("error %q does not mention %q", err, w)
				}
			}
		})
	}
}
