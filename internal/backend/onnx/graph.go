package onnx

import (
	"fmt"
	"slices"
	"strconv"
	"strings"

	"github.com/MeKo-Christian/go-laya/backend"
	"github.com/MeKo-Christian/go-laya/internal/onnxheader"
)

// anyWidth is an output width nothing pins, so no check applies to it.
const anyWidth = -1

// checkGraphIO requires the graph to declare exactly graphInputs and
// graphOutputs, in any order, and otherwise returns an error wrapping
// backend.ErrIncompatibleCheckpoint that names each missing, unexpected or
// duplicated tensor. It is the graph half of _verify_compatibility's intent
// (agent.py:49-93); upstream has no graph and checks state-dict keys instead.
func checkGraphIO(inputs, outputs []string) error {
	problems := slices.Concat(diffNames("input", inputs, graphInputs), diffNames("output", outputs, graphOutputs))
	if len(problems) == 0 {
		return nil
	}
	return fmt.Errorf("graph: %s: %w", strings.Join(problems, "; "), backend.ErrIncompatibleCheckpoint)
}

// diffNames lists what separates got from want: names missing from got,
// names got has that want does not, and names got declares twice.
func diffNames(kind string, got, want []string) []string {
	var missing, unexpected, duplicate []string
	for _, w := range want {
		if !slices.Contains(got, w) {
			missing = append(missing, w)
		}
	}
	seen := make(map[string]bool, len(got))
	for _, g := range got {
		switch {
		case seen[g]:
			duplicate = append(duplicate, g)
		case !slices.Contains(want, g):
			unexpected = append(unexpected, g)
		}
		seen[g] = true
	}

	var out []string
	for _, p := range []struct {
		what  string
		names []string
	}{{"missing", missing}, {"unexpected", unexpected}, {"duplicate", duplicate}} {
		if len(p.names) > 0 {
			out = append(out, fmt.Sprintf("%s %s %s", p.what, kind, strings.Join(p.names, ", ")))
		}
	}
	return out
}

// checkHeadWidth checks the act_logits width the graph declares against want,
// the len(act_costs)+1 the checkpoint config derives, and returns the
// declared width. A declared act_logits must be 2-D, and a static width other
// than want is an error wrapping backend.ErrIncompatibleCheckpoint. A
// symbolic or undeclared width is anyWidth and passes, leaving Forward to
// check the width the graph produces; want 0 checks nothing but the rank.
func checkHeadWidth(outputs []onnxheader.Tensor, want int) (int, error) {
	i := slices.IndexFunc(outputs, func(t onnxheader.Tensor) bool { return t.Name == "act_logits" })
	if i < 0 || len(outputs[i].Dims) == 0 {
		return anyWidth, nil
	}
	dims := outputs[i].Dims
	if len(dims) != 2 {
		return 0, fmt.Errorf("graph: act_logits is %s, want 2-D: %w", declared(dims), backend.ErrIncompatibleCheckpoint)
	}
	got := dims[1].Value
	if got < 0 {
		return anyWidth, nil
	}
	if want > 0 && got != int64(want) {
		return 0, fmt.Errorf("graph: act_logits is %s, but the config's act_costs make it %d wide: %w",
			declared(dims), want, backend.ErrIncompatibleCheckpoint)
	}
	return int(got), nil
}

// declared renders declared dims as [batch 2]: a static size, a symbolic name,
// or ? for neither.
func declared(dims []onnxheader.Dim) string {
	parts := make([]string, len(dims))
	for i, d := range dims {
		switch {
		case d.Value >= 0:
			parts[i] = strconv.FormatInt(d.Value, 10)
		case d.Param != "":
			parts[i] = d.Param
		default:
			parts[i] = "?"
		}
	}
	return "[" + strings.Join(parts, " ") + "]"
}
