package onnx

import (
	"fmt"
	"slices"
	"strings"

	"github.com/MeKo-Christian/go-laya/backend"
)

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
