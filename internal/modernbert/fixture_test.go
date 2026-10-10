package modernbert

import (
	"bufio"
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/golden"
	"github.com/MeKo-Christian/go-laya/internal/runtime/tensor"
)

// opsFixture is testdata/ops.json, written by scripts/dump_modernbert_ops.py from
// the real transformers modules of a tiny ModernBertModel. Cases stay raw because
// each op has its own fields: Task 8.3 appends MLP cases to the same file.
type opsFixture struct {
	Header opsHeader         `json:"header"`
	Cases  []json.RawMessage `json:"cases"`
}

type opsHeader struct {
	Fixture  string            `json:"fixture"`
	Versions map[string]string `json:"versions"`
	Seed     *int64            `json:"seed"`
	Threads  *int              `json:"torch_threads"`
	Config   struct {
		HiddenSize int64    `json:"hidden_size"`
		NormEps    *float64 `json:"norm_eps"`
		NormBias   *bool    `json:"norm_bias"`
	} `json:"config"`
}

// tensorRec is the dumper's tensor_rec: a row-major float32 buffer and its shape.
type tensorRec struct {
	DType string    `json:"dtype"`
	Shape []int64   `json:"shape"`
	Data  []float32 `json:"data"`
}

func (r tensorRec) tensor(tb testing.TB, what string) *tensor.Tensor {
	tb.Helper()

	if r.DType != "float32" {
		tb.Fatalf("%s: dtype %q, want float32", what, r.DType)
	}
	t, err := tensor.New(r.Data, r.Shape)
	if err != nil {
		tb.Fatalf("%s: %v", what, err)
	}
	return t
}

func loadOps(tb testing.TB) opsFixture {
	tb.Helper()

	raw, err := os.ReadFile(filepath.Join("testdata", "ops.json"))
	if err != nil {
		tb.Fatalf("read the fixture: %v", err)
	}
	var f opsFixture
	if err := json.Unmarshal(raw, &f); err != nil {
		tb.Fatalf("decode testdata/ops.json: %v", err)
	}
	if len(f.Cases) == 0 {
		tb.Fatal("testdata/ops.json has no cases; an empty fixture is a broken checkout")
	}
	return f
}

// casesOf returns the raw cases whose "op" is op, failing if there are none: a
// filter that silently selects nothing is a test that silently passes.
func casesOf(tb testing.TB, f opsFixture, op string) []json.RawMessage {
	tb.Helper()

	var out []json.RawMessage
	for i, raw := range f.Cases {
		var head struct {
			Op string `json:"op"`
		}
		if err := json.Unmarshal(raw, &head); err != nil {
			tb.Fatalf("case %d: %v", i, err)
		}
		if head.Op == op {
			out = append(out, raw)
		}
	}
	if len(out) == 0 {
		tb.Fatalf("testdata/ops.json has no %q cases", op)
	}
	return out
}

// requirementPins reads the name==version lines of scripts/requirements-ref.txt,
// the reference environment PLAN.md R6 calls the contract. Reading the file
// rather than a copy of it means a moved pin fails here until ops.json is
// regenerated under it.
func requirementPins(tb testing.TB) map[string]string {
	tb.Helper()

	path := filepath.Join(golden.Root(tb), "scripts", "requirements-ref.txt")
	raw, err := os.ReadFile(path)
	if err != nil {
		tb.Fatalf("read %s: %v", path, err)
	}
	pins := map[string]string{}
	sc := bufio.NewScanner(bytes.NewReader(raw))
	for sc.Scan() {
		if name, version, ok := strings.Cut(strings.TrimSpace(sc.Text()), "=="); ok {
			pins[name] = version
		}
	}
	return pins
}

// TestOpsProvenance asserts the fixture came from the pinned environment and
// the checkpoints' norm settings. The tolerances in the op tests are claims
// about float32 torch on one version; a regeneration elsewhere must fail here
// rather than quietly move the oracle.
func TestOpsProvenance(t *testing.T) {
	h := loadOps(t).Header
	if h.Fixture != "ops" {
		t.Errorf("header fixture = %q, want \"ops\"", h.Fixture)
	}
	if h.Seed == nil {
		t.Error("header records no seed")
	}
	// One thread, so torch's reductions do not depend on how it split them.
	if h.Threads == nil || *h.Threads != 1 {
		t.Errorf("header torch_threads = %v, want 1", h.Threads)
	}

	pins := requirementPins(t)
	for _, name := range []string{"torch", "transformers", "numpy"} {
		want, ok := pins[name]
		if !ok {
			t.Fatalf("scripts/requirements-ref.txt pins no %s", name)
		}
		if got := h.Versions[name]; got != want {
			t.Errorf("fixture %s = %q, requirements-ref.txt pins %q: regenerate "+
				"testdata/ops.json under .venv-ref, as a reviewed diff (R6)", name, got, want)
		}
	}

	c := h.Config
	if c.NormEps == nil || float32(*c.NormEps) != NormEps {
		t.Errorf("fixture config norm_eps = %v, want %g", c.NormEps, NormEps)
	}
	if c.NormBias == nil || *c.NormBias {
		t.Errorf("fixture config norm_bias = %v, want false", c.NormBias)
	}
	if c.HiddenSize <= 0 {
		t.Errorf("fixture config hidden_size = %d", c.HiddenSize)
	}
}
