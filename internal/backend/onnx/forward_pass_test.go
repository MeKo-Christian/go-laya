//go:build !windows && !js && !wasm

// One forward pass of the S1 export through github.com/shota3506/onnxruntime-purego (Spike S2).
//
// The build constraint is not caution about purego -- it supports Windows -- but about
// what has actually been exercised. go-pocket-tts stubs the binding out on Windows and
// js/wasm entirely (internal/onnx/runner_windows.go, runner_wasm.go), so claiming those
// platforms here would be a claim no command backs. Keeping the constraint on the test
// file also keeps the binding out of `GOOS=windows go build ./...`.

package onnx

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"

	ort "github.com/shota3506/onnxruntime-purego/onnxruntime"
)

// Tolerances for the fixture comparison, scaled -- see maxScaledDiff.
//
// A plain absolute tolerance does not work across both outputs: logits carry a -1e4
// masked_fill sentinel and act_logits comes out in the thousands, so the same number
// would be far too tight on one and meaningless on the other.
//
// tolORT covers two ONNX Runtime builds evaluating the same graph: the fixture comes from
// the pinned reference environment's onnxruntime, the Go side loads whatever
// libonnxruntime is installed, and the binding only speaks C API 23. tolTorch also
// carries the ONNX-vs-PyTorch error S1 measured for this checkpoint.
const (
	tolORT   = 1e-4
	tolTorch = 1e-3
)

// tensorJSON is one tensor as scripts/export_onnx.py --fixture writes it. Data stays raw
// because the five inputs are not all numbers: marker_mask is a bool tensor, and that is
// precisely the element type nothing in-house has ever passed through this binding.
type tensorJSON struct {
	DType string          `json:"dtype"`
	Shape []int64         `json:"shape"`
	Data  json.RawMessage `json:"data"`
}

type fixture struct {
	Checkpoint  string                `json:"checkpoint"`
	ONNX        string                `json:"onnx"`
	Versions    map[string]string     `json:"versions"`
	InputNames  []string              `json:"input_names"`
	OutputNames []string              `json:"output_names"`
	Inputs      map[string]tensorJSON `json:"inputs"`
	ORT         map[string]tensorJSON `json:"ort"`
	Torch       map[string]tensorJSON `json:"torch"`
}

func loadFixture(t *testing.T) fixture {
	t.Helper()

	raw, err := os.ReadFile(filepath.Join("testdata", "forward_pass.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}

	var fx fixture
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatalf("decode fixture: %v", err)
	}

	return fx
}

func requireORTLibrary(t testing.TB) string {
	t.Helper()

	lib, err := findORTLibrary()
	if err != nil {
		t.Skipf("no ONNX Runtime: %v", err)
	}

	return lib
}

// newRuntime opens the shared library and logs which one answered, so a failing run
// names the runtime it was talking to rather than leaving it to be guessed.
func newRuntime(t testing.TB, lib string) *ort.Runtime {
	t.Helper()

	rt, err := ort.NewRuntime(lib, APIVersion)
	if err != nil {
		t.Fatalf("NewRuntime(%q, %d): %v", lib, APIVersion, err)
	}

	t.Cleanup(func() {
		if err := rt.Close(); err != nil {
			t.Errorf("runtime close: %v", err)
		}
	})
	t.Logf("ONNX Runtime %s (C API %d) from %s", rt.GetVersionString(), rt.GetAPIVersion(), lib)

	return rt
}

// decode turns one fixture tensor into an ORT value. The dtype string is the exporter's
// numpy dtype, which is what makes the bool path explicit rather than inferred.
func decode(t *testing.T, rt *ort.Runtime, name string, tj tensorJSON) *ort.Value {
	t.Helper()

	var (
		v   *ort.Value
		err error
	)

	switch tj.DType {
	case "int64":
		var data []int64
		mustUnmarshal(t, name, tj.Data, &data)
		v, err = ort.NewTensorValue(rt, data, tj.Shape)
	case "bool":
		var data []bool
		mustUnmarshal(t, name, tj.Data, &data)
		v, err = ort.NewTensorValue(rt, data, tj.Shape)
	case "float32":
		var data []float32
		mustUnmarshal(t, name, tj.Data, &data)
		v, err = ort.NewTensorValue(rt, data, tj.Shape)
	default:
		t.Fatalf("%s: unhandled fixture dtype %q", name, tj.DType)
	}

	if err != nil {
		t.Fatalf("%s: NewTensorValue: %v", name, err)
	}

	t.Cleanup(v.Close)

	return v
}

func mustUnmarshal(t *testing.T, name string, raw json.RawMessage, into any) {
	t.Helper()

	if err := json.Unmarshal(raw, into); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
}

func wantFloats(t *testing.T, name string, tj tensorJSON) []float32 {
	t.Helper()

	var data []float32
	mustUnmarshal(t, name, tj.Data, &data)

	return data
}

// maxScaledDiff reports the largest |got-want| divided by max(1, |want|), or -1 on a
// length mismatch. The floor of 1 keeps it an absolute comparison for small values and
// turns it into a relative one for the large ones, so a single tolerance covers both the
// -1e4 mask sentinel in logits and the thousands act_logits comes out in.
func maxScaledDiff(got, want []float32) float64 {
	if len(got) != len(want) {
		return -1
	}

	worst := 0.0

	for i := range got {
		d := math.Abs(float64(got[i])-float64(want[i])) / max(1, math.Abs(float64(want[i])))
		if d > worst {
			worst = d
		}
	}

	return worst
}

// TestForwardPass is S2.1: load the S1 export and run one forward pass, CGO-free.
//
// "Green" means more than "it returned something". The graph's own input and output
// names are checked against the exporter's, and the values are compared against both
// the Python ORT run and PyTorch. Storing both is what separates a binding that fed the
// tensors wrong -- a bool packed as four bytes would do it -- from two ONNX Runtime
// builds disagreeing in the last digits.
func TestForwardPass(t *testing.T) {
	if testing.Short() {
		t.Skip("-short: needs an ONNX Runtime library and the 1.7 GB S1 export")
	}

	lib := requireORTLibrary(t)
	fx := loadFixture(t)

	model, err := findModel(fx.ONNX)
	if err != nil {
		t.Skipf("no export: %v (run scripts/export_onnx.py --all --dynamo and set LAYA_ONNX_DIR)", err)
	}

	t.Logf("checkpoint %s, fixture built with onnxruntime %s", fx.Checkpoint, fx.Versions["onnxruntime"])

	rt := newRuntime(t, lib)

	env, err := rt.NewEnv("laya-onnx", ort.LoggingLevelWarning)
	if err != nil {
		t.Fatalf("NewEnv: %v", err)
	}

	t.Cleanup(env.Close)

	// From a path, never a reader: the export is 1.7 GB and split into a graph file plus
	// an external-data blob that ORT resolves relative to the model path.
	sess, err := rt.NewSession(env, model, &ort.SessionOptions{IntraOpNumThreads: 1})
	if err != nil {
		t.Fatalf("NewSession(%q): %v", model, err)
	}

	t.Cleanup(sess.Close)

	if got := sess.InputNames(); !slices.Equal(got, fx.InputNames) {
		t.Fatalf("graph inputs = %v, fixture says %v", got, fx.InputNames)
	}

	if got := sess.OutputNames(); !slices.Equal(got, fx.OutputNames) {
		t.Fatalf("graph outputs = %v, fixture says %v", got, fx.OutputNames)
	}

	inputs := make(map[string]*ort.Value, len(fx.InputNames))
	for _, name := range fx.InputNames {
		inputs[name] = decode(t, rt, name, fx.Inputs[name])
	}

	outputs, err := sess.Run(context.Background(), inputs)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	for _, v := range outputs {
		t.Cleanup(v.Close)
	}

	for _, name := range fx.OutputNames {
		v, ok := outputs[name]
		if !ok {
			t.Fatalf("output %q missing from %v", name, outputNames(outputs))
		}

		got, shape, err := ort.GetTensorData[float32](v)
		if err != nil {
			t.Fatalf("%s: GetTensorData: %v", name, err)
		}

		if !slices.Equal(shape, fx.ORT[name].Shape) {
			t.Errorf("%s: shape = %v, want %v", name, shape, fx.ORT[name].Shape)
		}

		vsORT := maxScaledDiff(got, wantFloats(t, name, fx.ORT[name]))
		vsTorch := maxScaledDiff(got, wantFloats(t, name, fx.Torch[name]))
		t.Logf("%s%v: max scaled diff vs python-ORT %.3g, vs PyTorch %.3g", name, shape, vsORT, vsTorch)

		if vsORT < 0 || vsORT > tolORT {
			t.Errorf("%s: %v vs python-ORT exceeds %g (got %v, want %s)", name, vsORT, tolORT, got, fx.ORT[name].Data)
		}

		if vsTorch < 0 || vsTorch > tolTorch {
			t.Errorf("%s: %v vs PyTorch exceeds %g", name, vsTorch, tolTorch)
		}
	}
}

// outputNames lists what Run actually returned, so a missing output says what did come back.
func outputNames(m map[string]*ort.Value) []string {
	names := make([]string, 0, len(m))
	for k := range m {
		names = append(names, k)
	}

	slices.Sort(names)

	return names
}

// finalizerEnv opts into the subtest that abandons values instead of closing them.
// That subtest is a known-failing reproduction, not an assertion -- see TestValueCleanup.
const finalizerEnv = "LAYA_ORT_FINALIZER"

// TestValueCleanup is S2.2: does a finalizer defect reproduce under -race, in a loop?
//
// go-pocket-tts records one as "local ONNX-backed native parity tests can panic inside
// onnxruntime-purego with runtime.AddCleanup" (its PLAN.md:36) and works around it with a
// manual `-skip`. That panic is upstream PR #11 -- newValueFromPtr registered a cleanup
// whose closure captured the *Value being cleaned up, so it was never collectable -- and
// it is fixed in the commit go.mod pins. go-pocket-tts is pinned two commits earlier.
//
// A different defect is still there, and this test is what found it. The package uses no
// synchronisation at all: Runtime.Close writes r.apiFuncs = nil (runtime.go:178) while
// releaseValuePtr reads it from the GC's cleanup goroutine (value.go:152). The nil check
// in that guard is therefore a data race on a multi-word struct, not a safety net.
//
// The reproduction needs values that are dropped *without* Close, because Close calls
// cleanup.Stop() and takes the finalizer out of play entirely. Hence the two subtests:
// "closed" is the assertion, "abandoned" is the reproduction, and the difference between
// them is exactly the rule D5 records. Neither needs a model or a session, so -count can
// be large without loading 1.7 GB each time.
func TestValueCleanup(t *testing.T) {
	const iterations = 256

	t.Run("closed", func(t *testing.T) {
		rt := newRuntime(t, requireORTLibrary(t))

		for i := range iterations {
			v, err := ort.NewTensorValue(rt, []float32{1, 2, 3, 4}, []int64{2, 2})
			if err != nil {
				t.Fatalf("iteration %d: NewTensorValue: %v", i, err)
			}

			v.Close()
		}

		// Two cycles: the first makes the values unreachable, the second gives the
		// cleanup goroutine a scheduling point before Close runs from t.Cleanup.
		runtime.GC()
		runtime.GC()
	})

	t.Run("abandoned", func(t *testing.T) {
		if os.Getenv(finalizerEnv) == "" {
			t.Skipf("known race in the binding; set %s=1 to reproduce", finalizerEnv)
		}

		rt := newRuntime(t, requireORTLibrary(t))

		for i := range iterations {
			v, err := ort.NewTensorValue(rt, []float32{1, 2, 3, 4}, []int64{2, 2})
			if err != nil {
				t.Fatalf("iteration %d: NewTensorValue: %v", i, err)
			}

			_ = v // deliberately not closed: the finalizer path is the point
		}

		runtime.GC()
		runtime.GC()
	})
}
