package onnx

import (
	"context"
	"errors"
	"log/slog"
	"slices"
	"strings"
	"sync"
	"testing"
)

// Provider lists as GetAvailableProviders reports them for the builds this
// repo has run against: the CPU and GPU packages of ORT 1.23.0 on Linux, and
// what a macOS build with CoreML reports.
var (
	cpuBuild    = []string{"CPUExecutionProvider"}
	gpuBuild    = []string{"TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"}
	coremlBuild = []string{"CoreMLExecutionProvider", "CPUExecutionProvider"}
)

// TestDeviceResolve pins which device each request lands on and, above all,
// when that counts as a fallback: only an explicitly requested device that
// cannot be used is one (agent.py:156-165). Auto picks silently, as upstream's
// device=None does (agent.py:166-172), and an explicit "cpu" never warns
// (upstream e630a68).
func TestDeviceResolve(t *testing.T) {
	for _, tc := range []struct {
		name      string
		req       string
		available []string
		device    string
		providers []string
		fallback  bool
	}{
		{"auto on cpu build", "", cpuBuild, deviceCPU, nil, false},
		{"auto spelled out", "auto", cpuBuild, deviceCPU, nil, false},
		{"auto prefers coreml", "auto", coremlBuild, deviceCoreML, []string{"CoreML"}, false},
		// CUDA is available in the GPU build but the binding cannot enable it
		// (Task B.3); auto skips it without a warning, since nobody asked.
		{"auto skips unusable cuda", "auto", gpuBuild, deviceCPU, nil, false},
		{"cpu", "cpu", gpuBuild, deviceCPU, nil, false},
		{"coreml", "coreml", coremlBuild, deviceCoreML, []string{"CoreML"}, false},
		{"coreml missing", "coreml", cpuBuild, deviceCPU, nil, true},
		{"cuda missing", "cuda", cpuBuild, deviceCPU, nil, true},
		{"cuda present but not enableable", "cuda", gpuBuild, deviceCPU, nil, true},
		{"cuda with index", "cuda:1", gpuBuild, deviceCPU, nil, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := resolveDevice(tc.req, tc.available)
			if err != nil {
				t.Fatalf("resolveDevice(%q): %v", tc.req, err)
			}
			if got.device != tc.device || !slices.Equal(got.providers, tc.providers) {
				t.Errorf("resolveDevice(%q) = %s %v, want %s %v", tc.req, got.device, got.providers, tc.device, tc.providers)
			}
			if (got.fallback != "") != tc.fallback {
				t.Errorf("resolveDevice(%q) fallback = %q, want fallback %v", tc.req, got.fallback, tc.fallback)
			}
		})
	}
}

// TestDeviceRejectsUnknown: a typo must not silently run on the CPU, and names
// are case-sensitive as torch.device's are. "mps" is
// upstream's Apple device and a documented deviation (PLAN.md 7.6.3); CoreML
// is the ONNX Runtime equivalent.
func TestDeviceRejectsUnknown(t *testing.T) {
	for _, req := range []string{"mps", "gpu", "cuda:", "cuda:-1", "cuda:x", "cpu:0", " cpu", "CPU", "CUDA"} {
		if _, err := resolveDevice(req, gpuBuild); !errors.Is(err, ErrUnknownDevice) {
			t.Errorf("resolveDevice(%q) = %v, want ErrUnknownDevice", req, err)
		}
	}
}

// recorder is a slog.Handler that keeps every record, so a test can count the
// warnings Open emitted.
type recorder struct {
	mu   sync.Mutex
	recs []slog.Record
}

func (*recorder) Enabled(context.Context, slog.Level) bool { return true }
func (*recorder) WithAttrs([]slog.Attr) slog.Handler       { panic("unused") }
func (*recorder) WithGroup(string) slog.Handler            { panic("unused") }

func (r *recorder) Handle(_ context.Context, rec slog.Record) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.recs = append(r.recs, rec)
	return nil
}

func (r *recorder) warnings() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	var out []string
	for _, rec := range r.recs {
		if rec.Level < slog.LevelWarn {
			continue
		}
		var b strings.Builder
		b.WriteString(rec.Message)
		rec.Attrs(func(a slog.Attr) bool {
			b.WriteString(" " + a.String())
			return true
		})
		out = append(out, b.String())
	}
	return out
}

// warnAttrs returns, per warning, its message under "msg" and each attribute's
// value under its key, so a test can assert what a warning carries.
func (r *recorder) warnAttrs() []map[string]string {
	r.mu.Lock()
	defer r.mu.Unlock()
	var out []map[string]string
	for _, rec := range r.recs {
		if rec.Level < slog.LevelWarn {
			continue
		}
		m := map[string]string{"msg": rec.Message}
		rec.Attrs(func(a slog.Attr) bool {
			m[a.Key] = a.Value.String()
			return true
		})
		out = append(out, m)
	}
	return out
}

// TestFallbackWarning is the Go device-fallback policy as behaviour, replacing
// upstream's test_criteria.py:103-116, which grepped agent.py's source for it.
// What that test meant, asserted on placeSession's log output with a fake
// session constructor, so it runs without ONNX Runtime:
//
//   - the warning fires only on a fallback that happened (upstream's
//     fell_back_from = None ... if fell_back_from is not None, agent.py:203,
//     218): never for an explicit "cpu", never for an "auto" that settles on
//     the CPU without trying anything else;
//   - it fires exactly once, and carries the reason (upstream's "Reason: %s",
//     agent.py:221) and what was requested, which is the actionable part:
//     which device to fix or stop asking for;
//   - a session that fails on the chosen device is retried on the CPU, and
//     that too warns once with the failure as its reason (agent.py:205-216).
//
// Upstream's last check, no bare torch.cuda.is_available() probe, has no Go
// counterpart: availability comes from the providers the loaded library
// reports, which resolveDevice takes as an argument.
func TestFallbackWarning(t *testing.T) {
	errSession := errors.New("provider rejected the graph")
	for _, tc := range []struct {
		name      string
		req       string
		available []string
		failOn    string // a provider whose session fails; "" means none does
		device    string
		warn      bool
		reason    string // a substring the warning's reason must contain
		calls     [][]string
	}{
		{name: "explicit cpu", req: "cpu", available: gpuBuild, device: deviceCPU, calls: [][]string{nil}},
		{name: "auto settles on cpu", req: "auto", available: gpuBuild, device: deviceCPU, calls: [][]string{nil}},
		{name: "auto default", req: "", available: cpuBuild, device: deviceCPU, calls: [][]string{nil}},
		{
			name: "auto picks coreml", req: "auto", available: coremlBuild, device: deviceCoreML,
			calls: [][]string{{"CoreML"}},
		},
		{
			name: "explicit coreml", req: "coreml", available: coremlBuild, device: deviceCoreML,
			calls: [][]string{{"CoreML"}},
		},
		{
			name: "cuda not built", req: "cuda", available: cpuBuild, device: deviceCPU,
			warn: true, reason: "CUDAExecutionProvider is not in this ONNX Runtime build", calls: [][]string{nil},
		},
		{
			name: "cuda not enableable", req: "cuda:0", available: gpuBuild, device: deviceCPU,
			warn: true, reason: "cannot enable CUDA", calls: [][]string{nil},
		},
		{
			name: "coreml not built", req: "coreml", available: cpuBuild, device: deviceCPU,
			warn: true, reason: "CoreMLExecutionProvider", calls: [][]string{nil},
		},
		{
			name: "coreml session fails", req: "coreml", available: coremlBuild, failOn: "CoreML",
			device: deviceCPU, warn: true, reason: "session on coreml failed: " + errSession.Error(),
			calls: [][]string{{"CoreML"}, nil},
		},
		// An auto-picked device whose session fails is a fallback that
		// happened, as upstream's failed .to(device) is for device=None.
		{
			name: "auto coreml session fails", req: "auto", available: coremlBuild, failOn: "CoreML",
			device: deviceCPU, warn: true, reason: errSession.Error(),
			calls: [][]string{{"CoreML"}, nil},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var calls [][]string
			newSession := func(providers []string) (string, error) {
				calls = append(calls, providers)
				if tc.failOn != "" && slices.Contains(providers, tc.failOn) {
					return "", errSession
				}
				return "session", nil
			}
			rec := &recorder{}
			sess, device, err := placeSession(tc.req, tc.available, "model.onnx", newSession, slog.New(rec))
			if err != nil || sess != "session" {
				t.Fatalf("placeSession = %q, %v; want a session", sess, err)
			}
			if device != tc.device {
				t.Errorf("device = %q, want %q", device, tc.device)
			}
			if !slices.EqualFunc(calls, tc.calls, slices.Equal) {
				t.Errorf("sessions built with %q, want %q", calls, tc.calls)
			}

			warns := rec.warnAttrs()
			if !tc.warn {
				if len(warns) != 0 {
					t.Fatalf("warnings %q, want none", warns)
				}
				return
			}
			if len(warns) != 1 {
				t.Fatalf("%d warnings %q, want exactly 1", len(warns), warns)
			}
			w := warns[0]
			if w["msg"] != fallbackWarning {
				t.Errorf("message %q, want %q", w["msg"], fallbackWarning)
			}
			if w["requested"] != tc.req {
				t.Errorf("requested = %q, want %q", w["requested"], tc.req)
			}
			if w["reason"] == "" || !strings.Contains(w["reason"], tc.reason) {
				t.Errorf("reason = %q, want it to contain %q", w["reason"], tc.reason)
			}
		})
	}
}

// TestFallbackWarningErrors: an unknown device fails before any session is
// built, and a CPU session that fails, first try or retry, is an error naming
// the model, never a warning.
func TestFallbackWarningErrors(t *testing.T) {
	failing := func(calls *int) func([]string) (string, error) {
		return func([]string) (string, error) {
			*calls++
			return "", errors.New("boom")
		}
	}
	for _, tc := range []struct {
		name      string
		req       string
		available []string
		calls     int
		is        error
	}{
		{name: "unknown device", req: "mps", available: coremlBuild, is: ErrUnknownDevice},
		{name: "cpu fails", req: "cpu", available: cpuBuild, calls: 1},
		{name: "retry fails", req: "coreml", available: coremlBuild, calls: 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var calls int
			rec := &recorder{}
			_, _, err := placeSession(tc.req, tc.available, "model.onnx", failing(&calls), slog.New(rec))
			if err == nil {
				t.Fatal("placeSession succeeded, want an error")
			}
			if tc.is != nil && !errors.Is(err, tc.is) {
				t.Errorf("err = %v, want %v", err, tc.is)
			}
			if tc.is == nil && !strings.Contains(err.Error(), "session model.onnx: boom") {
				t.Errorf("err = %v, want it to name the model and the failure", err)
			}
			if calls != tc.calls {
				t.Errorf("%d sessions built, want %d", calls, tc.calls)
			}
			if w := rec.warnAttrs(); len(w) != 0 {
				t.Errorf("warnings %q, want none", w)
			}
		})
	}
}
