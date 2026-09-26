//go:build !android && !ios && ((darwin && (amd64 || arm64)) || (linux && (amd64 || arm64 || loong64)) || (netbsd && (amd64 || arm64)))

package onnx

import (
	"context"
	"errors"
	"log/slog"
	"runtime"
	"strings"
	"sync"
	"testing"

	ort "github.com/shota3506/onnxruntime-purego/onnxruntime"

	"github.com/MeKo-Christian/go-laya/backend"
	"github.com/MeKo-Christian/go-laya/internal/golden"
)

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

// TestOpenDevice runs device selection against the real library: a device
// that cannot be used falls back to the CPU with exactly one warning, and one
// that can be used warns never (upstream e630a68).
//
// Every case but one fixes the advertised providers, so the outcome does not
// depend on which build LAYA_ORT_LIB names: a CoreML-enabled ORT on macOS
// would otherwise make "auto" pick CoreML and an explicit CoreML session
// succeed. The session-failure cases need a library that rejects CoreML at
// session creation, which is Linux's; they skip elsewhere.
func TestOpenDevice(t *testing.T) {
	if testing.Short() {
		t.Skip("-short: needs an ONNX Runtime library and the S1 exports")
	}
	lib := requireORTLibrary(t)
	model, err := findModel("laya-" + golden.TypedDecisions + "-dynamo.onnx")
	if err != nil {
		t.Skipf("no export: %v", err)
	}

	coremlAdvertised := []string{"CoreMLExecutionProvider", "CPUExecutionProvider"}
	for _, tc := range []struct {
		name      string
		device    string
		advertise []string // replaces GetAvailableProviders' answer; nil keeps the library's
		linuxOnly bool     // needs ORT to reject CoreML when building the session
		want      string
		warnings  int
	}{
		// The one case on the library's own providers: "cpu" is the same
		// everywhere, and it keeps the real GetAvailableProviders call covered.
		{name: "cpu, library providers", device: "cpu", want: deviceCPU},
		{name: "auto", device: "", advertise: cpuBuild, want: deviceCPU},
		{name: "auto skips unusable cuda", device: "auto", advertise: gpuBuild, want: deviceCPU},
		{name: "cpu", device: "cpu", advertise: gpuBuild, want: deviceCPU},
		{name: "cuda not built", device: "cuda", advertise: cpuBuild, want: deviceCPU, warnings: 1},
		{name: "cuda not enableable", device: "cuda", advertise: gpuBuild, want: deviceCPU, warnings: 1},
		{name: "coreml not built", device: "coreml", advertise: cpuBuild, want: deviceCPU, warnings: 1},
		// A library that advertises CoreML but cannot create a session with
		// it: the session-creation fallback, upstream's failed .to(device)
		// (agent.py:203-216). Linux ORT rejects CoreML at append time, which
		// is exactly such a failure without needing Apple hardware.
		{
			name: "coreml advertised, session fails", device: "coreml",
			advertise: coremlAdvertised, linuxOnly: true, want: deviceCPU, warnings: 1,
		},
		// Auto picks the advertised CoreML and its session fails: a fallback
		// that happened, so it warns like the explicit request above, as
		// upstream's failed .to(device) warns for an auto-picked device.
		{
			name: "auto picks coreml, session fails", device: "auto",
			advertise: coremlAdvertised, linuxOnly: true, want: deviceCPU, warnings: 1,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.linuxOnly && runtime.GOOS != "linux" {
				t.Skipf("needs an ORT that rejects CoreML sessions; %s may build one", runtime.GOOS)
			}
			if tc.advertise != nil {
				saved := availableProviders
				availableProviders = func(*ort.Runtime) ([]string, error) { return tc.advertise, nil }
				t.Cleanup(func() { availableProviders = saved })
			}

			rec := &recorder{}
			b, err := Open(model, Options{Library: lib, Device: tc.device, Logger: slog.New(rec)})
			if err != nil {
				t.Fatalf("Open(device %q): %v", tc.device, err)
			}
			t.Cleanup(func() {
				if err := b.Close(); err != nil {
					t.Errorf("Close: %v", err)
				}
			})

			if got := b.Device(); got != tc.want {
				t.Errorf("Device() = %q, want %q", got, tc.want)
			}
			warns := rec.warnings()
			if len(warns) != tc.warnings {
				t.Errorf("%d warnings, want %d: %q", len(warns), tc.warnings, warns)
			}
			for _, w := range warns {
				t.Logf("warning: %s", w)
			}

			// The fallen-back session must still run.
			ids := [][]int64{{1, 2, 3, 4}}
			_, _, err = b.Forward(context.Background(), backend.Batch{
				InputIDs: ids, AttentionMask: [][]int64{{1, 1, 1, 1}},
				MarkerPos: [][]int64{{1, 2}}, MarkerMask: [][]bool{{true, true}}, QType: []int64{0},
			})
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
		})
	}
}

// TestOpenRejectsUnknownDevice: a bad device fails before the library is even
// loaded, so it cannot be mistaken for a missing runtime.
func TestOpenRejectsUnknownDevice(t *testing.T) {
	_, err := Open("testdata/forward_pass.json", Options{Library: "/nonexistent/libonnxruntime.so", Device: "mps"})
	if !errors.Is(err, ErrUnknownDevice) {
		t.Fatalf("Open(device mps) = %v, want ErrUnknownDevice", err)
	}
}
