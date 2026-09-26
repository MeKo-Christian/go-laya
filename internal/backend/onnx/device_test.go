package onnx

import (
	"errors"
	"slices"
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
		// (Task 6.3.6); auto skips it without a warning, since nobody asked.
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
// upstream's Apple device and a documented deviation (PLAN.md 7.5.3); CoreML
// is the ONNX Runtime equivalent.
func TestDeviceRejectsUnknown(t *testing.T) {
	for _, req := range []string{"mps", "gpu", "cuda:", "cuda:-1", "cuda:x", "cpu:0", " cpu", "CPU", "CUDA"} {
		if _, err := resolveDevice(req, gpuBuild); !errors.Is(err, ErrUnknownDevice) {
			t.Errorf("resolveDevice(%q) = %v, want ErrUnknownDevice", req, err)
		}
	}
}
