package onnx

import (
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"
)

// This file is pure stdlib and carries no build constraints, like resolve.go:
// which device a request lands on is decided without the binding, so it is
// testable on every target.

// The devices Options.Device accepts, besides "" and "auto". Upstream's "mps"
// is not one of them: CoreML is ONNX Runtime's Apple backend (PLAN.md 7.5.3).
const (
	deviceCPU    = "cpu"
	deviceCUDA   = "cuda"
	deviceCoreML = "coreml"
)

// ErrUnknownDevice is Open's error for an Options.Device that names no
// device. It is never a fallback: a typo would otherwise run on the CPU.
var ErrUnknownDevice = errors.New("onnx backend: unknown device")

// A deviceSpec ties a device to ONNX Runtime's execution provider for it.
type deviceSpec struct {
	name string
	// provider is the name GetAvailableProviders reports when the loaded
	// library was built with the device.
	provider string
	// appendName is what the binding passes to ORT's generic
	// SessionOptionsAppendExecutionProvider. Empty means the binding cannot
	// enable the device even where the library has it.
	appendName string
	// unusable says why, when appendName is empty.
	unusable string
}

// autoOrder is upstream's device=None preference (agent.py:166-172), with
// CoreML standing in for mps. The CPU provider is always there, and is ORT's
// default, so it is never appended.
var autoOrder = []deviceSpec{
	{
		name:     deviceCUDA,
		provider: "CUDAExecutionProvider",
		// ORT's generic append rejects CUDA ("Unknown provider name"); it
		// needs the dedicated _CUDA_V2 entry point, which the binding does not
		// register (PLAN.md Task 6.3.6).
		unusable: "the ONNX Runtime binding cannot enable CUDA",
	},
	{name: deviceCoreML, provider: "CoreMLExecutionProvider", appendName: "CoreML"},
	{name: deviceCPU, provider: "CPUExecutionProvider"},
}

// deviceChoice is where a request landed. providers is what to append to the
// session options; fallback is non-empty when an explicitly requested device
// could not be used, and says why.
type deviceChoice struct {
	requested string
	device    string
	providers []string
	fallback  string
}

// resolveDevice picks the device for req, given the execution providers the
// loaded library reports. Only an explicit request that cannot be met is a
// fallback; "auto" moves down its preference list silently, as upstream does.
func resolveDevice(req string, available []string) (deviceChoice, error) {
	name, err := parseDevice(req)
	if err != nil {
		return deviceChoice{}, err
	}

	usable := func(d deviceSpec) (bool, string) {
		switch {
		case d.name == deviceCPU:
			return true, ""
		case !slices.Contains(available, d.provider):
			return false, d.provider + " is not in this ONNX Runtime build"
		case d.appendName == "":
			return false, d.unusable
		}
		return true, ""
	}
	pick := func(d deviceSpec) deviceChoice {
		c := deviceChoice{requested: req, device: d.name}
		if d.appendName != "" {
			c.providers = []string{d.appendName}
		}
		return c
	}

	if name == "" {
		for _, d := range autoOrder {
			if ok, _ := usable(d); ok {
				return pick(d), nil
			}
		}
	}

	i := slices.IndexFunc(autoOrder, func(d deviceSpec) bool { return d.name == name })
	d := autoOrder[i]
	if ok, why := usable(d); !ok {
		c := pick(autoOrder[len(autoOrder)-1])
		c.fallback = why
		return c, nil
	}
	return pick(d), nil
}

// parseDevice validates req and returns the device name, or "" for auto.
// "cuda:N" is accepted, as torch.device takes it; the index is validated but
// has nothing to select until CUDA can be enabled at all.
func parseDevice(req string) (string, error) {
	switch req {
	case "", "auto":
		return "", nil
	case deviceCPU, deviceCUDA, deviceCoreML:
		return req, nil
	}
	if idx, ok := strings.CutPrefix(req, deviceCUDA+":"); ok {
		if n, err := strconv.Atoi(idx); err == nil && n >= 0 && idx == strconv.Itoa(n) {
			return deviceCUDA, nil
		}
	}
	return "", fmt.Errorf("%w %q (want cpu, cuda, cuda:N, coreml or auto)", ErrUnknownDevice, req)
}
