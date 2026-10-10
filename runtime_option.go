package laya

import (
	"fmt"
	"strconv"
	"strings"
)

// Runtime selects what runs a checkpoint's forward pass. The zero value is
// RuntimeONNX. Upstream has one runtime, PyTorch, so the choice is this
// port's own (D30).
type Runtime int

const (
	// RuntimeONNX runs a local ONNX export of the checkpoint on ONNX
	// Runtime (D24): the default.
	RuntimeONNX Runtime = iota

	// RuntimeNative runs the checkpoint's own model.safetensors on the
	// pure-Go backend, on the CPU, with no shared library (D8).
	RuntimeNative
)

// String is the runtime's name: "onnx", "native", or Runtime(N) for a value
// that is neither.
func (r Runtime) String() string {
	switch r {
	case RuntimeONNX:
		return "onnx"
	case RuntimeNative:
		return "native"
	}
	return "Runtime(" + strconv.Itoa(int(r)) + ")"
}

// check refuses a value that is neither runtime.
func (r Runtime) check() error {
	if r != RuntimeONNX && r != RuntimeNative {
		return fmt.Errorf("%w: %v", ErrUnknownRuntime, r)
	}
	return nil
}

// checkNative refuses, beside RuntimeNative, the options that only the ONNX
// backend reads, which would otherwise be silently ignored: the one naming an
// export (exportOpt, when exportSet) and a device other than the CPU
// (deviceOpt). Open and NewRouter pass their own options' names.
func (s loaderSettings) checkNative(exportSet bool, exportOpt, deviceOpt string) error {
	if s.runtime != RuntimeNative {
		return nil
	}
	var ignored []string
	if exportSet {
		ignored = append(ignored, exportOpt)
	}
	switch s.device {
	case "", "auto", "cpu":
	default:
		ignored = append(ignored, fmt.Sprintf("%s(%q)", deviceOpt, s.device))
	}
	if len(ignored) > 0 {
		return fmt.Errorf("%w: RuntimeNative runs the checkpoint's own weights on the CPU, so %s would be ignored",
			ErrConflictingOptions, strings.Join(ignored, " and "))
	}
	return nil
}
