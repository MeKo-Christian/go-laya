package onnx

import (
	"errors"
	"strings"
	"testing"
)

// TestCheckRuntimeVersion is Task 6.6.2's policy: any 1.x release from 1.23 on
// serves the C API the binding speaks, and anything else, including a version
// that could not be read at all, is ErrRuntimeVersion naming what was found.
func TestCheckRuntimeVersion(t *testing.T) {
	for _, v := range []string{"1.23.0", "1.23.2", "1.24.1", "1.30.0"} {
		if err := checkRuntimeVersion(v); err != nil {
			t.Errorf("checkRuntimeVersion(%q) = %v, want nil", v, err)
		}
	}
	for _, v := range []string{
		"1.22.2", "1.9.0", "0.23.0", "2.0.0", "", "1.23", "1.23.0.1", "1.x.0", "v1.23.0",
		"1.23.0-rc1", "unknown (API version 23)",
	} {
		err := checkRuntimeVersion(v)
		if !errors.Is(err, ErrRuntimeVersion) {
			t.Errorf("checkRuntimeVersion(%q) = %v, want ErrRuntimeVersion", v, err)
			continue
		}
		if !strings.Contains(err.Error(), "\""+v+"\"") {
			t.Errorf("checkRuntimeVersion(%q): error %q does not name the version", v, err)
		}
	}
}
