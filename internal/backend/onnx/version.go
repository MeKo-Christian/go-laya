package onnx

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
)

// MinRuntimeMinor is the oldest ONNX Runtime 1.x minor release Open accepts.
// 1.23 introduced C API 23, the only one the binding implements, and ORT keeps
// serving older API tables from newer releases, so any later 1.x loads too
// (PLAN.md Task 6.6.2, decided: 1.23 or newer). The pinned download
// (internal/ortlib) is exactly 1.23.0.
const MinRuntimeMinor = APIVersion

// ErrRuntimeVersion is Open's error for an ONNX Runtime library whose own
// version string is not a 1.x release from 1.23 on, or could not be read.
var ErrRuntimeVersion = errors.New("onnx backend: unsupported ONNX Runtime version")

// checkRuntimeVersion applies the Task 6.6.2 policy to the library's
// self-reported version, as OrtGetApiBase()->GetVersionString returns it
// ("1.23.0"). Anything that does not parse as MAJOR.MINOR.PATCH fails closed:
// a version that cannot be read is not one that has been verified.
func checkRuntimeVersion(v string) error {
	parts := strings.Split(v, ".")
	nums := make([]int, 0, 3)
	if len(parts) == 3 {
		for _, p := range parts {
			n, err := strconv.Atoi(p)
			if err != nil || n < 0 || strconv.Itoa(n) != p {
				break
			}
			nums = append(nums, n)
		}
	}
	if len(nums) != 3 {
		return fmt.Errorf("%w: %q is not a MAJOR.MINOR.PATCH version", ErrRuntimeVersion, v)
	}
	if nums[0] != 1 || nums[1] < MinRuntimeMinor {
		return fmt.Errorf("%w: %q, need 1.%d or a later 1.x (C API %d)",
			ErrRuntimeVersion, v, MinRuntimeMinor, APIVersion)
	}
	return nil
}
