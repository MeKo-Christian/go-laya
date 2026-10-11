//go:build amd64 && !purego && !amd64.v3

package native

// digestBuildPinned reports whether forwardDigests can hold for this build:
// amd64 with the kernels' assembly and without the FMA fusion GOAMD64=v3
// lets the compiler make in plain Go.
const digestBuildPinned = true
