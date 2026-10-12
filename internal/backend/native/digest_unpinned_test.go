//go:build !amd64 || purego || amd64.v3

package native

// digestBuildPinned is false where forwardDigests cannot hold: off amd64, in
// a purego build, which runs the kernels' pure-Go fallbacks, and under
// GOAMD64=v3, where the compiler may fuse a multiply and an add.
const digestBuildPinned = false
