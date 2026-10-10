//go:build unix

package safetensors

import "syscall"

// openNonblock makes opening a named pipe with no writer return at once
// instead of waiting for one. A regular file's reads ignore it.
const openNonblock = syscall.O_NONBLOCK
