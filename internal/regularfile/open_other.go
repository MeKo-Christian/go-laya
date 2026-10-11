//go:build !unix

package regularfile

// openNonblock is zero where opening a file cannot wait for a writer the way
// opening a Unix named pipe does.
const openNonblock = 0
