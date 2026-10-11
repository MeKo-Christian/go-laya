// Package regularfile opens a checkpoint's files so that a path naming a
// named pipe, a device or a directory is refused at once instead of read,
// or waited on, as a file.
package regularfile

import (
	"fmt"
	"os"
)

// Open opens path for reading and requires the opened descriptor to be a
// regular file. The type is checked on the descriptor, not on the path
// beforehand, so a path swapped for a named pipe or device between a check
// and the open cannot slip through. openNonblock keeps the open itself from
// waiting on a named pipe, which no context could interrupt; symlinks are
// followed, as a checkpoint may sit in a cache that links to its blobs.
func Open(path string) (*os.File, error) {
	// #nosec G304 -- path is a checkpoint file the caller chose to load;
	// validating it is the point.
	f, err := os.OpenFile(path, os.O_RDONLY|openNonblock, 0)
	if err != nil {
		return nil, err
	}
	fi, err := f.Stat()
	if err == nil && !fi.Mode().IsRegular() {
		err = fmt.Errorf("%s is not a regular file", fi.Mode().Type())
	}
	if err != nil {
		f.Close()
		return nil, err
	}
	return f, nil
}
