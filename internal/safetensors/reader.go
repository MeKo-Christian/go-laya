package safetensors

import (
	"cmp"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"maps"
	"math"
	"os"
	"slices"

	"github.com/MeKo-Christian/go-laya/backend"
)

// temperature is the one tensor laya's checkpoints carry but never read at
// inference: common.py:102 registers it and forward ignores it, while
// calibration comes from rl_agent_config.json (docs/ARCHITECTURE.md §1.1). The
// shipped checkpoints store it as F32, F32 and F16, so Open accepts it in any
// dtype ReadHeader accepts.
const temperature = "temperature"

// chunkBytes is how much tensor data Float32 reads per ReadAt and between two
// context checks. It is a multiple of every decoded dtype's width.
const chunkBytes = 1 << 20

// The dtypes Float32 decodes.
const (
	dtypeF16 = "F16"
	dtypeF32 = "F32"
)

// decodable are the dtypes Float32 converts to float32, with their widths.
// BF16 is left out on purpose: no shipped checkpoint stores it.
var decodable = map[string]int64{dtypeF16: 2, dtypeF32: 4}

// File is an open .safetensors file whose header ReadHeader's rules have
// validated, and whose tensors, temperature aside, are all F16 or F32.
//
// It reads tensor data with ReadAt on the descriptor the header was validated
// through, rather than mapping the file. Every tensor is converted to a fresh
// float32 slice anyway, which takes away mmap's zero-copy benefit; ReadAt is
// portable pure Go; and a file truncated under a mapping faults the process
// with SIGBUS, while under ReadAt it is an ordinary error. ReadAt is safe for
// concurrent use, and so is Float32.
type File struct {
	path   string
	f      *os.File
	header *Header
	names  []string // in data-buffer order
}

// Open opens and validates the .safetensors file at path. Beyond ReadHeader's
// rules, every tensor except temperature must be F16 or F32. Every failure
// wraps backend.ErrIncompatibleCheckpoint and names path.
func Open(path string) (*File, error) {
	f, err := openFile(path)
	if err != nil {
		return nil, fmt.Errorf("safetensors: %s: %w: %w", path, err, backend.ErrIncompatibleCheckpoint)
	}
	return f, nil
}

func openFile(path string) (*File, error) {
	// Opening a named pipe blocks until a writer appears, so the type is
	// checked first; readHeaderFrom then validates what was opened.
	if fi, err := os.Stat(path); err != nil {
		return nil, err
	} else if !fi.Mode().IsRegular() {
		return nil, fmt.Errorf("%s is not a regular file", fi.Mode().Type())
	}
	// #nosec G304 -- path is the weights file the caller chose to load;
	// validating it is the point.
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	h, err := readHeaderFrom(f)
	if err == nil {
		err = checkDTypes(h)
	}
	if err != nil {
		f.Close()
		return nil, err
	}

	names := slices.Collect(maps.Keys(h.Tensors))
	slices.SortFunc(names, func(a, b string) int {
		ta, tb := h.Tensors[a], h.Tensors[b]
		return cmp.Or(cmp.Compare(ta.Begin, tb.Begin), cmp.Compare(ta.End, tb.End), cmp.Compare(a, b))
	})
	return &File{path: path, f: f, header: h, names: names}, nil
}

// checkDTypes requires every tensor but temperature to be decodable, so a
// checkpoint Float32 cannot load is refused when it is opened rather than
// half-way through a load.
func checkDTypes(h *Header) error {
	for _, name := range slices.Sorted(maps.Keys(h.Tensors)) {
		if dt := h.Tensors[name].DType; name != temperature && decodable[dt] == 0 {
			return fmt.Errorf("tensor %q: dtype %s is neither F16 nor F32", name, dt)
		}
	}
	return nil
}

// Close closes the file.
func (f *File) Close() error {
	return f.f.Close()
}

// Names lists every tensor, temperature included, in the order of their data
// in the file, which is the order to read them in.
func (f *File) Names() []string {
	return slices.Clone(f.names)
}

// Info reports the named tensor's header entry.
func (f *File) Info(name string) (TensorInfo, bool) {
	t, ok := f.header.Tensors[name]
	t.Shape = slices.Clone(t.Shape)
	return t, ok
}

// Float32 reads the named tensor and returns its data as float32, row-major,
// with its shape. F16 is decoded exactly and F32 passed through, both read
// little-endian as the format specifies. temperature may be read by name: it
// decodes like any tensor when it is F16 or F32, and is an error otherwise.
//
// Reads stay inside the tensor's span of the data buffer, whose size
// ReadHeader checked against its dtype, shape and the file. ctx is checked
// before the read and between chunks of it, and a cancellation returns ctx's
// error alone. Every other failure wraps backend.ErrIncompatibleCheckpoint
// and names the tensor.
func (f *File) Float32(ctx context.Context, name string) ([]float32, []int64, error) {
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	data, err := f.read(ctx, name)
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return nil, nil, err
		}
		return nil, nil, fmt.Errorf("safetensors: %s: tensor %q: %w: %w", f.path, name, err, backend.ErrIncompatibleCheckpoint)
	}
	return data, slices.Clone(f.header.Tensors[name].Shape), nil
}

func (f *File) read(ctx context.Context, name string) ([]float32, error) {
	t, ok := f.header.Tensors[name]
	if !ok {
		return nil, errors.New("no such tensor")
	}
	width := decodable[t.DType]
	if width == 0 {
		return nil, fmt.Errorf("dtype %s is neither F16 nor F32", t.DType)
	}
	size := t.End - t.Begin
	// Only a 32-bit platform can fail this: the float32 copy's byte size
	// must fit in an int, or make panics.
	if size/width > math.MaxInt/4 {
		return nil, fmt.Errorf("%d elements do not fit in the address space as float32", size/width)
	}

	out := make([]float32, size/width)
	r := io.NewSectionReader(f.f, f.header.DataOffset+t.Begin, size)
	buf := make([]byte, min(size, chunkBytes))
	for i := 0; i < len(out); {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		chunk := buf[:min(int64(len(buf)), int64(len(out)-i)*width)]
		if _, err := io.ReadFull(r, chunk); err != nil {
			return nil, err
		}
		i += decode(out[i:], chunk, t.DType)
	}
	return out, nil
}

// decode converts the little-endian elements of src into dst, returning how
// many it wrote.
func decode(dst []float32, src []byte, dtype string) int {
	if dtype == dtypeF32 {
		n := len(src) / 4
		for k := range n {
			dst[k] = math.Float32frombits(binary.LittleEndian.Uint32(src[4*k:]))
		}
		return n
	}
	n := len(src) / 2
	for k := range n {
		dst[k] = f16ToF32(binary.LittleEndian.Uint16(src[2*k:]))
	}
	return n
}
