// Package safetensors reads and validates a .safetensors file's header by the
// rules of the safetensors library's own validation, so a downloaded weights
// file is known to be well-formed before anything maps its data.
package safetensors

import (
	"bytes"
	"cmp"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/bits"
	"os"
	"slices"

	"github.com/MeKo-Christian/go-laya/backend"
)

// maxHeaderSize is the safetensors library's MAX_HEADER_SIZE.
const maxHeaderSize = 100_000_000

// metadataKey is the header entry that holds free-form string metadata
// instead of a tensor.
const metadataKey = "__metadata__"

// dtypeSizes are the byte widths of the dtypes this package accepts. The
// sub-byte float formats are left out: nothing laya ships uses them.
var dtypeSizes = map[string]uint64{
	"BOOL": 1, "U8": 1, "I8": 1, "F8_E5M2": 1, "F8_E4M3": 1, "F8_E8M0": 1,
	"I16": 2, "U16": 2, "F16": 2, "BF16": 2,
	"I32": 4, "U32": 4, "F32": 4,
	"I64": 8, "U64": 8, "F64": 8, "C64": 8,
}

// TensorInfo is one tensor's header entry. Begin and End are byte offsets
// into the data buffer, which starts at Header.DataOffset.
type TensorInfo struct {
	DType      string
	Shape      []int64
	Begin, End int64
}

// Header is a validated safetensors header.
type Header struct {
	Tensors    map[string]TensorInfo
	Metadata   map[string]string
	DataOffset int64
}

// ReadHeader reads and validates the header of the .safetensors file at path.
// The header must be at most 100 MB, lie inside the file and be a JSON
// object. Each tensor needs a known dtype, a shape of non-negative dims and
// data_offsets whose span is exactly its dtype's width times its element
// count, and sorted by offset the tensors must tile the data buffer from its
// first byte to the end of the file, with no gap and no overlap. Every failure
// wraps backend.ErrIncompatibleCheckpoint and names path.
func ReadHeader(path string) (*Header, error) {
	h, err := readHeader(path)
	if err != nil {
		return nil, fmt.Errorf("safetensors: %s: %w: %w", path, err, backend.ErrIncompatibleCheckpoint)
	}
	return h, nil
}

func readHeader(path string) (*Header, error) {
	// #nosec G304 -- path is the weights file the caller chose to load;
	// validating it is the point.
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}
	size := fi.Size()
	if size < 8 {
		return nil, fmt.Errorf("file of %d bytes is shorter than 8 bytes", size)
	}

	var prefix [8]byte
	if _, err := io.ReadFull(f, prefix[:]); err != nil {
		return nil, err
	}
	n := binary.LittleEndian.Uint64(prefix[:])
	if n > maxHeaderSize {
		return nil, fmt.Errorf("header of %d bytes is larger than %d bytes", n, maxHeaderSize)
	}
	if n > uint64(size-8) {
		return nil, fmt.Errorf("header of %d bytes runs past the end of the file (%d bytes)", n, size)
	}
	raw := make([]byte, n)
	if _, err := io.ReadFull(f, raw); err != nil {
		return nil, err
	}

	h, err := parse(raw)
	if err != nil {
		return nil, err
	}
	h.DataOffset = 8 + int64(n)
	return h, tile(h.Tensors, size-h.DataOffset)
}

// parse decodes the header JSON and checks each entry on its own.
func parse(raw []byte) (*Header, error) {
	if !bytes.HasPrefix(raw, []byte("{")) {
		return nil, errors.New("header is not a JSON object")
	}
	var entries map[string]json.RawMessage
	if err := json.Unmarshal(raw, &entries); err != nil {
		return nil, fmt.Errorf("header: %w", err)
	}

	h := &Header{Tensors: make(map[string]TensorInfo, len(entries))}
	for name, e := range entries {
		if name == metadataKey {
			if err := json.Unmarshal(e, &h.Metadata); err != nil {
				return nil, fmt.Errorf("%s: %w", metadataKey, err)
			}
			continue
		}
		t, err := tensor(e)
		if err != nil {
			return nil, fmt.Errorf("tensor %q: %w", name, err)
		}
		h.Tensors[name] = t
	}
	return h, nil
}

// tensor decodes one entry and checks that its span is the size its dtype
// and shape say.
func tensor(e json.RawMessage) (TensorInfo, error) {
	var raw struct {
		DType   *string  `json:"dtype"`
		Shape   *[]int64 `json:"shape"`
		Offsets []int64  `json:"data_offsets"`
	}
	if err := json.Unmarshal(e, &raw); err != nil {
		return TensorInfo{}, err
	}
	if raw.DType == nil {
		return TensorInfo{}, errors.New("no dtype")
	}
	width, ok := dtypeSizes[*raw.DType]
	if !ok {
		return TensorInfo{}, fmt.Errorf("unknown dtype %q", *raw.DType)
	}
	if raw.Shape == nil {
		return TensorInfo{}, errors.New("no shape")
	}
	if len(raw.Offsets) != 2 || raw.Offsets[0] < 0 || raw.Offsets[1] < 0 {
		return TensorInfo{}, fmt.Errorf("data_offsets %v is not [begin, end]", raw.Offsets)
	}
	t := TensorInfo{DType: *raw.DType, Shape: *raw.Shape, Begin: raw.Offsets[0], End: raw.Offsets[1]}
	if t.End < t.Begin {
		return TensorInfo{}, fmt.Errorf("data_offsets [%d, %d] run backwards", t.Begin, t.End)
	}

	need := width
	for _, d := range t.Shape {
		if d < 0 {
			return TensorInfo{}, fmt.Errorf("shape %v has a negative dim", t.Shape)
		}
		hi, lo := bits.Mul64(need, uint64(d))
		if hi != 0 || lo > 1<<63-1 {
			return TensorInfo{}, fmt.Errorf("shape %v overflows", t.Shape)
		}
		need = lo
	}
	if got := uint64(t.End - t.Begin); got != need {
		return TensorInfo{}, fmt.Errorf("holds %d bytes, but %s %v needs %d bytes", got, t.DType, t.Shape, need)
	}
	return t, nil
}

// tile requires the tensors, sorted by offset, to cover the data buffer of
// size bytes exactly once.
func tile(tensors map[string]TensorInfo, size int64) error {
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	slices.SortFunc(names, func(a, b string) int {
		ta, tb := tensors[a], tensors[b]
		return cmp.Or(cmp.Compare(ta.Begin, tb.Begin), cmp.Compare(ta.End, tb.End), cmp.Compare(a, b))
	})

	var end int64
	for _, name := range names {
		t := tensors[name]
		if t.Begin != end {
			return fmt.Errorf("tensor %q starts at %d, but the previous tensor ends at %d", name, t.Begin, end)
		}
		end = t.End
	}
	if end != size {
		return fmt.Errorf("tensors cover %d of %d data bytes", end, size)
	}
	return nil
}
