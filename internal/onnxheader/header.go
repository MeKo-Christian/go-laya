// Package onnxheader reads and validates an ONNX graph file before ONNX
// Runtime sees it: its protobuf framing, IR version and opset, where each
// tensor's external data lives, and the graph's declared inputs and outputs.
//
// It decodes the wire format with protowire and no generated ONNX types, so
// it reads only the handful of fields below and skips the rest.
package onnxheader

import (
	"errors"
	"fmt"
	"math/bits"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"google.golang.org/protobuf/encoding/protowire"

	"github.com/MeKo-Christian/go-laya/backend"
)

// ONNX TensorProto.DataType values the tests name.
const (
	elemFloat = 1
	elemInt64 = 7
	elemBool  = 9
)

// dataLocationExternal is TensorProto.DataLocation.EXTERNAL.
const dataLocationExternal = 1

// The IR versions and default-domain opset the pinned ONNX Runtime (1.23.0)
// loads, found by handing it models at each bound: IR 12 and opset 24 fail.
const (
	minIRVersion = 3 // the first IR version with opset_import
	maxIRVersion = 11
	maxOpset     = 23
)

// maxDepth bounds how deep subgraphs (If/Loop/Scan bodies) may nest.
const maxDepth = 32

// maxFileSize caps the graph file. The exports keep their weights in an
// external-data file and their graphs are about 3 MB.
const maxFileSize = 64 << 20

// Dim is one axis of a declared shape: Value is its static size, or -1 when
// it is symbolic (Param names it) or unknown.
type Dim struct {
	Value int64
	Param string
}

// Tensor is one of the graph's declared inputs or outputs. ElemType is the
// ONNX TensorProto.DataType, 0 for a value that is not a tensor.
type Tensor struct {
	Name     string
	ElemType int32
	Dims     []Dim
}

// Header is what Read learns about a graph: its IR version, its default-domain
// opset, and the main graph's declared inputs and outputs.
type Header struct {
	IRVersion int64
	Opset     int64
	Inputs    []Tensor
	Outputs   []Tensor
}

// Read validates the graph file at path and returns its Header. The file must
// be a regular file of at most 64 MiB holding a well-formed ModelProto with a
// graph, an IR version and ai.onnx opset the pinned runtime supports, and every
// tensor it can reach (initializers, sparse initializers, node attributes,
// subgraphs, functions, training graphs) that keeps its data externally must
// keep it in a regular file inside path's directory, within that file's size.
// Every failure wraps backend.ErrIncompatibleCheckpoint and names path.
func Read(path string) (*Header, error) {
	h, err := read(path)
	if err != nil {
		return nil, fmt.Errorf("onnx header: %s: %w: %w", path, err, backend.ErrIncompatibleCheckpoint)
	}
	return h, nil
}

func read(path string) (*Header, error) {
	size, err := regular(path)
	if err != nil {
		return nil, err
	}
	if size > maxFileSize {
		return nil, fmt.Errorf("larger than %d bytes", maxFileSize)
	}
	// #nosec G304 G703 -- path is the graph the caller chose to load;
	// validating it is the point.
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	r := &reader{dir: filepath.Dir(path), sizes: map[string]int64{}}
	return r.model(b)
}

// regular returns the size of the file at path, failing unless it is a
// regular file. It does not follow a symlink.
func regular(path string) (int64, error) {
	// #nosec G703 -- stat only: the graph the caller chose, or an external-data
	// location already required to be local to the graph's directory.
	fi, err := os.Lstat(path)
	if err != nil {
		return 0, err
	}
	if !fi.Mode().IsRegular() {
		return 0, fmt.Errorf("%s is not a regular file", path)
	}
	return fi.Size(), nil
}

// local returns the size of the regular file loc names below dir. Every
// directory on the way must be a real directory: Lstat on the joined path
// alone would follow a symlinked parent out of dir.
func local(dir, loc string) (int64, error) {
	segs := strings.Split(filepath.Clean(loc), string(filepath.Separator))
	p := dir
	for _, seg := range segs[:len(segs)-1] {
		p = filepath.Join(p, seg)
		// #nosec G703 -- stat only, one component of a location already
		// required to be local to the graph's directory.
		fi, err := os.Lstat(p)
		if err != nil {
			return 0, err
		}
		if !fi.IsDir() {
			return 0, fmt.Errorf("%s is not a directory", p)
		}
	}
	return regular(filepath.Join(p, segs[len(segs)-1]))
}

// reader walks one graph file, remembering the external-data files it has
// already checked.
type reader struct {
	dir   string
	sizes map[string]int64
}

// field is one decoded protobuf field: v for a varint, b for bytes.
type field struct {
	num int32
	typ protowire.Type
	v   uint64
	b   []byte
}

var errWireType = errors.New("unexpected wire type")

// bytesOf returns the field's bytes, failing when it is not length-delimited.
func (f field) bytesOf() ([]byte, error) {
	if f.typ != protowire.BytesType {
		return nil, fmt.Errorf("field %d: %w", f.num, errWireType)
	}
	return f.b, nil
}

// varintOf returns the field's varint, failing when it is not one.
func (f field) varintOf() (uint64, error) {
	if f.typ != protowire.VarintType {
		return 0, fmt.Errorf("field %d: %w", f.num, errWireType)
	}
	return f.v, nil
}

// fields calls fn for each field of the message b, in order.
func fields(b []byte, fn func(field) error) error {
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		if n < 0 {
			return malformed(n)
		}
		b = b[n:]
		f := field{num: int32(num), typ: typ}
		switch typ {
		case protowire.VarintType:
			f.v, n = protowire.ConsumeVarint(b)
		case protowire.BytesType:
			f.b, n = protowire.ConsumeBytes(b)
		case protowire.Fixed32Type, protowire.Fixed64Type, protowire.StartGroupType, protowire.EndGroupType:
			n = protowire.ConsumeFieldValue(num, typ, b)
		default: // wire types 6 and 7 do not exist
			n = protowire.ConsumeFieldValue(num, typ, b)
		}
		if n < 0 {
			return malformed(n)
		}
		b = b[n:]
		if err := fn(f); err != nil {
			return err
		}
	}
	return nil
}

func malformed(n int) error {
	return fmt.Errorf("malformed protobuf: %w", protowire.ParseError(n))
}

// wrapWire turns a wrong wire type on a known field into a malformed error.
func wrapWire(err error) error {
	if errors.Is(err, errWireType) {
		return fmt.Errorf("malformed protobuf: %w", err)
	}
	return err
}

// modelParts is a ModelProto's fields that Read looks at.
type modelParts struct {
	ir, opset                   uint64
	hasOpset                    bool
	graphs, training, functions [][]byte
}

// model checks a ModelProto's versions, then walks everything that can hold
// a tensor.
func (r *reader) model(b []byte) (*Header, error) {
	m, err := splitModel(b)
	if err != nil {
		return nil, wrapWire(err)
	}
	if len(m.graphs) == 0 {
		return nil, errors.New("no graph")
	}
	if m.ir < minIRVersion || m.ir > maxIRVersion {
		return nil, fmt.Errorf("IR version %d is outside [%d, %d]", m.ir, minIRVersion, maxIRVersion)
	}
	if !m.hasOpset {
		return nil, errors.New("no ai.onnx opset")
	}
	if m.opset > maxOpset {
		return nil, fmt.Errorf("ai.onnx opset %d is newer than %d", m.opset, maxOpset)
	}

	h := &Header{IRVersion: int64(m.ir), Opset: int64(m.opset)}
	for _, g := range m.graphs {
		if err := r.graph(g, 0, h); err != nil {
			return nil, wrapWire(err)
		}
	}
	// TrainingInfoProto: initialization (1) and algorithm (2) are graphs.
	for _, t := range m.training {
		if err := each(t, 1, 2, func(g []byte) error { return r.graph(g, 1, nil) }); err != nil {
			return nil, wrapWire(err)
		}
	}
	for _, fn := range m.functions {
		if err := r.function(fn); err != nil {
			return nil, wrapWire(err)
		}
	}
	return h, nil
}

// splitModel collects a ModelProto's fields that Read looks at.
func splitModel(b []byte) (modelParts, error) {
	var m modelParts
	err := fields(b, func(f field) error {
		if f.num == 1 { // ir_version
			var err error
			m.ir, err = f.varintOf()
			return err
		}
		var dst *[][]byte
		switch f.num {
		case 7: // graph
			dst = &m.graphs
		case 8: // opset_import
			o, err := f.bytesOf()
			if err != nil {
				return err
			}
			domain, version, err := operatorSet(o)
			if err == nil && (domain == "" || domain == "ai.onnx") {
				m.opset, m.hasOpset = version, true
			}
			return err
		case 20: // training_info
			dst = &m.training
		case 25: // functions
			dst = &m.functions
		default:
			return nil
		}
		sub, err := f.bytesOf()
		*dst = append(*dst, sub)
		return err
	})
	return m, err
}

// each calls fn with the bytes of every field of b numbered a or c.
func each(b []byte, a, c int32, fn func([]byte) error) error {
	return fields(b, func(f field) error {
		if f.num != a && f.num != c {
			return nil
		}
		sub, err := f.bytesOf()
		if err != nil {
			return err
		}
		return fn(sub)
	})
}

// operatorSet decodes an OperatorSetIdProto.
func operatorSet(b []byte) (domain string, version uint64, err error) {
	err = fields(b, func(f field) error {
		switch f.num {
		case 1:
			d, err := f.bytesOf()
			domain = string(d)
			return err
		case 2:
			v, err := f.varintOf()
			version = v
			return err
		}
		return nil
	})
	return domain, version, err
}

// graph walks a GraphProto at the given subgraph depth. io collects the
// declared inputs and outputs of the main graph only.
func (r *reader) graph(b []byte, depth int, io *Header) error {
	if depth >= maxDepth {
		return fmt.Errorf("subgraphs nested deeper than %d", maxDepth)
	}
	return fields(b, func(f field) error {
		switch f.num {
		case 1, 5, 15, 11, 12:
		default:
			return nil
		}
		sub, err := f.bytesOf()
		if err != nil {
			return err
		}
		switch f.num {
		case 1: // node
			return r.node(sub, depth)
		case 5: // initializer
			return r.tensor(sub)
		case 15: // sparse_initializer
			return r.sparse(sub)
		case 11, 12: // input, output
			if io == nil {
				return nil
			}
			t, err := valueInfo(sub)
			if err != nil {
				return err
			}
			if f.num == 11 {
				io.Inputs = append(io.Inputs, t)
			} else {
				io.Outputs = append(io.Outputs, t)
			}
		}
		return nil
	})
}

// function walks a FunctionProto: its nodes (7) and the default values of
// its attributes (attribute_proto, 11), which are AttributeProtos too.
func (r *reader) function(b []byte) error {
	return fields(b, func(f field) error {
		if f.num != 7 && f.num != 11 {
			return nil
		}
		sub, err := f.bytesOf()
		if err != nil {
			return err
		}
		if f.num == 7 {
			return r.node(sub, 0)
		}
		return r.attribute(sub, 0)
	})
}

// node walks a NodeProto's attributes (5).
func (r *reader) node(b []byte, depth int) error {
	return each(b, 5, 5, func(attr []byte) error { return r.attribute(attr, depth) })
}

// attribute walks an AttributeProto's tensors and subgraphs.
func (r *reader) attribute(b []byte, depth int) error {
	return fields(b, func(f field) error {
		switch f.num {
		case 5, 10, 6, 11, 22, 23:
		default:
			return nil
		}
		sub, err := f.bytesOf()
		if err != nil {
			return err
		}
		switch f.num {
		case 5, 10: // t, tensors
			return r.tensor(sub)
		case 6, 11: // g, graphs
			return r.graph(sub, depth+1, nil)
		default: // sparse_tensor, sparse_tensors
			return r.sparse(sub)
		}
	})
}

// sparse walks a SparseTensorProto: values (1) and indices (2) are tensors.
func (r *reader) sparse(b []byte) error { return each(b, 1, 2, r.tensor) }

// tensor decodes the fields of a TensorProto that locate its data and checks
// any external data.
func (r *reader) tensor(b []byte) error {
	var dims []uint64
	var dtype, location uint64
	var name string
	var ext [][2]string
	err := fields(b, func(f field) error {
		var err error
		switch f.num {
		case 1: // dims, packed or not
			if f.typ == protowire.BytesType {
				for p := f.b; len(p) > 0; {
					v, n := protowire.ConsumeVarint(p)
					if n < 0 {
						return malformed(n)
					}
					dims, p = append(dims, v), p[n:]
				}
				return nil
			}
			var d uint64
			d, err = f.varintOf()
			dims = append(dims, d)
		case 2: // data_type
			dtype, err = f.varintOf()
		case 8: // name
			var n []byte
			n, err = f.bytesOf()
			name = string(n)
		case 13: // external_data
			var kv []byte
			if kv, err = f.bytesOf(); err == nil {
				var e [2]string
				err = fields(kv, func(f field) error {
					if f.num != 1 && f.num != 2 {
						return nil
					}
					s, err := f.bytesOf()
					e[f.num-1] = string(s)
					return err
				})
				ext = append(ext, e)
			}
		case 14: // data_location
			location, err = f.varintOf()
		}
		return err
	})
	if err != nil {
		return err
	}
	if location != dataLocationExternal && len(ext) == 0 {
		return nil
	}
	if err := r.external(dims, dtype, ext); err != nil {
		return fmt.Errorf("tensor %q: %w", name, err)
	}
	return nil
}

// external checks one tensor's external_data entries: a location inside the
// graph's directory naming a regular file, and a byte range inside it.
func (r *reader) external(dims []uint64, dtype uint64, ext [][2]string) error {
	var loc string
	var offset, length int64 = 0, -1
	for _, e := range ext {
		var err error
		switch e[0] {
		case "location":
			loc = e[1]
		case "offset":
			offset, err = nonNegative(e)
		case "length":
			length, err = nonNegative(e)
		case "checksum":
		default:
			return fmt.Errorf("unknown external_data key %q", e[0])
		}
		if err != nil {
			return err
		}
	}
	if loc == "" {
		return errors.New("external data has no location")
	}
	if !filepath.IsLocal(loc) {
		return fmt.Errorf("external data location %q is not inside the graph's directory", loc)
	}
	if length < 0 {
		n, ok := byteSize(dims, dtype)
		if !ok {
			return errors.New("external data has no length and cannot be sized from its dims")
		}
		length = n
	}

	size, ok := r.sizes[loc]
	if !ok {
		var err error
		if size, err = local(r.dir, loc); err != nil {
			return err
		}
		r.sizes[loc] = size
	}
	if offset > size || length > size-offset {
		return fmt.Errorf("external data [%d, +%d) runs past the end of %s (%d bytes)", offset, length, loc, size)
	}
	return nil
}

// nonNegative parses an external_data offset or length.
func nonNegative(e [2]string) (int64, error) {
	v, err := strconv.ParseInt(e[1], 10, 64)
	if err != nil || v < 0 {
		return 0, fmt.Errorf("external data %s %q is not a non-negative integer", e[0], e[1])
	}
	return v, nil
}

// elemSizes are the byte widths of the fixed-size ONNX data types.
var elemSizes = map[uint64]uint64{
	1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8, 9: 1, 10: 2, 11: 8,
	12: 4, 13: 8, 14: 8, 15: 16, 16: 2, 17: 1, 18: 1, 19: 1, 20: 1,
}

// byteSize is dims' element count times dtype's width, and false when dtype
// has no fixed width or the product overflows an int64.
func byteSize(dims []uint64, dtype uint64) (int64, bool) {
	n, ok := elemSizes[dtype]
	if !ok {
		return 0, false
	}
	for _, d := range dims {
		hi, lo := bits.Mul64(n, d)
		if hi != 0 || lo > 1<<63-1 {
			return 0, false
		}
		n = lo
	}
	return int64(n), true
}

// valueInfo decodes a ValueInfoProto's name and, for a tensor, its element
// type and shape.
func valueInfo(b []byte) (Tensor, error) {
	var t Tensor
	err := fields(b, func(f field) error {
		switch f.num {
		case 1: // name
			n, err := f.bytesOf()
			t.Name = string(n)
			return err
		case 2: // type → TypeProto.tensor_type (1)
			typ, err := f.bytesOf()
			if err != nil {
				return err
			}
			return fields(typ, func(f field) error {
				if f.num != 1 {
					return nil
				}
				tt, err := f.bytesOf()
				if err != nil {
					return err
				}
				return tensorType(tt, &t)
			})
		}
		return nil
	})
	return t, err
}

// tensorType decodes TypeProto.Tensor: elem_type (1) and shape (2).
func tensorType(b []byte, t *Tensor) error {
	return fields(b, func(f field) error {
		switch f.num {
		case 1:
			v, err := f.varintOf()
			t.ElemType = int32(v)
			return err
		case 2:
			shape, err := f.bytesOf()
			if err != nil {
				return err
			}
			t.Dims = nil
			return fields(shape, func(f field) error {
				if f.num != 1 {
					return nil
				}
				dim, err := f.bytesOf()
				if err != nil {
					return err
				}
				d := Dim{Value: -1}
				err = fields(dim, func(f field) error {
					switch f.num {
					case 1:
						v, err := f.varintOf()
						d.Value, d.Param = int64(v), ""
						return err
					case 2:
						p, err := f.bytesOf()
						d.Value, d.Param = -1, string(p)
						return err
					}
					return nil
				})
				t.Dims = append(t.Dims, d)
				return err
			})
		}
		return nil
	})
}
