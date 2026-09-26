package onnxheader

import (
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"google.golang.org/protobuf/encoding/protowire"

	"github.com/MeKo-Christian/go-laya/backend"
)

// msg builds protobuf wire bytes field by field, so the tests need neither a
// fixture file nor Python.
type msg []byte

func (m msg) varint(num protowire.Number, v uint64) msg {
	m = protowire.AppendTag(m, num, protowire.VarintType)
	return protowire.AppendVarint(m, v)
}

func (m msg) bytes(num protowire.Number, b []byte) msg {
	m = protowire.AppendTag(m, num, protowire.BytesType)
	return protowire.AppendBytes(m, b)
}

func (m msg) str(num protowire.Number, s string) msg { return m.bytes(num, []byte(s)) }

func opset(domain string, version uint64) msg { return msg{}.str(1, domain).varint(2, version) }

func model(ir, op uint64, graph msg) msg {
	m := msg{}.varint(1, ir).str(2, "pytorch")
	if graph != nil {
		m = m.bytes(7, graph)
	}
	return m.bytes(8, opset("", op))
}

// ioTensor is a tensor-typed ValueInfoProto; a dim is a number or a name.
func ioTensor(name string, elem uint64, dims ...any) msg {
	var shape msg
	for _, d := range dims {
		switch d := d.(type) {
		case int:
			shape = shape.bytes(1, msg{}.varint(1, uint64(d)))
		case string:
			shape = shape.bytes(1, msg{}.str(2, d))
		}
	}
	tensorType := msg{}.varint(1, elem).bytes(2, shape)
	return msg{}.str(1, name).bytes(2, msg{}.bytes(1, tensorType))
}

// exportIO is the graph IO scripts/export_onnx.py writes, as read from
// build/onnx/laya-english-dynamo.onnx.
func exportIO() msg {
	return msg{}.
		bytes(11, ioTensor("input_ids", elemInt64, "batch", "seq")).
		bytes(11, ioTensor("attention_mask", elemInt64, "batch", "seq")).
		bytes(11, ioTensor("marker_pos", elemInt64, "batch", "k")).
		bytes(11, ioTensor("marker_mask", elemBool, "batch", "k")).
		bytes(11, ioTensor("qtype", elemInt64, "batch")).
		bytes(12, ioTensor("logits", elemFloat, "batch", "k")).
		bytes(12, ioTensor("act_logits", elemFloat, "batch", 2))
}

// external is a float TensorProto of dims whose data lives in another file.
func external(name string, dims []uint64, kv ...string) msg {
	t := msg{}
	for _, d := range dims {
		t = t.varint(1, d)
	}
	t = t.varint(2, elemFloat).str(8, name)
	for i := 0; i+1 < len(kv); i += 2 {
		t = t.bytes(13, msg{}.str(1, kv[i]).str(2, kv[i+1]))
	}
	return t.varint(14, dataLocationExternal)
}

// inSubgraph hides g inside an If node's then_branch attribute.
func inSubgraph(g msg) msg {
	attr := msg{}.str(1, "then_branch").bytes(6, g).varint(20, 5)
	return msg{}.bytes(1, msg{}.str(4, "If").bytes(5, attr))
}

// nested wraps g in depth levels of subgraph.
func nested(g msg, depth int) msg {
	for range depth {
		g = inSubgraph(g)
	}
	return g
}

const dataFile = "model.onnx.data"

// writeModel writes body as dir/model.onnx beside a 64-byte data file and
// returns the model's path.
func writeModel(t *testing.T, body []byte) string {
	t.Helper()

	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, dataFile), make([]byte, 64), 0o600); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "model.onnx")
	if err := os.WriteFile(path, body, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestRead pins what counts as a graph file safe to hand to ONNX Runtime:
// well-formed protobuf, an IR version and default opset the pinned runtime
// supports, and every tensor's external data inside a regular file beside
// the graph. Everything else is ErrIncompatibleCheckpoint naming the problem.
func TestRead(t *testing.T) {
	ok := exportIO()
	ext := func(kv ...string) msg { return ok.bytes(5, external("w", []uint64{4, 2}, kv...)) }
	off := func(offset, length int) msg {
		return ext("location", dataFile, "offset", strconv.Itoa(offset), "length", strconv.Itoa(length))
	}

	tests := []struct {
		name string
		body []byte
		want []string // substrings of the error; nil means no error
	}{
		{name: "export", body: model(10, 18, ok)},
		{name: "oldest ir", body: model(minIRVersion, 18, ok)},
		{name: "newest ir and opset", body: model(maxIRVersion, maxOpset, ok)},
		{name: "external whole file", body: model(10, 18, off(0, 64))},
		{name: "external at an offset", body: model(10, 18, off(32, 32))},
		{name: "external sized from dims", body: model(10, 18, ext("location", dataFile, "offset", "32"))},
		{name: "external in a subgraph", body: model(10, 18, slices.Concat(ok, inSubgraph(off(0, 32))))},
		{name: "ai.onnx domain", body: msg{}.varint(1, 10).bytes(7, ok).bytes(8, opset("ai.onnx", 18))},
		{name: "other domain beside", body: model(10, 18, ok).bytes(8, opset("com.microsoft", 99))},
		{name: "subgraphs at the limit", body: model(10, 18, slices.Concat(ok, nested(off(0, 8), maxDepth-1)))},

		{name: "empty", body: nil, want: []string{"no graph"}},
		{name: "no graph", body: model(10, 18, nil), want: []string{"no graph"}},
		{name: "truncated varint", body: []byte{0x08, 0x80}, want: []string{"malformed"}},
		{name: "length past end", body: []byte{0x3a, 0x10, 0x00}, want: []string{"malformed"}},
		{name: "bad wire type", body: []byte{0x0f}, want: []string{"malformed"}},
		{name: "field zero", body: []byte{0x00, 0x00}, want: []string{"malformed"}},
		{name: "ir too old", body: model(minIRVersion-1, 18, ok), want: []string{"IR version 2"}},
		{name: "ir too new", body: model(maxIRVersion+1, 18, ok), want: []string{"IR version 12"}},
		{name: "opset too new", body: model(10, maxOpset+1, ok), want: []string{"opset 24"}},
		{
			name: "no default opset", body: msg{}.varint(1, 10).bytes(7, ok).bytes(8, opset("com.microsoft", 1)),
			want: []string{"no ai.onnx opset"},
		},
		{name: "absolute location", body: model(10, 18, ext("location", "/etc/passwd")), want: []string{`"/etc/passwd"`}},
		{name: "parent location", body: model(10, 18, ext("location", "../"+dataFile)), want: []string{`"../model.onnx.data"`}},
		{name: "no location", body: model(10, 18, ext("offset", "0", "length", "32")), want: []string{"no location"}},
		{name: "missing file", body: model(10, 18, ext("location", "absent.data")), want: []string{"absent.data"}},
		{name: "past the end", body: model(10, 18, off(40, 32)), want: []string{`"w"`, "past the end"}},
		{name: "sized past the end", body: model(10, 18, ext("location", dataFile, "offset", "40")), want: []string{"past the end"}},
		{
			name: "overflowing range", body: model(10, 18, off(1<<62, 1<<62)),
			want: []string{"past the end"},
		},
		{name: "negative offset", body: model(10, 18, off(-1, 8)), want: []string{"offset"}},
		{name: "unknown key", body: model(10, 18, ext("location", dataFile, "sha", "x")), want: []string{`"sha"`}},
		{
			name: "subgraph escapes", body: model(10, 18, slices.Concat(ok, inSubgraph(ok.bytes(5, external("w", []uint64{8}, "location", "../x"))))),
			want: []string{`"../x"`},
		},
		{
			name: "function escapes", body: model(10, 18, ok).bytes(25, msg{}.bytes(7, msg{}.bytes(5,
				msg{}.str(1, "value").bytes(5, external("c", []uint64{8}, "location", "/x"))))),
			want: []string{`"/x"`},
		},
		{
			name: "function default attribute escapes", body: model(10, 18, ok).bytes(25, msg{}.bytes(11,
				msg{}.str(1, "value").bytes(5, external("c", []uint64{8}, "location", "/z")))),
			want: []string{`"/z"`},
		},
		{
			name: "sparse initializer escapes", body: model(10, 18, ok.bytes(15, msg{}.bytes(1, external("s", []uint64{8}, "location", "/y")))),
			want: []string{`"/y"`},
		},
		{
			name: "subgraphs past the limit", body: model(10, 18, slices.Concat(ok, nested(ok, maxDepth))),
			want: []string{"nested deeper than"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := writeModel(t, tt.body)
			h, err := Read(path)
			if tt.want == nil {
				if err != nil {
					t.Fatalf("Read = %v, want nil", err)
				}
				if h == nil {
					t.Fatal("Read returned a nil *Header and no error")
				}
				return
			}
			if !errors.Is(err, backend.ErrIncompatibleCheckpoint) {
				t.Fatalf("Read = %v, want ErrIncompatibleCheckpoint", err)
			}
			for _, w := range append([]string{path}, tt.want...) {
				if !strings.Contains(err.Error(), w) {
					t.Errorf("error %q does not mention %q", err, w)
				}
			}
		})
	}
}

// Header carries the graph's IO so 6.4.5 can check act_logits' static width
// without parsing the file again.
func TestReadHeader(t *testing.T) {
	h, err := Read(writeModel(t, model(10, 18, exportIO())))
	if err != nil {
		t.Fatal(err)
	}
	if h.IRVersion != 10 || h.Opset != 18 {
		t.Errorf("IRVersion, Opset = %d, %d; want 10, 18", h.IRVersion, h.Opset)
	}
	checkExportIO(t, h)
}

// checkExportIO asserts the IO scripts/export_onnx.py writes.
func checkExportIO(t *testing.T, h *Header) {
	t.Helper()

	sym := func(p string) Dim { return Dim{Value: -1, Param: p} }
	want := map[string]Tensor{
		"input_ids":      {Name: "input_ids", ElemType: elemInt64, Dims: []Dim{sym("batch"), sym("seq")}},
		"attention_mask": {Name: "attention_mask", ElemType: elemInt64, Dims: []Dim{sym("batch"), sym("seq")}},
		"marker_pos":     {Name: "marker_pos", ElemType: elemInt64, Dims: []Dim{sym("batch"), sym("k")}},
		"marker_mask":    {Name: "marker_mask", ElemType: elemBool, Dims: []Dim{sym("batch"), sym("k")}},
		"qtype":          {Name: "qtype", ElemType: elemInt64, Dims: []Dim{sym("batch")}},
		"logits":         {Name: "logits", ElemType: elemFloat, Dims: []Dim{sym("batch"), sym("k")}},
		"act_logits":     {Name: "act_logits", ElemType: elemFloat, Dims: []Dim{sym("batch"), {Value: 2}}},
	}
	got := slices.Concat(h.Inputs, h.Outputs)
	if len(got) != len(want) || len(h.Outputs) != 2 {
		t.Fatalf("Inputs, Outputs = %v, %v; want 5 and 2", h.Inputs, h.Outputs)
	}
	for _, g := range got {
		w := want[g.Name]
		if g.Name != w.Name || g.ElemType != w.ElemType || !slices.Equal(g.Dims, w.Dims) {
			t.Errorf("tensor %+v, want %+v", g, w)
		}
	}
}

// A graph file must be a regular file of sane size, and so must its data.
func TestReadFile(t *testing.T) {
	t.Run("missing", func(t *testing.T) {
		if _, err := Read(filepath.Join(t.TempDir(), "absent.onnx")); !errors.Is(err, backend.ErrIncompatibleCheckpoint) {
			t.Errorf("Read = %v, want ErrIncompatibleCheckpoint", err)
		}
	})
	t.Run("oversized", func(t *testing.T) {
		path := writeModel(t, nil)
		if err := os.Truncate(path, maxFileSize+1); err != nil {
			t.Fatal(err)
		}
		_, err := Read(path)
		if !errors.Is(err, backend.ErrIncompatibleCheckpoint) || !strings.Contains(err.Error(), "larger than") {
			t.Errorf("Read = %v, want ErrIncompatibleCheckpoint naming the size", err)
		}
	})
	t.Run("symlinked data", func(t *testing.T) {
		path := writeModel(t, model(10, 18, exportIO().bytes(5, external("w", []uint64{2}, "location", "link.data"))))
		if err := os.Symlink(filepath.Join(filepath.Dir(path), dataFile), filepath.Join(filepath.Dir(path), "link.data")); err != nil {
			t.Skip(err)
		}
		_, err := Read(path)
		if !errors.Is(err, backend.ErrIncompatibleCheckpoint) || !strings.Contains(err.Error(), "not a regular file") {
			t.Errorf("Read = %v, want ErrIncompatibleCheckpoint for a symlink", err)
		}
	})
	t.Run("symlinked directory", func(t *testing.T) {
		path := writeModel(t, model(10, 18, exportIO().bytes(5, external("w", []uint64{2}, "location", "linked/"+dataFile))))
		if err := os.Symlink(filepath.Dir(path), filepath.Join(filepath.Dir(path), "linked")); err != nil {
			t.Skip(err)
		}
		_, err := Read(path)
		if !errors.Is(err, backend.ErrIncompatibleCheckpoint) || !strings.Contains(err.Error(), "not a directory") {
			t.Errorf("Read = %v, want ErrIncompatibleCheckpoint for a symlinked parent", err)
		}
	})
	t.Run("real subdirectory", func(t *testing.T) {
		path := writeModel(t, model(10, 18, exportIO().bytes(5, external("w", []uint64{2}, "location", "sub/"+dataFile))))
		sub := filepath.Join(filepath.Dir(path), "sub")
		if err := os.Mkdir(sub, 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(sub, dataFile), make([]byte, 8), 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := Read(path); err != nil {
			t.Errorf("Read = %v, want nil", err)
		}
	})
	t.Run("data is a directory", func(t *testing.T) {
		path := writeModel(t, model(10, 18, exportIO().bytes(5, external("w", []uint64{2}, "location", "sub"))))
		if err := os.Mkdir(filepath.Join(filepath.Dir(path), "sub"), 0o700); err != nil {
			t.Fatal(err)
		}
		_, err := Read(path)
		if !errors.Is(err, backend.ErrIncompatibleCheckpoint) || !strings.Contains(err.Error(), "not a regular file") {
			t.Errorf("Read = %v, want ErrIncompatibleCheckpoint for a directory", err)
		}
	})
}

// The three dynamo exports pass, and their IO is what export_onnx.py writes.
func TestReadExports(t *testing.T) {
	dir := os.Getenv("LAYA_ONNX_DIR")
	if testing.Short() || dir == "" {
		t.Skip("set LAYA_ONNX_DIR to the exports (just test-onnx does)")
	}
	for _, name := range []string{"english", "multilingual", "typed-decisions"} {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(dir, "laya-"+name+"-dynamo.onnx")
			if _, err := os.Stat(path); err != nil {
				t.Skip(err)
			}
			h, err := Read(path)
			if err != nil {
				t.Fatal(err)
			}
			checkExportIO(t, h)
		})
	}
}

// Read is a parser for downloaded bytes: whatever they are, it returns
// rather than panics.
func FuzzRead(f *testing.F) {
	f.Add([]byte(model(10, 18, exportIO())))
	f.Add([]byte(model(10, 18, exportIO().bytes(5, external("w", []uint64{2}, "location", dataFile)))))
	f.Add([]byte(model(10, 18, nested(exportIO(), 3))))
	f.Add([]byte{0x3a, 0x10, 0x00})
	dir := f.TempDir()
	if err := os.WriteFile(filepath.Join(dir, dataFile), make([]byte, 64), 0o600); err != nil {
		f.Fatal(err)
	}
	path := filepath.Join(dir, "model.onnx")
	f.Fuzz(func(t *testing.T, body []byte) {
		if err := os.WriteFile(path, body, 0o600); err != nil {
			t.Fatal(err)
		}
		if h, err := Read(path); (h == nil) == (err == nil) {
			t.Fatalf("Read = %v, %v; want exactly one of them", h, err)
		}
	})
}
