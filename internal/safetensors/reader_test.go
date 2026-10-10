package safetensors

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/MeKo-Christian/go-laya/backend"
)

// spec is one tensor of a synthetic file: its header entry and raw bytes.
type spec struct {
	name, dtype string
	shape       []int64
	data        []byte
}

// build frames specs as a .safetensors file, laying their data out in the
// order given.
func build(t testing.TB, specs ...spec) []byte {
	t.Helper()

	header := make(map[string]any, len(specs))
	var data []byte
	for _, s := range specs {
		header[s.name] = map[string]any{
			"dtype": s.dtype, "shape": s.shape,
			"data_offsets": []int{len(data), len(data) + len(s.data)},
		}
		data = append(data, s.data...)
	}
	raw, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	b := binary.LittleEndian.AppendUint64(nil, uint64(len(raw)))
	b = append(b, raw...)
	return append(b, data...)
}

func f16Bytes(hs ...uint16) []byte {
	var b []byte
	for _, h := range hs {
		b = binary.LittleEndian.AppendUint16(b, h)
	}
	return b
}

func f32Bytes(fs ...float32) []byte {
	var b []byte
	for _, f := range fs {
		b = binary.LittleEndian.AppendUint32(b, math.Float32bits(f))
	}
	return b
}

func u32Bytes(us ...uint32) []byte {
	var b []byte
	for _, u := range us {
		b = binary.LittleEndian.AppendUint32(b, u)
	}
	return b
}

func fromBits(us ...uint32) []float32 {
	fs := make([]float32, len(us))
	for i, u := range us {
		fs[i] = math.Float32frombits(u)
	}
	return fs
}

func open(t testing.TB, body []byte) *File {
	t.Helper()

	f, err := Open(write(t, body))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.Close() })
	return f
}

// equalBits compares float32 slices bit for bit, so -0 and 0 differ.
func equalBits(a, b []float32) bool {
	return slices.EqualFunc(a, b, func(x, y float32) bool { return math.Float32bits(x) == math.Float32bits(y) })
}

func wantIncompatible(t *testing.T, err error, mentions ...string) {
	t.Helper()

	if !errors.Is(err, backend.ErrIncompatibleCheckpoint) {
		t.Fatalf("err = %v, want ErrIncompatibleCheckpoint", err)
	}
	for _, m := range mentions {
		if !strings.Contains(err.Error(), m) {
			t.Errorf("error %q does not mention %q", err, m)
		}
	}
}

// F16 data decodes and F32 data passes through, both read little-endian from
// the tensor's own span of the data buffer, which starts after the header.
func TestFloat32Decodes(t *testing.T) {
	f := open(t, build(
		t,
		spec{"b", "F32", []int64{3}, f32Bytes(1.5, -3.25, 1e-30)},
		spec{"w", "F16", []int64{2, 3}, f16Bytes(0x3c00, 0xc000, 0x3800, 0x7bff, 0x0001, 0x8000)},
		spec{"s", "F16", []int64{}, f16Bytes(0x3e00)},
		spec{"e", "F16", []int64{2, 0}, nil},
		spec{"nan16", "F16", []int64{2}, f16Bytes(0x7e01, 0xfd00)},
		spec{"nan32", "F32", []int64{3}, u32Bytes(0x7fa00001, 0xffc00000, 0x80000000)},
	))

	tests := []struct {
		name  string
		shape []int64
		want  []float32
	}{
		{"b", []int64{3}, []float32{1.5, -3.25, 1e-30}},
		{"w", []int64{2, 3}, []float32{1, -2, 0.5, 65504, 0x1p-24, float32(math.Copysign(0, -1))}},
		{"s", []int64{}, []float32{1.5}},
		{"e", []int64{2, 0}, []float32{}},
		// NaNs keep sign and payload, a signalling one included; F32 bits
		// pass through untouched.
		{"nan16", []int64{2}, fromBits(0x7fc02000, 0xffa00000)},
		{"nan32", []int64{3}, fromBits(0x7fa00001, 0xffc00000, 0x80000000)},
	}
	for _, tt := range tests {
		data, shape, err := f.Float32(t.Context(), tt.name)
		if err != nil {
			t.Fatalf("Float32(%q) = %v", tt.name, err)
		}
		if data == nil || !equalBits(data, tt.want) {
			t.Errorf("Float32(%q) = %v, want %v", tt.name, data, tt.want)
		}
		if !slices.Equal(shape, tt.shape) {
			t.Errorf("Float32(%q) shape = %v, want %v", tt.name, shape, tt.shape)
		}
	}
}

// The returned shape is the caller's to keep: changing it does not change the
// file's view of the tensor.
func TestFloat32ShapeIsACopy(t *testing.T) {
	f := open(t, build(t, spec{"w", "F32", []int64{2}, f32Bytes(1, 2)}))
	_, shape, err := f.Float32(t.Context(), "w")
	if err != nil {
		t.Fatal(err)
	}
	shape[0] = 99
	if info, _ := f.Info("w"); info.Shape[0] != 2 {
		t.Errorf("Info shape = %v after the caller edited its copy", info.Shape)
	}
}

// Names lists every tensor in file order, temperature included; Info reports
// a tensor's header entry.
func TestNamesAndInfo(t *testing.T) {
	f := open(t, build(
		t,
		spec{"z", "F32", []int64{1}, f32Bytes(1)},
		spec{"temperature", "F32", []int64{3}, f32Bytes(1, 1, 1)},
		spec{"a", "F16", []int64{2}, f16Bytes(0, 0)},
	))
	if got, want := f.Names(), []string{"z", "temperature", "a"}; !slices.Equal(got, want) {
		t.Errorf("Names = %v, want %v", got, want)
	}
	info, ok := f.Info("a")
	if !ok || info.DType != "F16" || !slices.Equal(info.Shape, []int64{2}) || info.Begin != 16 || info.End != 20 {
		t.Errorf("Info(a) = %+v, %v", info, ok)
	}
	if _, ok := f.Info("absent"); ok {
		t.Error("Info(absent) reported a tensor")
	}
}

// laya never reads temperature at inference, and the shipped checkpoints
// store it as F32, F32 and F16, so Open tolerates it in any dtype and every
// other tensor still loads. Reading it by name decodes F16/F32 like any tensor
// and is an error in any other dtype.
func TestTemperatureAnyDType(t *testing.T) {
	tests := []struct {
		dtype string
		data  []byte
		want  []float32 // nil: reading temperature is an error
	}{
		{"F32", f32Bytes(1, 0.5, 2), []float32{1, 0.5, 2}},
		{"F16", f16Bytes(0x3c00, 0x3800, 0x4000), []float32{1, 0.5, 2}},
		{"BF16", f16Bytes(0x3f80, 0x3f00, 0x4000), nil},
		{"I64", make([]byte, 24), nil},
	}
	for _, tt := range tests {
		t.Run(tt.dtype, func(t *testing.T) {
			f := open(t, build(
				t,
				spec{"w", "F16", []int64{2}, f16Bytes(0x3c00, 0xc000)},
				spec{"temperature", tt.dtype, []int64{3}, tt.data},
				spec{"b", "F32", []int64{1}, f32Bytes(7)},
			))
			for _, name := range f.Names() {
				if name == "temperature" {
					continue
				}
				if _, _, err := f.Float32(t.Context(), name); err != nil {
					t.Errorf("Float32(%q) = %v", name, err)
				}
			}

			data, _, err := f.Float32(t.Context(), "temperature")
			if tt.want == nil {
				wantIncompatible(t, err, `"temperature"`, tt.dtype)
				return
			}
			if err != nil {
				t.Fatalf("Float32(temperature) = %v", err)
			}
			if !equalBits(data, tt.want) {
				t.Errorf("Float32(temperature) = %v, want %v", data, tt.want)
			}
		})
	}
}

// Any tensor other than temperature must be F16 or F32, and Open says which
// one is not. The exemption is for the exact name only.
func TestOpenRejectsUndecodableDType(t *testing.T) {
	for _, s := range []spec{
		{"encoder.w", "BF16", []int64{2}, f16Bytes(0x3f80, 0)},
		{"encoder.w", "F64", []int64{1}, make([]byte, 8)},
		{"ids", "I64", []int64{1}, make([]byte, 8)},
		{"head.temperature", "BF16", []int64{1}, f16Bytes(0x3f80)},
	} {
		t.Run(s.name+"/"+s.dtype, func(t *testing.T) {
			path := write(t, build(t, spec{"ok", "F16", []int64{1}, f16Bytes(0x3c00)}, s))
			f, err := Open(path)
			if f != nil {
				t.Error("Open returned a *File with its error")
			}
			wantIncompatible(t, err, path, `"`+s.name+`"`, s.dtype)
		})
	}
}

// A file whose header does not describe its bytes is refused by ReadHeader's
// rules before any tensor data is decoded.
func TestOpenRejectsTamperedFile(t *testing.T) {
	good := build(t, spec{"w", "F16", []int64{2}, f16Bytes(0x3c00, 0x4000)})
	tests := []struct {
		name string
		body []byte
		want string
	}{
		{"truncated", good[:len(good)-1], "4 of 3"},
		{"extended", append(slices.Clone(good), 0), "4 of 5"},
		{"offsets past the end", encode(`{"w":{"dtype":"F16","shape":[2],"data_offsets":[2,6]}}`, 4), "starts at 2"},
		{"shape larger than span", encode(`{"w":{"dtype":"F16","shape":[4],"data_offsets":[0,4]}}`, 4), "8 bytes"},
		{"header length past the end", append(binary.LittleEndian.AppendUint64(nil, 1<<20), '{', '}'), "past the end"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := write(t, tt.body)
			f, err := Open(path)
			if f != nil {
				t.Error("Open returned a *File with its error")
			}
			wantIncompatible(t, err, path, tt.want)
		})
	}
}

// A path that is not a regular file is refused before it is opened, so a
// named pipe cannot block Open.
func TestOpenNotARegularFile(t *testing.T) {
	dir := t.TempDir()
	_, err := Open(dir)
	wantIncompatible(t, err, dir, "not a regular file")
}

// On a 32-bit platform a tensor whose float32 copy would not fit in the
// address space is an error, not a makeslice panic. The file is sparse.
func TestFloat32TooLargeFor32Bit(t *testing.T) {
	if math.MaxInt > math.MaxInt32 {
		t.Skip("64-bit: the tensor would be allocated")
	}
	const n = 1 << 30 // 2 GiB of F16, 4 GiB as float32
	path := write(t, encode(`{"w":{"dtype":"F16","shape":[1073741824],"data_offsets":[0,2147483648]}}`, 0))
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Truncate(path, fi.Size()+2*n); err != nil {
		t.Fatal(err)
	}
	f, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	_, _, err = f.Float32(t.Context(), "w")
	wantIncompatible(t, err, `"w"`, "address space")
}

func TestOpenMissingFile(t *testing.T) {
	_, err := Open(t.TempDir() + "/absent.safetensors")
	wantIncompatible(t, err, "absent.safetensors")
}

func TestFloat32UnknownTensor(t *testing.T) {
	f := open(t, build(t, spec{"w", "F16", []int64{1}, f16Bytes(0)}))
	_, _, err := f.Float32(t.Context(), "v")
	wantIncompatible(t, err, `"v"`)
}

// A file cut short after Open is an error, not zeros or a crash: reads go
// through the descriptor and stop at its end.
func TestFloat32FileShrunkAfterOpen(t *testing.T) {
	body := build(t, spec{"a", "F32", []int64{1}, f32Bytes(1)}, spec{"w", "F16", []int64{4}, f16Bytes(1, 2, 3, 4)})
	path := write(t, body)
	f, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := os.Truncate(path, int64(len(body)-2)); err != nil {
		t.Fatal(err)
	}

	if _, _, err := f.Float32(t.Context(), "a"); err != nil {
		t.Errorf("Float32(a), still inside the file, = %v", err)
	}
	data, _, err := f.Float32(t.Context(), "w")
	if data != nil {
		t.Errorf("Float32(w) returned data %v with its error", data)
	}
	wantIncompatible(t, err, `"w"`)
}

func TestFloat32AfterClose(t *testing.T) {
	f, err := Open(write(t, build(t, spec{"w", "F16", []int64{1}, f16Bytes(0)})))
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	if _, _, err := f.Float32(t.Context(), "w"); err == nil {
		t.Error("Float32 after Close succeeded")
	}
}

// bigTensor spans several read chunks, with an odd tail, so the chunk
// boundaries are exercised.
func bigTensor(dtype string) (spec, []float32) {
	width := 2
	if dtype == "F32" {
		width = 4
	}
	n := (2*chunkBytes + 6) / width
	want := make([]float32, n)
	var data []byte
	for i := range n {
		if dtype == "F32" {
			want[i] = float32(i) * 0.25
			data = binary.LittleEndian.AppendUint32(data, math.Float32bits(want[i]))
			continue
		}
		h := uint16(i*37) & 0x7bff // finite, both signs never needed here
		want[i] = f16ToF32(h)
		data = binary.LittleEndian.AppendUint16(data, h)
	}
	return spec{"big", dtype, []int64{int64(n)}, data}, want
}

func TestFloat32AcrossChunks(t *testing.T) {
	for _, dtype := range []string{"F16", "F32"} {
		t.Run(dtype, func(t *testing.T) {
			s, want := bigTensor(dtype)
			f := open(t, build(t, spec{"pre", "F16", []int64{3}, f16Bytes(1, 2, 3)}, s))
			data, _, err := f.Float32(t.Context(), "big")
			if err != nil {
				t.Fatal(err)
			}
			if !equalBits(data, want) {
				t.Error("Float32(big) differs from the data written")
			}
		})
	}
}

// cancelAfter is a context that reports cancellation once Err has been asked
// more than n times, so a test can cancel a read part-way through
// deterministically.
type cancelAfter struct{ n, calls int }

func (*cancelAfter) Deadline() (time.Time, bool) { return time.Time{}, false }
func (*cancelAfter) Done() <-chan struct{}       { return nil }
func (*cancelAfter) Value(any) any               { return nil }

func (c *cancelAfter) Err() error {
	c.calls++
	if c.calls > c.n {
		return context.Canceled
	}
	return nil
}

// A cancelled context stops a read before it starts and inside a large
// tensor, and the error is the context's, not the checkpoint's.
func TestFloat32Cancel(t *testing.T) {
	s, _ := bigTensor("F16")
	f := open(t, build(t, s))

	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, _, err := f.Float32(ctx, "big"); !errors.Is(err, context.Canceled) || errors.Is(err, backend.ErrIncompatibleCheckpoint) {
		t.Errorf("Float32 on a cancelled context = %v, want context.Canceled alone", err)
	}

	// The context is checked on entry and before each of the tensor's three
	// chunks; allowing one check fewer cancels the read before its last
	// chunk.
	probe := &cancelAfter{n: math.MaxInt}
	if _, _, err := f.Float32(probe, "big"); err != nil {
		t.Fatal(err)
	}
	if probe.calls != 1+3 {
		t.Fatalf("a read of 3 chunks checked the context %d times, want 4", probe.calls)
	}
	data, _, err := f.Float32(&cancelAfter{n: probe.calls - 1}, "big")
	if !errors.Is(err, context.Canceled) || data != nil {
		t.Errorf("Float32 cancelled before its last chunk = %d values, %v, want nil, context.Canceled", len(data), err)
	}
}

// Open reads downloaded bytes: whatever they are, it and every read after it
// return rather than panic.
func FuzzOpen(f *testing.F) {
	f.Add(build(f, spec{"w", "F16", []int64{2}, f16Bytes(0x3c00, 0x7e00)}))
	f.Add(build(f, spec{"temperature", "BF16", []int64{1}, f16Bytes(0)}, spec{"b", "F32", []int64{1}, f32Bytes(1)}))
	f.Add([]byte{1, 0, 0, 0, 0, 0, 0, 0, '{'})
	path := f.TempDir() + "/model.safetensors"
	f.Fuzz(func(t *testing.T, body []byte) {
		if err := os.WriteFile(path, body, 0o600); err != nil {
			t.Fatal(err)
		}
		st, err := Open(path)
		if err != nil {
			return
		}
		defer st.Close()
		for _, name := range st.Names() {
			_, _, _ = st.Float32(t.Context(), name)
		}
	})
}
