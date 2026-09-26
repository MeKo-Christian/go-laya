package safetensors

import (
	"encoding/binary"
	"errors"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/MeKo-Christian/go-laya/backend"
	"github.com/MeKo-Christian/go-laya/internal/golden"
)

// encode frames header and data bytes of data as a .safetensors file.
func encode(header string, data int) []byte {
	b := binary.LittleEndian.AppendUint64(nil, uint64(len(header)))
	b = append(b, header...)
	return append(b, make([]byte, data)...)
}

func write(t testing.TB, body []byte) string {
	t.Helper()

	path := filepath.Join(t.TempDir(), "model.safetensors")
	if err := os.WriteFile(path, body, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestReadHeader pins safetensors' own validation rules: a bounded JSON
// object header, known dtypes, non-negative shapes, and tensors that tile the
// data buffer exactly, each the size its dtype and shape say. Anything else is
// ErrIncompatibleCheckpoint naming the tensor.
func TestReadHeader(t *testing.T) {
	two := `{"a":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},` +
		`"b":{"dtype":"I64","shape":[1],"data_offsets":[24,32]}}`

	tests := []struct {
		name string
		body []byte
		want []string // substrings of the error; nil means no error
	}{
		{name: "two tensors", body: encode(two, 32)},
		{name: "out of order", body: encode(`{"b":{"dtype":"U8","shape":[2],"data_offsets":[4,6]},"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}`, 6)},
		{name: "metadata", body: encode(`{"__metadata__":{"format":"pt"},"a":{"dtype":"BOOL","shape":[3],"data_offsets":[0,3]}}`, 3)},
		{name: "scalar", body: encode(`{"a":{"dtype":"F64","shape":[],"data_offsets":[0,8]}}`, 8)},
		{name: "empty tensor", body: encode(`{"a":{"dtype":"F32","shape":[0,4],"data_offsets":[0,0]}}`, 0)},
		{name: "no tensors", body: encode(`{}`, 0)},
		{name: "padded header", body: encode(`{"a":{"dtype":"I8","shape":[1],"data_offsets":[0,1]}}   `, 1)},

		{name: "too short", body: []byte{1, 0, 0}, want: []string{"shorter than"}},
		{
			name: "header over the cap", body: binary.LittleEndian.AppendUint64(nil, maxHeaderSize+1),
			want: []string{"larger than"},
		},
		{name: "header past the end", body: binary.LittleEndian.AppendUint64(nil, 10), want: []string{"past the end"}},
		{name: "not an object", body: encode(`[]`, 0), want: []string{"not a JSON object"}},
		{name: "invalid JSON", body: encode(`{"a":`, 0), want: []string{"header"}},
		{name: "unknown dtype", body: encode(`{"a":{"dtype":"F128","shape":[1],"data_offsets":[0,16]}}`, 16), want: []string{`"a"`, `"F128"`}},
		{name: "no dtype", body: encode(`{"a":{"shape":[1],"data_offsets":[0,4]}}`, 4), want: []string{`"a"`, "dtype"}},
		{name: "no offsets", body: encode(`{"a":{"dtype":"F32","shape":[1]}}`, 4), want: []string{`"a"`, "data_offsets"}},
		{name: "three offsets", body: encode(`{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,2,4]}}`, 4), want: []string{`"a"`, "data_offsets"}},
		{name: "negative dim", body: encode(`{"a":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}}`, 4), want: []string{`"a"`, "negative dim"}},
		{name: "size mismatch", body: encode(`{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}`, 4), want: []string{`"a"`, "8 bytes"}},
		{name: "reversed offsets", body: encode(`{"a":{"dtype":"U8","shape":[0],"data_offsets":[4,0]}}`, 4), want: []string{`"a"`}},
		{
			name: "gap", body: encode(`{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"b":{"dtype":"U8","shape":[2],"data_offsets":[4,6]}}`, 6),
			want: []string{`"b"`, "starts at 4"},
		},
		{
			name: "overlap", body: encode(`{"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"b":{"dtype":"U8","shape":[2],"data_offsets":[2,4]}}`, 4),
			want: []string{`"b"`, "starts at 2"},
		},
		{name: "does not start at 0", body: encode(`{"a":{"dtype":"U8","shape":[2],"data_offsets":[2,4]}}`, 4), want: []string{`"a"`, "starts at 2"}},
		{name: "trailing byte", body: encode(`{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}`, 3), want: []string{"2 of 3"}},
		{name: "short data", body: encode(`{"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}`, 2), want: []string{"4 of 2"}},
		{
			name: "element count overflows", body: encode(`{"a":{"dtype":"F32","shape":[4294967296,4294967296],"data_offsets":[0,0]}}`, 0),
			want: []string{`"a"`, "overflows"},
		},
		{name: "metadata not strings", body: encode(`{"__metadata__":{"n":1}}`, 0), want: []string{"__metadata__"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := write(t, tt.body)
			h, err := ReadHeader(path)
			if tt.want == nil {
				if err != nil {
					t.Fatalf("ReadHeader = %v, want nil", err)
				}
				if h == nil {
					t.Fatal("ReadHeader returned a nil *Header and no error")
				}
				return
			}
			if !errors.Is(err, backend.ErrIncompatibleCheckpoint) {
				t.Fatalf("ReadHeader = %v, want ErrIncompatibleCheckpoint", err)
			}
			for _, w := range append([]string{path}, tt.want...) {
				if !strings.Contains(err.Error(), w) {
					t.Errorf("error %q does not mention %q", err, w)
				}
			}
		})
	}
}

// The Header reports what the file declares.
func TestReadHeaderContents(t *testing.T) {
	h, err := ReadHeader(write(t, encode(`{"__metadata__":{"format":"pt"},`+
		`"b":{"dtype":"I64","shape":[1],"data_offsets":[24,32]},`+
		`"a":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}`, 32)))
	if err != nil {
		t.Fatal(err)
	}
	if !maps.Equal(h.Metadata, map[string]string{"format": "pt"}) {
		t.Errorf("Metadata = %v", h.Metadata)
	}
	a, b := h.Tensors["a"], h.Tensors["b"]
	if len(h.Tensors) != 2 || a.DType != "F32" || !slices.Equal(a.Shape, []int64{2, 3}) || a.Begin != 0 || a.End != 24 ||
		b.DType != "I64" || !slices.Equal(b.Shape, []int64{1}) || b.Begin != 24 || b.End != 32 {
		t.Errorf("Tensors = %+v", h.Tensors)
	}
	if h.DataOffset != 8+int64(len(`{"__metadata__":{"format":"pt"},`+
		`"b":{"dtype":"I64","shape":[1],"data_offsets":[24,32]},`+
		`"a":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}`)) {
		t.Errorf("DataOffset = %d", h.DataOffset)
	}
}

func TestReadHeaderMissingFile(t *testing.T) {
	if _, err := ReadHeader(filepath.Join(t.TempDir(), "absent.safetensors")); !errors.Is(err, backend.ErrIncompatibleCheckpoint) {
		t.Errorf("ReadHeader = %v, want ErrIncompatibleCheckpoint", err)
	}
}

// The three shipped checkpoints' weights pass.
func TestReadHeaderShipped(t *testing.T) {
	root := golden.SkipWithoutModels(t)
	for _, ck := range []string{golden.English, golden.Multilingual, golden.TypedDecisions} {
		t.Run(ck, func(t *testing.T) {
			h, err := ReadHeader(filepath.Join(golden.CheckpointDir(root, ck), "model.safetensors"))
			if err != nil {
				t.Fatal(err)
			}
			if len(h.Tensors) == 0 {
				t.Error("no tensors")
			}
		})
	}
}

// ReadHeader parses downloaded bytes: whatever they are, it returns rather
// than panics.
func FuzzReadHeader(f *testing.F) {
	f.Add(encode(`{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}`, 8))
	f.Add(encode(`{"__metadata__":{"k":"v"}}`, 0))
	f.Add([]byte{1, 0, 0, 0, 0, 0, 0, 0, '{'})
	path := filepath.Join(f.TempDir(), "model.safetensors")
	f.Fuzz(func(t *testing.T, body []byte) {
		if err := os.WriteFile(path, body, 0o600); err != nil {
			t.Fatal(err)
		}
		if h, err := ReadHeader(path); (h == nil) == (err == nil) {
			t.Fatalf("ReadHeader = %v, %v; want exactly one of them", h, err)
		}
	})
}
