package evalset

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"google.golang.org/protobuf/encoding/protowire"
)

// stampedGraph writes a ModelProto to path that carries only what
// int8Source reads: an ir_version and, unless source is "", the laya.int8
// stamp scripts/export_onnx.py --int8 puts into metadata_props (field 14).
func stampedGraph(t *testing.T, path, source string) {
	t.Helper()
	b := protowire.AppendTag(nil, 1, protowire.VarintType)
	b = protowire.AppendVarint(b, 10)
	entry := func(key, value string) []byte {
		e := protowire.AppendTag(nil, 1, protowire.BytesType)
		e = protowire.AppendString(e, key)
		e = protowire.AppendTag(e, 2, protowire.BytesType)
		return protowire.AppendString(e, value)
	}
	b = protowire.AppendTag(b, 14, protowire.BytesType)
	b = protowire.AppendBytes(b, entry("laya.attn", "sdpa"))
	if source != "" {
		stamp, err := json.Marshal(map[string]any{"method": "quantize_dynamic", "source": source})
		if err != nil {
			t.Fatal(err)
		}
		b = protowire.AppendTag(b, 14, protowire.BytesType)
		b = protowire.AppendBytes(b, entry("laya.int8", string(stamp)))
	}
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatal(err)
	}
}

// exportDir is an fp32 export directory holding both a canonical and a
// -dynamo export of english, each with its external data, as build/onnx can.
func exportDir(t *testing.T) string {
	t.Helper()
	dir := filepath.Join(t.TempDir(), "onnx")
	if err := os.Mkdir(dir, 0o750); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"laya-english.onnx", "laya-english-dynamo.onnx"} {
		for _, f := range []string{name, name + ".data"} {
			if err := os.WriteFile(filepath.Join(dir, f), []byte(f), 0o600); err != nil {
				t.Fatal(err)
			}
		}
	}
	return dir
}

// TestStageBaseline is the guard behind the fp32-vs-int8 deltas: the fp32
// side opens exactly the export the int8 graph was quantized from, as the
// graph's own stamp names it, even where a canonical export of another
// exporter sits beside it under the name the loader prefers.
func TestStageBaseline(t *testing.T) {
	fp32Dir := exportDir(t)
	int8Graph := filepath.Join(t.TempDir(), "laya-english.onnx")
	stampedGraph(t, int8Graph, "laya-english-dynamo.onnx")

	staged, src, err := stageBaseline(fp32Dir, int8Graph, "english", t.TempDir())
	if err != nil {
		t.Fatalf("stageBaseline: %v", err)
	}
	if want := filepath.Join(fp32Dir, "laya-english-dynamo.onnx"); src != want {
		t.Errorf("source = %s, want %s", src, want)
	}
	for name, from := range map[string]string{
		"laya-english.onnx":             "laya-english-dynamo.onnx",
		"laya-english-dynamo.onnx.data": "laya-english-dynamo.onnx.data",
	} {
		want, err := os.Stat(filepath.Join(fp32Dir, from))
		if err != nil {
			t.Fatal(err)
		}
		got, err := os.Stat(filepath.Join(staged, name))
		if err != nil || !os.SameFile(got, want) {
			t.Errorf("staged %s is not %s (%v)", name, from, err)
		}
	}
}

func TestStageBaselineRejects(t *testing.T) {
	tests := []struct {
		name, source, want string
	}{
		{"no stamp", "", "laya.int8"},
		{"source outside the directory", "../laya-english-dynamo.onnx", "plain name"},
		{"source of another checkpoint", "laya-multilingual-dynamo.onnx", "does not hold"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			int8Graph := filepath.Join(t.TempDir(), "laya-english.onnx")
			stampedGraph(t, int8Graph, tt.source)
			_, _, err := stageBaseline(exportDir(t), int8Graph, "english", t.TempDir())
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("stageBaseline: err = %v, want one naming %q", err, tt.want)
			}
		})
	}
}
