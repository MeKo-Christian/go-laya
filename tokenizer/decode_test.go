package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/golden"
)

// oracleModelJSON is modelJSON exactly as it was before Task B.1, when
// encoding/json decoded the vocabulary into a map and the merges into a slice of
// string pairs. It is the oracle the streaming decode is held to, and it lives
// here rather than in load.go because production must not keep a second path.
type oracleModelJSON struct {
	Type                    string           `json:"type"`
	Dropout                 *float64         `json:"dropout"`
	UnkToken                *string          `json:"unk_token"`
	ContinuingSubwordPrefix *string          `json:"continuing_subword_prefix"`
	EndOfWordSuffix         *string          `json:"end_of_word_suffix"`
	FuseUnk                 bool             `json:"fuse_unk"`
	ByteFallback            bool             `json:"byte_fallback"`
	IgnoreMerges            bool             `json:"ignore_merges"`
	Vocab                   map[string]int32 `json:"vocab"`
	Merges                  [][2]string      `json:"merges"`
}

// oracleBuildMerges is the pre-B.1 buildMerges, verbatim.
func oracleBuildMerges(pairs [][2]string, vocab map[string]int32) (map[[2]int32]merge, error) {
	out := make(map[[2]int32]merge, len(pairs))
	for rank, p := range pairs {
		left, ok := vocab[p[0]]
		if !ok {
			return nil, fmt.Errorf("%w: merge %d has %q on the left, which is not in the vocabulary",
				ErrUnsupported, rank, p[0])
		}
		right, ok := vocab[p[1]]
		if !ok {
			return nil, fmt.Errorf("%w: merge %d has %q on the right, which is not in the vocabulary",
				ErrUnsupported, rank, p[1])
		}
		joined, ok := vocab[p[0]+p[1]]
		if !ok {
			return nil, fmt.Errorf("%w: merge %d produces %q, which is not in the vocabulary",
				ErrUnsupported, rank, p[0]+p[1])
		}
		key := [2]int32{left, right}
		if _, dup := out[key]; dup {
			continue
		}
		out[key] = merge{rank: int32(rank), newID: joined}
	}
	return out, nil
}

// oracleVocabMerges is the old path end to end over one "model" object.
func oracleVocabMerges(rawModel []byte) (map[string]int32, map[[2]int32]merge, error) {
	var m oracleModelJSON
	if err := json.Unmarshal(rawModel, &m); err != nil {
		return nil, nil, err
	}
	merges, err := oracleBuildMerges(m.Merges, m.Vocab)
	if err != nil {
		return nil, nil, err
	}
	return m.Vocab, merges, nil
}

// streamVocabMerges is the new path over the same object, in the order OpenFS
// and build run it.
func streamVocabMerges(rawModel []byte) (map[string]int32, map[[2]int32]merge, error) {
	var m modelJSON
	if err := json.Unmarshal(rawModel, &m); err != nil {
		return nil, nil, err
	}
	vocab, err := decodeVocab(m.Vocab)
	if err != nil {
		return nil, nil, err
	}
	merges, err := decodeMerges(m.Merges, vocab)
	if err != nil {
		return nil, nil, err
	}
	return vocab, merges, nil
}

// sameTables fails unless both paths error, or both succeed with identical
// tables. reflect.DeepEqual rather than maps.Equal on purpose: it also tells a
// nil vocabulary from an empty one.
func sameTables(tb testing.TB, rawModel []byte) (map[string]int32, map[[2]int32]merge, error) {
	tb.Helper()

	wantVocab, wantMerges, wantErr := oracleVocabMerges(rawModel)
	gotVocab, gotMerges, gotErr := streamVocabMerges(rawModel)
	switch {
	case (wantErr == nil) != (gotErr == nil):
		tb.Fatalf("model %q:\nencoding/json err = %v\nstreaming err     = %v", rawModel, wantErr, gotErr)
	case wantErr != nil:
		return nil, nil, gotErr
	}
	if !reflect.DeepEqual(gotVocab, wantVocab) {
		tb.Fatalf("model %q: vocab differs\nencoding/json = %#v\nstreaming     = %#v", rawModel, wantVocab, gotVocab)
	}
	if !reflect.DeepEqual(gotMerges, wantMerges) {
		tb.Fatalf("model %q: merges differ\nencoding/json = %v\nstreaming     = %v", rawModel, wantMerges, gotMerges)
	}
	return gotVocab, gotMerges, nil
}

// rawModelOf pulls the "model" object out of a tokenizer.json.
func rawModelOf(tb testing.TB, path string) []byte {
	tb.Helper()

	raw, err := os.ReadFile(path)
	if err != nil {
		tb.Fatalf("read %s: %v", path, err)
	}
	var doc struct {
		Model json.RawMessage `json:"model"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		tb.Fatalf("decode %s: %v", path, err)
	}
	return doc.Model
}

// TestDecodeMatchesEncodingJSON is B.1's acceptance check: on every
// tokenizer.json this package actually loads, the streaming decode builds the
// same vocabulary and merge table encoding/json did.
func TestDecodeMatchesEncodingJSON(t *testing.T) {
	for _, c := range []struct {
		name       string
		checkpoint string // empty for a fixture under testdata/
	}{
		{name: "mini_en"},
		{name: "mini_ml"},
		{name: golden.English, checkpoint: golden.English},
		{name: golden.Multilingual, checkpoint: golden.Multilingual},
		{name: golden.TypedDecisions, checkpoint: golden.TypedDecisions},
	} {
		t.Run(c.name, func(t *testing.T) {
			path := filepath.Join("testdata", c.name, fileTokenizerJSON)
			if c.checkpoint != "" {
				root := golden.SkipWithoutModels(t)
				path = filepath.Join(golden.CheckpointDir(root, c.checkpoint), "tokenizer", fileTokenizerJSON)
			}
			vocab, merges, err := sameTables(t, rawModelOf(t, path))
			if err != nil {
				t.Fatalf("both paths refused %s: %v", path, err)
			}
			// Two empty tables are equal too; that would prove nothing.
			if len(vocab) == 0 || len(merges) == 0 {
				t.Fatalf("vacuous: %d vocabulary entries, %d merges", len(vocab), len(merges))
			}
		})
	}
}

// decodeEdgeCases are the inputs where a hand-written scanner and encoding/json
// plausibly part ways. Each is a "model" object; the fuzz corpus starts from
// them too.
var decodeEdgeCases = []struct {
	name    string
	model   string
	wantErr bool
	// Spot checks on top of the differential, so a case cannot pass merely
	// because both paths are wrong in the same way.
	vocab  map[string]int32
	merges map[[2]int32]merge
}{
	{
		name: "escapes",
		model: `{"vocab":{"\u00e9":1,"\ud83d\ude00":2,"\ud800":3,"\"":4,"\\":5,"\/":6,"a":7,"a\u00e9":8},` +
			`"merges":[["a","\u00e9"]]}`,
		vocab: map[string]int32{
			"é": 1, "😀": 2, "\uFFFD": 3, `"`: 4, `\`: 5, "/": 6, "a": 7, "aé": 8,
		},
		merges: map[[2]int32]merge{{7, 1}: {rank: 0, newID: 8}},
	},
	{
		name:  "invalid UTF-8 becomes U+FFFD",
		model: "{\"vocab\":{\"\xff\":1,\"a\xc3\":2},\"merges\":[]}",
		vocab: map[string]int32{"\uFFFD": 1, "a\uFFFD": 2},
	},
	{
		name:  "raw multi-byte UTF-8 is kept as is",
		model: `{"vocab":{"▁":1,"é":2,"▁é":3},"merges":[["▁","é"]]}`,
		vocab: map[string]int32{"▁": 1, "é": 2, "▁é": 3},
	},
	{
		name:  "duplicate vocabulary key: last wins",
		model: `{"vocab":{"a":1,"b":2,"a":3},"merges":[]}`,
		vocab: map[string]int32{"a": 3, "b": 2},
	},
	{
		name:   "duplicate merge: earlier rank wins",
		model:  `{"vocab":{"a":1,"b":2,"ab":3},"merges":[["a","b"],["a","b"]]}`,
		merges: map[[2]int32]merge{{1, 2}: {rank: 0, newID: 3}},
	},
	{
		name:   "one-element merge pairs with the empty string",
		model:  `{"vocab":{"":0,"a":1},"merges":[["a"]]}`,
		merges: map[[2]int32]merge{{1, 0}: {rank: 0, newID: 1}},
	},
	{
		name:   "empty merge pair",
		model:  `{"vocab":{"":0},"merges":[[]]}`,
		merges: map[[2]int32]merge{{0, 0}: {rank: 0, newID: 0}},
	},
	{
		name:  "three-element merge ignores the third, whatever its type",
		model: `{"vocab":{"a":1,"b":2,"ab":3},"merges":[["a","b","zzz"],["b","a",{"x":[1,"]"]}]]}`,
		// ["b","a"] produces "ba", which is absent; the third element is never
		// looked at, so the error is the resolution one.
		wantErr: true,
	},
	{
		name:   "three-element merge with a non-string third",
		model:  `{"vocab":{"a":1,"b":2,"ab":3},"merges":[["a","b",7,null,true,[]]]}`,
		merges: map[[2]int32]merge{{1, 2}: {rank: 0, newID: 3}},
	},
	{
		name:   "null merge pair is two empty strings",
		model:  `{"vocab":{"":0,"a":1},"merges":[null,["a",""]]}`,
		merges: map[[2]int32]merge{{0, 0}: {rank: 0, newID: 0}, {1, 0}: {rank: 1, newID: 1}},
	},
	{
		name:   "null inside a merge pair is the empty string",
		model:  `{"vocab":{"":0,"a":1},"merges":[[null,"a"]]}`,
		merges: map[[2]int32]merge{{0, 1}: {rank: 0, newID: 1}},
	},
	{name: "merge pair with a number", model: `{"vocab":{"a":1},"merges":[["a",1]]}`, wantErr: true},
	{name: "merge pair with a bool", model: `{"vocab":{"a":1},"merges":[[true,"a"]]}`, wantErr: true},
	{name: "merge pair with an object", model: `{"vocab":{"a":1},"merges":[[{},"a"]]}`, wantErr: true},
	{name: "merge as one spaced string", model: `{"vocab":{"a":1,"b":2,"ab":3},"merges":["a b"]}`, wantErr: true},
	{name: "merge as a number", model: `{"vocab":{},"merges":[1]}`, wantErr: true},
	{name: "merge as an object", model: `{"vocab":{},"merges":[{}]}`, wantErr: true},
	{name: "merges as an object", model: `{"vocab":{},"merges":{}}`, wantErr: true},
	{name: "merges as a string", model: `{"vocab":{},"merges":"x"}`, wantErr: true},
	{name: "left side missing", model: `{"vocab":{"b":2},"merges":[["a","b"]]}`, wantErr: true},
	{name: "right side missing", model: `{"vocab":{"a":1},"merges":[["a","b"]]}`, wantErr: true},
	{name: "concatenation missing", model: `{"vocab":{"a":1,"b":2},"merges":[["a","b"]]}`, wantErr: true},
	{
		name:    "type error after a resolution error is still an error",
		model:   `{"vocab":{"a":1},"merges":[["a","x"],[1]]}`,
		wantErr: true,
	},
	{name: "fractional id", model: `{"vocab":{"a":1.5}}`, wantErr: true},
	{name: "integral fraction", model: `{"vocab":{"a":1.0}}`, wantErr: true},
	{name: "exponent id", model: `{"vocab":{"a":1e2}}`, wantErr: true},
	{name: "string id", model: `{"vocab":{"a":"3"}}`, wantErr: true},
	{name: "bool id", model: `{"vocab":{"a":true}}`, wantErr: true},
	{name: "array id", model: `{"vocab":{"a":[1]}}`, wantErr: true},
	{name: "object id", model: `{"vocab":{"a":{}}}`, wantErr: true},
	{name: "id past MaxInt32", model: `{"vocab":{"a":2147483648}}`, wantErr: true},
	{name: "id below MinInt32", model: `{"vocab":{"a":-2147483649}}`, wantErr: true},
	{name: "id with many digits", model: `{"vocab":{"a":100000000000000000000000000000}}`, wantErr: true},
	{
		name:  "ids at the int32 limits",
		model: `{"vocab":{"a":2147483647,"b":-2147483648,"c":-0,"d":0}}`,
		vocab: map[string]int32{"a": 2147483647, "b": -2147483648, "c": 0, "d": 0},
	},
	{
		name:  "null id is zero",
		model: `{"vocab":{"a":null,"b":5}}`,
		vocab: map[string]int32{"a": 0, "b": 5},
	},
	{
		name:  "null id overwrites an earlier duplicate",
		model: `{"vocab":{"a":5,"a":null}}`,
		vocab: map[string]int32{"a": 0},
	},
	{name: "null vocab", model: `{"vocab":null,"merges":null}`},
	{name: "absent vocab and merges", model: `{"type":"BPE"}`},
	{name: "empty vocab and merges", model: `{"vocab":{},"merges":[]}`, vocab: map[string]int32{}},
	{name: "vocab as an array", model: `{"vocab":[]}`, wantErr: true},
	{name: "vocab as a string", model: `{"vocab":"a"}`, wantErr: true},
	{name: "vocab as a number", model: `{"vocab":1}`, wantErr: true},
	{
		name: "whitespace everywhere",
		model: "{ \"vocab\" :\t{\r\n \"a\" \t: 1 ,\n\"b\":\r2\t,\"ab\"  :  3 \n} ,\n \"merges\" : [ \t" +
			"[ \"a\" ,\r\n \"b\" ] \n, [\n] , null \t] \r\n}",
		vocab: map[string]int32{"a": 1, "b": 2, "ab": 3},
		// [] and null are ("",""), which is not in this vocabulary.
		wantErr: true,
	},
	{
		name:   "whitespace everywhere, resolvable",
		model:  "{\"vocab\":\n{ \"a\" : 1 , \"b\" : 2 , \"ab\" : 3 }\n,\"merges\"\n:\n[ [ \"a\" , \"b\" ] ]\n}",
		vocab:  map[string]int32{"a": 1, "b": 2, "ab": 3},
		merges: map[[2]int32]merge{{1, 2}: {rank: 0, newID: 3}},
	},
	// encoding/json matches field names case-insensitively and decodes every
	// occurrence of a repeated key into the same field: a map accumulates, a
	// null resets it, and a slice is overwritten in place.
	{
		name:  "repeated vocab key accumulates",
		model: `{"vocab":{"a":1,"b":2},"Vocab":{"b":3,"c":4}}`,
		vocab: map[string]int32{"a": 1, "b": 3, "c": 4},
	},
	{name: "repeated vocab key reset by null", model: `{"vocab":{"a":1},"VOCAB":null}`},
	{
		name:  "repeated vocab key after null",
		model: `{"vocab":{"a":1},"vocab":null,"vocab":{"b":2}}`,
		vocab: map[string]int32{"b": 2},
	},
	{name: "repeated vocab key with a type error", model: `{"vocab":{"a":1},"vocab":[]}`, wantErr: true},
	{
		name: "repeated merges key: a null pair keeps the earlier occurrence's pair",
		model: `{"vocab":{"":0,"a":1,"b":2,"ab":3,"ba":4},` +
			`"merges":[["a","b"],["b","a"]],"Merges":[null]}`,
		merges: map[[2]int32]merge{{1, 2}: {rank: 0, newID: 3}},
	},
	{
		name: "repeated merges key: shorter occurrence truncates",
		model: `{"vocab":{"a":1,"b":2,"ab":3,"ba":4},` +
			`"merges":[["a","b"],["b","a"]],"merges":[["b","a"]]}`,
		merges: map[[2]int32]merge{{2, 1}: {rank: 0, newID: 4}},
	},
	{
		name:  "repeated merges key reset by null",
		model: `{"vocab":{"a":1,"b":2,"ab":3},"merges":[["a","b"]],"merges":null}`,
		vocab: map[string]int32{"a": 1, "b": 2, "ab": 3}, merges: map[[2]int32]merge{},
	},
}

func TestDecodeEdgeCases(t *testing.T) {
	for _, c := range decodeEdgeCases {
		t.Run(c.name, func(t *testing.T) {
			vocab, merges, err := sameTables(t, []byte(c.model))
			if (err != nil) != c.wantErr {
				t.Fatalf("err = %v, wantErr %v", err, c.wantErr)
			}
			if err != nil {
				return
			}
			if c.vocab != nil && !reflect.DeepEqual(vocab, c.vocab) {
				t.Errorf("vocab = %#v, want %#v", vocab, c.vocab)
			}
			if c.merges != nil && !reflect.DeepEqual(merges, c.merges) {
				t.Errorf("merges = %v, want %v", merges, c.merges)
			}
		})
	}
}

// TestDecodePresizeIsCapped holds the map presizing to a bound. The hint is a
// separator count over the raw value, and tokenizer.json comes from an
// untrusted Hub repo: one token of 8 M commas (or ']') must not presize a map
// of 8 M entries, hundreds of MB, before a single id has been checked.
func TestDecodePresizeIsCapped(t *testing.T) {
	const separators = 8 << 20
	const budget = 128 << 20
	cases := []struct {
		name string
		raw  []byte
		scan func(raw []byte) error
	}{
		{
			name: "vocab",
			raw:  []byte(`{"` + strings.Repeat(",", separators) + `":0}`),
			scan: func(raw []byte) error { _, err := scanVocab(raw); return err },
		},
		{
			// The pair fails to resolve against the empty vocab, but only
			// after the table has been allocated.
			name: "merges",
			raw:  []byte(`[["` + strings.Repeat("]", separators) + `","x"]]`),
			scan: func(raw []byte) error { _, err := scanMerges(raw, map[string]int32{}); return err },
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var before, after runtime.MemStats
			runtime.GC()
			runtime.ReadMemStats(&before)
			_ = c.scan(c.raw)
			runtime.ReadMemStats(&after)
			if got := after.TotalAlloc - before.TotalAlloc; got > budget {
				t.Errorf("scan of a %d-separator token allocated %d MB, want at most %d MB",
					separators, got>>20, budget>>20)
			}
		})
	}
}

// fuzzVocab is what the direct scanMerges calls resolve against: small, but
// with the empty string, so null and short pairs can resolve.
var fuzzVocab = map[string]int32{"": 0, "a": 1, "b": 2, "ab": 3, "ba": 4, "é": 5, "aé": 6}

// FuzzDecodeVocabMerges holds the streaming decode to encoding/json on
// arbitrary input, at two levels: a whole "model" object (which covers repeated
// and case-folded keys) and the bare vocab and merges values. On invalid JSON
// the scanners only have to fail without panicking; production never hands
// them any, because encoding/json validates the whole document first.
func FuzzDecodeVocabMerges(f *testing.F) {
	for _, c := range decodeEdgeCases {
		f.Add([]byte(c.model))
	}
	for _, dir := range []string{"mini_en", "mini_ml"} {
		f.Add(rawModelOf(f, filepath.Join("testdata", dir, fileTokenizerJSON)))
	}
	for _, s := range []string{
		`{"a":1}`, `[["a","b"]]`, `null`, ` [ ] `, `{`, `[`, `"`, `[["a`, `{"a":`, `{"a":1,}`, `[["a"],]`,
		`{"a\`, `[["\u00`, `{"a":-}`, `{"a":1}x`, `[[]]x`, `[[[[[[[[`, `]`, `}`, "\x00",
	} {
		f.Add([]byte(s))
	}

	f.Fuzz(func(t *testing.T, data []byte) {
		_, _, _ = sameTables(t, data)

		// The bare values: must not panic, and on valid JSON must agree.
		gotVocab, gotVocabErr := scanVocab(data)
		gotMerges, gotMergesErr := scanMerges(data, fuzzVocab)
		if !json.Valid(data) {
			return
		}

		var wantVocab map[string]int32
		wantVocabErr := json.Unmarshal(data, &wantVocab)
		if (wantVocabErr == nil) != (gotVocabErr == nil) {
			t.Fatalf("vocab %q: encoding/json err = %v, scanVocab err = %v", data, wantVocabErr, gotVocabErr)
		}
		if wantVocabErr == nil && !reflect.DeepEqual(gotVocab, wantVocab) {
			t.Fatalf("vocab %q: encoding/json = %#v, scanVocab = %#v", data, wantVocab, gotVocab)
		}

		var pairs [][2]string
		wantMergesErr := json.Unmarshal(data, &pairs)
		var wantMerges map[[2]int32]merge
		if wantMergesErr == nil {
			wantMerges, wantMergesErr = oracleBuildMerges(pairs, fuzzVocab)
		}
		if (wantMergesErr == nil) != (gotMergesErr == nil) {
			t.Fatalf("merges %q: encoding/json err = %v, scanMerges err = %v", data, wantMergesErr, gotMergesErr)
		}
		if wantMergesErr == nil && !reflect.DeepEqual(gotMerges, wantMerges) {
			t.Fatalf("merges %q: encoding/json = %v, scanMerges = %v", data, wantMerges, gotMerges)
		}
	})
}
