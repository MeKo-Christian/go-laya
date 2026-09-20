package tokenizer

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/golden"
)

func TestOpenMiniFixtures(t *testing.T) {
	for _, c := range []struct {
		dir        string
		vocab      int
		merges     int
		byteFall   bool
		fuseUnk    bool
		normalizer string
	}{
		{"mini_en", 271, 5, false, false, "NFC"},
		{"mini_ml", 40, 4, true, true, "Replace"},
	} {
		t.Run(c.dir, func(t *testing.T) {
			tok, err := Open(filepath.Join("testdata", c.dir))
			if err != nil {
				t.Fatalf("Open: %v", err)
			}
			if got := len(tok.vocab); got != c.vocab {
				t.Errorf("vocab = %d entries, want %d", got, c.vocab)
			}
			if got := len(tok.merges); got != c.merges {
				t.Errorf("merges = %d, want %d", got, c.merges)
			}
			if tok.byteFallback != c.byteFall {
				t.Errorf("byteFallback = %v, want %v", tok.byteFallback, c.byteFall)
			}
			if tok.fuseUnk != c.fuseUnk {
				t.Errorf("fuseUnk = %v, want %v", tok.fuseUnk, c.fuseUnk)
			}
		})
	}
}

// TestOpenRefusesWhatItCannotReproduce. A loader that ignores a field it does
// not understand is R1's failure mode exactly: the tokenizer keeps working and
// returns different ids. Each case mutates one field of a fixture that loads
// cleanly, so the mutation is the only difference.
func TestOpenRefusesWhatItCannotReproduce(t *testing.T) {
	for _, c := range []struct {
		name   string
		mutate func(m map[string]any)
	}{
		{"ignore_merges", func(m map[string]any) {
			m["model"].(map[string]any)["ignore_merges"] = true
		}},
		{"dropout", func(m map[string]any) {
			m["model"].(map[string]any)["dropout"] = 0.1
		}},
		{"continuing_subword_prefix", func(m map[string]any) {
			m["model"].(map[string]any)["continuing_subword_prefix"] = "##"
		}},
		{"end_of_word_suffix", func(m map[string]any) {
			m["model"].(map[string]any)["end_of_word_suffix"] = "</w>"
		}},
		{"truncation", func(m map[string]any) {
			m["truncation"] = map[string]any{"max_length": 8}
		}},
		{"padding", func(m map[string]any) {
			m["padding"] = map[string]any{"strategy": "BatchLongest"}
		}},
		{"unknown model type", func(m map[string]any) {
			m["model"].(map[string]any)["type"] = "Unigram"
		}},
		{"unknown normalizer", func(m map[string]any) {
			m["normalizer"] = map[string]any{"type": "BertNormalizer"}
		}},
		{"unknown pre_tokenizer", func(m map[string]any) {
			m["pre_tokenizer"] = map[string]any{"type": "Whitespace"}
		}},
		{"merge out of vocabulary", func(m map[string]any) {
			model := m["model"].(map[string]any)
			model["merges"] = append(model["merges"].([]any), []any{"a", "zzz"})
		}},
	} {
		t.Run(c.name, func(t *testing.T) {
			dir := mutatedFixture(t, "mini_en", c.mutate)

			_, err := Open(dir)
			if err == nil {
				t.Fatalf("Open accepted a tokenizer.json with %s changed", c.name)
			}
			if !errors.Is(err, ErrUnsupported) {
				t.Errorf("Open: %v, want it to wrap ErrUnsupported", err)
			}
		})
	}
}

// TestOpenAcceptsDeadStages: both real checkpoints carry a post_processor and a
// decoder, and laya uses neither -- it always passes add_special_tokens=false
// and never decodes. Refusing them would refuse every real checkpoint.
func TestOpenAcceptsDeadStages(t *testing.T) {
	dir := mutatedFixture(t, "mini_en", func(m map[string]any) {
		m["post_processor"] = map[string]any{"type": "TemplateProcessing"}
		m["decoder"] = map[string]any{"type": "ByteLevel"}
	})

	if _, err := Open(dir); err != nil {
		t.Errorf("Open rejected a checkpoint for stages it never runs: %v", err)
	}
}

// TestOpenRefusesMalformedIDs. Every id in tokenizer.json indexes indexVocab's
// dense table directly, and Open parses a file that may have come off the Hub
// -- AGENTS.md names the deserialization path as the attack surface. Left
// unchecked, a negative id panics on the slice write, an id near MaxInt32 makes
// a two-kilobyte file allocate tens of gigabytes, and one past MaxInt32 wraps
// the int32 conversion into some other token's slot.
func TestOpenRefusesMalformedIDs(t *testing.T) {
	for _, c := range []struct {
		name   string
		mutate func(m map[string]any)
	}{
		{"negative vocabulary id", func(m map[string]any) {
			m["model"].(map[string]any)["vocab"].(map[string]any)["zz"] = -1
		}},
		{"negative added-token id", func(m map[string]any) {
			m["added_tokens"].([]any)[0].(map[string]any)["id"] = -1
		}},
		{"vocabulary id past the ceiling", func(m map[string]any) {
			m["model"].(map[string]any)["vocab"].(map[string]any)["zz"] = maxTokenID + 1
		}},
		{"added-token id past MaxInt32", func(m map[string]any) {
			m["added_tokens"].([]any)[0].(map[string]any)["id"] = int64(math.MaxInt32) + 1
		}},
	} {
		t.Run(c.name, func(t *testing.T) {
			dir := mutatedFixture(t, "mini_en", c.mutate)

			_, err := Open(dir)
			if err == nil {
				t.Fatalf("Open accepted a tokenizer.json with a %s", c.name)
			}
			if !errors.Is(err, ErrUnsupported) {
				t.Errorf("Open: %v, want it to wrap ErrUnsupported", err)
			}
		})
	}
}

// TestOpenRefusesEmptyMetaspaceReplacement. splitMergedWithNext scans for the
// replacement and never advances when it is empty, so accepting one turns the
// first Encode into an infinite loop instead of a load error. The guard belongs
// at load because that is where the file is still in hand to name.
func TestOpenRefusesEmptyMetaspaceReplacement(t *testing.T) {
	dir := mutatedFixture(t, "mini_ml", func(m map[string]any) {
		m["pre_tokenizer"].(map[string]any)["replacement"] = ""
	})

	_, err := Open(dir)
	if err == nil {
		t.Fatal("Open accepted a Metaspace with an empty replacement")
	}
	if !errors.Is(err, ErrUnsupported) {
		t.Errorf("Open: %v, want it to wrap ErrUnsupported", err)
	}
}

// TestByteCoverageOfTheRealCheckpoints is PLAN.md 1.4 item 3, asserted at load
// because it is a property of the file and not of any one input. Gated: it
// needs the 3.5 and 34 MB tokenizer.json.
func TestByteCoverageOfTheRealCheckpoints(t *testing.T) {
	root := golden.SkipWithoutModels(t)

	t.Run("english drops 13 byte-chars", func(t *testing.T) {
		tok := openCheckpoint(t, root, golden.English)
		var missing []byte
		for b := range 256 {
			if _, ok := tok.vocab[string(byteToRune[b])]; !ok {
				missing = append(missing, byte(b))
			}
		}
		want := []byte{0xC0, 0xC1, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD, 0xFE, 0xFF}
		if string(missing) != string(want) {
			t.Errorf("missing byte-chars = % X, want % X", missing, want)
		}
	})

	t.Run("multilingual drops only <0x09>", func(t *testing.T) {
		tok := openCheckpoint(t, root, golden.Multilingual)
		var missing []string
		for b := range 256 {
			name := byteFallbackToken(byte(b))
			if _, ok := tok.vocab[name]; !ok {
				missing = append(missing, name)
			}
		}
		if len(missing) != 1 || missing[0] != "<0x09>" {
			t.Errorf("missing byte tokens = %v, want only <0x09>", missing)
		}
		// Harmless precisely because U+0009 is itself a vocabulary entry, so
		// byte fallback is never asked for it.
		if _, ok := tok.vocab["\t"]; !ok {
			t.Error(`<0x09> is missing and "\t" is not in the vocabulary either`)
		}
	})
}

// TestRealCheckpointShape pins the counts PLAN.md 1.4 states, so a checkpoint
// swap is loud rather than silent.
func TestRealCheckpointShape(t *testing.T) {
	root := golden.SkipWithoutModels(t)

	for _, c := range []struct {
		checkpoint string
		vocab      int
		merges     int
		added      int
		byteFall   bool
	}{
		{golden.English, 50280, 50009, 116, false},
		{golden.Multilingual, 256000, 580604, 249, true},
		{golden.TypedDecisions, 50280, 50009, 116, false},
	} {
		t.Run(c.checkpoint, func(t *testing.T) {
			tok := openCheckpoint(t, root, c.checkpoint)
			if got := len(tok.vocab); got != c.vocab {
				t.Errorf("vocab = %d, want %d", got, c.vocab)
			}
			if got := len(tok.merges); got != c.merges {
				t.Errorf("merges = %d, want %d", got, c.merges)
			}
			if got := len(tok.added); got != c.added {
				t.Errorf("added tokens = %d, want %d", got, c.added)
			}
			if tok.byteFallback != c.byteFall {
				t.Errorf("byteFallback = %v, want %v", tok.byteFallback, c.byteFall)
			}
		})
	}
}

func openCheckpoint(tb testing.TB, root, checkpoint string) *HF {
	tb.Helper()

	tok, err := Open(filepath.Join(golden.CheckpointDir(root, checkpoint), "tokenizer"))
	if err != nil {
		tb.Fatalf("Open %s: %v", checkpoint, err)
	}
	return tok
}

// mutatedFixture copies a fixture into a temp directory with one field changed.
func mutatedFixture(tb testing.TB, base string, mutate func(map[string]any)) string {
	tb.Helper()

	raw, err := os.ReadFile(filepath.Join("testdata", base, "tokenizer.json"))
	if err != nil {
		tb.Fatalf("read the base fixture: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		tb.Fatalf("decode the base fixture: %v", err)
	}
	mutate(m)

	out, err := json.Marshal(m)
	if err != nil {
		tb.Fatalf("re-encode the fixture: %v", err)
	}
	dir := tb.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), out, 0o600); err != nil {
		tb.Fatalf("write the mutated fixture: %v", err)
	}
	cfg, err := os.ReadFile(filepath.Join("testdata", base, "tokenizer_config.json"))
	if err != nil {
		tb.Fatalf("read the base config: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), cfg, 0o600); err != nil {
		tb.Fatalf("write the config: %v", err)
	}
	return dir
}
