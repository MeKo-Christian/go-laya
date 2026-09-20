package tokenizer

import (
	"strings"
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/golden"
)

// tokensOf renders ids as their token strings, which is what makes a BPE
// failure readable: ["a","b"] against ["ab"] names the merge that did or did
// not fire.
func tokensOf(t *HF, ids []int64) []string {
	out := make([]string, len(ids))
	for i, id := range ids {
		out[i] = t.IDToToken(id)
	}
	return out
}

func TestBPEMergesByRank(t *testing.T) {
	tok := mustOpenMini(t, "mini_ml")

	for _, c := range []struct {
		name string
		word string
		want []string
	}{
		{"single merge", "▁a", []string{"▁a"}},
		{"chained merge picks the lower rank first", "▁ab", []string{"▁ab"}},
		{"no merge available", "xy", []string{"x", "y"}},
		{"bare replacement", "▁", []string{"▁"}},
		{"empty", "", nil},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := tokensOf(tok, tok.bpe(c.word)); !equalStrings(got, c.want) {
				t.Errorf("bpe(%q) = %q, want %q", c.word, got, c.want)
			}
		})
	}
}

// TestBPEByteFallback: a character absent from the vocabulary becomes one
// <0xXX> token per UTF-8 byte, pushed before merging so the bytes can still
// participate in merges.
func TestBPEByteFallback(t *testing.T) {
	tok := mustOpenMini(t, "mini_ml")

	for _, c := range []struct {
		name string
		word string
		want []string
	}{
		{"NUL", "▁a\x00b", []string{"▁a", "<0x00>", "b"}},
		{"astral plane", "▁a\U0002000Bb", []string{"▁a", "<0xF0>", "<0xA0>", "<0x80>", "<0x8B>", "b"}},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := tokensOf(tok, tok.bpe(c.word)); !equalStrings(got, c.want) {
				t.Errorf("bpe(%q) = %q, want %q", c.word, got, c.want)
			}
		})
	}
}

// TestBPEFuseUnk is unreachable on both real checkpoints -- multilingual's only
// missing byte token is <0x09>, needed only by U+0009, which is itself in the
// vocabulary -- so it is tested by turning byte fallback off on the fixture,
// which is the one configuration that forces the unk path.
func TestBPEFuseUnk(t *testing.T) {
	dir := mutatedFixture(t, "mini_ml", func(m map[string]any) {
		m["model"].(map[string]any)["byte_fallback"] = false
	})
	tok, err := Open(dir)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	// Two consecutive unknown characters fuse into one <unk>; a known token
	// between them flushes the run.
	for _, c := range []struct {
		word string
		want []string
	}{
		{"▁a??b", []string{"▁a", "<unk>", "b"}},
		{"??", []string{"<unk>"}},
		{"?a?", []string{"<unk>", "a", "<unk>"}},
	} {
		if got := tokensOf(tok, tok.bpe(c.word)); !equalStrings(got, c.want) {
			t.Errorf("bpe(%q) = %q, want %q", c.word, got, c.want)
		}
	}
}

// TestBPEWithoutUnkDropsTheCharacter is the rule the review caught in HF's
// source: the byte-fallback branch sits *inside* `if let Some(unk_token)`, so
// byte_fallback with a null unk_token falls back to nothing at all and the
// character is silently dropped. That is what makes PLAN.md 1.4 item 3's
// "silently dropped" true on English, and why Encode sanitises to valid UTF-8
// at the API edge rather than trusting the vocabulary to catch it.
func TestBPEWithoutUnkDropsTheCharacter(t *testing.T) {
	dir := mutatedFixture(t, "mini_ml", func(m map[string]any) {
		m["model"].(map[string]any)["unk_token"] = nil
	})
	tok, err := Open(dir)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	// Plain drop, with nothing to merge across the gap.
	if got := tokensOf(tok, tok.bpe("x?y")); !equalStrings(got, []string{"x", "y"}) {
		t.Errorf("bpe with no unk = %q, want the unknown character dropped", got)
	}

	// And the consequence worth pinning: dropping a character makes its
	// neighbours adjacent, so a merge that could not otherwise have fired now
	// can. "▁a?b" loses the ? and then merges ▁+a and ▁a+b into a
	// single token. A silent drop is therefore not a local error -- it changes
	// the tokenisation of the text around it.
	if got := tokensOf(tok, tok.bpe("▁a?b")); !equalStrings(got, []string{"▁ab"}) {
		t.Errorf("bpe across a dropped character = %q, want [▁ab]", got)
	}
}

// TestBPEPartialByteCoverageFallsThrough: byte fallback is all-or-nothing per
// character. If any of its bytes lacks a <0xXX> token the character takes the
// unk path instead, rather than emitting a partial byte sequence.
func TestBPEPartialByteCoverageFallsThrough(t *testing.T) {
	tok := mustOpenMini(t, "mini_ml")

	// U+00E9 is two bytes, 0xC3 0xA9; the fixture has neither, so the
	// character cannot be represented byte-wise and becomes unk.
	if got := tokensOf(tok, tok.bpe("▁aé")); !equalStrings(got, []string{"▁a", "<unk>"}) {
		t.Errorf("partial byte coverage = %q, want the unk path", got)
	}
}

// TestBPEOnTheRealVocabulary exercises the 50009- and 580604-merge tables,
// where a wrong heap order shows up and a five-merge fixture cannot.
func TestBPEOnTheRealVocabulary(t *testing.T) {
	root := golden.SkipWithoutModels(t)

	t.Run("english tab run", func(t *testing.T) {
		tok := openCheckpoint(t, root, golden.English)
		// 25 tabs, the pre-token PLAN.md 4.3.4 shows the scanner producing.
		word := strings.Repeat("ĉ", 25)
		got := tokensOf(tok, tok.bpe(word))
		want := []string{
			"ĉĉĉĉĉĉĉĉ",
			"ĉĉĉĉĉĉĉĉ",
			"ĉĉĉĉ", "ĉĉĉĉĉ",
		}
		if !equalStrings(got, want) {
			t.Errorf("bpe(25 tabs) = %q, want %q", got, want)
		}
	})

	t.Run("multilingual byte fallback", func(t *testing.T) {
		tok := openCheckpoint(t, root, golden.Multilingual)
		if got := tokensOf(tok, tok.bpe("▁a\U0002000Bb")); !equalStrings(got,
			[]string{"▁a", "<0xF0>", "<0xA0>", "<0x80>", "<0x8B>", "b"}) {
			t.Errorf("bpe = %q", got)
		}
	})
}
