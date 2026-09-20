package tokenizer

import (
	"strings"
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/golden"
)

// checkEncodeInvariants is task 4.5.1's list. It asserts properties rather
// than values, so it needs no oracle and can run on inputs no corpus covers.
func checkEncodeInvariants(t *testing.T, tok *HF, text string, roundTrips bool) {
	t.Helper()

	ids := tok.Encode(text) // must not panic

	for i, id := range ids {
		if id < 0 || id >= tok.VocabSize() {
			t.Fatalf("Encode(%q)[%d] = %d, outside [0, %d)", text, i, id, tok.VocabSize())
		}
	}

	// Deterministic. A map iteration leaking into the output would show up
	// here and nowhere else.
	if again := tok.Encode(text); !equalIDs(ids, again) {
		t.Fatalf("Encode(%q) is not deterministic: %v then %v", text, ids, again)
	}

	// Anti-runaway. The bound is 2n+1, not n+1, and the fuzzer is what
	// established that: byte fallback costs at most one token per byte, but
	// Metaspace prepends its replacement per *segment*, and every added token
	// splits the input into more segments. "0\n0" on the multilingual pipeline
	// is three segments and five tokens for three bytes, because \n is an added
	// token and each neighbour gets its own prepend. Segments are at least one
	// byte each, so 2n+1 holds; anything above it is a real blowup.
	sanitised := strings.ToValidUTF8(text, "�")
	if len(ids) > 2*len(sanitised)+1 {
		t.Fatalf("Encode(%q) produced %d ids for %d bytes", text, len(ids), len(sanitised))
	}

	if !roundTrips {
		return
	}
	// On a ByteLevel pipeline the token strings concatenate back to the
	// byte-encoded input, so decoding the join returns the input exactly --
	// with two conditions. No added token may have matched, since an added
	// token contributes its raw content rather than a byte-encoded piece. And
	// the comparison is against the *normalized* text, not the input: NFC
	// composes, so decomposed Hangul jamo and an NFD "e" plus combining acute
	// legitimately come back in composed form.
	for _, seg := range tok.splitAdded(sanitised) {
		if seg.token != nil {
			return
		}
	}
	normalized := tok.norm.normalize(sanitised)
	var joined strings.Builder
	for _, id := range ids {
		joined.WriteString(tok.IDToToken(id))
	}
	decoded, err := byteDecode(joined.String())
	if err != nil {
		t.Fatalf("Encode(%q) produced a token outside the byte alphabet: %v", text, err)
	}
	if decoded != normalized {
		t.Fatalf("Encode(%q) does not round trip: got %q, want %q (normalized)",
			text, decoded, normalized)
	}
}

// FuzzEncodeMini runs on the synthetic fixtures, so it runs in CI and its seed
// corpus is exercised on every `go test` even without -fuzz.
func FuzzEncodeMini(f *testing.F) {
	for _, c := range corpusTexts(f) {
		f.Add(c)
	}
	for _, c := range []string{
		"", " ", "[MASK]", "  [MASK]", "a [MASK]", "don't", "▁▁",
		"a\x00b", "\U0002000B", "�", "[unused0][unused0]",
	} {
		f.Add(c)
	}

	en := mustOpenMini(f, "mini_en")
	ml := mustOpenMini(f, "mini_ml")

	f.Fuzz(func(t *testing.T, text string) {
		checkEncodeInvariants(t, en, text, true)
		checkEncodeInvariants(t, ml, text, false)
	})
}

// FuzzEncode is the same invariants against the real vocabularies, where the
// added-token tables and the merge tables are the real ones. Gated on the
// checkpoints being present.
func FuzzEncode(f *testing.F) {
	for _, c := range corpusTexts(f) {
		f.Add(c)
	}

	root := golden.SkipWithoutModels(f)
	en := openCheckpoint(f, root, golden.English)
	ml := openCheckpoint(f, root, golden.Multilingual)

	f.Fuzz(func(t *testing.T, text string) {
		checkEncodeInvariants(t, en, text, true)
		checkEncodeInvariants(t, ml, text, false)
	})
}
