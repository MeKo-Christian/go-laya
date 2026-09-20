package tokenizer

import (
	"strings"
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/golden"
)

// goldenCase is one record of testdata/tokenizer_{en,ml}.jsonl.
type goldenCase struct {
	Text                   string   `json:"text"`
	IDs                    []int64  `json:"ids"`
	Tokens                 []string `json:"tokens"`
	IDsWithLeadingSpace    []int64  `json:"ids_with_leading_space"`
	TokensWithLeadingSpace []string `json:"tokens_with_leading_space"`
}

// TestGoldenCorpora is task 4.4.10: both corpora, ids *and* token strings,
// against the real checkpoints. This is the assertion M5's gate names.
//
// Every case is checked twice, bare and with a leading space, because
// common.py:68 tokenises every option with one and the two checkpoints
// disagree about whether it survives.
func TestGoldenCorpora(t *testing.T) {
	root := golden.SkipWithoutModels(t)

	for _, c := range []struct{ fixture, checkpoint string }{
		{"tokenizer_en", golden.English},
		{"tokenizer_ml", golden.Multilingual},
	} {
		t.Run(c.fixture, func(t *testing.T) {
			tok := openCheckpoint(t, root, c.checkpoint)

			for _, rec := range golden.Load(t, c.fixture) {
				var want goldenCase
				rec.Unmarshal(t, &want)

				t.Run(rec.Name, func(t *testing.T) {
					assertEncoding(t, tok, want.Text, want.IDs, want.Tokens)
					assertEncoding(t, tok, " "+want.Text,
						want.IDsWithLeadingSpace, want.TokensWithLeadingSpace)
				})
			}
		})
	}
}

// assertEncoding compares ids and token strings, reporting the first
// divergence with both. An id diff on its own says nothing; ["a","b"] against
// ["ab"] says which stage broke (AGENTS.md).
func assertEncoding(tb testing.TB, tok *HF, text string, wantIDs []int64, wantTokens []string) {
	tb.Helper()

	gotIDs := tok.Encode(text)
	gotTokens := tokensOf(tok, gotIDs)

	if equalIDs(gotIDs, wantIDs) {
		return
	}
	tb.Errorf("Encode(%q)\n got ids %v\n     tokens %q\nwant ids %v\n     tokens %q\nfirst divergence at %d",
		text, gotIDs, gotTokens, wantIDs, wantTokens, firstDiff(gotIDs, wantIDs))
}

func equalIDs(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func firstDiff(a, b []int64) int {
	for i := range min(len(a), len(b)) {
		if a[i] != b[i] {
			return i
		}
	}
	return min(len(a), len(b))
}

// TestEncodeSanitisesInvalidUTF8 is task 4.4.11. A lone surrogate cannot round
// trip through the JSON corpus -- Python writes "\ud800" and Go's decoder
// yields U+FFFD, so a vector would assert against a corrupted input -- so the
// boundary is tested here instead, without one.
//
// It matters because English's BPE has unk_token null: an out-of-alphabet byte
// would be silently dropped rather than replaced, shifting every marker
// position after it.
func TestEncodeSanitisesInvalidUTF8(t *testing.T) {
	tok := mustOpenMini(t, "mini_en")

	for _, c := range []struct {
		name string
		in   string
	}{
		{"lone continuation byte", "a\x80b"},
		{"truncated sequence", "a\xe2\x96b"},
		{"bare 0xFF", "a\xffb"},
	} {
		t.Run(c.name, func(t *testing.T) {
			got := tok.Encode(c.in)
			want := tok.Encode(strings.ToValidUTF8(c.in, "�"))
			if !equalIDs(got, want) {
				t.Errorf("Encode(%q) = %v, want it sanitised to %v", c.in, got, want)
			}
			if len(got) == 0 {
				t.Errorf("Encode(%q) dropped everything", c.in)
			}
		})
	}
}

// TestEncodeEmpty: [] on both pipelines, including multilingual, where
// prepend_scheme "always" might be expected to emit a lone U+2581.
func TestEncodeEmpty(t *testing.T) {
	for _, name := range []string{"mini_en", "mini_ml"} {
		if got := mustOpenMini(t, name).Encode(""); len(got) != 0 {
			t.Errorf("%s: Encode(%q) = %v, want no ids", name, "", got)
		}
	}
}
