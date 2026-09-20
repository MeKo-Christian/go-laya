package tokenizer

import (
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/golden"
)

// pretokCase is one record of testdata/pretok_{en,ml}.jsonl (task 4.4.13).
type pretokCase struct {
	Text       string   `json:"text"`
	Normalized string   `json:"normalized"`
	Pretokens  []string `json:"pretokens"`
}

// TestNormalizerGolden runs the normalizer half of the per-stage corpus. It
// needs no checkpoint: the vectors were dumped from the real tokenizers, but a
// normalizer is a pure string-to-string function, so this is the CI-side half
// of the parity claim.
func TestNormalizerGolden(t *testing.T) {
	for _, c := range []struct {
		fixture string
		norm    normalizer
	}{
		{"pretok_en", nfcNormalizer{}},
		{"pretok_ml", replaceNormalizer{pattern: " ", content: "▁"}},
	} {
		t.Run(c.fixture, func(t *testing.T) {
			for _, rec := range golden.Load(t, c.fixture) {
				var got pretokCase
				rec.Unmarshal(t, &got)

				if out := c.norm.normalize(got.Text); out != got.Normalized {
					t.Errorf("%s: normalize(%q)\n got %q\nwant %q",
						rec.Name, got.Text, out, got.Normalized)
				}
			}
		})
	}
}

// TestNFCIsTheOnlyCompositionApplied guards against reaching for norm.NFKC,
// which would fold the fullwidth and ligature cases the corpus deliberately
// keeps distinct: English normalises NFC only, so "ﬁnal" stays one codepoint.
func TestNFCIsTheOnlyCompositionApplied(t *testing.T) {
	for _, c := range []struct{ in, want string }{
		{"ﬁnal", "ﬁnal"},  // ligature fi survives NFC, folds under NFKC
		{"ＡＢＣ", "ＡＢＣ"},    // fullwidth ABC likewise
		{"café", "café"}, // NFD composes
		{"", ""},
	} {
		if got := (nfcNormalizer{}).normalize(c.in); got != c.want {
			t.Errorf("normalize(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestReplaceIsLiteralNotRegex: the multilingual normalizer is
// Replace{pattern:{String:" "}}, a literal string replacement. Treating the
// pattern as a regex would be a silent behaviour change on any input holding a
// metacharacter.
func TestReplaceIsLiteralNotRegex(t *testing.T) {
	r := replaceNormalizer{pattern: " ", content: "▁"}
	for _, c := range []struct{ in, want string }{
		{"a b", "a▁b"},
		{"  ", "▁▁"},
		{"a.*b", "a.*b"},
		{"▁", "▁"},
		{"", ""},
	} {
		if got := r.normalize(c.in); got != c.want {
			t.Errorf("normalize(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}
