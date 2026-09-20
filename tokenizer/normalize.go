package tokenizer

import (
	"strings"

	"golang.org/x/text/unicode/norm"
)

// normalizer is the first stage of an HF pipeline, applied to each segment that
// the added-token matcher did not claim.
//
// Only the two forms the laya checkpoints actually declare are implemented:
// NFC on English and typed-decisions, Replace{" " -> U+2581} on multilingual.
// HF's list is much longer, and porting the rest would be code no vector pins.
// The loader refuses a tokenizer.json asking for anything else rather than
// silently skipping normalisation, which would shift every token.
type normalizer interface {
	normalize(text string) string
}

// nfcNormalizer is HF's {"type":"NFC"}.
//
// NFC, never NFKC: the compatibility fold would collapse the fullwidth and
// ligature cases the corpus keeps distinct, so "final" spelled with U+FB01 must
// survive as one codepoint.
type nfcNormalizer struct{}

func (nfcNormalizer) normalize(text string) string {
	// norm.NFC.String returns the input unchanged, without allocating, when it
	// is already normalised -- which serialised JSON state almost always is.
	return norm.NFC.String(text)
}

// replaceNormalizer is HF's {"type":"Replace","pattern":{"String":...}}.
//
// The pattern is a literal string, not a regex: HF's Replace carries either a
// String or a Regex pattern and multilingual declares String. Compiling it as a
// pattern would change behaviour on any input containing a metacharacter, which
// serialised user JSON certainly does.
//
// This is what makes tok(" "+text) behave oppositely on the two checkpoints
// (PLAN.md 1.4 item 1): the leading space is already U+2581 by the time
// Metaspace looks at it, so Metaspace's prepend guard suppresses itself.
type replaceNormalizer struct {
	pattern string
	content string
}

func (r replaceNormalizer) normalize(text string) string {
	return strings.ReplaceAll(text, r.pattern, r.content)
}
