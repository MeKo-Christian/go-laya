package tokenizer

import (
	"regexp"
	"strings"
	"testing"
	"unicode"
)

// whiteSpaceClass spells out Unicode White_Space as a character class. Go's
// regexp understands \p{L} and \p{N} (categories) but not \p{White_Space},
// which is a binary property, so the set is written out. It is asserted
// against unicode.IsSpace below rather than trusted.
const whiteSpaceClass = `\x{0009}-\x{000D}\x{0020}\x{0085}\x{00A0}\x{1680}` +
	`\x{2000}-\x{200A}\x{2028}\x{2029}\x{202F}\x{205F}\x{3000}`

// oraclePattern is HF's pattern with two edits: \s is the explicit White_Space
// set, and the `\s+(?!\S)` branch is dropped because RE2 has no negative
// lookahead. The give-back that branch encodes is applied afterwards, in
// oracleSplit, which is the only hand-written part of the oracle.
var oraclePattern = regexp.MustCompile(
	`\A(?:'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^` + whiteSpaceClass + `\p{L}\p{N}]+|[` +
		whiteSpaceClass + `]+)`,
)

// oracleSplit is a second implementation of splitByteLevel that shares no code
// with it: the alternation is resolved by Go's regexp engine (leftmost-first,
// so branch order is honoured) and only the give-back is written out. Two
// implementations agreeing under fuzzing is as close to Oniguruma as Go gets
// without cgo.
func oracleSplit(text string) []string {
	var out []string
	for pos := 0; pos < len(text); {
		loc := oraclePattern.FindStringIndex(text[pos:])
		if loc == nil {
			// Cannot happen: the last two branches between them accept every
			// rune. Surface it rather than loop forever.
			panic("oracle: no branch matched at " + text[pos:])
		}
		n := loc[1]
		piece := text[pos : pos+n]

		// `\s+(?!\S)`: a whitespace run followed by a non-space gives back its
		// last rune, and only when at least one rune would remain.
		if isAllSpace(piece) && pos+n < len(text) && len([]rune(piece)) >= 2 {
			_, size := utf8DecodeLast(piece)
			n -= size
			piece = text[pos : pos+n]
		}
		out = append(out, piece)
		pos += n
	}
	return out
}

func isAllSpace(s string) bool {
	for _, r := range s {
		if !unicode.IsSpace(r) {
			return false
		}
	}
	return s != ""
}

func utf8DecodeLast(s string) (rune, int) {
	r := []rune(s)
	last := r[len(r)-1]
	return last, len(string(last))
}

// TestWhiteSpaceClassMatchesGo guards the oracle's own premise: the written-out
// set must be exactly what unicode.IsSpace accepts, or the two implementations
// would agree by sharing a mistake.
func TestWhiteSpaceClassMatchesGo(t *testing.T) {
	re := regexp.MustCompile(`\A[` + whiteSpaceClass + `]\z`)
	for r := rune(0); r <= unicode.MaxRune; r++ {
		if !utf8Valid(r) {
			continue
		}
		if got, want := re.MatchString(string(r)), unicode.IsSpace(r); got != want {
			t.Fatalf("U+%04X: class says %v, unicode.IsSpace says %v", r, got, want)
		}
	}
}

func utf8Valid(r rune) bool { return r < 0xD800 || r > 0xDFFF }

// TestOracleAgreesOnTheCorpus is the cheap half of the cross-check: the two
// implementations must agree on every case the golden corpus already holds,
// before the fuzzer looks for cases it does not.
func TestOracleAgreesOnTheCorpus(t *testing.T) {
	for _, c := range corpusTexts(t) {
		if got, want := splitByteLevel(c), oracleSplit(c); !equalStrings(got, want) {
			t.Errorf("%q:\nscanner %q\noracle  %q", c, got, want)
		}
	}
}

// FuzzByteLevelScannerAgainstOracle is task 4.3.7. It asserts only that the two
// implementations agree -- neither is the reference, so a disagreement is a bug
// in whichever one the Python oracle then contradicts.
func FuzzByteLevelScannerAgainstOracle(f *testing.F) {
	for _, c := range corpusTexts(f) {
		f.Add(c)
	}
	for _, c := range []string{
		"", " ", "  ", "\t\t", "a\u00a0\u00a0b", "don't", "DON'T", " 's",
		"\u3000\u3000x", "a\u2003\u2003", "1 23", "\u0085\u0085",
	} {
		f.Add(c)
	}

	f.Fuzz(func(t *testing.T, text string) {
		// The scanner's contract starts after the API edge sanitises input.
		text = strings.ToValidUTF8(text, "\uFFFD")

		got, want := splitByteLevel(text), oracleSplit(text)
		if !equalStrings(got, want) {
			t.Fatalf("%q:\nscanner %q\noracle  %q", text, got, want)
		}
		if strings.Join(got, "") != text {
			t.Fatalf("%q: pieces do not reassemble: %q", text, got)
		}
	})
}
