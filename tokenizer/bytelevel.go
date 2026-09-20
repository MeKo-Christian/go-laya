package tokenizer

import (
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"
)

// HF's ByteLevel pre-tokenizer splits on this pattern:
//
//	's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//
// It is scanned by hand rather than compiled, for two independent reasons.
// RE2 has no negative lookahead, so `\s+(?!\S)` cannot be expressed at all. And
// Go's `\s` is ASCII-only while the Oniguruma build HF links treats it as
// Unicode White_Space -- measured, not assumed: U+0085 and U+3000 runs split
// per the give-back rule instead of being swallowed by the greedy
// ` ?[^\s\p{L}\p{N}]+` alternative.
//
// The alternation is tried in order at each position, exactly as a regex engine
// would. Shortcuts are what break it; see splitByteLevel.

// contractions are the seven literals, in pattern order. They are lowercase
// only, which is why "DON'T" splits into DON, ' and T while "don't" keeps "'t".
var contractions = [...]string{"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"}

// isLetter, isNumber and isSpaceRune are named rather than inlined so that a
// Unicode-version skew against Oniguruma (R1's residual risk, findable only by
// task 4.5.3's differential run) is a one-file change rather than a rewrite.
func isLetter(r rune) bool { return unicode.IsLetter(r) }

func isNumber(r rune) bool { return unicode.IsNumber(r) }

// isSpaceRune is Unicode White_Space, which is exactly what unicode.IsSpace
// implements -- and exactly what the probe above showed Oniguruma using.
func isSpaceRune(r rune) bool { return unicode.IsSpace(r) }

// preTokenizeByteLevel splits text and maps each piece through the byte table,
// which is what HF's pre_tokenize_str returns.
func preTokenizeByteLevel(text string) []string {
	pieces := splitByteLevel(text)
	out := make([]string, len(pieces))
	for i, p := range pieces {
		out[i] = byteEncode(p)
	}
	return out
}

// splitByteLevel applies the pattern, returning the raw (unmapped) pieces.
func splitByteLevel(text string) []string {
	var out []string
	for pos := 0; pos < len(text); {
		n := matchAt(text, pos)
		out = append(out, text[pos:pos+n])
		pos += n
	}
	return out
}

// matchAt returns the byte length of the alternative that matches at pos. It
// never returns 0: the last alternative, `\s+`, matches any whitespace, and
// ` ?[^\s\p{L}\p{N}]+` matches any other non-letter non-digit, so every rune is
// covered by some branch.
func matchAt(text string, pos int) int {
	if n := matchContraction(text, pos); n > 0 {
		return n
	}
	// ` ?\p{L}+`, ` ?\p{N}+`, ` ?[^\s\p{L}\p{N}]+` -- in this order. The
	// optional prefix is a literal U+0020, not "any whitespace", which is why a
	// tab that the give-back rule hands back is never absorbed by the next
	// piece while a space is.
	for _, class := range [...]func(rune) bool{isLetter, isNumber, isOther} {
		if n := matchOptionalSpaceRun(text, pos, class); n > 0 {
			return n
		}
	}
	return matchWhitespace(text, pos)
}

func matchContraction(text string, pos int) int {
	if text[pos] != '\'' {
		return 0
	}
	for _, c := range contractions {
		if strings.HasPrefix(text[pos:], c) {
			return len(c)
		}
	}
	return 0
}

// isOther is `[^\s\p{L}\p{N}]`.
func isOther(r rune) bool { return !isSpaceRune(r) && !isLetter(r) && !isNumber(r) }

// matchOptionalSpaceRun implements ` ?X+`: an optional literal space followed
// by one or more runes in the class. It returns 0 when no rune of the class
// follows, because ` ?X+` requires at least one -- a lone space does not match
// and falls through to the whitespace branch.
func matchOptionalSpaceRun(text string, pos int, class func(rune) bool) int {
	i := pos
	if text[i] == ' ' {
		i++
	}
	start := i
	for i < len(text) {
		r, size := utf8.DecodeRuneInString(text[i:])
		if !class(r) {
			break
		}
		i += size
	}
	if i == start {
		return 0
	}
	return i - pos
}

// matchWhitespace implements `\s+(?!\S)|\s+`.
//
// `\s+` is greedy, so it first takes the whole run; the lookahead then requires
// the next character not to be a non-space. When the run is followed by a
// non-space the engine backtracks one character, which succeeds only if at
// least one character remains -- so a run of length 1 fails the first
// alternative outright and the plain `\s+` takes it whole.
func matchWhitespace(text string, pos int) int {
	end := pos
	count := 0
	lastSize := 0
	for end < len(text) {
		r, size := utf8.DecodeRuneInString(text[end:])
		if !isSpaceRune(r) {
			break
		}
		end += size
		lastSize = size
		count++
	}
	if end == pos {
		// Not whitespace and not matched by any earlier branch. Unreachable
		// while isOther is the complement of the other three classes, but
		// returning 0 here would spin forever, so fail loudly instead.
		panic(fmt.Sprintf("tokenizer: no ByteLevel alternative matched at byte %d of %q", pos, text))
	}
	if end < len(text) && count >= 2 {
		return end - lastSize - pos // give the last character back
	}
	return end - pos
}
