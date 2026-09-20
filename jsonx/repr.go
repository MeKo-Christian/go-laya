package jsonx

import (
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

// ReprString formats a string the way Python's repr does, which is what the
// `%r` conversion in an interpolated message produces.
//
// It exists because laya builds user-visible strings with `%r` and the bytes
// matter: router.py's route reasons embed the caller's model, task and lang
// arguments and the detected language code that way (router.py:256, 261, 267,
// 272, 285). Go's nearest reflex, strconv.Quote, disagrees on three counts --
// it always double-quotes, it emits Go's \a \b \f \v shorthands, and it escapes
// printable non-ASCII -- and each disagreement produces a string that still
// reads like a reason, so the mistake survives review.
//
// The rules, from CPython's unicode_repr:
//
//   - The delimiter is ' unless the string contains a ' and no ", in which case
//     it is ", so that the common apostrophe case needs no backslash.
//   - Backslash and the active delimiter are escaped; \n, \r and \t get letter
//     escapes. Nothing else does -- U+0007 is \x07, not \a.
//   - Every other non-printable code point becomes \xNN, \uNNNN or \UNNNNNNNN
//     in lowercase hex, chosen by magnitude.
//   - Printable non-ASCII passes through literally. Python 3 does not escape
//     it, the same posture Marshal takes with ensure_ascii=False.
//
// Go's unicode.IsPrint agrees with Python's str.isprintable() on the categories
// that decide this: both count ASCII space as printable and every other Zs, and
// all of Cc, Cf, Cs, Co and Cn, as not.
//
// A byte that is not valid UTF-8 has no Python counterpart, since a Python str
// cannot hold one. It is escaped as \xNN so that the result stays unambiguous
// and a mangled argument cannot smuggle a delimiter into the message.
func ReprString(s string) string {
	quote := byte('\'')
	if strings.ContainsRune(s, '\'') && !strings.ContainsRune(s, '"') {
		quote = '"'
	}

	var b strings.Builder
	b.Grow(len(s) + 2)
	b.WriteByte(quote)

	for i := 0; i < len(s); {
		r, size := utf8.DecodeRuneInString(s[i:])
		if r == utf8.RuneError && size == 1 {
			writeHexEscape(&b, rune(s[i]))
			i++
			continue
		}
		writeReprRune(&b, r, rune(quote))
		i += size
	}

	b.WriteByte(quote)
	return b.String()
}

func writeReprRune(b *strings.Builder, r, quote rune) {
	switch {
	case r == '\\' || r == quote:
		b.WriteByte('\\')
		b.WriteRune(r)
	case r == '\n':
		b.WriteString(`\n`)
	case r == '\r':
		b.WriteString(`\r`)
	case r == '\t':
		b.WriteString(`\t`)
	case unicode.IsPrint(r):
		b.WriteRune(r)
	default:
		writeHexEscape(b, r)
	}
}

// writeHexEscape picks the escape width by magnitude, as CPython does: \xNN
// below U+0100, \uNNNN below U+10000, \UNNNNNNNN above. The hex is lowercase
// and the width is fixed, so \x0b never collapses to \xb.
func writeHexEscape(b *strings.Builder, r rune) {
	switch {
	case r < 0x100:
		b.WriteString(`\x`)
		writePaddedHex(b, r, 2)
	case r < 0x10000:
		b.WriteString(`\u`)
		writePaddedHex(b, r, 4)
	default:
		b.WriteString(`\U`)
		writePaddedHex(b, r, 8)
	}
}

func writePaddedHex(b *strings.Builder, r rune, width int) {
	h := strconv.FormatInt(int64(r), 16)
	for n := len(h); n < width; n++ {
		b.WriteByte('0')
	}
	b.WriteString(h)
}
