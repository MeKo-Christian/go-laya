//nolint:gosmopolitan // the CJK literal is a fixture under test, not UI text
package jsonx

import (
	"strconv"
	"testing"
)

// The want column is the output of a real CPython 3.12 `repr(s)`, generated
// against the pinned reference interpreter and pasted in. There is no golden
// fixture for it: repr() is not part of any corpus dump_python_parity.py
// produces, and lang/lang_test.go sets the precedent for pinning a
// hand-transcribed Python table where no fixture exists.
//
// Note what Python does *not* do: repr has no \a, \b, \f or \v shorthand. Only
// \n, \r, \t, the backslash and the active quote get a letter escape; every
// other unprintable code point becomes \xNN, \uNNNN or \UNNNNNNNN in lowercase
// hex. `bell`, `backspace`, `formfeed` and `vtab` below are the cases that pin
// it, because writing them as \a\b\f\v is the obvious mistake.
func TestReprStringMatchesPythonRepr(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"plain", "multilingual", "'multilingual'"},
		{"empty", "", "''"},
		{"lang-code", "de", "'de'"},
		{"workflow", "customer_service", "'customer_service'"},
		{"path", "/tmp/ml", "'/tmp/ml'"},
		{"has-apostrophe", "it's", "\"it's\""},
		{"has-double-quote", "say \"hi\"", "'say \"hi\"'"},
		{"has-both", "both's \"x\"", "'both\\'s \"x\"'"},
		{"only-apostrophes", "'''", "\"'''\""},
		{"quote-then-apostrophe", "\"'", "'\"\\''"},
		{"backslash", "a\\b", "'a\\\\b'"},
		{"newline", "a\x0ab", "'a\\nb'"},
		{"carriage-return", "a\x0db", "'a\\rb'"},
		{"tab", "a\x09b", "'a\\tb'"},
		{"bell", "a\x07b", "'a\\x07b'"},
		{"backspace", "a\x08b", "'a\\x08b'"},
		{"formfeed", "a\x0cb", "'a\\x0cb'"},
		{"vtab", "a\x0b", "'a\\x0b'"},
		{"nul", "a\x00b", "'a\\x00b'"},
		{"esc", "a\x1bb", "'a\\x1bb'"},
		{"del", "a\x7fb", "'a\\x7fb'"},
		{"c1-nel", "a\u0085b", "'a\\x85b'"},
		{"nbsp", "a\u00a0b", "'a\\xa0b'"},
		{"soft-hyphen", "a\u00adb", "'a\\xadb'"},
		{"non-ascii-letter", "ünïcode", "'ünïcode'"},
		{"cjk", "日本語", "'日本語'"},
		{"emoji", "😀", "'😀'"},
		{"line-sep", "a\u2028b", "'a\\u2028b'"},
		{"para-sep", "a\u2029b", "'a\\u2029b'"},
		{"zwsp", "a\u200bb", "'a\\u200bb'"},
		{"ideographic-space", "a\u3000b", "'a\\u3000b'"},
		{"private-use", "a\ue000b", "'a\\ue000b'"},
		{"astral-nonprint", "a\U000e0001b", "'a\\U000e0001b'"},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := ReprString(c.in); got != c.want {
				t.Errorf("ReprString(%q) = %s, python repr = %s", c.in, got, c.want)
			}
		})
	}
}

// strconv.Quote is the reflex that would be reached for here, and it disagrees
// on the delimiter and on Go's \a\b\f\v shorthands. A reason string built with
// it would still read plausibly, which is why this is pinned rather than
// commented.
//
// "it's" is deliberately absent: it is the one shape where the two agree, since
// Python switches to a double quote for exactly the reason Go always uses one.
func TestReprStringIsNotStrconvQuote(t *testing.T) {
	for _, s := range []string{"multilingual", "ünïcode", "a\x07b", "a\x0bb"} {
		if ReprString(s) == strconv.Quote(s) {
			t.Errorf("ReprString(%q) agrees with strconv.Quote; one of them is not doing its job", s)
		}
	}
}

// Python strings are sequences of code points and cannot hold an unpaired byte,
// so there is no upstream behaviour to copy here. Escaping the raw byte keeps
// the output round-trippable and keeps a mangled model name from smuggling a
// quote into a reason string.
func TestReprStringEscapesInvalidUTF8(t *testing.T) {
	got := ReprString("a\xffb")
	if want := "'a\\xffb'"; got != want {
		t.Errorf("ReprString of an invalid byte = %s, want %s", got, want)
	}
}
